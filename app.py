"""
Smart Attendance System — Flask Backend (v3)
Three-tier access control:
  admin       → Full system control
  dept_head   → Manages dept: faculty, timetable, sessions, leave/sub
  faculty     → Calendar view of sessions, mark attendance

Persistent store: system_db.json
Attendance log:   attendance_log.csv
"""

from flask import Flask, Response, jsonify, request, session
import cv2, numpy as np, csv, json, datetime, threading, time, os, hashlib, io, base64
import google.generativeai as genai

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️  insightface not installed – face recognition disabled (demo mode)")

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR = "known_faces"
CSV_FILE        = "attendance_log.csv"
DB_FILE         = "system_db.json"
SECRET_KEY      = "your-random-flask-secret-2025"
GOOGLE_API_KEY  = "AIzaSyBpElFLwdXIj0nROmuWGxtCgKPLk0igtIM"

DEPARTMENTS = ["CSE", "ECE", "EEE", "MECH", "CIVIL"]
ALL_SECTIONS = {
    "CSE":   [f"CSE-{i}" for i in range(1, 10)],
    "ECE":   [f"ECE-{i}" for i in range(1, 7)],
    "EEE":   [f"EEE-{i}" for i in range(1, 5)],
    "MECH":  [f"MECH-{i}" for i in range(1, 5)],
    "CIVIL": [f"CIVIL-{i}" for i in range(1, 4)],
}

# Timetable slots 9:30–4:30, each 50 mins
TIME_SLOTS = [
    {"id": "slot1",  "label": "Period 1",  "start": "09:30", "end": "10:20"},
    {"id": "slot2",  "label": "Period 2",  "start": "10:20", "end": "11:10"},
    {"id": "slot3",  "label": "Period 3",  "start": "11:10", "end": "12:00"},
    {"id": "slot4",  "label": "Lunch",     "start": "12:00", "end": "12:40"},
    {"id": "slot5",  "label": "Period 4",  "start": "12:40", "end": "13:30"},
    {"id": "slot6",  "label": "Period 5",  "start": "13:30", "end": "14:20"},
    {"id": "slot7",  "label": "Period 6",  "start": "14:20", "end": "15:10"},
    {"id": "slot8",  "label": "Period 7",  "start": "15:10", "end": "16:00"},
    {"id": "slot9",  "label": "Period 8",  "start": "16:00", "end": "16:30"},
]
BREAK_SLOTS = {"slot4"}

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

CAMERA_SOURCES = [
    {
        "id": "default",
        "label": "Hikvision CCTV",
        "source": "rtsp://admin:Thub%40project@192.168.11.9:554/Streaming/Channels/101"
    },
]

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
genai.configure(api_key=GOOGLE_API_KEY)
app.secret_key = SECRET_KEY

# ── In-memory state ───────────────────────────────────────────────────────────
frame_lock    = threading.Lock()
camera_frames = {}
_state_lock   = threading.Lock()
attendance_today = {}

face_app         = None
known_embeddings = {}

# ═══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
def _default_db():
    return {
        "users": {
            "admin": {
                "password_hash": _hash("admin123"),
                "role": "admin",
                "department": None,
                "name": "System Administrator",
                "sections": []
            }
        },
        "timetable": {},
        "students":  {s: [] for dept in ALL_SECTIONS for s in ALL_SECTIONS[dept]},
        "camera_map": {s: "default" for dept in ALL_SECTIONS for s in ALL_SECTIONS[dept]},
        "sessions": [],
        "attendance_records": {},
    }

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def load_db() -> dict:
    default = _default_db()
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE) as f:
                saved = json.load(f)
            for k, v in default.items():
                if k not in saved:
                    saved[k] = v
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        saved[k].setdefault(k2, v2)
            return saved
        except Exception as e:
            print(f"⚠️  DB load error: {e} — using default")
    return default

def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

db = load_db()

# ── CSV ───────────────────────────────────────────────────────────────────────
def setup_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Roll", "Name", "Section", "Department", "Slot", "Subject",
                 "Faculty", "Date", "Time", "Status"])

def write_attendance_csv(roll, name, section, dept, slot_id, subject, faculty, status="Present"):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow(
            [roll, name, section, dept, slot_id, subject, faculty, date_str, time_str, status])

# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION / TIMETABLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def current_slot_id():
    now = datetime.datetime.now().strftime("%H:%M")
    for slot in TIME_SLOTS:
        if slot["start"] <= now < slot["end"]:
            return slot["id"]
    return None

def current_day() -> str:
    return datetime.datetime.now().strftime("%A")

def generate_sessions_for_week(faculty_username: str, dept: str):
    tt = db["timetable"].get(faculty_username, {})
    if not tt:
        return
    today = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday())
    day_offsets = {d: i for i, d in enumerate(DAYS)}

    existing_keys = set()
    for s in db["sessions"]:
        existing_keys.add(f"{s['faculty_username']}::{s['date']}::{s['slot_id']}::{s['section']}")

    new_sessions = []
    for day_name, slots in tt.items():
        offset = day_offsets.get(day_name)
        if offset is None:
            continue
        date_obj = monday + datetime.timedelta(days=offset)
        date_str = date_obj.isoformat()
        for slot_id, entry in slots.items():
            if slot_id in BREAK_SLOTS:
                continue
            section = entry.get("section", "")
            subject  = entry.get("subject", "")
            if not section:
                continue
            key = f"{faculty_username}::{date_str}::{slot_id}::{section}"
            if key not in existing_keys:
                slot_info = next((s for s in TIME_SLOTS if s["id"] == slot_id), {})
                new_sessions.append({
                    "id": f"{faculty_username}_{date_str}_{slot_id}_{section}",
                    "date": date_str,
                    "section": section,
                    "slot_id": slot_id,
                    "slot_label": slot_info.get("label", slot_id),
                    "slot_start": slot_info.get("start", ""),
                    "slot_end": slot_info.get("end", ""),
                    "subject": subject,
                    "faculty_username": faculty_username,
                    "dept": dept,
                    "status": "scheduled",
                    "substitute": None,
                    "att_key": f"{section}::{slot_id}::{date_str}",
                })
                existing_keys.add(key)

    if new_sessions:
        db["sessions"].extend(new_sessions)
        save_db()

def get_faculty_sessions(faculty_username: str, week_offset: int = 0):
    today = datetime.date.today()
    monday = today - datetime.timedelta(days=today.weekday()) + datetime.timedelta(weeks=week_offset)
    saturday = monday + datetime.timedelta(days=5)
    result = []
    for s in db["sessions"]:
        if s.get("faculty_username") != faculty_username and s.get("substitute") != faculty_username:
            continue
        try:
            d = datetime.date.fromisoformat(s["date"])
        except Exception:
            continue
        if monday <= d <= saturday:
            result.append(s)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
#  PHOTO HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def get_student_photo_b64(roll: str):
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(KNOWN_FACES_DIR, roll + ext)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = base64.b64encode(f.read()).decode()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
                return f"data:{mime};base64,{data}"
            except Exception:
                pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
def camera_loop(cam_id: str, source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {cam_id}")
        return
    print(f"📷 Camera ready: {cam_id}")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        if INSIGHTFACE_AVAILABLE and face_app:
            try:
                for face in face_app.get(frame):
                    x1,y1,x2,y2 = face.bbox.astype(int)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,80),2)
            except Exception:
                pass
        with frame_lock:
            camera_frames[cam_id] = frame.copy()
        time.sleep(0.03)

def gen_frames(cam_id: str):
    while True:
        with frame_lock:
            frame = camera_frames.get(cam_id)
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.04)

# ═══════════════════════════════════════════════════════════════════════════════
#  FACE RECOGNITION
# ═══════════════════════════════════════════════════════════════════════════════
def init_face_model():
    global face_app
    if not INSIGHTFACE_AVAILABLE:
        return
    print("🤖 Loading InsightFace …")
    face_app = FaceAnalysis(name="buffalo_l",
                             providers=["CoreMLExecutionProvider","CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640,640))
    print("✓ InsightFace ready")

def load_known_faces():
    if not (INSIGHTFACE_AVAILABLE and face_app):
        return
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    for fname in os.listdir(KNOWN_FACES_DIR):
        if not fname.lower().endswith((".png",".jpg",".jpeg")):
            continue
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, fname))
        if img is None:
            continue
        try:
            faces = face_app.get(img)
            if faces:
                name = os.path.splitext(fname)[0]
                known_embeddings[name] = faces[0].embedding
                print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ❌ {fname}: {e}")

def cosine_dist(e1, e2):
    return 1 - np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-9)

def match_face(emb):
    best, dist = None, float("inf")
    for name, ref in known_embeddings.items():
        d = cosine_dist(emb, ref)
        if d < dist:
            dist, best = d, name
    return best, dist

def liveness(f1, f2):
    if not (INSIGHTFACE_AVAILABLE and face_app):
        return True, 0.5
    try:
        fa1,fa2 = face_app.get(f1), face_app.get(f2)
        if not fa1 or not fa2:
            return False, 0.0
        lm1,lm2 = fa1[0].landmark_2d_106, fa2[0].landmark_2d_106
        if lm1 is None or lm2 is None:
            return True, 0.5
        pts=[28,30,35,40,52,64]
        ds=[np.linalg.norm(lm2[i]-lm1[i]) for i in pts if i<len(lm1)]
        avg=float(np.mean(ds)) if ds else 0.0
        return avg>1.5, avg
    except Exception:
        return True, 0.5

def verify(cam_id: str, section: str, slot_id: str, dept: str, faculty: str) -> dict:
    TOLE = 0.6
    if not (INSIGHTFACE_AVAILABLE and face_app):
        return {"success": False, "message": "Face recognition not available"}
    with frame_lock:
        f1 = camera_frames.get(cam_id)
    if f1 is None:
        return {"success": False, "message": "No camera frame"}
    f1 = f1.copy()
    time.sleep(3)
    with frame_lock:
        f2 = camera_frames.get(cam_id)
    if f2 is None:
        return {"success": False, "message": "Camera frame lost"}
    f2 = f2.copy()

    fa1, fa2 = face_app.get(f1), face_app.get(f2)
    if not fa1: return {"success": False, "message": "No face detected"}
    if not fa2: return {"success": False, "message": "Face lost between frames"}
    if not known_embeddings: return {"success": False, "message": "No faces in database"}

    alive, mv = liveness(f1, f2)
    mc, md = {}, {}
    for ffs in (fa1, fa2):
        for face in ffs:
            n, d = match_face(face.embedding)
            if n and d < TOLE:
                mc[n] = mc.get(n,0)+1
                md.setdefault(n,[]).append(d)

    confirmed = [n for n,c in mc.items() if c>=2 and alive]
    if not confirmed:
        return {"success": False, "message": "No confident match (liveness or threshold failed)"}

    date_str = datetime.date.today().isoformat()
    stu_map = {s["name"]: s["roll"] for s in db["students"].get(section, [])}
    key = f"{section}::{slot_id}::{date_str}"
    marked_new, already = [], []

    with _state_lock:
        att_recs = db["attendance_records"].setdefault(key, [])
        existing_rolls = {r["roll"] for r in att_recs}
        now = datetime.datetime.now().strftime("%H:%M:%S")
        for name in confirmed:
            roll = stu_map.get(name, name)
            if roll not in existing_rolls:
                att_recs.append({"roll": roll, "name": name, "status": "Present",
                                 "time": now, "marked_by": "face"})
                existing_rolls.add(roll)
                write_attendance_csv(roll, name, section, dept, slot_id, "", faculty)
                marked_new.append(name)
            else:
                already.append(name)

    save_db()
    return {
        "success": True,
        "marked": sorted(marked_new),
        "already_marked": sorted(already),
        "message": f"Marked: {', '.join(sorted(marked_new))}" if marked_new else "Already recorded"
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def me():
    return db["users"].get(session.get("username"), {})

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — AUTH
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/login", methods=["POST"])
def api_login():
    d = request.get_json(silent=True) or {}
    uname = d.get("username","").strip()
    pwd   = d.get("password","")
    user  = db["users"].get(uname)
    if not user or user["password_hash"] != _hash(pwd):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
    session.update(logged_in=True, username=uname,
                   role=user["role"], department=user.get("department"),
                   name=user.get("name",""))
    return jsonify({
        "success":    True,
        "role":       user["role"],
        "department": user.get("department"),
        "name":       user.get("name",""),
        "sections":   user.get("sections", []),
        "username":   uname,
    })

@app.route("/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/check_session")
def api_check_session():
    u = me()
    if not u:
        return jsonify({"logged_in": False})
    return jsonify({
        "logged_in":  True,
        "role":       u.get("role"),
        "department": u.get("department"),
        "name":       u.get("name",""),
        "sections":   u.get("sections",[]),
        "username":   session.get("username"),
    })

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/cameras")
def api_cameras():
    if not me():
        return jsonify({"cameras": []}), 401
    return jsonify({"cameras": CAMERA_SOURCES})

@app.route("/video_feed")
def video_feed():
    if not me():
        return Response("Unauthorised", status=401)
    cam_id = request.args.get("cam", "default")
    return Response(gen_frames(cam_id),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — FACULTY
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/faculty/calendar")
def faculty_calendar():
    u = me()
    if not u:
        return jsonify({"sessions": []}), 401
    week_offset = int(request.args.get("week", 0))
    uname = session.get("username")
    sessions = get_faculty_sessions(uname, week_offset)
    for s in sessions:
        key = s.get("att_key","")
        recs = db["attendance_records"].get(key, [])
        total = len(db["students"].get(s["section"], []))
        present = sum(1 for r in recs if r["status"] == "Present")
        s["att_count"] = present
        s["total_students"] = total
        s["att_marked"] = len(recs) > 0
        # Resolve original faculty display name (used when this session is a substitute)
        orig_user = db["users"].get(s.get("faculty_username",""), {})
        s["faculty_name"] = orig_user.get("name", s.get("faculty_username",""))
        # Resolve substitute display name
        if s.get("substitute"):
            sub_user = db["users"].get(s["substitute"], {})
            s["substitute_name"] = sub_user.get("name", s["substitute"])
        else:
            s["substitute_name"] = None
    return jsonify({"sessions": sessions, "time_slots": TIME_SLOTS})

@app.route("/faculty/session_detail")
def faculty_session_detail():
    u = me()
    if not u:
        return jsonify({}), 401
    session_id = request.args.get("id","")
    sess = next((s for s in db["sessions"] if s["id"] == session_id), None)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    uname = session.get("username")
    if sess["faculty_username"] != uname and sess.get("substitute") != uname:
        if u.get("role") not in ("admin","dept_head"):
            return jsonify({"error": "Forbidden"}), 403
    key = sess.get("att_key","")
    recs = db["attendance_records"].get(key, [])
    all_students = db["students"].get(sess["section"], [])
    present_rolls = {r["roll"] for r in recs if r["status"] == "Present"}
    student_list = []
    for stu in all_students:
        student_list.append({
            "roll": stu["roll"],
            "name": stu["name"],
            "status": "Present" if stu["roll"] in present_rolls else "Absent",
            "photo": get_student_photo_b64(stu["roll"]),
        })
    return jsonify({
        "session": sess,
        "students": student_list,
        "present_count": len(present_rolls),
        "total": len(all_students),
    })

@app.route("/faculty/mark_attendance", methods=["POST"])
def faculty_mark_attendance():
    u = me()
    if not u:
        return jsonify({"success": False, "message": "Not logged in"}), 401
    d       = request.get_json(silent=True) or {}
    section = d.get("section","")
    slot_id = d.get("slot_id","")
    cam_id  = d.get("cam_id","default")
    dept    = u.get("department","")
    result  = verify(cam_id, section, slot_id, dept, u.get("name",""))
    return jsonify(result)

@app.route("/faculty/save_attendance", methods=["POST"])
def faculty_save_attendance():
    u = me()
    if not u:
        return jsonify({"success": False}), 401
    d        = request.get_json(silent=True) or {}
    section  = d.get("section","")
    slot_id  = d.get("slot_id","")
    date_str = d.get("date", datetime.date.today().isoformat())
    records  = d.get("records",[])
    dept     = u.get("department","")

    key = f"{section}::{slot_id}::{date_str}"
    with _state_lock:
        db["attendance_records"][key] = records
    save_db()
    for r in records:
        write_attendance_csv(r["roll"], r.get("name",""), section, dept,
                             slot_id, "", u.get("name",""), r.get("status","Present"))
    return jsonify({"success": True, "saved": len(records)})

@app.route("/faculty/attendance_list")
def faculty_attendance_list():
    if not me():
        return jsonify({"records": []}), 401
    section  = request.args.get("section","")
    slot_id  = request.args.get("slot","")
    date_str = request.args.get("date", datetime.date.today().isoformat())
    key = f"{section}::{slot_id}::{date_str}"
    recs = db["attendance_records"].get(key, [])
    for r in recs:
        if "photo" not in r:
            r["photo"] = get_student_photo_b64(r["roll"])
    return jsonify({"records": recs, "count": len(recs)})

@app.route("/faculty/slot_info")
def faculty_slot_info():
    if not me():
        return jsonify({}), 401
    sid = current_slot_id()
    day = current_day()
    return jsonify({
        "slot_id":    sid,
        "day":        day,
        "slot_label": next((s["label"] for s in TIME_SLOTS if s["id"]==sid), "—"),
        "slot_time":  next((f"{s['start']}–{s['end']}" for s in TIME_SLOTS if s["id"]==sid), ""),
        "time_slots": TIME_SLOTS,
    })

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — DEPT HEAD
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/dept/faculty_list")
def dept_faculty_list():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    dept = u["department"] if u["role"]=="dept_head" else request.args.get("dept","CSE")
    users = [
        {"username": un, "name": ud["name"], "role": ud["role"],
         "sections": ud.get("sections",[]), "department": ud.get("department")}
        for un, ud in db["users"].items()
        if ud.get("department") == dept and ud["role"] in ("faculty","dept_head")
    ]
    return jsonify({"faculty": users, "sections": ALL_SECTIONS.get(dept,[])})

@app.route("/dept/create_faculty", methods=["POST"])
def dept_create_faculty():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    uname    = d.get("username","").strip()
    password = d.get("password","")
    name     = d.get("name","")
    role     = d.get("role","faculty")
    dept     = d.get("department","")
    sections = d.get("sections",[])

    if u["role"]=="dept_head":
        dept = u["department"]
        if role == "admin":
            return jsonify({"error":"Cannot create admin"}),403
    if not uname or not password:
        return jsonify({"error":"Username and password required"}),400
    if uname in db["users"]:
        return jsonify({"error":"Username already exists"}),409

    db["users"][uname] = {
        "password_hash": _hash(password),
        "role": role,
        "department": dept,
        "name": name or uname,
        "sections": sections,
    }
    save_db()
    return jsonify({"success": True})

@app.route("/dept/update_faculty", methods=["POST"])
def dept_update_faculty():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    uname = d.get("username","")
    if uname not in db["users"]:
        return jsonify({"error":"User not found"}),404
    target = db["users"][uname]
    if u["role"]=="dept_head" and target.get("department")!=u["department"]:
        return jsonify({"error":"Cross-department forbidden"}),403
    if "name"     in d: target["name"]     = d["name"]
    if "sections" in d: target["sections"] = d["sections"]
    if "role"     in d and u["role"]=="admin": target["role"] = d["role"]
    if "password" in d and d["password"]: target["password_hash"] = _hash(d["password"])
    save_db()
    return jsonify({"success": True})

@app.route("/dept/delete_faculty", methods=["POST"])
def dept_delete_faculty():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    uname = d.get("username","")
    if uname == "admin":
        return jsonify({"error":"Cannot delete root admin"}),403
    if uname not in db["users"]:
        return jsonify({"error":"Not found"}),404
    del db["users"][uname]
    save_db()
    return jsonify({"success": True})

# ── Timetable ──────────────────────────────────────────────────────────────────
@app.route("/dept/timetable")
def dept_timetable():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    dept = u["department"] if u["role"]=="dept_head" else request.args.get("dept","CSE")
    faculty_username = request.args.get("faculty","")
    day = request.args.get("day","Monday")
    tt = db["timetable"].get(faculty_username, {}).get(day, {}) if faculty_username else {}
    faculty_list = [
        {"username": un, "name": ud["name"]}
        for un, ud in db["users"].items()
        if ud.get("department")==dept and ud["role"] in ("faculty","dept_head")
    ]
    return jsonify({
        "timetable": tt,
        "time_slots": TIME_SLOTS,
        "days": DAYS,
        "sections": ALL_SECTIONS.get(dept,[]),
        "cameras": CAMERA_SOURCES,
        "faculty_list": faculty_list,
    })

@app.route("/dept/timetable_full")
def dept_timetable_full():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    faculty_username = request.args.get("faculty","")
    if not faculty_username:
        return jsonify({"error":"faculty required"}),400
    tt = db["timetable"].get(faculty_username, {})
    return jsonify({"timetable": tt, "time_slots": TIME_SLOTS, "days": DAYS})

@app.route("/dept/save_timetable", methods=["POST"])
def dept_save_timetable():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    faculty_username = d.get("faculty_username","")
    day     = d.get("day","Monday")
    entries = d.get("entries",{})

    if not faculty_username:
        return jsonify({"error":"faculty_username required"}),400

    db["timetable"].setdefault(faculty_username, {})
    db["timetable"][faculty_username][day] = entries
    save_db()

    faculty_user = db["users"].get(faculty_username, {})
    dept = faculty_user.get("department", u.get("department",""))
    generate_sessions_for_week(faculty_username, dept)

    return jsonify({"success": True})

@app.route("/dept/upload_timetable_excel", methods=["POST"])
def dept_upload_timetable_excel():
    """
    Bulk-import timetable for ALL faculty from a single Excel file.
    Expected columns (case-insensitive):
        Faculty - faculty username (required)
        Day     - Monday ... Saturday (required)
        Period  - slot label OR slot id e.g. 'Period 1'/'slot1' (required)
        Section - section code e.g. 'CSE-3' (optional)
        Subject - subject name (optional)
        Camera  - camera id (optional, default 'default')
    """
    u = me()
    if not u or u["role"] not in ("admin", "dept_head"):
        return jsonify({"error": "Forbidden"}), 403
    if not OPENPYXL_AVAILABLE:
        return jsonify({"error": "openpyxl not installed"}), 500
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    if not file.filename.lower().endswith(".xlsx"):
        return jsonify({"error": "Only .xlsx files accepted"}), 400
    slot_by_label = {s["label"].lower(): s["id"] for s in TIME_SLOTS}
    slot_by_id    = {s["id"]: s["id"] for s in TIME_SLOTS}
    try:
        wb = openpyxl.load_workbook(file)
        ws = wb.active
        raw_headers = [str(c.value).strip().lower() if c.value else "" for c in ws[1]]
        def col(keyword):
            return next((i for i, h in enumerate(raw_headers) if keyword in h), None)
        fac_col  = col("faculty")
        day_col  = col("day")
        per_col  = next((i for i, h in enumerate(raw_headers) if "period" in h or "slot" in h), None)
        sec_col  = col("section")
        subj_col = col("subject")
        cam_col  = col("camera")
        if fac_col is None or day_col is None or per_col is None:
            return jsonify({"error": "Required columns missing. Need: Faculty, Day, Period (or Slot)"}), 400
        dept_for_check = u.get("department") if u["role"] == "dept_head" else None
        updated_faculty = set()
        skipped = []
        for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            fac_uname = str(row[fac_col]).strip() if row[fac_col] else ""
            day_val   = str(row[day_col]).strip().capitalize() if row[day_col] else ""
            period_raw= str(row[per_col]).strip() if row[per_col] else ""
            section   = str(row[sec_col]).strip() if sec_col is not None and row[sec_col] else ""
            subject   = str(row[subj_col]).strip() if subj_col is not None and row[subj_col] else ""
            cam_id    = str(row[cam_col]).strip() if cam_col is not None and row[cam_col] else "default"
            if not fac_uname or fac_uname.lower() == "none":
                continue
            if day_val not in DAYS:
                skipped.append(f"Row {row_idx}: unknown day '{day_val}'")
                continue
            slot_id = slot_by_id.get(period_raw.lower()) or slot_by_label.get(period_raw.lower())
            if not slot_id:
                skipped.append(f"Row {row_idx}: unknown period '{period_raw}'")
                continue
            if slot_id in BREAK_SLOTS:
                continue
            fac_user = db["users"].get(fac_uname)
            if not fac_user:
                skipped.append(f"Row {row_idx}: faculty '{fac_uname}' not found")
                continue
            if dept_for_check and fac_user.get("department") != dept_for_check:
                skipped.append(f"Row {row_idx}: faculty '{fac_uname}' not in your dept")
                continue
            db["timetable"].setdefault(fac_uname, {}).setdefault(day_val, {})
            db["timetable"][fac_uname][day_val][slot_id] = {
                "section": section, "subject": subject, "cam_id": cam_id,
            }
            updated_faculty.add(fac_uname)
        for fac_uname in updated_faculty:
            fac_user = db["users"].get(fac_uname, {})
            dept = fac_user.get("department", u.get("department", ""))
            generate_sessions_for_week(fac_uname, dept)
        save_db()
        return jsonify({"success": True, "faculty_updated": len(updated_faculty), "skipped": skipped})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/dept/timetable_template_excel")
def dept_timetable_template_excel():
    """Download a pre-filled template Excel for bulk timetable upload."""
    u = me()
    if not u or u["role"] not in ("admin", "dept_head"):
        return Response("Forbidden", status=403)
    if not OPENPYXL_AVAILABLE:
        return Response("openpyxl not installed", status=500)
    dept = u["department"] if u["role"] == "dept_head" else "CSE"
    faculty_list = [
        (un, ud["name"])
        for un, ud in db["users"].items()
        if ud.get("department") == dept and ud["role"] in ("faculty", "dept_head")
    ]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Timetable"
    ws.append(["Faculty", "Day", "Period", "Section", "Subject", "Camera"])
    period_slots = [s for s in TIME_SLOTS if s["id"] not in BREAK_SLOTS]
    for fac_un, _ in faculty_list:
        for day in DAYS:
            for slot in period_slots:
                ws.append([fac_un, day, slot["label"], "", "", "default"])
    ws2 = wb.create_sheet("Instructions")
    ws2.append(["Column", "Description", "Valid Values"])
    ws2.append(["Faculty", "Faculty username (login ID)", "Any existing faculty username"])
    ws2.append(["Day", "Day of week", ", ".join(DAYS)])
    ws2.append(["Period", "Period label", ", ".join(s["label"] for s in period_slots)])
    ws2.append(["Section", "Class section", "e.g. CSE-1, ECE-3"])
    ws2.append(["Subject", "Subject name", "Free text"])
    ws2.append(["Camera", "Camera ID (optional)", "default"])
    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return Response(
        out.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=timetable_template_{dept}.xlsx"}
    )


# ── Sessions ──────────────────────────────────────────────────────────────────
@app.route("/dept/sessions")
def dept_sessions():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    dept = u["department"] if u["role"]=="dept_head" else request.args.get("dept","CSE")
    section = request.args.get("section","")
    date_filter = request.args.get("date","")
    sessions = [
        s for s in db["sessions"]
        if s.get("dept")==dept
        and (not section or s.get("section")==section)
        and (not date_filter or s.get("date")==date_filter)
    ]
    for s in sessions:
        key = s.get("att_key","")
        recs = db["attendance_records"].get(key, [])
        total = len(db["students"].get(s["section"], []))
        present = sum(1 for r in recs if r["status"] == "Present")
        s["att_count"] = present
        s["total_students"] = total
        orig_user = db["users"].get(s.get("faculty_username",""), {})
        s["faculty_name"] = orig_user.get("name", s.get("faculty_username",""))
        if s.get("substitute"):
            sub_user = db["users"].get(s["substitute"], {})
            s["substitute_name"] = sub_user.get("name", s["substitute"])
        else:
            s["substitute_name"] = None
    return jsonify({"sessions": sessions})

@app.route("/dept/assign_substitute", methods=["POST"])
def dept_assign_substitute():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    session_id  = d.get("session_id","")
    substitute  = d.get("substitute","")

    sess = next((s for s in db["sessions"] if s["id"] == session_id), None)
    if not sess:
        return jsonify({"error":"Session not found"}),404

    sess["substitute"] = substitute
    save_db()
    return jsonify({"success": True})

# ── Analytics ─────────────────────────────────────────────────────────────────
@app.route("/dept/analytics")
def dept_analytics():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    dept    = u["department"] if u["role"]=="dept_head" else request.args.get("dept","CSE")
    section = request.args.get("section","")
    date    = request.args.get("date", datetime.date.today().isoformat())

    total_stu = len(db["students"].get(section,[]))
    summary = []
    for slot in TIME_SLOTS:
        if slot["id"] in BREAK_SLOTS:
            continue
        key = f"{section}::{slot['id']}::{date}"
        recs = db["attendance_records"].get(key, [])
        present = sum(1 for r in recs if r.get("status") == "Present")
        absent  = max(0, total_stu - present)
        summary.append({
            "slot_id":   slot["id"],
            "label":     slot["label"],
            "time":      f"{slot['start']}–{slot['end']}",
            "present":   present,
            "absent":    absent,
            "total":     total_stu,
            "pct":       round(present/total_stu*100,1) if total_stu else 0,
            "records":   recs,
        })
    return jsonify({"summary": summary, "section": section, "date": date, "dept": dept})

@app.route("/dept/attendance_section")
def dept_attendance_section():
    u = me()
    if not u or u["role"] not in ("admin","dept_head","faculty"):
        return jsonify({"error":"Forbidden"}),403
    section  = request.args.get("section","")
    slot_id  = request.args.get("slot","")
    date_str = request.args.get("date", datetime.date.today().isoformat())
    key = f"{section}::{slot_id}::{date_str}"
    recs = db["attendance_records"].get(key, [])
    all_students = db["students"].get(section, [])
    present_rolls = {r["roll"] for r in recs if r["status"] == "Present"}
    result = []
    for stu in all_students:
        result.append({
            "roll":   stu["roll"],
            "name":   stu["name"],
            "status": "Present" if stu["roll"] in present_rolls else "Absent",
            "photo":  get_student_photo_b64(stu["roll"]),
        })
    return jsonify({"records": result, "section": section, "slot_id": slot_id, "date": date_str})

@app.route("/dept/update_attendance", methods=["POST"])
def dept_update_attendance():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d        = request.get_json(silent=True) or {}
    section  = d.get("section","")
    slot_id  = d.get("slot_id","")
    date_str = d.get("date", datetime.date.today().isoformat())
    records  = d.get("records",[])

    try:
        att_date = datetime.date.fromisoformat(date_str)
        cutoff   = datetime.datetime.combine(att_date + datetime.timedelta(days=1),
                                              datetime.time(18, 0))
        if datetime.datetime.now() > cutoff:
            return jsonify({"error": "Attendance can only be modified until 6 PM the next day"}), 403
    except Exception:
        pass

    key = f"{section}::{slot_id}::{date_str}"
    db["attendance_records"][key] = records
    save_db()
    return jsonify({"success": True})

# ── Student search ────────────────────────────────────────────────────────────
@app.route("/dept/search_attendance")
def dept_search_attendance():
    u = me()
    if not u or u["role"] not in ("admin","dept_head","faculty"):
        return jsonify({"error":"Forbidden"}),403
    query   = request.args.get("q","").strip().lower()
    section = request.args.get("section","")
    dept    = u["department"] if u["role"] in ("dept_head","faculty") else request.args.get("dept","")

    results = {}
    for key, recs in db["attendance_records"].items():
        parts = key.split("::")
        if len(parts) != 3:
            continue
        sec, slot_id, date_str = parts
        if section and sec != section:
            continue
        for r in recs:
            roll = r.get("roll","").lower()
            name = r.get("name","").lower()
            if query and query not in roll and query not in name:
                continue
            results.setdefault(r["roll"], {
                "roll": r["roll"],
                "name": r.get("name",""),
                "photo": get_student_photo_b64(r["roll"]),
                "records": []
            })
            results[r["roll"]]["records"].append({
                "section": sec,
                "slot_id": slot_id,
                "date": date_str,
                "status": r.get("status","Present"),
            })

    return jsonify({"results": list(results.values())})

# ── Excel upload ──────────────────────────────────────────────────────────────
@app.route("/dept/upload_students_excel", methods=["POST"])
def dept_upload_students_excel():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    if not OPENPYXL_AVAILABLE:
        return jsonify({"error":"openpyxl not installed"}),500

    file    = request.files.get("file")
    section = request.form.get("section","")

    if not file:
        return jsonify({"error":"No file uploaded"}),400
    if not file.filename.lower().endswith(".xlsx"):
        return jsonify({"error":"Only .xlsx files accepted"}),400

    try:
        wb = openpyxl.load_workbook(file)
        ws = wb.active
        headers = [str(c.value).strip().lower() if c.value else "" for c in ws[1]]
        roll_col = next((i for i,h in enumerate(headers) if "roll" in h), None)
        name_col = next((i for i,h in enumerate(headers) if "name" in h), None)
        if roll_col is None:
            return jsonify({"error":"No 'Roll' column found in Excel"}),400

        students = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            roll = str(row[roll_col]).strip() if row[roll_col] else ""
            name = str(row[name_col]).strip() if (name_col is not None and row[name_col]) else roll
            if not roll or roll.lower() == "none":
                continue
            students.append({"roll": roll, "name": name})

        db["students"][section] = students
        save_db()
        return jsonify({"success": True, "count": len(students)})
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route("/dept/upload_excel", methods=["POST"])
def dept_upload_excel():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    if not OPENPYXL_AVAILABLE:
        return jsonify({"error":"openpyxl not installed"}),500

    file    = request.files.get("file")
    section = request.form.get("section","")
    slot_id = request.form.get("slot_id","")
    date_str = request.form.get("date", datetime.date.today().isoformat())
    dept    = u["department"] if u["role"]=="dept_head" else request.form.get("dept","")

    if not file:
        return jsonify({"error":"No file uploaded"}),400
    if not file.filename.lower().endswith(".xlsx"):
        return jsonify({"error":"Only .xlsx files accepted"}),400

    try:
        wb = openpyxl.load_workbook(file)
        ws = wb.active
        headers = [str(c.value).strip().lower() if c.value else "" for c in ws[1]]
        roll_col   = next((i for i,h in enumerate(headers) if "roll" in h), None)
        name_col   = next((i for i,h in enumerate(headers) if "name" in h), None)
        status_col = next((i for i,h in enumerate(headers) if "status" in h), None)
        if roll_col is None:
            return jsonify({"error":"No 'Roll' column found"}),400

        stu_map = {s["roll"]: s["name"] for s in db["students"].get(section,[])}
        key = f"{section}::{slot_id}::{date_str}"
        existing_rolls = {r["roll"] for r in db["attendance_records"].get(key,[])}
        db["attendance_records"].setdefault(key, [])
        count = 0
        now = datetime.datetime.now().strftime("%H:%M:%S")
        for row in ws.iter_rows(min_row=2, values_only=True):
            roll = str(row[roll_col]).strip() if row[roll_col] else ""
            if not roll or roll.lower() == "none":
                continue
            name   = str(row[name_col]).strip() if (name_col is not None and row[name_col]) else stu_map.get(roll, roll)
            status = str(row[status_col]).strip() if (status_col is not None and row[status_col]) else "Present"
            if roll not in existing_rolls:
                db["attendance_records"][key].append({"roll":roll,"name":name,"status":status,"time":now,"marked_by":"excel"})
                existing_rolls.add(roll)
                write_attendance_csv(roll, name, section, dept, slot_id, "", u.get("name",""), status)
                count += 1

        save_db()
        return jsonify({"success": True, "marked": count})
    except Exception as e:
        return jsonify({"error": str(e)}),500

# ── Students CRUD ─────────────────────────────────────────────────────────────
@app.route("/dept/students")
def dept_students():
    u = me()
    if not u or u["role"] not in ("admin","dept_head","faculty"):
        return jsonify({"error":"Forbidden"}),403
    section = request.args.get("section","")
    students = db["students"].get(section,[])
    result = []
    for s in students:
        result.append({**s, "photo": get_student_photo_b64(s["roll"])})
    return jsonify({"students": result})

@app.route("/dept/save_students", methods=["POST"])
def dept_save_students():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    section  = d.get("section","")
    students = d.get("students",[])
    db["students"][section] = students
    save_db()
    return jsonify({"success": True})

@app.route("/dept/add_student", methods=["POST"])
def dept_add_student():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    section = d.get("section","")
    roll    = d.get("roll","").strip()
    name    = d.get("name","").strip()
    if not section or not roll:
        return jsonify({"error":"section and roll required"}),400
    students = db["students"].setdefault(section, [])
    if any(s["roll"] == roll for s in students):
        return jsonify({"error":"Roll number already exists in this section"}),409
    students.append({"roll": roll, "name": name or roll})
    save_db()
    return jsonify({"success": True})

@app.route("/dept/remove_student", methods=["POST"])
def dept_remove_student():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return jsonify({"error":"Forbidden"}),403
    d = request.get_json(silent=True) or {}
    section = d.get("section","")
    roll    = d.get("roll","")
    if section in db["students"]:
        db["students"][section] = [s for s in db["students"][section] if s["roll"] != roll]
        save_db()
    return jsonify({"success": True})

# ── Download attendance ───────────────────────────────────────────────────────
@app.route("/dept/download_attendance")
def dept_download_attendance():
    u = me()
    if not u or u["role"] not in ("admin","dept_head","faculty"):
        return Response("Forbidden",status=403)
    if not OPENPYXL_AVAILABLE:
        return Response("openpyxl not installed",status=500)

    section = request.args.get("section","")
    date    = request.args.get("date", datetime.date.today().isoformat())

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{section} {date}"
    ws.append(["Roll", "Name", "Period 1", "Period 2", "Period 3",
               "Period 4", "Period 5", "Period 6", "Period 7", "Period 8"])

    all_students = db["students"].get(section, [])
    period_slots = [s for s in TIME_SLOTS if s["id"] not in BREAK_SLOTS]

    for stu in all_students:
        row = [stu["roll"], stu["name"]]
        for slot in period_slots:
            key = f"{section}::{slot['id']}::{date}"
            recs = db["attendance_records"].get(key, [])
            present_rolls = {r["roll"] for r in recs if r["status"] == "Present"}
            row.append("P" if stu["roll"] in present_rolls else "A")
        ws.append(row)

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    fname = f"attendance_{section}_{date}.xlsx"
    return Response(
        out.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )

@app.route("/dept/download_student_attendance")
def dept_download_student_attendance():
    u = me()
    if not u or u["role"] not in ("admin","dept_head","faculty"):
        return Response("Forbidden",status=403)
    if not OPENPYXL_AVAILABLE:
        return Response("openpyxl not installed",status=500)

    query   = request.args.get("q","").strip().lower()
    section = request.args.get("section","")

    results = {}
    for key, recs in db["attendance_records"].items():
        parts = key.split("::")
        if len(parts) != 3:
            continue
        sec, slot_id, date_str = parts
        if section and sec != section:
            continue
        slot_label = next((s["label"] for s in TIME_SLOTS if s["id"]==slot_id), slot_id)
        for r in recs:
            roll = r.get("roll","").lower()
            name = r.get("name","").lower()
            if query and query not in roll and query not in name:
                continue
            results.setdefault(r["roll"], {
                "roll": r["roll"], "name": r.get("name",""), "records": []
            })
            results[r["roll"]]["records"].append({
                "section": sec, "slot_id": slot_id, "slot_label": slot_label,
                "date": date_str, "status": r.get("status","Present"),
            })

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Student Attendance"
    ws.append(["Roll", "Name", "Section", "Date", "Period", "Status"])
    for stu in results.values():
        for rec in sorted(stu["records"], key=lambda x: (x["date"],x["slot_id"])):
            ws.append([stu["roll"], stu["name"], rec["section"], rec["date"], rec["slot_label"], rec["status"]])

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    fname = f"student_attendance_{query or 'all'}.xlsx"
    return Response(
        out.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )

# ── Admin routes ──────────────────────────────────────────────────────────────
@app.route("/admin/all_users")
def admin_all_users():
    u = me()
    if not u or u["role"] != "admin":
        return jsonify({"error":"Forbidden"}),403
    users = [
        {"username": un, **{k:v for k,v in ud.items() if k!="password_hash"}}
        for un, ud in db["users"].items()
    ]
    return jsonify({"users": users, "departments": DEPARTMENTS})

@app.route("/admin/export_csv")
def admin_export_csv():
    u = me()
    if not u or u["role"] not in ("admin","dept_head"):
        return Response("Forbidden",status=403)
    section = request.args.get("section","")
    dept    = u["department"] if u["role"]=="dept_head" else request.args.get("dept","")

    out_rows = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if section and row.get("Section")!=section:
                    continue
                if dept and row.get("Department")!=dept:
                    continue
                out_rows.append(row)

    out = io.StringIO()
    if out_rows:
        w = csv.DictWriter(out, fieldnames=out_rows[0].keys())
        w.writeheader(); w.writerows(out_rows)
    fname = f"attendance_{section or dept or 'all'}.csv"
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})

@app.route("/")
def index():
    try:
        with open("index.html") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    except FileNotFoundError:
        return "<h2>Put index.html alongside app.py</h2>", 200

# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    setup_csv()
    init_face_model()
    load_known_faces()
    for cam in CAMERA_SOURCES:
        threading.Thread(target=camera_loop, args=(cam["id"],cam["source"]), daemon=True).start()
    print("\n🌐  http://localhost:8080")
    print("🔑  Default admin login: admin / admin123\n")
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)