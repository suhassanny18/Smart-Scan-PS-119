"""
Unit tests for data models in the Anti-Cheat Detection System.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock

from anti_cheat_system.models.data_models import (
    BoundingBox, FaceDetection, ObjectDetection, StudentTrack, StudentIdentity,
    GazeAnalysis, TalkingAnalysis, PostureAnalysis, ActionAnalysis,
    BehaviorEvent, StudentState, Alert, EmailData, FrameMetadata,
    StreamStatus, SystemMetrics, TemporalScores
)
from anti_cheat_system.models.enums import (
    AlertLevel, AttendanceStatus, BehaviorEventType, LookingDirection,
    ActionType, StreamHealth, DEFAULT_BEHAVIOR_WEIGHTS
)


class TestBoundingBox:
    """Test BoundingBox data model."""
    
    def test_bounding_box_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 100.0
        assert bbox.y2 == 200.0
    
    def test_bounding_box_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        assert bbox.width == 90.0
        assert bbox.height == 180.0
        assert bbox.center == (55.0, 110.0)
        assert bbox.area == 16200.0
    
    def test_bounding_box_zero_area(self):
        """Test bounding box with zero area."""
        bbox = BoundingBox(x1=50.0, y1=50.0, x2=50.0, y2=50.0)
        assert bbox.area == 0.0


class TestFaceDetection:
    """Test FaceDetection data model."""
    
    def test_face_detection_creation(self):
        """Test basic face detection creation."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        face_crop = np.random.rand(180, 90, 3)
        landmarks = np.random.rand(468, 3)
        
        detection = FaceDetection(
            bbox=bbox,
            confidence=0.95,
            face_crop=face_crop,
            landmarks=landmarks
        )
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.95
        assert np.array_equal(detection.face_crop, face_crop)
        assert np.array_equal(detection.landmarks, landmarks)
        assert isinstance(detection.timestamp, datetime)
    
    def test_face_detection_without_optional_fields(self):
        """Test face detection without optional fields."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        detection = FaceDetection(bbox=bbox, confidence=0.85)
        
        assert detection.bbox == bbox
        assert detection.confidence == 0.85
        assert detection.face_crop is None
        assert detection.landmarks is None
    
    def test_face_detection_invalid_crop(self):
        """Test face detection with invalid face crop."""
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=100.0, y2=200.0)
        
        with pytest.raises(ValueError, match="face_crop must be a numpy array"):
            FaceDetection(bbox=bbox, confidence=0.85, face_crop="invalid")


class TestObjectDetection:
    """Test ObjectDetection data model."""
    
    def test_object_detection_creation(self):
        """Test basic object detection creation."""
        bbox = BoundingBox(x1=50.0, y1=60.0, x2=150.0, y2=160.0)
        
        detection = ObjectDetection(
            bbox=bbox,
            class_id=67,
            class_name="cell phone",
            confidence=0.88,
            is_suspicious=True
        )
        
        assert detection.bbox == bbox
        assert detection.class_id == 67
        assert detection.class_name == "cell phone"
        assert detection.confidence == 0.88
        assert detection.is_suspicious is True
        assert isinstance(detection.timestamp, datetime)


class TestStudentTrack:
    """Test StudentTrack data model."""
    
    def test_student_track_creation(self):
        """Test basic student track creation."""
        bbox = BoundingBox(x1=30.0, y1=40.0, x2=130.0, y2=240.0)
        last_seen = datetime.now()
        
        track = StudentTrack(
            track_id=1,
            bbox=bbox,
            confidence=0.92,
            age=15,
            last_seen=last_seen,
            is_confirmed=True,
            velocity=(2.5, -1.0)
        )
        
        assert track.track_id == 1
        assert track.bbox == bbox
        assert track.confidence == 0.92
        assert track.age == 15
        assert track.last_seen == last_seen
        assert track.is_confirmed is True
        assert track.velocity == (2.5, -1.0)
    
    def test_student_track_negative_age(self):
        """Test student track with negative age."""
        bbox = BoundingBox(x1=30.0, y1=40.0, x2=130.0, y2=240.0)
        
        with pytest.raises(ValueError, match="Track age cannot be negative"):
            StudentTrack(
                track_id=1,
                bbox=bbox,
                confidence=0.92,
                age=-5,
                last_seen=datetime.now()
            )


class TestStudentIdentity:
    """Test StudentIdentity data model."""
    
    def test_student_identity_creation(self):
        """Test basic student identity creation."""
        embedding = np.random.rand(512)
        
        identity = StudentIdentity(
            roll_number="CS2021001",
            name="John Doe",
            class_id="CS2021",
            confidence=0.95,
            embedding=embedding
        )
        
        assert identity.roll_number == "CS2021001"
        assert identity.name == "John Doe"
        assert identity.class_id == "CS2021"
        assert identity.confidence == 0.95
        assert np.array_equal(identity.embedding, embedding)
    
    def test_student_identity_empty_roll_number(self):
        """Test student identity with empty roll number."""
        with pytest.raises(ValueError, match="Roll number cannot be empty"):
            StudentIdentity(
                roll_number="",
                name="John Doe",
                class_id="CS2021",
                confidence=0.95
            )
    
    def test_student_identity_invalid_confidence(self):
        """Test student identity with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            StudentIdentity(
                roll_number="CS2021001",
                name="John Doe",
                class_id="CS2021",
                confidence=1.5
            )


class TestBehaviorAnalysisModels:
    """Test behavioral analysis data models."""
    
    def test_gaze_analysis_creation(self):
        """Test gaze analysis creation."""
        gaze = GazeAnalysis(
            yaw=15.5,
            pitch=-10.2,
            roll=2.1,
            looking_direction=LookingDirection.RIGHT,
            deviation_percentage=0.25,
            sustained_duration=3.5
        )
        
        assert gaze.yaw == 15.5
        assert gaze.pitch == -10.2
        assert gaze.roll == 2.1
        assert gaze.looking_direction == LookingDirection.RIGHT
        assert gaze.deviation_percentage == 0.25
        assert gaze.sustained_duration == 3.5
    
    def test_talking_analysis_creation(self):
        """Test talking analysis creation."""
        talking = TalkingAnalysis(
            mouth_open_ratio=0.15,
            lip_motion_detected=True,
            talking_confidence=0.85,
            duration=2.3
        )
        
        assert talking.mouth_open_ratio == 0.15
        assert talking.lip_motion_detected is True
        assert talking.talking_confidence == 0.85
        assert talking.duration == 2.3
    
    def test_posture_analysis_creation(self):
        """Test posture analysis creation."""
        posture = PostureAnalysis(
            shoulder_angle=25.0,
            is_leaning=True,
            lean_direction="right",
            proximity_to_others=[("CS2021002", 85.5), ("CS2021003", 120.0)]
        )
        
        assert posture.shoulder_angle == 25.0
        assert posture.is_leaning is True
        assert posture.lean_direction == "right"
        assert len(posture.proximity_to_others) == 2
        assert posture.proximity_to_others[0] == ("CS2021002", 85.5)
    
    def test_action_analysis_creation(self):
        """Test action analysis creation."""
        bbox = BoundingBox(x1=40.0, y1=50.0, x2=140.0, y2=150.0)
        
        action = ActionAnalysis(
            action_type=ActionType.HAND_TO_FACE,
            confidence=0.78,
            duration=1.5,
            bbox=bbox
        )
        
        assert action.action_type == ActionType.HAND_TO_FACE
        assert action.confidence == 0.78
        assert action.duration == 1.5
        assert action.bbox == bbox


class TestBehaviorEvent:
    """Test BehaviorEvent data model."""
    
    def test_behavior_event_creation(self):
        """Test basic behavior event creation."""
        timestamp = datetime.now()
        
        event = BehaviorEvent(
            event_type=BehaviorEventType.PHONE_DETECTED,
            confidence=0.92,
            timestamp=timestamp,
            duration=2.5,
            evidence_path="/evidence/screenshot_001.jpg",
            metadata={"frame_number": 1500, "camera_id": "cam_01"}
        )
        
        assert event.event_type == BehaviorEventType.PHONE_DETECTED
        assert event.confidence == 0.92
        assert event.timestamp == timestamp
        assert event.duration == 2.5
        assert event.evidence_path == "/evidence/screenshot_001.jpg"
        assert event.metadata["frame_number"] == 1500
    
    def test_behavior_event_invalid_confidence(self):
        """Test behavior event with invalid confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            BehaviorEvent(
                event_type=BehaviorEventType.TALKING,
                confidence=1.2,
                timestamp=datetime.now(),
                duration=1.0
            )
    
    def test_behavior_event_negative_duration(self):
        """Test behavior event with negative duration."""
        with pytest.raises(ValueError, match="Duration cannot be negative"):
            BehaviorEvent(
                event_type=BehaviorEventType.TALKING,
                confidence=0.8,
                timestamp=datetime.now(),
                duration=-1.0
            )


class TestStudentState:
    """Test StudentState data model."""
    
    def test_student_state_creation(self):
        """Test basic student state creation."""
        state = StudentState(
            roll_number="CS2021001",
            name="John Doe",
            class_id="CS2021",
            track_id=5,
            looking_away_frames=10,
            talking_frames=5,
            phone_detected_frames=2,
            suspicion_score=45.5,
            attendance_status=AttendanceStatus.PRESENT,
            alert_status=AlertLevel.WARNING
        )
        
        assert state.roll_number == "CS2021001"
        assert state.name == "John Doe"
        assert state.class_id == "CS2021"
        assert state.track_id == 5
        assert state.looking_away_frames == 10
        assert state.talking_frames == 5
        assert state.phone_detected_frames == 2
        assert state.suspicion_score == 45.5
        assert state.attendance_status == AttendanceStatus.PRESENT
        assert state.alert_status == AlertLevel.WARNING
    
    def test_student_state_empty_roll_number(self):
        """Test student state with empty roll number."""
        with pytest.raises(ValueError, match="Roll number cannot be empty"):
            StudentState(
                roll_number="",
                name="John Doe",
                class_id="CS2021"
            )
    
    def test_student_state_negative_suspicion_score(self):
        """Test student state with negative suspicion score."""
        with pytest.raises(ValueError, match="Suspicion score cannot be negative"):
            StudentState(
                roll_number="CS2021001",
                name="John Doe",
                class_id="CS2021",
                suspicion_score=-10.0
            )


class TestAlert:
    """Test Alert data model."""
    
    def test_alert_creation(self):
        """Test basic alert creation."""
        timestamp = datetime.now()
        
        alert = Alert(
            alert_id="alert_001",
            roll_number="CS2021001",
            student_name="John Doe",
            alert_level=AlertLevel.CRITICAL,
            alert_type="phone_detection",
            composite_score=95.5,
            contributing_behaviors=["phone_detected", "looking_away"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=timestamp,
            email_sent=True,
            email_recipients=["admin@school.edu", "proctor@school.edu"]
        )
        
        assert alert.alert_id == "alert_001"
        assert alert.roll_number == "CS2021001"
        assert alert.student_name == "John Doe"
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.alert_type == "phone_detection"
        assert alert.composite_score == 95.5
        assert alert.contributing_behaviors == ["phone_detected", "looking_away"]
        assert alert.evidence_screenshot == "/evidence/alert_001.jpg"
        assert alert.timestamp == timestamp
        assert alert.email_sent is True
        assert len(alert.email_recipients) == 2
    
    def test_alert_empty_id(self):
        """Test alert with empty ID."""
        with pytest.raises(ValueError, match="Alert ID cannot be empty"):
            Alert(
                alert_id="",
                roll_number="CS2021001",
                student_name="John Doe",
                alert_level=AlertLevel.CRITICAL,
                alert_type="phone_detection",
                composite_score=95.5,
                contributing_behaviors=["phone_detected"],
                evidence_screenshot="/evidence/alert_001.jpg",
                timestamp=datetime.now()
            )
    
    def test_alert_negative_score(self):
        """Test alert with negative composite score."""
        with pytest.raises(ValueError, match="Composite score cannot be negative"):
            Alert(
                alert_id="alert_001",
                roll_number="CS2021001",
                student_name="John Doe",
                alert_level=AlertLevel.CRITICAL,
                alert_type="phone_detection",
                composite_score=-10.0,
                contributing_behaviors=["phone_detected"],
                evidence_screenshot="/evidence/alert_001.jpg",
                timestamp=datetime.now()
            )


class TestEmailData:
    """Test EmailData data model."""
    
    def test_email_data_creation(self):
        """Test basic email data creation."""
        alert = Alert(
            alert_id="alert_001",
            roll_number="CS2021001",
            student_name="John Doe",
            alert_level=AlertLevel.CRITICAL,
            alert_type="phone_detection",
            composite_score=95.5,
            contributing_behaviors=["phone_detected"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=datetime.now()
        )
        
        email_data = EmailData(
            alert=alert,
            recipients=["admin@school.edu", "proctor@school.edu"],
            subject="Critical Alert: Phone Detected",
            body="A critical alert has been generated...",
            attachments=["/evidence/alert_001.jpg"],
            retry_count=1,
            max_retries=3
        )
        
        assert email_data.alert == alert
        assert len(email_data.recipients) == 2
        assert email_data.subject == "Critical Alert: Phone Detected"
        assert email_data.body == "A critical alert has been generated..."
        assert len(email_data.attachments) == 1
        assert email_data.retry_count == 1
        assert email_data.max_retries == 3
    
    def test_email_data_empty_recipients(self):
        """Test email data with empty recipients."""
        alert = Alert(
            alert_id="alert_001",
            roll_number="CS2021001",
            student_name="John Doe",
            alert_level=AlertLevel.CRITICAL,
            alert_type="phone_detection",
            composite_score=95.5,
            contributing_behaviors=["phone_detected"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=datetime.now()
        )
        
        with pytest.raises(ValueError, match="Recipients list cannot be empty"):
            EmailData(
                alert=alert,
                recipients=[],
                subject="Test Subject",
                body="Test Body"
            )


class TestSystemMetrics:
    """Test SystemMetrics data model."""
    
    def test_system_metrics_creation(self):
        """Test basic system metrics creation."""
        timestamp = datetime.now()
        
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_usage=65.5,
            memory_usage=78.2,
            gpu_usage=45.0,
            fps=18.5,
            active_tracks=12,
            alerts_generated=3,
            processing_latency=0.15
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 65.5
        assert metrics.memory_usage == 78.2
        assert metrics.gpu_usage == 45.0
        assert metrics.fps == 18.5
        assert metrics.active_tracks == 12
        assert metrics.alerts_generated == 3
        assert metrics.processing_latency == 0.15
    
    def test_system_metrics_invalid_cpu_usage(self):
        """Test system metrics with invalid CPU usage."""
        with pytest.raises(ValueError, match="CPU usage must be between 0 and 100"):
            SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=150.0,
                memory_usage=50.0,
                gpu_usage=None,
                fps=15.0,
                active_tracks=5,
                alerts_generated=1,
                processing_latency=0.1
            )
    
    def test_system_metrics_invalid_gpu_usage(self):
        """Test system metrics with invalid GPU usage."""
        with pytest.raises(ValueError, match="GPU usage must be between 0 and 100"):
            SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=50.0,
                memory_usage=50.0,
                gpu_usage=150.0,
                fps=15.0,
                active_tracks=5,
                alerts_generated=1,
                processing_latency=0.1
            )


class TestTemporalScores:
    """Test TemporalScores data model."""
    
    def test_temporal_scores_creation(self):
        """Test basic temporal scores creation."""
        scores = TemporalScores(
            looking_away_score=10.0,
            talking_score=15.0,
            phone_score=50.0,
            chit_score=0.0,
            action_score=20.0,
            missing_score=0.0,
            proximity_score=5.0
        )
        
        assert scores.looking_away_score == 10.0
        assert scores.talking_score == 15.0
        assert scores.phone_score == 50.0
        assert scores.chit_score == 0.0
        assert scores.action_score == 20.0
        assert scores.missing_score == 0.0
        assert scores.proximity_score == 5.0
        assert scores.composite_score == 0.0  # Not calculated yet
    
    def test_temporal_scores_calculate_composite(self):
        """Test composite score calculation."""
        scores = TemporalScores(
            looking_away_score=2.0,  # 2 * 5 = 10
            talking_score=1.0,       # 1 * 15 = 15
            phone_score=1.0,         # 1 * 50 = 50
            chit_score=0.0,          # 0 * 45 = 0
            action_score=1.0,        # 1 * 20 = 20
            missing_score=0.0,       # 0 * 25 = 0
            proximity_score=1.0      # 1 * 10 = 10
        )
        
        composite = scores.calculate_composite(DEFAULT_BEHAVIOR_WEIGHTS)
        expected = 10 + 15 + 50 + 0 + 20 + 0 + 10  # 105
        
        assert composite == expected
        assert scores.composite_score == expected
    
    def test_temporal_scores_custom_weights(self):
        """Test composite score calculation with custom weights."""
        scores = TemporalScores(
            looking_away_score=1.0,
            talking_score=1.0,
            phone_score=1.0,
            chit_score=1.0,
            action_score=1.0,
            missing_score=1.0,
            proximity_score=1.0
        )
        
        custom_weights = {
            BehaviorEventType.LOOKING_AWAY: 10,
            BehaviorEventType.TALKING: 20,
            BehaviorEventType.PHONE_DETECTED: 100,
            BehaviorEventType.CHIT_DETECTED: 90,
            BehaviorEventType.SUSPICIOUS_ACTION: 30,
            BehaviorEventType.MISSING: 40,
            BehaviorEventType.PROXIMITY_VIOLATION: 15,
        }
        
        composite = scores.calculate_composite(custom_weights)
        expected = 10 + 20 + 100 + 90 + 30 + 40 + 15  # 305
        
        assert composite == expected
        assert scores.composite_score == expected


class TestFrameMetadata:
    """Test FrameMetadata data model."""
    
    def test_frame_metadata_creation(self):
        """Test basic frame metadata creation."""
        timestamp = datetime.now()
        
        metadata = FrameMetadata(
            frame_id="frame_001",
            timestamp=timestamp,
            camera_id="cam_01",
            frame_number=1500,
            processing_time=0.05,
            detections_count=3,
            tracks_count=2
        )
        
        assert metadata.frame_id == "frame_001"
        assert metadata.timestamp == timestamp
        assert metadata.camera_id == "cam_01"
        assert metadata.frame_number == 1500
        assert metadata.processing_time == 0.05
        assert metadata.detections_count == 3
        assert metadata.tracks_count == 2
    
    def test_frame_metadata_negative_frame_number(self):
        """Test frame metadata with negative frame number."""
        with pytest.raises(ValueError, match="Frame number cannot be negative"):
            FrameMetadata(
                frame_id="frame_001",
                timestamp=datetime.now(),
                camera_id="cam_01",
                frame_number=-1,
                processing_time=0.05,
                detections_count=3,
                tracks_count=2
            )


class TestStreamStatus:
    """Test StreamStatus data model."""
    
    def test_stream_status_creation(self):
        """Test basic stream status creation."""
        last_frame_time = datetime.now()
        
        status = StreamStatus(
            stream_id="stream_01",
            url="rtsp://camera1.local/stream",
            health=StreamHealth.HEALTHY,
            last_frame_time=last_frame_time,
            fps=25.0,
            reconnect_attempts=0,
            error_message=None
        )
        
        assert status.stream_id == "stream_01"
        assert status.url == "rtsp://camera1.local/stream"
        assert status.health == StreamHealth.HEALTHY
        assert status.last_frame_time == last_frame_time
        assert status.fps == 25.0
        assert status.reconnect_attempts == 0
        assert status.error_message is None
    
    def test_stream_status_negative_fps(self):
        """Test stream status with negative FPS."""
        with pytest.raises(ValueError, match="FPS cannot be negative"):
            StreamStatus(
                stream_id="stream_01",
                url="rtsp://camera1.local/stream",
                health=StreamHealth.HEALTHY,
                last_frame_time=datetime.now(),
                fps=-5.0
            )
    
    def test_stream_status_negative_reconnect_attempts(self):
        """Test stream status with negative reconnect attempts."""
        with pytest.raises(ValueError, match="Reconnect attempts cannot be negative"):
            StreamStatus(
                stream_id="stream_01",
                url="rtsp://camera1.local/stream",
                health=StreamHealth.HEALTHY,
                last_frame_time=datetime.now(),
                fps=25.0,
                reconnect_attempts=-1
            )