"""
Core data models for the Anti-Cheat Detection System.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from .enums import (
    AlertLevel, AttendanceStatus, BehaviorEventType, 
    LookingDirection, ActionType, StreamHealth
)


@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: BoundingBox
    confidence: float
    face_crop: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if self.face_crop is not None and not isinstance(self.face_crop, np.ndarray):
            raise ValueError("face_crop must be a numpy array")


@dataclass
class ObjectDetection:
    """Object detection result."""
    bbox: BoundingBox
    class_id: int
    class_name: str
    confidence: float
    is_suspicious: bool
    timestamp: datetime = field(default_factory=datetime.now)
    object_type: Optional[str] = None  # Type of suspicious object


@dataclass
class StudentTrack:
    """Student tracking information."""
    track_id: int
    bbox: BoundingBox
    confidence: float
    age: int  # Number of frames this track has existed
    last_seen: datetime
    is_confirmed: bool = False
    velocity: Optional[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.age < 0:
            raise ValueError("Track age cannot be negative")


@dataclass
class StudentIdentity:
    """Student identity information."""
    roll_number: str
    name: str
    class_id: str
    confidence: float
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.roll_number:
            raise ValueError("Roll number cannot be empty")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class GazeAnalysis:
    """Gaze analysis result."""
    yaw: float
    pitch: float
    roll: float
    looking_direction: LookingDirection
    deviation_percentage: float
    sustained_duration: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TalkingAnalysis:
    """Talking detection result."""
    mouth_open_ratio: float
    lip_motion_detected: bool
    talking_confidence: float
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PostureAnalysis:
    """Posture analysis result."""
    shoulder_angle: float
    is_leaning: bool
    lean_direction: str
    proximity_to_others: List[Tuple[str, float]]  # (roll_number, distance)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActionAnalysis:
    """Suspicious action analysis result."""
    action_type: ActionType
    confidence: float
    duration: float
    bbox: BoundingBox
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BehaviorEvent:
    """Individual behavior event."""
    event_type: BehaviorEventType
    confidence: float
    timestamp: datetime
    duration: float
    evidence_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.duration < 0:
            raise ValueError("Duration cannot be negative")


@dataclass
class StudentState:
    """Complete student monitoring state."""
    roll_number: str
    name: str
    class_id: str
    track_id: Optional[int] = None
    
    # Behavioral counters
    looking_away_frames: int = 0
    talking_frames: int = 0
    phone_detected_frames: int = 0
    chit_detected_frames: int = 0
    suspicious_action_frames: int = 0
    missing_frames: int = 0
    proximity_violation_frames: int = 0
    
    # Temporal analysis
    suspicion_score: float = 0.0
    last_behavior_update: datetime = field(default_factory=datetime.now)
    last_recognition_time: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Status tracking
    attendance_status: AttendanceStatus = AttendanceStatus.UNKNOWN
    alert_status: AlertLevel = AlertLevel.NORMAL
    cooldown_until: Optional[datetime] = None
    
    # Evidence tracking
    recent_screenshots: List[str] = field(default_factory=list)
    behavior_history: List[BehaviorEvent] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.roll_number:
            raise ValueError("Roll number cannot be empty")
        if self.suspicion_score < 0:
            raise ValueError("Suspicion score cannot be negative")


@dataclass
class Alert:
    """Alert notification."""
    alert_id: str
    roll_number: str
    student_name: str
    alert_level: AlertLevel
    alert_type: str
    composite_score: float
    contributing_behaviors: List[str]
    evidence_screenshot: str
    timestamp: datetime
    email_sent: bool = False
    email_recipients: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.alert_id:
            raise ValueError("Alert ID cannot be empty")
        if self.composite_score < 0:
            raise ValueError("Composite score cannot be negative")


@dataclass
class EmailData:
    """Email notification data."""
    alert: Alert
    recipients: List[str]
    subject: str
    body: str
    attachments: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.recipients:
            raise ValueError("Recipients list cannot be empty")
        if self.retry_count < 0:
            raise ValueError("Retry count cannot be negative")


@dataclass
class FrameMetadata:
    """Frame processing metadata."""
    frame_id: str
    timestamp: datetime
    camera_id: str
    frame_number: int
    processing_time: float
    detections_count: int
    tracks_count: int
    
    def __post_init__(self):
        if self.frame_number < 0:
            raise ValueError("Frame number cannot be negative")
        if self.processing_time < 0:
            raise ValueError("Processing time cannot be negative")


@dataclass
class StreamStatus:
    """RTSP stream status."""
    stream_id: str
    url: str
    health: StreamHealth
    last_frame_time: datetime
    fps: float
    reconnect_attempts: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.fps < 0:
            raise ValueError("FPS cannot be negative")
        if self.reconnect_attempts < 0:
            raise ValueError("Reconnect attempts cannot be negative")


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    fps: float
    active_tracks: int
    alerts_generated: int
    processing_latency: float
    
    def __post_init__(self):
        if self.cpu_usage < 0 or self.cpu_usage > 100:
            raise ValueError("CPU usage must be between 0 and 100")
        if self.memory_usage < 0 or self.memory_usage > 100:
            raise ValueError("Memory usage must be between 0 and 100")
        if self.gpu_usage is not None and (self.gpu_usage < 0 or self.gpu_usage > 100):
            raise ValueError("GPU usage must be between 0 and 100")


@dataclass
class TemporalScores:
    """Temporal behavior analysis scores."""
    looking_away_score: float = 0.0
    talking_score: float = 0.0
    phone_score: float = 0.0
    chit_score: float = 0.0
    action_score: float = 0.0
    missing_score: float = 0.0
    proximity_score: float = 0.0
    composite_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_composite(self, weights: Dict[BehaviorEventType, int]) -> float:
        """Calculate composite score using provided weights."""
        total_score = (
            self.looking_away_score * weights.get(BehaviorEventType.LOOKING_AWAY, 5) +
            self.talking_score * weights.get(BehaviorEventType.TALKING, 15) +
            self.phone_score * weights.get(BehaviorEventType.PHONE_DETECTED, 50) +
            self.chit_score * weights.get(BehaviorEventType.CHIT_DETECTED, 45) +
            self.action_score * weights.get(BehaviorEventType.SUSPICIOUS_ACTION, 20) +
            self.missing_score * weights.get(BehaviorEventType.MISSING, 25) +
            self.proximity_score * weights.get(BehaviorEventType.PROXIMITY_VIOLATION, 10)
        )
        self.composite_score = total_score
        return total_score