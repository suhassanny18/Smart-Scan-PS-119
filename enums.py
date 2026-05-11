"""
Enums and constants for the Anti-Cheat Detection System.
"""

from enum import Enum, IntEnum
from typing import Dict, Any


class AlertLevel(Enum):
    """Alert severity levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class AttendanceStatus(Enum):
    """Student attendance status."""
    PRESENT = "present"
    ABSENT = "absent"
    LATE = "late"
    UNKNOWN = "unknown"


class BehaviorEventType(Enum):
    """Types of suspicious behavior events."""
    LOOKING_AWAY = "looking_away"
    TALKING = "talking"
    PHONE_DETECTED = "phone_detected"
    CHIT_DETECTED = "chit_detected"
    SUSPICIOUS_ACTION = "suspicious_action"
    MISSING = "missing"
    PROXIMITY_VIOLATION = "proximity_violation"
    HAND_TO_FACE = "hand_to_face"
    REACHING_DOWN = "reaching_down"
    PASSING_OBJECT = "passing_object"


class LookingDirection(Enum):
    """Gaze direction classifications."""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    DOWN = "down"
    UP = "up"
    AWAY = "away"


class ActionType(Enum):
    """Suspicious action types."""
    HAND_TO_FACE = "hand_to_face"
    REACHING_DOWN = "reaching_down"
    PASSING_OBJECT = "passing_object"
    CONCEALMENT = "concealment"
    UNUSUAL_ARM_MOVEMENT = "unusual_arm_movement"


class StreamHealth(Enum):
    """RTSP stream health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECONNECTING = "reconnecting"


class TrackerType(Enum):
    """Multi-object tracking algorithm types."""
    BYTETRACK = "bytetrack"
    DEEPSORT = "deepsort"


class ModelType(Enum):
    """AI model types."""
    YOLOV8_FACE = "yolov8-face"
    RETINAFACE = "retinaface"
    YOLOV8N = "yolov8n"
    FACENET = "facenet"


# Suspicious object class IDs for YOLO detection
SUSPICIOUS_OBJECT_CLASSES = {
    67: "cell phone",
    76: "book",
    77: "clock",
    # Add more suspicious object class IDs as needed
}

# Behavior scoring weights (configurable)
DEFAULT_BEHAVIOR_WEIGHTS = {
    BehaviorEventType.LOOKING_AWAY: 5,
    BehaviorEventType.TALKING: 15,
    BehaviorEventType.PHONE_DETECTED: 50,
    BehaviorEventType.CHIT_DETECTED: 45,
    BehaviorEventType.SUSPICIOUS_ACTION: 20,
    BehaviorEventType.MISSING: 25,
    BehaviorEventType.PROXIMITY_VIOLATION: 10,
}

# Default alert thresholds
DEFAULT_ALERT_THRESHOLDS = {
    AlertLevel.WARNING: 60.0,
    AlertLevel.CRITICAL: 85.0,
}