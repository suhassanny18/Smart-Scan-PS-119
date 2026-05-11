"""
Configuration models for the Anti-Cheat Detection System.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pydantic import validator
from pydantic_settings import BaseSettings
from .enums import (
    TrackerType, ModelType, BehaviorEventType, AlertLevel,
    DEFAULT_BEHAVIOR_WEIGHTS, DEFAULT_ALERT_THRESHOLDS
)


class SystemSettings(BaseSettings):
    """Main system configuration using Pydantic settings."""
    
    # CCTV Configuration
    rtsp_streams: List[str] = field(default_factory=list)
    reconnect_timeout: int = 30
    frame_skip_rate: int = 1
    max_frame_queue_size: int = 100
    
    # Detection Thresholds
    face_detection_confidence: float = 0.7
    object_detection_confidence: float = 0.5
    recognition_confidence: float = 0.8
    
    # Tracking Configuration
    tracker_type: TrackerType = TrackerType.BYTETRACK
    max_students_per_camera: int = 20
    track_timeout: int = 30
    
    # Behavioral Thresholds
    gaze_deviation_threshold: float = 0.22
    talking_threshold: float = 0.6
    posture_angle_threshold: float = 20.0
    proximity_distance_threshold: float = 100.0
    
    # Temporal Analysis
    behavior_frame_window: int = 90  # 3 seconds at 30 FPS
    score_decay_rate: float = 0.95
    recognition_interval: int = 30  # frames
    
    # Alert Thresholds
    warning_threshold: float = 60.0
    critical_threshold: float = 85.0
    alert_cooldown_seconds: int = 300  # 5 minutes
    
    # Email Configuration
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    email_retry_attempts: int = 3
    email_recipients: List[str] = field(default_factory=list)
    
    # Database Configuration
    database_url: str = "postgresql://user:password@localhost/anticheat"
    redis_url: str = "redis://localhost:6379"
    
    # Performance Configuration
    target_fps: int = 15
    worker_thread_count: int = 4
    gpu_enabled: bool = True
    batch_size: int = 4
    
    # Security Configuration
    secret_key: str = "your-secret-key-here"
    admin_username: str = "admin"
    admin_password: str = "admin123"
    
    # File Storage
    evidence_storage_path: str = "./evidence"
    model_storage_path: str = "./models"
    log_storage_path: str = "./logs"
    
    class Config:
        env_file = ".env"
        env_prefix = "ANTICHEAT_"
    
    @validator('rtsp_streams')
    def validate_rtsp_streams(cls, v):
        if not v:
            return ["0"]  # Default to webcam if no RTSP streams provided
        return v
    
    @validator('face_detection_confidence', 'object_detection_confidence', 'recognition_confidence')
    def validate_confidence_thresholds(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence thresholds must be between 0.0 and 1.0")
        return v
    
    @validator('gaze_deviation_threshold')
    def validate_gaze_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Gaze deviation threshold must be between 0.0 and 1.0")
        return v


@dataclass
class BehaviorConfig:
    """Behavioral analysis configuration."""
    
    # Gaze Analysis
    gaze_deviation_threshold: float = 0.22
    sustained_gaze_duration: float = 3.0  # seconds
    gaze_smoothing_window: int = 5  # frames
    
    # Talking Detection
    mouth_open_threshold: float = 0.02
    lip_motion_threshold: float = 0.01
    talking_duration_threshold: float = 2.0  # seconds
    
    # Posture Analysis
    shoulder_angle_threshold: float = 20.0  # degrees
    leaning_threshold: float = 15.0  # degrees
    proximity_threshold: float = 100.0  # pixels
    
    # Action Detection
    hand_face_threshold: float = 0.7
    reaching_threshold: float = 0.6
    object_passing_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.gaze_deviation_threshold <= 1.0:
            raise ValueError("Gaze deviation threshold must be between 0.0 and 1.0")
        if self.sustained_gaze_duration <= 0:
            raise ValueError("Sustained gaze duration must be positive")


@dataclass
class ScoringConfig:
    """Suspicion scoring configuration."""
    
    # Behavior weights
    behavior_weights: Dict[BehaviorEventType, int] = field(
        default_factory=lambda: DEFAULT_BEHAVIOR_WEIGHTS.copy()
    )
    
    # Alert thresholds
    alert_thresholds: Dict[AlertLevel, float] = field(
        default_factory=lambda: DEFAULT_ALERT_THRESHOLDS.copy()
    )
    
    # Temporal parameters
    score_decay_rate: float = 0.95
    frame_window_size: int = 90  # 3 seconds at 30 FPS
    cooldown_period: int = 300  # 5 minutes in seconds
    
    # Score normalization
    max_score_per_behavior: float = 100.0
    score_smoothing_factor: float = 0.8
    
    def __post_init__(self):
        """Validate scoring configuration."""
        if not 0.0 <= self.score_decay_rate <= 1.0:
            raise ValueError("Score decay rate must be between 0.0 and 1.0")
        if self.frame_window_size <= 0:
            raise ValueError("Frame window size must be positive")
        if self.cooldown_period <= 0:
            raise ValueError("Cooldown period must be positive")


@dataclass
class StateConfig:
    """Student state management configuration."""
    
    # State lifecycle
    state_timeout: int = 1800  # 30 minutes in seconds
    cleanup_interval: int = 300  # 5 minutes in seconds
    max_behavior_history: int = 1000
    
    # Memory management
    max_screenshots_per_student: int = 10
    screenshot_retention_hours: int = 24
    
    # Recovery settings
    track_recovery_timeout: int = 60  # seconds
    identity_confidence_threshold: float = 0.8
    
    def __post_init__(self):
        """Validate state configuration."""
        if self.state_timeout <= 0:
            raise ValueError("State timeout must be positive")
        if self.cleanup_interval <= 0:
            raise ValueError("Cleanup interval must be positive")


@dataclass
class AlertConfig:
    """Alert generation configuration."""
    
    # Alert rules
    enable_warning_alerts: bool = True
    enable_critical_alerts: bool = True
    enable_email_notifications: bool = True
    
    # Evidence capture
    capture_evidence_screenshots: bool = True
    annotate_screenshots: bool = True
    screenshot_quality: int = 85  # JPEG quality 0-100
    
    # Cooldown settings
    alert_cooldown_seconds: int = 300  # 5 minutes
    duplicate_suppression_window: int = 60  # 1 minute
    
    # Escalation rules
    escalation_threshold_multiplier: float = 1.5
    max_alerts_per_student_per_hour: int = 10
    
    def __post_init__(self):
        """Validate alert configuration."""
        if not 0 <= self.screenshot_quality <= 100:
            raise ValueError("Screenshot quality must be between 0 and 100")
        if self.alert_cooldown_seconds <= 0:
            raise ValueError("Alert cooldown must be positive")


@dataclass
class SMTPConfig:
    """SMTP email configuration."""
    
    server: str
    port: int = 587
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30
    
    # Email settings
    sender_email: str = ""
    sender_name: str = "Anti-Cheat Detection System"
    
    def __post_init__(self):
        """Validate SMTP configuration."""
        if not self.server:
            raise ValueError("SMTP server cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("SMTP port must be between 1 and 65535")


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Processing settings
    target_fps: int = 15
    max_processing_latency: float = 2.0  # seconds
    frame_skip_threshold: float = 0.5  # skip frames if processing is slow
    
    # GPU settings
    gpu_enabled: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    
    # Batch processing
    detection_batch_size: int = 4
    recognition_batch_size: int = 8
    
    # Threading
    worker_threads: int = 4
    io_threads: int = 2
    
    # Memory management
    max_memory_usage_mb: int = 4096
    garbage_collection_interval: int = 100  # frames
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.target_fps <= 0:
            raise ValueError("Target FPS must be positive")
        if not 0.0 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("GPU memory fraction must be between 0.0 and 1.0")


# Global configuration instance
_config_instance: Optional[SystemSettings] = None


def get_config() -> SystemSettings:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = SystemSettings()
    return _config_instance


def set_config(config: SystemSettings) -> None:
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config