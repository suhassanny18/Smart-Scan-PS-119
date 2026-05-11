"""
Head Pose Analysis Engine for detecting suspicious head movements and orientations.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

from ..models.data_models import HeadPose, BehaviorEvent
from ..models.enums import BehaviorType, SuspicionLevel

logger = logging.getLogger(__name__)


class PosePattern(Enum):
    """Head pose pattern classifications."""
    NORMAL = "normal"
    TILTED = "tilted"
    TURNED_LEFT = "turned_left"
    TURNED_RIGHT = "turned_right"
    LOOKING_UP = "looking_up"
    LOOKING_DOWN = "looking_down"
    EXTREME_POSE = "extreme_pose"
    UNSTABLE = "unstable"


@dataclass
class PoseAnalysisConfig:
    """Configuration for head pose analysis."""
    # Pose angle thresholds (in degrees)
    normal_yaw_threshold: float = 20.0  # Degrees from center considered normal
    normal_pitch_threshold: float = 15.0  # Degrees up/down considered normal
    normal_roll_threshold: float = 10.0  # Degrees tilt considered normal
    
    extreme_yaw_threshold: float = 60.0  # Degrees considered extreme turning
    extreme_pitch_threshold: float = 45.0  # Degrees considered extreme up/down
    extreme_roll_threshold: float = 30.0  # Degrees considered extreme tilt
    
    # Movement analysis
    stability_window_seconds: float = 5.0  # Window for stability analysis
    movement_threshold: float = 10.0  # Degrees/second for significant movement
    jitter_threshold: float = 2.0  # Degrees for jitter detection
    
    # Pattern detection
    suspicious_duration_threshold: float = 3.0  # Seconds of suspicious pose
    instability_threshold: float = 15.0  # Degrees std dev for instability
    rapid_movement_threshold: float = 30.0  # Degrees/second for rapid movement
    
    # Quality filters
    min_confidence_threshold: float = 0.6  # Minimum pose detection confidence
    pose_history_size: int = 150  # Number of pose samples to keep
    
    # Scoring parameters
    base_suspicion_score: float = 0.1
    extreme_pose_score: float = 0.6
    instability_score: float = 0.4
    rapid_movement_score: float = 0.3


@dataclass
class PoseMetrics:
    """Metrics for pose analysis performance."""
    total_pose_samples: int = 0
    normal_pose_percentage: float = 0.0
    extreme_pose_percentage: float = 0.0
    
    # Movement metrics
    average_movement_speed: float = 0.0
    max_movement_speed: float = 0.0
    instability_periods: int = 0
    rapid_movements: int = 0
    
    # Pose distribution
    average_yaw: float = 0.0
    average_pitch: float = 0.0
    average_roll: float = 0.0
    yaw_std: float = 0.0
    pitch_std: float = 0.0
    roll_std: float = 0.0
    
    # Quality metrics
    low_confidence_samples: int = 0
    missing_data_periods: int = 0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PoseSample:
    """Individual pose sample with metadata."""
    timestamp: datetime
    yaw: float  # Head rotation left/right
    pitch: float  # Head rotation up/down
    roll: float  # Head tilt
    confidence: float
    pattern: PosePattern
    suspicion_score: float = 0.0
    movement_speed: float = 0.0


class PoseAnalyzer:
    """
    Head pose analysis engine for detecting suspicious head movements and orientations.
    
    Features:
    - Real-time head pose classification
    - Movement pattern analysis
    - Stability and jitter detection
    - Extreme pose detection
    - Temporal behavior tracking
    - Configurable sensitivity thresholds
    """
    
    def __init__(self, config: PoseAnalysisConfig):
        """
        Initialize pose analyzer.
        
        Args:
            config: Pose analysis configuration
        """
        self.config = config
        
        # Pose history for temporal analysis
        self.pose_history: deque = deque(maxlen=config.pose_history_size)
        
        # Current analysis state
        self.current_pattern = PosePattern.NORMAL
        self.suspicious_start_time: Optional[datetime] = None
        self.last_pose_sample: Optional[PoseSample] = None
        
        # Movement tracking
        self.movement_history: List[float] = []
        self.instability_periods: List[datetime] = []
        
        # Performance metrics
        self.metrics = PoseMetrics()
        
        logger.info("PoseAnalyzer initialized")
    
    def analyze_pose(self, head_pose: HeadPose, timestamp: datetime) -> List[BehaviorEvent]:
        """
        Analyze head pose and detect suspicious patterns.
        
        Args:
            head_pose: Head pose data
            timestamp: Current timestamp
            
        Returns:
            List of detected behavior events
        """
        events = []
        
        try:
            # Validate pose data quality
            if not self._is_valid_pose_data(head_pose):
                self._handle_missing_pose_data(timestamp)
                return events
            
            # Create pose sample
            pose_sample = self._create_pose_sample(head_pose, timestamp)
            
            # Add to history
            self.pose_history.append(pose_sample)
            
            # Detect rapid movements
            if self.last_pose_sample:
                rapid_movement_event = self._detect_rapid_movement(pose_sample, self.last_pose_sample)
                if rapid_movement_event:
                    events.append(rapid_movement_event)
            
            # Detect instability
            instability_event = self._detect_instability(pose_sample, timestamp)
            if instability_event:
                events.append(instability_event)
            
            # Detect prolonged extreme poses
            extreme_pose_event = self._detect_prolonged_extreme_pose(pose_sample, timestamp)
            if extreme_pose_event:
                events.append(extreme_pose_event)
            
            # Update current state
            self.current_pattern = pose_sample.pattern
            self.last_pose_sample = pose_sample
            
            # Update metrics
            self._update_metrics(pose_sample)
            
        except Exception as e:
            logger.error(f"Error in pose analysis: {e}")
        
        return events
    
    def _is_valid_pose_data(self, head_pose: HeadPose) -> bool:
        """
        Validate pose data quality.
        
        Args:
            head_pose: Head pose data
            
        Returns:
            True if data is valid for analysis
        """
        if head_pose.confidence < self.config.min_confidence_threshold:
            self.metrics.low_confidence_samples += 1
            return False
        
        # Check for reasonable angle ranges
        if (abs(head_pose.yaw) > 180 or 
            abs(head_pose.pitch) > 90 or 
            abs(head_pose.roll) > 180):
            return False
        
        return True
    
    def _create_pose_sample(self, head_pose: HeadPose, timestamp: datetime) -> PoseSample:
        """
        Create pose sample from head pose data.
        
        Args:
            head_pose: Head pose data
            timestamp: Current timestamp
            
        Returns:
            Pose sample with pattern classification
        """
        # Classify pose pattern
        pattern = self._classify_pose_pattern(head_pose.yaw, head_pose.pitch, head_pose.roll)
        
        # Calculate movement speed
        movement_speed = 0.0
        if self.last_pose_sample:
            movement_speed = self._calculate_movement_speed(head_pose, self.last_pose_sample, timestamp)
        
        # Calculate suspicion score
        suspicion_score = self._calculate_pose_suspicion_score(pattern, head_pose, movement_speed)
        
        return PoseSample(
            timestamp=timestamp,
            yaw=head_pose.yaw,
            pitch=head_pose.pitch,
            roll=head_pose.roll,
            confidence=head_pose.confidence,
            pattern=pattern,
            suspicion_score=suspicion_score,
            movement_speed=movement_speed
        )
    
    def _classify_pose_pattern(self, yaw: float, pitch: float, roll: float) -> PosePattern:
        """
        Classify pose pattern based on angles.
        
        Args:
            yaw: Head rotation left/right
            pitch: Head rotation up/down
            roll: Head tilt
            
        Returns:
            Pose pattern classification
        """
        # Check for extreme poses first
        if (abs(yaw) > self.config.extreme_yaw_threshold or
            abs(pitch) > self.config.extreme_pitch_threshold or
            abs(roll) > self.config.extreme_roll_threshold):
            return PosePattern.EXTREME_POSE
        
        # Check for normal pose
        if (abs(yaw) <= self.config.normal_yaw_threshold and
            abs(pitch) <= self.config.normal_pitch_threshold and
            abs(roll) <= self.config.normal_roll_threshold):
            return PosePattern.NORMAL
        
        # Classify based on dominant angle
        max_angle = max(abs(yaw), abs(pitch), abs(roll))
        
        if abs(yaw) == max_angle:
            return PosePattern.TURNED_LEFT if yaw < 0 else PosePattern.TURNED_RIGHT
        elif abs(pitch) == max_angle:
            return PosePattern.LOOKING_UP if pitch > 0 else PosePattern.LOOKING_DOWN
        else:  # roll is dominant
            return PosePattern.TILTED
    
    def _calculate_movement_speed(self, current_pose: HeadPose, last_sample: PoseSample, 
                                timestamp: datetime) -> float:
        """
        Calculate head movement speed.
        
        Args:
            current_pose: Current head pose
            last_sample: Previous pose sample
            timestamp: Current timestamp
            
        Returns:
            Movement speed in degrees per second
        """
        time_diff = (timestamp - last_sample.timestamp).total_seconds()
        if time_diff <= 0:
            return 0.0
        
        # Calculate angular differences
        yaw_diff = abs(current_pose.yaw - last_sample.yaw)
        pitch_diff = abs(current_pose.pitch - last_sample.pitch)
        roll_diff = abs(current_pose.roll - last_sample.roll)
        
        # Handle angle wrapping for yaw and roll
        yaw_diff = min(yaw_diff, 360 - yaw_diff)
        roll_diff = min(roll_diff, 360 - roll_diff)
        
        # Calculate total angular movement
        total_movement = np.sqrt(yaw_diff**2 + pitch_diff**2 + roll_diff**2)
        
        return total_movement / time_diff
    
    def _calculate_pose_suspicion_score(self, pattern: PosePattern, head_pose: HeadPose, 
                                      movement_speed: float) -> float:
        """
        Calculate suspicion score for pose pattern.
        
        Args:
            pattern: Pose pattern
            head_pose: Head pose data
            movement_speed: Movement speed
            
        Returns:
            Suspicion score (0.0 to 1.0)
        """
        if pattern == PosePattern.NORMAL:
            return 0.0
        
        if pattern == PosePattern.EXTREME_POSE:
            return self.config.extreme_pose_score
        
        # Base score for non-normal poses
        base_score = self.config.base_suspicion_score
        
        # Add movement component
        if movement_speed > self.config.rapid_movement_threshold:
            base_score += self.config.rapid_movement_score
        
        # Scale by angle magnitude
        max_angle = max(abs(head_pose.yaw), abs(head_pose.pitch), abs(head_pose.roll))
        angle_factor = min(max_angle / self.config.extreme_yaw_threshold, 1.0)
        
        return min(base_score * (1 + angle_factor), 1.0)
    
    def _detect_rapid_movement(self, current: PoseSample, previous: PoseSample) -> Optional[BehaviorEvent]:
        """
        Detect rapid head movements.
        
        Args:
            current: Current pose sample
            previous: Previous pose sample
            
        Returns:
            Behavior event if rapid movement detected
        """
        if current.movement_speed > self.config.rapid_movement_threshold:
            self.metrics.rapid_movements += 1
            
            return BehaviorEvent(
                behavior_type=BehaviorType.RAPID_HEAD_MOVEMENT,
                confidence=min(current.movement_speed / (self.config.rapid_movement_threshold * 2), 1.0),
                timestamp=current.timestamp,
                duration=(current.timestamp - previous.timestamp).total_seconds(),
                metadata={
                    "movement_speed": current.movement_speed,
                    "yaw_change": abs(current.yaw - previous.yaw),
                    "pitch_change": abs(current.pitch - previous.pitch),
                    "roll_change": abs(current.roll - previous.roll)
                }
            )
        
        return None
    
    def _detect_instability(self, current: PoseSample, timestamp: datetime) -> Optional[BehaviorEvent]:
        """
        Detect head pose instability (jittery movements).
        
        Args:
            current: Current pose sample
            timestamp: Current timestamp
            
        Returns:
            Behavior event if instability detected
        """
        # Need sufficient history for stability analysis
        if len(self.pose_history) < 10:
            return None
        
        # Get recent samples within stability window
        cutoff_time = timestamp - timedelta(seconds=self.config.stability_window_seconds)
        recent_samples = [s for s in self.pose_history if s.timestamp > cutoff_time]
        
        if len(recent_samples) < 5:
            return None
        
        # Calculate standard deviation of angles
        yaw_values = [s.yaw for s in recent_samples]
        pitch_values = [s.pitch for s in recent_samples]
        roll_values = [s.roll for s in recent_samples]
        
        yaw_std = np.std(yaw_values)
        pitch_std = np.std(pitch_values)
        roll_std = np.std(roll_values)
        
        max_std = max(yaw_std, pitch_std, roll_std)
        
        if max_std > self.config.instability_threshold:
            self.instability_periods.append(timestamp)
            self.metrics.instability_periods += 1
            
            return BehaviorEvent(
                behavior_type=BehaviorType.HEAD_INSTABILITY,
                confidence=min(max_std / (self.config.instability_threshold * 2), 1.0),
                timestamp=timestamp,
                duration=self.config.stability_window_seconds,
                metadata={
                    "yaw_std": yaw_std,
                    "pitch_std": pitch_std,
                    "roll_std": roll_std,
                    "max_std": max_std,
                    "sample_count": len(recent_samples)
                }
            )
        
        return None
    
    def _detect_prolonged_extreme_pose(self, current: PoseSample, timestamp: datetime) -> Optional[BehaviorEvent]:
        """
        Detect prolonged extreme head poses.
        
        Args:
            current: Current pose sample
            timestamp: Current timestamp
            
        Returns:
            Behavior event if prolonged extreme pose detected
        """
        is_extreme = current.pattern == PosePattern.EXTREME_POSE
        
        if is_extreme:
            if self.suspicious_start_time is None:
                self.suspicious_start_time = timestamp
        else:
            # Check if we had a prolonged extreme pose
            if self.suspicious_start_time is not None:
                duration = (timestamp - self.suspicious_start_time).total_seconds()
                
                if duration >= self.config.suspicious_duration_threshold:
                    confidence = min(duration / (self.config.suspicious_duration_threshold * 2), 1.0)
                    confidence *= current.suspicion_score
                    
                    event = BehaviorEvent(
                        behavior_type=BehaviorType.EXTREME_HEAD_POSE,
                        confidence=confidence,
                        timestamp=self.suspicious_start_time,
                        duration=duration,
                        metadata={
                            "end_timestamp": timestamp.isoformat(),
                            "pose_pattern": current.pattern.value,
                            "max_yaw": max(abs(s.yaw) for s in self.pose_history[-10:]),
                            "max_pitch": max(abs(s.pitch) for s in self.pose_history[-10:]),
                            "max_roll": max(abs(s.roll) for s in self.pose_history[-10:])
                        }
                    )
                    
                    self.suspicious_start_time = None
                    return event
                
                self.suspicious_start_time = None
        
        return None
    
    def _handle_missing_pose_data(self, timestamp: datetime) -> None:
        """
        Handle missing or invalid pose data.
        
        Args:
            timestamp: Current timestamp
        """
        self.metrics.missing_data_periods += 1
        
        # Reset extreme pose tracking if too much missing data
        if self.suspicious_start_time and self.last_pose_sample:
            missing_duration = (timestamp - self.last_pose_sample.timestamp).total_seconds()
            if missing_duration > 1.0:  # 1 second threshold
                self.suspicious_start_time = None
    
    def _update_metrics(self, pose_sample: PoseSample) -> None:
        """
        Update performance metrics.
        
        Args:
            pose_sample: Current pose sample
        """
        self.metrics.total_pose_samples += 1
        
        # Update movement statistics
        if pose_sample.movement_speed > 0:
            self.movement_history.append(pose_sample.movement_speed)
            
            # Keep only recent movement data
            if len(self.movement_history) > 100:
                self.movement_history = self.movement_history[-100:]
            
            self.metrics.average_movement_speed = np.mean(self.movement_history)
            self.metrics.max_movement_speed = max(self.metrics.max_movement_speed, pose_sample.movement_speed)
        
        # Update pose angle statistics
        if len(self.pose_history) > 0:
            yaw_values = [s.yaw for s in self.pose_history]
            pitch_values = [s.pitch for s in self.pose_history]
            roll_values = [s.roll for s in self.pose_history]
            
            self.metrics.average_yaw = np.mean(yaw_values)
            self.metrics.average_pitch = np.mean(pitch_values)
            self.metrics.average_roll = np.mean(roll_values)
            
            self.metrics.yaw_std = np.std(yaw_values)
            self.metrics.pitch_std = np.std(pitch_values)
            self.metrics.roll_std = np.std(roll_values)
        
        # Update pattern percentages
        normal_count = sum(1 for s in self.pose_history if s.pattern == PosePattern.NORMAL)
        extreme_count = sum(1 for s in self.pose_history if s.pattern == PosePattern.EXTREME_POSE)
        
        total_samples = len(self.pose_history)
        if total_samples > 0:
            self.metrics.normal_pose_percentage = normal_count / total_samples
            self.metrics.extreme_pose_percentage = extreme_count / total_samples
        
        self.metrics.timestamp = datetime.now()
    
    def get_current_pose_state(self) -> Dict[str, Any]:
        """
        Get current pose analysis state.
        
        Returns:
            Dictionary with current state information
        """
        return {
            "current_pattern": self.current_pattern.value if self.current_pattern else None,
            "is_extreme_pose": self.suspicious_start_time is not None,
            "extreme_pose_duration": (
                (datetime.now() - self.suspicious_start_time).total_seconds()
                if self.suspicious_start_time else 0.0
            ),
            "recent_pose_samples": len(self.pose_history),
            "current_movement_speed": (
                self.last_pose_sample.movement_speed if self.last_pose_sample else 0.0
            ),
            "last_sample_confidence": (
                self.last_pose_sample.confidence if self.last_pose_sample else 0.0
            )
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            # Sample statistics
            "total_pose_samples": self.metrics.total_pose_samples,
            "normal_pose_percentage": self.metrics.normal_pose_percentage,
            "extreme_pose_percentage": self.metrics.extreme_pose_percentage,
            
            # Movement metrics
            "average_movement_speed": self.metrics.average_movement_speed,
            "max_movement_speed": self.metrics.max_movement_speed,
            "instability_periods": self.metrics.instability_periods,
            "rapid_movements": self.metrics.rapid_movements,
            
            # Pose distribution
            "average_yaw": self.metrics.average_yaw,
            "average_pitch": self.metrics.average_pitch,
            "average_roll": self.metrics.average_roll,
            "yaw_std": self.metrics.yaw_std,
            "pitch_std": self.metrics.pitch_std,
            "roll_std": self.metrics.roll_std,
            
            # Quality metrics
            "low_confidence_samples": self.metrics.low_confidence_samples,
            "missing_data_periods": self.metrics.missing_data_periods,
            "data_quality_ratio": (
                (self.metrics.total_pose_samples - self.metrics.low_confidence_samples) /
                max(self.metrics.total_pose_samples, 1)
            ),
            
            # Configuration
            "normal_yaw_threshold": self.config.normal_yaw_threshold,
            "extreme_yaw_threshold": self.config.extreme_yaw_threshold,
            "stability_window_seconds": self.config.stability_window_seconds,
            
            "last_update": self.metrics.timestamp.isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PoseMetrics()
        self.movement_history.clear()
        logger.info("Pose analyzer metrics reset")
    
    def update_config(self, new_config: PoseAnalysisConfig) -> None:
        """
        Update analysis configuration.
        
        Args:
            new_config: New configuration
        """
        self.config = new_config
        
        # Adjust history size if needed
        if len(self.pose_history) > new_config.pose_history_size:
            samples_to_keep = list(self.pose_history)[-new_config.pose_history_size:]
            self.pose_history = deque(samples_to_keep, maxlen=new_config.pose_history_size)
        else:
            self.pose_history = deque(self.pose_history, maxlen=new_config.pose_history_size)
        
        logger.info("Pose analyzer configuration updated")
    
    def cleanup(self) -> None:
        """Clean up analyzer resources."""
        logger.info("Cleaning up PoseAnalyzer")
        
        self.pose_history.clear()
        self.movement_history.clear()
        self.instability_periods.clear()
        
        self.current_pattern = PosePattern.NORMAL
        self.suspicious_start_time = None
        self.last_pose_sample = None
        
        logger.info("PoseAnalyzer cleanup completed")