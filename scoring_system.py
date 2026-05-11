"""
Composite scoring system for the Anti-Cheat Detection System
Combines scores from all detection engines and determines alert levels
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque

from anti_cheat_system.models import (
    SystemResult,
    DetectionResult,
    FaceMeshResult,
    PoseResult,
    DetectionEvent,
    AlertLevel,
    ScoringConfig
)


class ScoringSystem:
    """
    Composite scoring system that combines detection results
    and determines alert levels based on weighted scores
    """
    
    def __init__(self, config: ScoringConfig = None):
        """
        Initialize scoring system
        
        Args:
            config: ScoringConfig with weights and thresholds
        """
        self.config = config or ScoringConfig()
        
        # Score history for smoothing
        self.score_history = deque(maxlen=10)
        self.smoothed_score = 0.0
        
        # Event logging
        self.detection_events = []
        self.alert_history = deque(maxlen=100)
        
        # Performance tracking
        self.total_frames_processed = 0
        self.alert_counts = {
            AlertLevel.NORMAL: 0,
            AlertLevel.AMBER: 0,
            AlertLevel.RED: 0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Scoring system initialized")
    
    def calculate_composite_score(
        self,
        object_score: float,
        gaze_score: float,
        posture_score: float
    ) -> float:
        """
        Calculate weighted composite score from individual detector scores
        
        Args:
            object_score: Score from object detection (0.0 to 1.0)
            gaze_score: Score from gaze tracking (0.0 to 1.0)
            posture_score: Score from posture analysis (0.0 to 1.0)
            
        Returns:
            float: Composite score (0.0 to 1.0)
        """
        # Validate input scores
        object_score = max(0.0, min(1.0, object_score))
        gaze_score = max(0.0, min(1.0, gaze_score))
        posture_score = max(0.0, min(1.0, posture_score))
        
        # Calculate weighted composite score
        composite_score = (
            (object_score * self.config.object_weight) +
            (gaze_score * self.config.gaze_weight) +
            (posture_score * self.config.posture_weight)
        )
        
        # Apply smoothing if enabled
        if self.config.score_smoothing_factor > 0:
            composite_score = self._apply_smoothing(composite_score)
        else:
            # No smoothing - just add to history
            self.score_history.append(composite_score)
        
        # Ensure score is in valid range
        composite_score = max(0.0, min(1.0, composite_score))
        
        return composite_score
    
    def _apply_smoothing(self, current_score: float) -> float:
        """
        Apply exponential smoothing to reduce score fluctuations
        
        Args:
            current_score: Current frame's composite score
            
        Returns:
            float: Smoothed composite score
        """
        if self.config.score_smoothing_factor == 0.0:
            # No smoothing - return current score directly
            self.score_history.append(current_score)
            return current_score
        
        if len(self.score_history) == 0:
            self.smoothed_score = current_score
        else:
            alpha = 1.0 - self.config.score_smoothing_factor
            self.smoothed_score = (alpha * current_score + 
                                 self.config.score_smoothing_factor * self.smoothed_score)
        
        self.score_history.append(current_score)
        return self.smoothed_score
    
    def determine_alert_level(self, composite_score: float) -> AlertLevel:
        """
        Determine alert level based on composite score
        
        Args:
            composite_score: Composite score (0.0 to 1.0)
            
        Returns:
            AlertLevel: Alert level (NORMAL, AMBER, or RED)
        """
        if composite_score >= self.config.red_alert_threshold:
            return AlertLevel.RED
        elif composite_score >= self.config.amber_alert_threshold:
            return AlertLevel.AMBER
        else:
            return AlertLevel.NORMAL
    
    def process_detection_results(
        self,
        detection_result: DetectionResult,
        face_mesh_result: FaceMeshResult,
        pose_result: PoseResult,
        object_score: float,
        gaze_score: float,
        posture_score: float,
        frame_number: int = 0
    ) -> SystemResult:
        """
        Process all detection results and create system result
        
        Args:
            detection_result: Object detection results
            face_mesh_result: Gaze tracking results
            pose_result: Posture analysis results
            object_score: Object detection score
            gaze_score: Gaze tracking score
            posture_score: Posture analysis score
            frame_number: Current frame number
            
        Returns:
            SystemResult: Complete system result with composite score and alert level
        """
        start_time = time.time()
        self.total_frames_processed += 1
        
        # Calculate composite score
        composite_score = self.calculate_composite_score(
            object_score, gaze_score, posture_score
        )
        
        # Determine alert level
        alert_level = self.determine_alert_level(composite_score)
        
        # Update alert counts
        self.alert_counts[alert_level] += 1
        
        # Create system result
        system_result = SystemResult(
            object_score=object_score,
            gaze_score=gaze_score,
            posture_score=posture_score,
            composite_score=composite_score,
            alert_level=alert_level,
            timestamp=datetime.now(),
            detection_result=detection_result,
            face_mesh_result=face_mesh_result,
            pose_result=pose_result,
            frame_number=frame_number,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
        # Log detection events if alert level is not normal
        if alert_level != AlertLevel.NORMAL:
            self._log_detection_event(system_result)
        
        # Update alert history
        self.alert_history.append((datetime.now(), alert_level, composite_score))
        
        return system_result
    
    def _log_detection_event(self, system_result: SystemResult):
        """
        Log detection event for non-normal alert levels
        
        Args:
            system_result: System result to log
        """
        # Create detection event details
        details = {
            'object_score': system_result.object_score,
            'gaze_score': system_result.gaze_score,
            'posture_score': system_result.posture_score,
            'composite_score': system_result.composite_score,
            'processing_time_ms': system_result.processing_time_ms
        }
        
        # Add specific detection details
        if system_result.detection_result and system_result.detection_result.suspicious_objects_found:
            details['objects_detected'] = len(system_result.detection_result.objects_detected)
            details['object_classes'] = [obj.get('class_id', 'unknown') 
                                        for obj in system_result.detection_result.objects_detected]
        
        if system_result.face_mesh_result and system_result.face_mesh_result.face_detected:
            details['gaze_deviation'] = system_result.face_mesh_result.gaze_deviation_percent
            details['sustained_deviation'] = system_result.face_mesh_result.sustained_deviation
        
        if system_result.pose_result and system_result.pose_result.pose_detected:
            details['shoulder_angle'] = system_result.pose_result.shoulder_angle
            details['is_leaning'] = system_result.pose_result.is_leaning
            details['proximity_detected'] = system_result.pose_result.proximity_detected
        
        # Create and store detection event
        event = DetectionEvent(
            timestamp=system_result.timestamp,
            event_type="composite",
            score=system_result.composite_score,
            alert_level=system_result.alert_level,
            details=details,
            frame_number=system_result.frame_number
        )
        
        self.detection_events.append(event)
        
        # Log the event
        self.logger.warning(
            f"{system_result.alert_level.value} ALERT - Frame {system_result.frame_number}: "
            f"Composite Score: {system_result.composite_score:.3f} "
            f"(Object: {system_result.object_score:.3f}, "
            f"Gaze: {system_result.gaze_score:.3f}, "
            f"Posture: {system_result.posture_score:.3f})"
        )
    
    def get_scoring_stats(self) -> Dict:
        """
        Get scoring system statistics and performance metrics
        
        Returns:
            dict: Statistics dictionary
        """
        total_alerts = sum(self.alert_counts.values())
        
        # Calculate alert percentages
        alert_percentages = {}
        for level, count in self.alert_counts.items():
            alert_percentages[level.value.lower() + '_percent'] = (
                (count / max(total_alerts, 1)) * 100
            )
        
        # Calculate recent score statistics
        recent_scores = list(self.score_history)
        avg_recent_score = sum(recent_scores) / max(len(recent_scores), 1)
        
        return {
            'total_frames_processed': self.total_frames_processed,
            'total_detection_events': len(self.detection_events),
            'alert_counts': {level.value: count for level, count in self.alert_counts.items()},
            **alert_percentages,
            'current_smoothed_score': self.smoothed_score,
            'avg_recent_score': round(avg_recent_score, 4),
            'score_history_length': len(self.score_history),
            'config': {
                'red_threshold': self.config.red_alert_threshold,
                'amber_threshold': self.config.amber_alert_threshold,
                'object_weight': self.config.object_weight,
                'gaze_weight': self.config.gaze_weight,
                'posture_weight': self.config.posture_weight,
                'smoothing_factor': self.config.score_smoothing_factor
            }
        }
    
    def get_recent_events(self, limit: int = 10) -> List[DetectionEvent]:
        """
        Get recent detection events
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List[DetectionEvent]: Recent detection events
        """
        return self.detection_events[-limit:] if self.detection_events else []
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent alert history
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List[Dict]: Recent alert history
        """
        recent_alerts = list(self.alert_history)[-limit:]
        return [
            {
                'timestamp': timestamp.isoformat(),
                'alert_level': alert_level.value,
                'composite_score': round(score, 4)
            }
            for timestamp, alert_level, score in recent_alerts
        ]
    
    def reset_statistics(self):
        """Reset all statistics and history"""
        self.score_history.clear()
        self.detection_events.clear()
        self.alert_history.clear()
        self.smoothed_score = 0.0
        self.total_frames_processed = 0
        self.alert_counts = {
            AlertLevel.NORMAL: 0,
            AlertLevel.AMBER: 0,
            AlertLevel.RED: 0
        }
        self.logger.info("Scoring system statistics reset")
    
    def update_config(self, new_config: ScoringConfig):
        """
        Update scoring configuration
        
        Args:
            new_config: New scoring configuration
        """
        self.config = new_config
        self.logger.info("Scoring system configuration updated")