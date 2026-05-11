"""
Unit tests for configuration models
"""

import pytest
from anti_cheat_system.models import (
    ObjectDetectionConfig,
    GazeTrackingConfig,
    PostureAnalysisConfig,
    ScoringConfig,
    VideoConfig,
    LoggingConfig,
    SystemConfig,
    DEFAULT_CONFIG
)


class TestObjectDetectionConfig:
    """Test ObjectDetectionConfig"""
    
    def test_default_config(self):
        """Test default ObjectDetectionConfig values"""
        config = ObjectDetectionConfig()
        
        assert config.model_path == "yolov8n.pt"
        assert config.confidence_threshold == 0.5
        assert config.weight == 0.50
        assert 67 in config.suspicious_classes  # cell phone
        assert config.duration_gate_frames == 3
    
    def test_custom_config(self):
        """Test custom ObjectDetectionConfig values"""
        config = ObjectDetectionConfig(
            confidence_threshold=0.7,
            weight=0.6,
            suspicious_classes=[67, 73]
        )
        
        assert config.confidence_threshold == 0.7
        assert config.weight == 0.6
        assert config.suspicious_classes == [67, 73]


class TestGazeTrackingConfig:
    """Test GazeTrackingConfig"""
    
    def test_default_config(self):
        """Test default GazeTrackingConfig values"""
        config = GazeTrackingConfig()
        
        assert config.max_num_faces == 1
        assert config.gaze_deviation_threshold == 0.22
        assert config.gaze_duration_threshold == 3.0
        assert config.weight == 0.30
    
    def test_landmark_indices(self):
        """Test landmark indices are properly set"""
        config = GazeTrackingConfig()
        
        assert config.nose_tip_idx == 1
        assert config.left_eye_idx == 33
        assert config.right_eye_idx == 263


class TestPostureAnalysisConfig:
    """Test PostureAnalysisConfig"""
    
    def test_default_config(self):
        """Test default PostureAnalysisConfig values"""
        config = PostureAnalysisConfig()
        
        assert config.shoulder_angle_threshold == 20.0
        assert config.proximity_threshold == 0.3
        assert config.weight == 0.20
        assert config.model_complexity == 1
    
    def test_pose_indices(self):
        """Test pose landmark indices are properly set"""
        config = PostureAnalysisConfig()
        
        assert config.left_shoulder_idx == 11
        assert config.right_shoulder_idx == 12
        assert config.nose_idx == 0


class TestScoringConfig:
    """Test ScoringConfig"""
    
    def test_default_config(self):
        """Test default ScoringConfig values"""
        config = ScoringConfig()
        
        assert config.red_alert_threshold == 0.85
        assert config.amber_alert_threshold == 0.60
        assert config.object_weight == 0.50
        assert config.gaze_weight == 0.30
        assert config.posture_weight == 0.20
    
    def test_weight_validation(self):
        """Test that weights must sum to 1.0"""
        # Valid weights
        config = ScoringConfig(
            object_weight=0.5,
            gaze_weight=0.3,
            posture_weight=0.2
        )
        # Should not raise an exception
        
        # Invalid weights
        with pytest.raises(ValueError):
            ScoringConfig(
                object_weight=0.6,
                gaze_weight=0.3,
                posture_weight=0.2  # Sum = 1.1
            )


class TestVideoConfig:
    """Test VideoConfig"""
    
    def test_default_config(self):
        """Test default VideoConfig values"""
        config = VideoConfig()
        
        assert config.camera_index == 0
        assert config.frame_width == 640
        assert config.frame_height == 480
        assert config.fps == 30
        assert config.target_fps == 15
        assert config.show_video is True


class TestLoggingConfig:
    """Test LoggingConfig"""
    
    def test_default_config(self):
        """Test default LoggingConfig values"""
        config = LoggingConfig()
        
        assert config.log_level == "INFO"
        assert config.log_to_file is True
        assert config.log_directory == "logs"
        assert config.log_detection_events is True


class TestSystemConfig:
    """Test SystemConfig"""
    
    def test_default_system_config(self):
        """Test default SystemConfig initialization"""
        config = SystemConfig()
        
        # Check that all sub-configs are initialized
        assert config.object_detection is not None
        assert config.gaze_tracking is not None
        assert config.posture_analysis is not None
        assert config.scoring is not None
        assert config.video is not None
        assert config.logging is not None
        
        # Check default values
        assert config.debug_mode is False
        assert config.performance_monitoring is True
    
    def test_custom_system_config(self):
        """Test SystemConfig with custom sub-configs"""
        custom_scoring = ScoringConfig(red_alert_threshold=0.9)
        
        config = SystemConfig(
            scoring=custom_scoring,
            debug_mode=True
        )
        
        assert config.scoring.red_alert_threshold == 0.9
        assert config.debug_mode is True
    
    def test_config_validation(self):
        """Test SystemConfig validation"""
        config = SystemConfig()
        
        # Should validate successfully with default config
        assert config.validate() is True
        
        # Test with invalid scoring weights - this should raise an exception during creation
        with pytest.raises(ValueError):
            invalid_scoring = ScoringConfig(
                object_weight=0.6,
                gaze_weight=0.6,
                posture_weight=0.2
            )
        
        # Test validation with a manually created invalid config
        config.scoring.object_weight = 0.6
        config.scoring.gaze_weight = 0.6
        config.scoring.posture_weight = 0.2
        
        # Should fail validation due to invalid weights
        assert config.validate() is False


class TestDefaultConfig:
    """Test the default configuration instance"""
    
    def test_default_config_instance(self):
        """Test that DEFAULT_CONFIG is properly initialized"""
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, SystemConfig)
        assert DEFAULT_CONFIG.validate() is True
    
    def test_default_config_weights(self):
        """Test that default config has correct weights"""
        config = DEFAULT_CONFIG
        
        assert config.scoring.object_weight == 0.50
        assert config.scoring.gaze_weight == 0.30
        assert config.scoring.posture_weight == 0.20
        
        # Weights should sum to 1.0
        total = (config.scoring.object_weight + 
                config.scoring.gaze_weight + 
                config.scoring.posture_weight)
        assert abs(total - 1.0) < 0.01
    
    def test_default_config_thresholds(self):
        """Test that default config has correct thresholds"""
        config = DEFAULT_CONFIG
        
        assert config.scoring.red_alert_threshold == 0.85
        assert config.scoring.amber_alert_threshold == 0.60
        assert config.gaze_tracking.gaze_deviation_threshold == 0.22
        assert config.posture_analysis.shoulder_angle_threshold == 20.0