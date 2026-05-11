"""
Unit tests for the main AntiCheatSystem controller
Tests system initialization, frame processing, and coordination
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from anti_cheat_system.main import AntiCheatSystem
from anti_cheat_system.models import (
    SystemConfig,
    SystemResult,
    DetectionResult,
    FaceMeshResult,
    PoseResult,
    AlertLevel
)


class TestAntiCheatSystem:
    """Test cases for AntiCheatSystem class"""
    
    def test_anti_cheat_system_init_default_config(self):
        """Test AntiCheatSystem initialization with default config"""
        system = AntiCheatSystem()
        
        assert system.config is not None
        assert system.is_initialized is False
        assert system.is_running is False
        assert system.should_stop is False
        assert system.frame_count == 0
        assert system.error_count == 0
        assert system.video_capture is None
        assert system.object_detector is None
        assert system.gaze_tracker is None
        assert system.posture_analyzer is None
        assert system.scoring_system is None
    
    def test_anti_cheat_system_init_custom_config(self):
        """Test AntiCheatSystem initialization with custom config"""
        config = SystemConfig()
        config.debug_mode = True
        config.logging.log_level = "DEBUG"
        
        system = AntiCheatSystem(config)
        
        assert system.config.debug_mode is True
        assert system.config.logging.log_level == "DEBUG"
    
    @patch('anti_cheat_system.main.VideoCapture')
    @patch('anti_cheat_system.main.ObjectDetector')
    @patch('anti_cheat_system.main.GazeTracker')
    @patch('anti_cheat_system.main.PostureAnalyzer')
    @patch('anti_cheat_system.main.ScoringSystem')
    def test_initialize_components_success(self, mock_scoring, mock_posture, mock_gaze, mock_object, mock_video):
        """Test successful component initialization"""
        # Mock all components as successfully initialized
        mock_video_instance = Mock()
        mock_video_instance.initialize_camera.return_value = True
        mock_video.return_value = mock_video_instance
        
        mock_object_instance = Mock()
        mock_object_instance.is_initialized = True
        mock_object.return_value = mock_object_instance
        
        mock_gaze_instance = Mock()
        mock_gaze_instance.is_initialized = True
        mock_gaze.return_value = mock_gaze_instance
        
        mock_posture_instance = Mock()
        mock_posture_instance.is_initialized = True
        mock_posture.return_value = mock_posture_instance
        
        mock_scoring_instance = Mock()
        mock_scoring.return_value = mock_scoring_instance
        
        system = AntiCheatSystem()
        result = system.initialize_components()
        
        assert result is True
        assert system.is_initialized is True
        assert system.video_capture is not None
        assert system.object_detector is not None
        assert system.gaze_tracker is not None
        assert system.posture_analyzer is not None
        assert system.scoring_system is not None
    
    @patch('anti_cheat_system.main.VideoCapture')
    def test_initialize_components_video_failure(self, mock_video):
        """Test component initialization with video capture failure"""
        mock_video_instance = Mock()
        mock_video_instance.initialize_camera.return_value = False
        mock_video.return_value = mock_video_instance
        
        system = AntiCheatSystem()
        result = system.initialize_components()
        
        assert result is False
        assert system.is_initialized is False
    
    @patch('anti_cheat_system.main.VideoCapture')
    @patch('anti_cheat_system.main.ObjectDetector')
    def test_initialize_components_object_detector_failure(self, mock_object, mock_video):
        """Test component initialization with object detector failure"""
        mock_video_instance = Mock()
        mock_video_instance.initialize_camera.return_value = True
        mock_video.return_value = mock_video_instance
        
        mock_object_instance = Mock()
        mock_object_instance.is_initialized = False
        mock_object.return_value = mock_object_instance
        
        system = AntiCheatSystem()
        result = system.initialize_components()
        
        assert result is False
        assert system.is_initialized is False
    
    def test_process_frame_not_initialized(self):
        """Test frame processing when system is not initialized"""
        system = AntiCheatSystem()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = system.process_frame(frame)
        
        assert result is None
    
    def test_process_frame_success(self):
        """Test successful frame processing"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock all components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Mock detection results
        detection_result = DetectionResult([], [], [], 0, False, False)
        face_mesh_result = FaceMeshResult(None, 0.0, 0.0, 0.1, False, False)
        pose_result = PoseResult(None, 5.0, False, False, False)
        
        system.object_detector.detect_objects.return_value = detection_result
        system.object_detector.calculate_object_score.return_value = 0.1
        
        system.gaze_tracker.detect_face_mesh.return_value = face_mesh_result
        system.gaze_tracker.calculate_gaze_score.return_value = 0.2
        
        system.posture_analyzer.detect_pose.return_value = pose_result
        system.posture_analyzer.calculate_posture_score.return_value = 0.1
        
        # Mock system result
        system_result = SystemResult(
            object_score=0.1,
            gaze_score=0.2,
            posture_score=0.1,
            composite_score=0.13,
            alert_level=AlertLevel.NORMAL,
            timestamp=datetime.now()
        )
        system.scoring_system.process_detection_results.return_value = system_result
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = system.process_frame(frame)
        
        assert result is not None
        assert isinstance(result, SystemResult)
        assert result.alert_level == AlertLevel.NORMAL
        assert system.frame_count == 1
    
    def test_process_frame_exception_handling(self):
        """Test frame processing exception handling"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock component that raises exception
        system.object_detector = Mock()
        system.object_detector.detect_objects.side_effect = Exception("Detection failed")
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = system.process_frame(frame)
        
        assert result is None
        assert system.error_handler.error_count >= 1
    
    def test_fps_counter_update(self):
        """Test FPS counter functionality"""
        system = AntiCheatSystem()
        
        # Simulate time passage
        system.last_fps_time = time.time() - 2.0  # 2 seconds ago
        system.fps_counter = 30  # 30 frames in 2 seconds
        
        system._update_fps_counter()
        
        # Allow for small timing variations
        assert abs(system.current_fps - 15.0) < 1.0  # Approximately 15 FPS
        assert system.fps_counter == 0  # Reset after update
    
    def test_error_handling_max_errors(self):
        """Test error handling with maximum error threshold"""
        system = AntiCheatSystem()
        system.max_errors = 3
        
        # Trigger multiple errors
        for i in range(4):
            system._handle_processing_error()
        
        assert system.error_count == 4
        assert system.should_stop is True
    
    def test_stop_system(self):
        """Test system stop functionality"""
        system = AntiCheatSystem()
        system.is_running = True
        
        system.stop()
        
        assert system.should_stop is True
        assert system.is_running is False
    
    @patch('anti_cheat_system.main.cv2')
    def test_cleanup(self, mock_cv2):
        """Test system cleanup"""
        system = AntiCheatSystem()
        system.video_capture = Mock()
        system.is_running = True
        
        # Mock OpenCV window check
        mock_cv2.getWindowProperty.return_value = 1  # Window exists
        
        system._cleanup()
        
        system.video_capture.release.assert_called_once()
        mock_cv2.destroyAllWindows.assert_called_once()
        assert system.is_running is False
    
    def test_get_system_stats_basic(self):
        """Test basic system statistics retrieval"""
        system = AntiCheatSystem()
        system.frame_count = 100
        system.current_fps = 15.5
        system.error_count = 2
        
        stats = system.get_system_stats()
        
        assert 'system' in stats
        assert stats['system']['frame_count'] == 100
        assert stats['system']['current_fps'] == 15.5
        assert stats['system']['error_count'] == 2
        assert stats['system']['is_initialized'] is False
        assert stats['system']['is_running'] is False
    
    def test_get_system_stats_with_components(self):
        """Test system statistics with component stats"""
        system = AntiCheatSystem()
        
        # Mock components with stats
        system.object_detector = Mock()
        system.object_detector.get_detection_stats.return_value = {'detections': 10}
        
        system.gaze_tracker = Mock()
        system.gaze_tracker.get_gaze_stats.return_value = {'faces': 5}
        
        system.posture_analyzer = Mock()
        system.posture_analyzer.get_posture_stats.return_value = {'poses': 8}
        
        system.scoring_system = Mock()
        system.scoring_system.get_scoring_stats.return_value = {'alerts': 3}
        
        stats = system.get_system_stats()
        
        assert 'object_detector' in stats
        assert 'gaze_tracker' in stats
        assert 'posture_analyzer' in stats
        assert 'scoring_system' in stats
        assert stats['object_detector']['detections'] == 10
        assert stats['gaze_tracker']['faces'] == 5
        assert stats['posture_analyzer']['poses'] == 8
        assert stats['scoring_system']['alerts'] == 3
    
    def test_reset_statistics(self):
        """Test statistics reset functionality"""
        system = AntiCheatSystem()
        
        # Set some statistics
        system.frame_count = 100
        system.error_count = 5
        system.fps_counter = 10
        system.current_fps = 15.0
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        system.reset_statistics()
        
        assert system.frame_count == 0
        assert system.error_count == 0
        assert system.fps_counter == 0
        assert system.current_fps == 0.0
        
        # Verify component reset calls
        system.object_detector.reset_detection_history.assert_called_once()
        system.gaze_tracker.reset_deviation_tracking.assert_called_once()
        system.posture_analyzer.reset_posture_tracking.assert_called_once()
        system.scoring_system.reset_statistics.assert_called_once()
    
    @patch('anti_cheat_system.main.cv2')
    def test_create_display_frame_basic(self, mock_cv2):
        """Test basic display frame creation"""
        system = AntiCheatSystem()
        system.config.video.show_detections = False
        system.config.video.show_landmarks = False
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        system_result = SystemResult(
            object_score=0.1,
            gaze_score=0.2,
            posture_score=0.1,
            composite_score=0.13,
            alert_level=AlertLevel.NORMAL,
            timestamp=datetime.now(),
            processing_time_ms=10.5
        )
        
        display_frame = system._create_display_frame(frame, system_result)
        
        assert display_frame is not None
        assert display_frame.shape == frame.shape
    
    @patch('anti_cheat_system.main.cv2')
    def test_create_display_frame_with_detections(self, mock_cv2):
        """Test display frame creation with detection overlays"""
        system = AntiCheatSystem()
        system.config.video.show_detections = True
        system.config.video.show_landmarks = True
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        
        system.object_detector.draw_detections.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        system.gaze_tracker.draw_face_mesh.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        system.posture_analyzer.draw_pose.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create system result with detections
        detection_result = DetectionResult([{'class_id': 67}], [0.8], [(10, 10, 50, 50)], 1, True, True)
        face_mesh_result = FaceMeshResult(np.zeros((468, 3)), 5.0, 20.0, 0.3, True, True)
        pose_result = PoseResult(np.zeros((33, 3)), 25.0, True, False, True)
        
        system_result = SystemResult(
            object_score=0.8,
            gaze_score=0.6,
            posture_score=0.4,
            composite_score=0.66,
            alert_level=AlertLevel.AMBER,
            timestamp=datetime.now(),
            detection_result=detection_result,
            face_mesh_result=face_mesh_result,
            pose_result=pose_result
        )
        
        display_frame = system._create_display_frame(frame, system_result)
        
        # Verify drawing methods were called
        system.object_detector.draw_detections.assert_called_once()
        system.gaze_tracker.draw_face_mesh.assert_called_once()
        system.posture_analyzer.draw_pose.assert_called_once()
    
    @patch('anti_cheat_system.main.cv2')
    def test_draw_system_status(self, mock_cv2):
        """Test system status drawing"""
        system = AntiCheatSystem()
        system.frame_count = 150
        system.current_fps = 25.5
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        system_result = SystemResult(
            object_score=0.8,
            gaze_score=0.6,
            posture_score=0.4,
            composite_score=0.66,
            alert_level=AlertLevel.AMBER,
            timestamp=datetime.now(),
            processing_time_ms=15.2
        )
        
        system._draw_system_status(frame, system_result)
        
        # Verify cv2.putText was called multiple times for different status elements
        assert mock_cv2.putText.call_count >= 5  # Alert, score, individual scores, FPS, processing time
    
    def test_context_manager(self):
        """Test context manager functionality"""
        with AntiCheatSystem() as system:
            assert isinstance(system, AntiCheatSystem)
        
        # System should be stopped after context exit
        assert system.should_stop is True
        assert system.is_running is False
    
    @patch('argparse.ArgumentParser')
    @patch('anti_cheat_system.main.AntiCheatSystem')
    def test_main_function_basic(self, mock_system_class, mock_parser_class):
        """Test main function basic execution"""
        from anti_cheat_system.main import main
        
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = None
        mock_args.debug = False
        mock_args.no_display = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser
        
        # Mock system
        mock_system = Mock()
        mock_system.initialize_components.return_value = True
        mock_system.run.return_value = True
        mock_system_class.return_value.__enter__.return_value = mock_system
        mock_system_class.return_value.__exit__.return_value = None
        
        result = main()
        
        assert result == 0
        mock_system.initialize_components.assert_called_once()
        mock_system.run.assert_called_once()
    
    @patch('argparse.ArgumentParser')
    @patch('anti_cheat_system.main.AntiCheatSystem')
    def test_main_function_initialization_failure(self, mock_system_class, mock_parser_class):
        """Test main function with initialization failure"""
        from anti_cheat_system.main import main
        
        # Mock argument parser
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = None
        mock_args.debug = False
        mock_args.no_display = False
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser
        
        # Mock system with initialization failure
        mock_system = Mock()
        mock_system.initialize_components.return_value = False
        mock_system_class.return_value.__enter__.return_value = mock_system
        mock_system_class.return_value.__exit__.return_value = None
        
        result = main()
        
        assert result == 1
        mock_system.initialize_components.assert_called_once()
        mock_system.run.assert_not_called()
    
    @patch('argparse.ArgumentParser')
    @patch('anti_cheat_system.main.AntiCheatSystem')
    def test_main_function_debug_mode(self, mock_system_class, mock_parser_class):
        """Test main function with debug mode enabled"""
        from anti_cheat_system.main import main
        
        # Mock argument parser with debug flag
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.config = None
        mock_args.debug = True
        mock_args.no_display = True
        mock_parser.parse_args.return_value = mock_args
        mock_parser_class.return_value = mock_parser
        
        # Mock system
        mock_system = Mock()
        mock_system.initialize_components.return_value = True
        mock_system_class.return_value.__enter__.return_value = mock_system
        mock_system_class.return_value.__exit__.return_value = None
        
        result = main()
        
        assert result == 0
        # Verify system was created with debug configuration
        mock_system_class.assert_called_once()
        config_arg = mock_system_class.call_args[0][0]
        assert config_arg.debug_mode is True
        assert config_arg.logging.log_level == "DEBUG"
        assert config_arg.video.show_video is False