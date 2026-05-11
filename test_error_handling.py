#!/usr/bin/env python3
"""
Tests for comprehensive error handling and logging system
"""

import pytest
import tempfile
import time
import logging
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from anti_cheat_system.error_handler import (
    ErrorHandler, 
    ResourceMonitor, 
    ErrorEvent, 
    ErrorSeverity,
    setup_error_handling
)
from anti_cheat_system.models.config import LoggingConfig
from anti_cheat_system.models.enums import SystemState


class TestErrorHandler:
    """Test ErrorHandler class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LoggingConfig(
            log_directory=self.temp_dir,
            log_filename="test.log",
            log_level="DEBUG",
            log_to_file=True,
            log_to_console=False
        )
        self.error_handler = ErrorHandler(self.config)
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization"""
        assert self.error_handler.config == self.config
        assert len(self.error_handler.error_history) == 0
        assert len(self.error_handler.component_error_counts) == 0
        assert self.error_handler.system_state == SystemState.INITIALIZING
        assert self.error_handler.logger is not None
    
    def test_handle_error_basic(self):
        """Test basic error handling"""
        test_error = Exception("Test error")
        
        result = self.error_handler.handle_error(
            "test_component",
            test_error,
            ErrorSeverity.MEDIUM
        )
        
        # Should handle error successfully
        assert result is True
        
        # Check error was recorded
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.component_error_counts["test_component"] == 1
        
        # Check error event details
        error_event = self.error_handler.error_history[0]
        assert error_event.component == "test_component"
        assert error_event.error_type == "Exception"
        assert error_event.message == "Test error"
        assert error_event.severity == ErrorSeverity.MEDIUM
    
    def test_handle_error_with_context(self):
        """Test error handling with context information"""
        test_error = Exception("Test error with context")
        context = {"frame_number": 123, "processing_time": 45.6}
        
        self.error_handler.handle_error(
            "test_component",
            test_error,
            ErrorSeverity.HIGH,
            context=context
        )
        
        error_event = self.error_handler.error_history[0]
        assert error_event.context == context
    
    def test_handle_error_with_recovery(self):
        """Test error handling with recovery strategy"""
        # Register a mock recovery strategy
        recovery_strategy = Mock(return_value=True)
        self.error_handler.register_recovery_strategy("test_component", recovery_strategy)
        
        test_error = Exception("Recoverable error")
        
        result = self.error_handler.handle_error(
            "test_component",
            test_error,
            ErrorSeverity.MEDIUM,
            attempt_recovery=True
        )
        
        # Should handle error successfully with recovery
        assert result is True
        
        # Check recovery was attempted
        error_event = self.error_handler.error_history[0]
        assert error_event.recovery_attempted is True
        assert error_event.recovery_successful is True
        
        # Check recovery strategy was called
        recovery_strategy.assert_called_once_with(test_error, {})
    
    def test_handle_error_recovery_failure(self):
        """Test error handling when recovery fails"""
        # Register a failing recovery strategy
        recovery_strategy = Mock(return_value=False)
        self.error_handler.register_recovery_strategy("test_component", recovery_strategy)
        
        test_error = Exception("Non-recoverable error")
        
        result = self.error_handler.handle_error(
            "test_component",
            test_error,
            ErrorSeverity.HIGH,
            attempt_recovery=True
        )
        
        # Should still handle error but recovery failed
        assert result is False
        
        error_event = self.error_handler.error_history[0]
        assert error_event.recovery_attempted is True
        assert error_event.recovery_successful is False
    
    def test_error_threshold_exceeded(self):
        """Test system stop when error threshold is exceeded"""
        # Generate many errors to exceed threshold
        for i in range(self.error_handler.max_total_errors):
            result = self.error_handler.handle_error(
                "test_component",
                Exception(f"Error {i}"),
                ErrorSeverity.MEDIUM
            )
            
            if i < self.error_handler.max_total_errors - 1:
                assert result is True
            else:
                # Last error should trigger system stop
                assert result is False
                assert self.error_handler.system_state == SystemState.ERROR
    
    def test_component_error_threshold(self):
        """Test component-specific error threshold"""
        # Generate many errors for single component
        for i in range(self.error_handler.max_component_errors):
            result = self.error_handler.handle_error(
                "failing_component",
                Exception(f"Component error {i}"),
                ErrorSeverity.MEDIUM
            )
            
            if i < self.error_handler.max_component_errors - 1:
                assert result is True
            else:
                # Last error should trigger system stop
                assert result is False
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        # Generate various errors
        self.error_handler.handle_error("comp1", Exception("Error 1"), ErrorSeverity.LOW)
        self.error_handler.handle_error("comp1", Exception("Error 2"), ErrorSeverity.MEDIUM)
        self.error_handler.handle_error("comp2", Exception("Error 3"), ErrorSeverity.HIGH)
        self.error_handler.handle_error("comp2", Exception("Error 4"), ErrorSeverity.CRITICAL)
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats['total_errors'] == 4
        assert stats['severity_breakdown']['low'] == 1
        assert stats['severity_breakdown']['medium'] == 1
        assert stats['severity_breakdown']['high'] == 1
        assert stats['severity_breakdown']['critical'] == 1
        
        assert stats['component_breakdown']['comp1']['total_errors'] == 2
        assert stats['component_breakdown']['comp2']['total_errors'] == 2
    
    def test_reset_error_counts(self):
        """Test resetting error counts"""
        # Generate some errors
        self.error_handler.handle_error("comp1", Exception("Error 1"), ErrorSeverity.MEDIUM)
        self.error_handler.handle_error("comp2", Exception("Error 2"), ErrorSeverity.MEDIUM)
        
        assert len(self.error_handler.error_history) == 2
        assert len(self.error_handler.component_error_counts) == 2
        
        # Reset specific component
        self.error_handler.reset_error_counts("comp1")
        assert self.error_handler.component_error_counts["comp1"] == 0
        assert self.error_handler.component_error_counts["comp2"] == 1
        
        # Reset all
        self.error_handler.reset_error_counts()
        assert len(self.error_handler.component_error_counts) == 0
        assert len(self.error_handler.error_history) == 0
    
    def test_logging_setup(self):
        """Test logging configuration"""
        # Check log file was created
        log_file = Path(self.temp_dir) / "test.log"
        
        # Generate a log message
        self.error_handler.logger.info("Test log message")
        
        # Check file exists and has content
        assert log_file.exists()
        
        # Check error log file
        error_log_file = Path(self.temp_dir) / "errors.log"
        self.error_handler.logger.error("Test error message")
        assert error_log_file.exists()


class TestResourceMonitor:
    """Test ResourceMonitor class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = LoggingConfig(
            log_directory=self.temp_dir,
            log_to_console=False
        )
        self.error_handler = ErrorHandler(self.config)
        self.resource_monitor = ResourceMonitor(self.error_handler)
    
    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor initialization"""
        assert self.resource_monitor.error_handler == self.error_handler
        assert self.resource_monitor.memory_threshold_percent == 85.0
        assert self.resource_monitor.cpu_threshold_percent == 90.0
        assert self.resource_monitor.check_interval == 30
    
    @patch('anti_cheat_system.error_handler.psutil')
    def test_check_resources_normal(self, mock_psutil):
        """Test resource checking under normal conditions"""
        # Mock normal resource usage
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 30.0
        
        mock_disk = Mock()
        mock_disk.used = 50 * 1024**3  # 50GB
        mock_disk.total = 100 * 1024**3  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024**2  # 100MB
        mock_psutil.Process.return_value = mock_process
        
        # Force check by resetting last check time
        self.resource_monitor.last_check_time = 0
        
        resource_info = self.resource_monitor.check_resources()
        
        assert resource_info['memory_percent'] == 50.0
        assert resource_info['cpu_percent'] == 30.0
        assert resource_info['disk_percent'] == 50.0
        assert resource_info['process_memory_mb'] == 100.0
        
        # Should not generate any errors
        assert len(self.error_handler.error_history) == 0
    
    @patch('anti_cheat_system.error_handler.psutil')
    def test_check_resources_high_usage(self, mock_psutil):
        """Test resource checking with high usage"""
        # Mock high resource usage
        mock_memory = Mock()
        mock_memory.percent = 90.0  # Above threshold
        mock_psutil.virtual_memory.return_value = mock_memory
        mock_psutil.cpu_percent.return_value = 95.0  # Above threshold
        
        mock_disk = Mock()
        mock_disk.used = 98 * 1024**3  # 98GB
        mock_disk.total = 100 * 1024**3  # 100GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 500 * 1024**2  # 500MB
        mock_psutil.Process.return_value = mock_process
        
        # Force check by resetting last check time
        self.resource_monitor.last_check_time = 0
        
        resource_info = self.resource_monitor.check_resources()
        
        # Should generate errors for high usage
        assert len(self.error_handler.error_history) >= 2  # Memory and CPU errors
        
        # Check error types
        error_messages = [e.message for e in self.error_handler.error_history]
        assert any("High memory usage" in msg for msg in error_messages)
        assert any("High CPU usage" in msg for msg in error_messages)
    
    def test_cleanup_resources(self):
        """Test resource cleanup"""
        # Should not raise any exceptions
        self.resource_monitor.cleanup_resources()
        
        # Should log cleanup completion
        # (We can't easily test garbage collection effects)


class TestSetupErrorHandling:
    """Test setup_error_handling function"""
    
    def test_setup_error_handling(self):
        """Test error handling setup"""
        config = LoggingConfig()
        error_handler = setup_error_handling(config)
        
        assert isinstance(error_handler, ErrorHandler)
        assert error_handler.config == config
        
        # Check recovery strategies were registered
        assert "video_capture" in error_handler.recovery_strategies
        assert "object_detector" in error_handler.recovery_strategies
        assert "gaze_tracker" in error_handler.recovery_strategies
        assert "posture_analyzer" in error_handler.recovery_strategies
    
    def test_recovery_strategies(self):
        """Test default recovery strategies"""
        config = LoggingConfig()
        error_handler = setup_error_handling(config)
        
        # Test camera recovery strategy
        camera_strategy = error_handler.recovery_strategies["video_capture"]
        result = camera_strategy(Exception("Camera error"), {})
        assert result is True  # Simplified recovery always succeeds
        
        # Test model recovery strategy
        model_strategy = error_handler.recovery_strategies["object_detector"]
        result = model_strategy(Exception("Model error"), {})
        assert result is True  # Simplified recovery always succeeds


class TestErrorEvent:
    """Test ErrorEvent dataclass"""
    
    def test_error_event_creation(self):
        """Test ErrorEvent creation"""
        timestamp = datetime.now()
        event = ErrorEvent(
            timestamp=timestamp,
            component="test_component",
            error_type="ValueError",
            severity=ErrorSeverity.HIGH,
            message="Test error message",
            traceback_info="Test traceback",
            context={"key": "value"},
            recovery_attempted=True,
            recovery_successful=False
        )
        
        assert event.timestamp == timestamp
        assert event.component == "test_component"
        assert event.error_type == "ValueError"
        assert event.severity == ErrorSeverity.HIGH
        assert event.message == "Test error message"
        assert event.traceback_info == "Test traceback"
        assert event.context == {"key": "value"}
        assert event.recovery_attempted is True
        assert event.recovery_successful is False


class TestErrorSeverity:
    """Test ErrorSeverity enum"""
    
    def test_error_severity_values(self):
        """Test ErrorSeverity enum values"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_severity_comparison(self):
        """Test ErrorSeverity comparison"""
        # Test that we can compare severity levels
        severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        
        for i, severity in enumerate(severities):
            assert severity in ErrorSeverity
            assert isinstance(severity.value, str)


class TestIntegrationWithMainSystem:
    """Test integration with main system"""
    
    def test_main_system_error_handling_integration(self):
        """Test that main system properly integrates error handling"""
        from anti_cheat_system.main import AntiCheatSystem
        from anti_cheat_system.models import SystemConfig
        
        config = SystemConfig()
        config.logging.log_to_console = False
        config.logging.log_to_file = False
        
        system = AntiCheatSystem(config)
        
        # Check error handler was initialized
        assert hasattr(system, 'error_handler')
        assert isinstance(system.error_handler, ErrorHandler)
        
        # Check resource monitor was initialized
        assert hasattr(system, 'resource_monitor')
        assert isinstance(system.resource_monitor, ResourceMonitor)
        
        # Check logger is from error handler
        assert system.logger == system.error_handler.logger
    
    def test_recovery_strategies_integration(self):
        """Test that recovery strategies work with main system"""
        from anti_cheat_system.main import AntiCheatSystem
        from anti_cheat_system.models import SystemConfig
        
        config = SystemConfig()
        config.logging.log_to_console = False
        config.logging.log_to_file = False
        
        system = AntiCheatSystem(config)
        
        # Check that recovery strategies are registered
        assert "video_capture" in system.error_handler.recovery_strategies
        assert "object_detector" in system.error_handler.recovery_strategies
        assert "gaze_tracker" in system.error_handler.recovery_strategies
        assert "posture_analyzer" in system.error_handler.recovery_strategies
        assert "scoring_system" in system.error_handler.recovery_strategies
        assert "ui_display" in system.error_handler.recovery_strategies
    
    @patch('anti_cheat_system.error_handler.cv2')
    def test_camera_recovery_strategy(self, mock_cv2):
        """Test camera recovery strategy"""
        from anti_cheat_system.error_handler import _camera_recovery_strategy
        
        # Mock successful camera recovery
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        
        error = Exception("Camera connection failed")
        context = {'camera_index': 0}
        
        result = _camera_recovery_strategy(error, context)
        assert result is True
        assert 'recovered_camera_index' in context
    
    @patch('anti_cheat_system.error_handler.torch')
    @patch('anti_cheat_system.error_handler.Path')
    def test_model_recovery_strategy(self, mock_path, mock_torch):
        """Test model recovery strategy"""
        from anti_cheat_system.error_handler import _model_recovery_strategy
        
        # Mock successful model recovery
        mock_torch.cuda.is_available.return_value = True
        mock_path.return_value.exists.return_value = True
        
        error = Exception("Model loading failed")
        context = {'model_path': 'yolov8n.pt'}
        
        with patch('anti_cheat_system.error_handler.YOLO'):
            result = _model_recovery_strategy(error, context)
            assert result is True
    
    @patch('anti_cheat_system.error_handler.mediapipe')
    def test_mediapipe_recovery_strategy(self, mock_mp):
        """Test MediaPipe recovery strategy"""
        from anti_cheat_system.error_handler import _mediapipe_recovery_strategy
        
        # Mock successful MediaPipe recovery
        mock_face_mesh = Mock()
        mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
        
        error = Exception("MediaPipe initialization failed")
        context = {'component_type': 'face_mesh'}
        
        result = _mediapipe_recovery_strategy(error, context)
        assert result is True
    
    def test_error_statistics_comprehensive(self):
        """Test comprehensive error statistics"""
        config = LoggingConfig(log_to_console=False, log_to_file=False)
        error_handler = ErrorHandler(config)
        
        # Generate various types of errors
        components = ['video_capture', 'object_detector', 'gaze_tracker', 'posture_analyzer']
        severities = [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        
        for i, component in enumerate(components):
            for j, severity in enumerate(severities):
                error_handler.handle_error(
                    component,
                    Exception(f"Error {i}-{j}"),
                    severity,
                    context={'test_id': f"{i}-{j}"}
                )
        
        stats = error_handler.get_error_statistics()
        
        # Check total counts
        assert stats['total_errors'] == len(components) * len(severities)
        
        # Check component breakdown
        for component in components:
            assert component in stats['component_breakdown']
            assert stats['component_breakdown'][component]['total_errors'] == len(severities)
        
        # Check severity breakdown
        for severity in severities:
            assert stats['severity_breakdown'][severity.value] == len(components)
    
    def test_resource_monitoring_callbacks(self):
        """Test resource monitoring with callbacks"""
        config = LoggingConfig(log_to_console=False, log_to_file=False)
        error_handler = ErrorHandler(config)
        resource_monitor = ResourceMonitor(error_handler)
        
        # Add callback
        callback_called = []
        def test_callback(stats):
            callback_called.append(stats)
        
        resource_monitor.add_callback(test_callback)
        
        # Force resource check
        resource_monitor.last_check_time = 0
        with patch('anti_cheat_system.error_handler.psutil'):
            resource_monitor.check_resources()
        
        # Start monitoring briefly
        resource_monitor.start_monitoring(0.1)
        time.sleep(0.2)
        resource_monitor.stop_monitoring()
        
        # Check callback was called
        assert len(callback_called) > 0