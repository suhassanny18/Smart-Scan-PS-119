"""
Main application controller for the Anti-Cheat Detection System
Coordinates all detection engines and provides real-time monitoring
"""

import cv2
import numpy as np
import logging
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from anti_cheat_system.video_capture import VideoCapture
from anti_cheat_system.scoring_system import ScoringSystem
from anti_cheat_system.detectors import ObjectDetector, GazeTracker, PostureAnalyzer
from anti_cheat_system.ui_display_simple import UIDisplay

# Import error handling with fallback
try:
    from anti_cheat_system.error_handler import ErrorHandler, ResourceMonitor, ErrorSeverity, setup_error_handling
except ImportError:
    # Fallback error handling
    from enum import Enum
    import logging
    
    class ErrorSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class ErrorHandler:
        def __init__(self, config=None):
            self.logger = logging.getLogger("anti_cheat_system")
            self.error_count = 0
            self.max_errors = 20
        
        def handle_error(self, component, error, severity=None, context=None, attempt_recovery=True):
            self.error_count += 1
            self.logger.error(f"[{component}] {type(error).__name__}: {str(error)}")
            
            # For HIGH and CRITICAL errors, return False to stop initialization
            if severity == ErrorSeverity.HIGH or severity == ErrorSeverity.CRITICAL:
                return False
            
            return self.error_count < self.max_errors
    
    class ResourceMonitor:
        def __init__(self, error_handler):
            self.error_handler = error_handler
            self.stats = {}
        
        def check_resources(self):
            pass
        
        def get_stats(self):
            return {}
    
    def setup_error_handling(config=None):
        return ErrorHandler(config)

from anti_cheat_system.models import (
    SystemConfig,
    SystemResult,
    AlertLevel,
    DetectionEvent
)
from anti_cheat_system.models.enums import SystemState


class AntiCheatSystem:
    """
    Main application controller that coordinates all detection engines
    and provides real-time anti-cheat monitoring
    """
    
    def __init__(self, config: SystemConfig = None, debug: bool = False):
        """
        Initialize the anti-cheat system
        
        Args:
            config: System configuration
            debug: Enable debug mode
        """
        self.config = config or SystemConfig()
        
        # Setup comprehensive error handling first
        self.error_handler = setup_error_handling(self.config.logging)
        self.resource_monitor = ResourceMonitor(self.error_handler)
        
        # Component initialization flags
        self.is_initialized = False
        self.is_running = False
        self.should_stop = False
        self.system_state = SystemState.INITIALIZING
        
        # Core components
        self.video_capture = None
        self.object_detector = None
        self.gaze_tracker = None
        self.posture_analyzer = None
        self.scoring_system = None
        self.ui_display = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Enhanced error handling (legacy support)
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = None
        
        # Threading
        self.processing_thread = None
        self.display_thread = None
        self.stop_event = threading.Event()
        
        # Setup logging (enhanced)
        self.logger = self.error_handler.logger
        
        self.logger.info("Anti-Cheat System initialized with enhanced error handling")
    
    def initialize(self) -> bool:
        """
        Initialize the system (wrapper for initialize_components)
        
        Returns:
            bool: True if initialization successful
        """
        return self.initialize_components()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.logging.log_to_file:
            # Create logs directory if it doesn't exist
            log_dir = Path(self.config.logging.log_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure file handler
            log_file = log_dir / self.config.logging.log_filename
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, self.config.logging.log_level))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
        
        if self.config.logging.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.config.logging.log_level))
            self.logger.addHandler(console_handler)
        
        self.logger.setLevel(getattr(logging, self.config.logging.log_level))
    
    def initialize_components(self) -> bool:
        """
        Initialize all system components with comprehensive error handling
        
        Returns:
            bool: True if all components initialized successfully
        """
        try:
            self.logger.info("Initializing system components...")
            self.system_state = SystemState.INITIALIZING
            
            # Validate configuration
            if not self.config.validate():
                self.error_handler.handle_error(
                    "system_config",
                    Exception("Configuration validation failed"),
                    ErrorSeverity.CRITICAL,
                    attempt_recovery=False
                )
                return False
            
            # Initialize video capture with error handling
            self.logger.info("Initializing video capture...")
            try:
                self.video_capture = VideoCapture(self.config.video)
                if not self.video_capture.initialize_camera():
                    raise Exception("Camera initialization failed")
            except Exception as e:
                context = {
                    'camera_index': self.config.video.camera_index,
                    'frame_width': self.config.video.frame_width,
                    'frame_height': self.config.video.frame_height
                }
                if not self.error_handler.handle_error("video_capture", e, ErrorSeverity.HIGH, context):
                    return False
                # If recovery was successful, try reinitializing with recovered settings
                if hasattr(self.error_handler, 'error_history') and self.error_handler.error_history:
                    last_error = self.error_handler.error_history[-1]
                    if (last_error.recovery_successful and 
                        'recovered_camera_index' in last_error.context):
                        self.config.video.camera_index = last_error.context['recovered_camera_index']
                        self.video_capture = VideoCapture(self.config.video)
            
            # Initialize object detector with error handling
            self.logger.info("Initializing object detector...")
            try:
                self.object_detector = ObjectDetector(self.config.object_detection)
                if not self.object_detector.is_initialized:
                    raise Exception("Object detector initialization failed")
                self.logger.info("Object detector initialized successfully")
            except Exception as e:
                self.logger.warning(f"Real object detector failed: {e}")
                self.logger.warning("Using dummy object detector as fallback")
                self.object_detector = self._create_dummy_object_detector()
            
            # Initialize gaze tracker with error handling
            self.logger.info("Initializing gaze tracker...")
            try:
                self.gaze_tracker = GazeTracker(self.config.gaze_tracking)
                if not self.gaze_tracker.is_initialized:
                    raise Exception("Gaze tracker initialization failed")
                self.logger.info("Gaze tracker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Real gaze tracker failed: {e}")
                self.logger.warning("Using dummy gaze tracker as fallback")
                self.gaze_tracker = self._create_dummy_gaze_tracker()
            
            # Initialize posture analyzer with error handling
            self.logger.info("Initializing posture analyzer...")
            try:
                self.posture_analyzer = PostureAnalyzer(self.config.posture_analysis)
                if not self.posture_analyzer.is_initialized:
                    raise Exception("Posture analyzer initialization failed")
                self.logger.info("Posture analyzer initialized successfully")
            except Exception as e:
                self.logger.warning(f"Real posture analyzer failed: {e}")
                self.logger.warning("Using dummy posture analyzer as fallback")
                self.posture_analyzer = self._create_dummy_posture_analyzer()
            
            # Initialize scoring system
            self.logger.info("Initializing scoring system...")
            try:
                self.scoring_system = ScoringSystem(self.config.scoring)
            except Exception as e:
                if not self.error_handler.handle_error("scoring_system", e, ErrorSeverity.MEDIUM):
                    return False
            
            # Initialize UI display
            self.logger.info("Initializing UI display...")
            try:
                self.ui_display = UIDisplay(self.config)
            except Exception as e:
                # UI display failure is not critical
                self.error_handler.handle_error("ui_display", e, ErrorSeverity.LOW)
                self.ui_display = None
            
            self.is_initialized = True
            self.system_state = SystemState.RUNNING
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(
                "system_initialization",
                e,
                ErrorSeverity.CRITICAL,
                attempt_recovery=False
            )
            self.system_state = SystemState.ERROR
            return False
    
    def process_frame(self, frame: np.ndarray) -> Optional[SystemResult]:
        """
        Process a single frame through all detection engines with comprehensive error handling
        
        Args:
            frame: Input video frame
            
        Returns:
            SystemResult: Complete system result or None if processing failed
        """
        if not self.is_initialized:
            self.error_handler.handle_error(
                "frame_processing",
                Exception("System not initialized"),
                ErrorSeverity.HIGH,
                attempt_recovery=False
            )
            return None
        
        try:
            start_time = time.time()
            self.frame_count += 1
            
            # Monitor system resources periodically
            if self.frame_count % 30 == 0:  # Every 30 frames
                self.resource_monitor.check_resources()
            
            # Initialize results with defaults for error recovery
            detection_result = None
            face_mesh_result = None
            pose_result = None
            object_score = 0.0
            gaze_score = 0.0
            posture_score = 0.0
            
            # Object detection with error handling
            try:
                detection_result = self.object_detector.detect_objects(frame)
                object_score = self.object_detector.calculate_object_score(detection_result)
            except Exception as e:
                self.error_handler.handle_error(
                    "object_detector",
                    e,
                    ErrorSeverity.MEDIUM,
                    {"frame_number": self.frame_count}
                )
                # Continue with default values
            
            # Gaze tracking with error handling
            try:
                face_mesh_result = self.gaze_tracker.detect_face_mesh(frame)
                gaze_score = self.gaze_tracker.calculate_gaze_score(face_mesh_result)
            except Exception as e:
                self.error_handler.handle_error(
                    "gaze_tracker",
                    e,
                    ErrorSeverity.MEDIUM,
                    {"frame_number": self.frame_count}
                )
                # Continue with default values
            
            # Posture analysis with error handling
            try:
                pose_result = self.posture_analyzer.detect_pose(frame)
                posture_score = self.posture_analyzer.calculate_posture_score(pose_result)
            except Exception as e:
                self.error_handler.handle_error(
                    "posture_analyzer",
                    e,
                    ErrorSeverity.MEDIUM,
                    {"frame_number": self.frame_count}
                )
                # Continue with default values
            
            # Composite scoring with error handling
            try:
                system_result = self.scoring_system.process_detection_results(
                    detection_result=detection_result,
                    face_mesh_result=face_mesh_result,
                    pose_result=pose_result,
                    object_score=object_score,
                    gaze_score=gaze_score,
                    posture_score=posture_score,
                    frame_number=self.frame_count
                )
            except Exception as e:
                self.error_handler.handle_error(
                    "scoring_system",
                    e,
                    ErrorSeverity.HIGH,
                    {"frame_number": self.frame_count}
                )
                return None
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            system_result.processing_time_ms = processing_time
            
            # Update FPS counter
            self._update_fps_counter()
            
            # Log high-priority alerts
            if system_result.alert_level == AlertLevel.RED:
                self.logger.critical(f"RED ALERT - Frame {self.frame_count}: Score {system_result.composite_score:.3f}")
            elif system_result.alert_level == AlertLevel.AMBER:
                self.logger.warning(f"AMBER ALERT - Frame {self.frame_count}: Score {system_result.composite_score:.3f}")
            
            return system_result
            
        except Exception as e:
            self.error_handler.handle_error(
                "frame_processing",
                e,
                ErrorSeverity.HIGH,
                {"frame_number": self.frame_count}
            )
            self._handle_processing_error()
            return None
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _handle_processing_error(self):
        """Handle processing errors with graceful degradation"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        if self.error_count >= self.max_errors:
            self.logger.critical(f"Too many errors ({self.error_count}), stopping system")
            self.should_stop = True
    
    def run(self) -> bool:
        """
        Run the anti-cheat system in real-time mode
        
        Returns:
            bool: True if system ran successfully
        """
        if not self.is_initialized:
            if not self.initialize_components():
                return False
        
        try:
            self.logger.info("Starting anti-cheat system...")
            self.is_running = True
            self.should_stop = False
            self.start_time = time.time()
            
            # Main processing loop
            while not self.should_stop:
                # Get frame from video capture
                ret, frame = self.video_capture.get_frame()
                if not ret or frame is None:
                    self.logger.warning("Failed to get frame from video capture")
                    time.sleep(0.1)  # Brief pause before retrying
                    continue
                
                # Process frame
                system_result = self.process_frame(frame)
                if system_result is None:
                    continue
                
                # Display results if enabled
                if self.config.video.show_video:
                    display_frame = self._create_display_frame(frame, system_result)
                    cv2.imshow('Anti-Cheat Detection System', display_frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        self.logger.info("Exit key pressed")
                        break
                
                # Frame rate control
                if self.config.video.target_fps > 0:
                    target_delay = 1.0 / self.config.video.target_fps
                    time.sleep(max(0, target_delay - (time.time() - self.start_time)))
            
            self.logger.info("Anti-cheat system stopped")
            return True
            
        except KeyboardInterrupt:
            self.logger.info("System interrupted by user")
            return True
        except Exception as e:
            self.logger.error(f"System runtime error: {e}")
            return False
        finally:
            self._cleanup()
    
    def _create_display_frame(self, frame: np.ndarray, system_result: SystemResult) -> np.ndarray:
        """
        Create display frame with detection results and alerts
        
        Args:
            frame: Original video frame
            system_result: System detection results
            
        Returns:
            np.ndarray: Frame with overlays
        """
        if self.ui_display is None:
            # Fallback to basic display if UI display not initialized
            return self._create_basic_display_frame(frame, system_result)
        
        # Get system statistics for enhanced display
        detection_stats = self.get_system_stats()
        
        # Use enhanced UI display with dashboard
        if hasattr(self.config.video, 'show_dashboard') and self.config.video.show_dashboard:
            return self.ui_display.create_enhanced_display(frame, system_result, detection_stats)
        else:
            return self.ui_display.create_simple_display(frame, system_result)
    
    def _create_basic_display_frame(self, frame: np.ndarray, system_result: SystemResult) -> np.ndarray:
        """
        Create basic display frame (fallback method)
        
        Args:
            frame: Original video frame
            system_result: System detection results
            
        Returns:
            np.ndarray: Frame with basic overlays
        """
        display_frame = frame.copy()
        
        # Draw object detections
        if (self.config.video.show_detections and 
            system_result.detection_result and 
            system_result.detection_result.suspicious_objects_found):
            display_frame = self.object_detector.draw_detections(
                display_frame, system_result.detection_result
            )
        
        # Draw gaze tracking
        if (self.config.video.show_landmarks and 
            system_result.face_mesh_result and 
            system_result.face_mesh_result.face_detected):
            display_frame = self.gaze_tracker.draw_face_mesh(
                display_frame, system_result.face_mesh_result
            )
        
        # Draw pose analysis
        if (self.config.video.show_landmarks and 
            system_result.pose_result and 
            system_result.pose_result.pose_detected):
            display_frame = self.posture_analyzer.draw_pose(
                display_frame, system_result.pose_result
            )
        
        # Draw system status and alerts
        self._draw_system_status(display_frame, system_result)
        
        return display_frame
    
    def _draw_system_status(self, frame: np.ndarray, system_result: SystemResult):
        """
        Draw system status information on frame
        
        Args:
            frame: Frame to draw on
            system_result: System results
        """
        height, width = frame.shape[:2]
        
        # Alert level indicator
        alert_color = {
            AlertLevel.NORMAL: (0, 255, 0),    # Green
            AlertLevel.AMBER: (0, 165, 255),   # Orange
            AlertLevel.RED: (0, 0, 255)        # Red
        }
        
        color = alert_color.get(system_result.alert_level, (255, 255, 255))
        
        # Alert status
        alert_text = f"ALERT: {system_result.alert_level.value}"
        cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Composite score
        score_text = f"Score: {system_result.composite_score:.3f}"
        cv2.putText(frame, score_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Individual scores
        scores_text = f"Obj:{system_result.object_score:.2f} Gaze:{system_result.gaze_score:.2f} Pose:{system_result.posture_score:.2f}"
        cv2.putText(frame, scores_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS and frame count
        fps_text = f"FPS: {self.current_fps:.1f} | Frame: {self.frame_count}"
        cv2.putText(frame, fps_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Processing time
        time_text = f"Processing: {system_result.processing_time_ms:.1f}ms"
        cv2.putText(frame, time_text, (width - 200, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def stop(self):
        """Stop the anti-cheat system"""
        self.logger.info("Stopping anti-cheat system...")
        self.should_stop = True
        self.is_running = False
        
        if self.stop_event:
            self.stop_event.set()
    
    def _cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up system resources...")
        
        try:
            if self.video_capture:
                self.video_capture.release()
            
            if cv2.getWindowProperty('Anti-Cheat Detection System', cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyAllWindows()
            
            self.is_running = False
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        
        Returns:
            dict: System statistics
        """
        runtime = time.time() - self.start_time if self.start_time else 0
        
        stats = {
            'system': {
                'is_initialized': self.is_initialized,
                'is_running': self.is_running,
                'runtime_seconds': runtime,
                'frame_count': self.frame_count,
                'current_fps': self.current_fps,
                'error_count': self.error_count,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
            }
        }
        
        # Add component statistics
        if self.object_detector:
            stats['object_detector'] = self.object_detector.get_detection_stats()
        
        if self.gaze_tracker:
            stats['gaze_tracker'] = self.gaze_tracker.get_gaze_stats()
        
        if self.posture_analyzer:
            stats['posture_analyzer'] = self.posture_analyzer.get_posture_stats()
        
        if self.scoring_system:
            stats['scoring_system'] = self.scoring_system.get_scoring_stats()
        
        return stats
    
    def reset_statistics(self):
        """Reset all system statistics"""
        self.frame_count = 0
        self.error_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.last_error_time = None
        
        if self.object_detector:
            self.object_detector.reset_detection_history()
        
        if self.gaze_tracker:
            self.gaze_tracker.reset_deviation_tracking()
        
        if self.posture_analyzer:
            self.posture_analyzer.reset_posture_tracking()
        
        if self.scoring_system:
            self.scoring_system.reset_statistics()
        
        self.logger.info("System statistics reset")
    
    def _create_dummy_object_detector(self):
        """Create a dummy object detector for testing when real detector fails"""
        class DummyObjectDetector:
            def __init__(self):
                self.is_initialized = True
            
            def detect_objects(self, frame):
                return None
            
            def calculate_object_score(self, detection_result):
                return 0.0
            
            def draw_detections(self, frame, detection_result):
                return frame
            
            def get_detection_stats(self):
                return {"dummy": True, "detections": 0}
        
        return DummyObjectDetector()
    
    def _create_dummy_gaze_tracker(self):
        """Create a dummy gaze tracker for testing when real tracker fails"""
        class DummyGazeTracker:
            def __init__(self):
                self.is_initialized = True
            
            def detect_face_mesh(self, frame):
                return None
            
            def calculate_gaze_score(self, face_mesh_result):
                return 0.0
            
            def draw_face_mesh(self, frame, face_mesh_result):
                return frame
            
            def get_gaze_stats(self):
                return {"dummy": True, "gaze_deviations": 0}
        
        return DummyGazeTracker()
    
    def _create_dummy_posture_analyzer(self):
        """Create a dummy posture analyzer for testing when real analyzer fails"""
        class DummyPostureAnalyzer:
            def __init__(self):
                self.is_initialized = True
            
            def detect_pose(self, frame):
                return None
            
            def calculate_posture_score(self, pose_result):
                return 0.0
            
            def draw_pose(self, frame, pose_result):
                return frame
            
            def get_posture_stats(self):
                return {"dummy": True, "posture_violations": 0}
        
        return DummyPostureAnalyzer()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        self._cleanup()


def main():
    """Main entry point for the application"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Anti-Cheat Detection System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig()
    if args.debug:
        config.debug_mode = True
        config.logging.log_level = "DEBUG"
    
    if args.no_display:
        config.video.show_video = False
    
    # Create and run system
    with AntiCheatSystem(config) as system:
        if system.initialize_components():
            system.run()
        else:
            print("Failed to initialize system components")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())