"""
Unit tests for gaze tracking engine
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from anti_cheat_system.detectors.gaze_tracker import GazeTracker
from anti_cheat_system.models import GazeTrackingConfig, FaceMeshResult, FaceLandmarks


class TestGazeTracker:
    """Test GazeTracker class"""
    
    def test_gaze_tracker_init_default_config(self):
        """Test GazeTracker initialization with default config"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            
            assert tracker.config is not None
            assert tracker.config.max_num_faces == 1
            assert tracker.config.gaze_deviation_threshold == 0.22
            assert tracker.config.weight == 0.30
    
    def test_gaze_tracker_init_custom_config(self):
        """Test GazeTracker initialization with custom config"""
        config = GazeTrackingConfig(
            max_num_faces=2,
            gaze_deviation_threshold=0.3,
            gaze_duration_threshold=5.0
        )
        
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker(config)
            
            assert tracker.config.max_num_faces == 2
            assert tracker.config.gaze_deviation_threshold == 0.3
            assert tracker.config.gaze_duration_threshold == 5.0
    
    def test_face_mesh_initialization_success(self):
        """Test successful MediaPipe Face Mesh initialization"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            
            assert tracker.is_initialized is True
            mock_mp.solutions.face_mesh.FaceMesh.assert_called_once()
    
    def test_face_mesh_initialization_failure(self):
        """Test MediaPipe Face Mesh initialization failure"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_mp.solutions.face_mesh.FaceMesh.side_effect = Exception("MediaPipe init failed")
            
            tracker = GazeTracker()
            
            assert tracker.is_initialized is False
    
    def test_detect_face_mesh_no_face(self):
        """Test face mesh detection with no face detected"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_results = Mock()
            mock_results.multi_face_landmarks = None
            mock_face_mesh.process.return_value = mock_results
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            result = tracker.detect_face_mesh(frame)
            
            assert isinstance(result, FaceMeshResult)
            assert result.face_detected is False
            assert result.landmarks is None
            assert result.gaze_deviation_percent == 0.0
    
    def test_detect_face_mesh_with_face(self):
        """Test face mesh detection with face detected"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            # Mock face landmarks
            mock_landmark = Mock()
            mock_landmark.x = 0.5
            mock_landmark.y = 0.5
            mock_landmark.z = 0.0
            
            mock_face_landmarks = Mock()
            mock_face_landmarks.landmark = [mock_landmark] * 468  # 468 face landmarks
            
            mock_results = Mock()
            mock_results.multi_face_landmarks = [mock_face_landmarks]
            
            mock_face_mesh = Mock()
            mock_face_mesh.process.return_value = mock_results
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            result = tracker.detect_face_mesh(frame)
            
            assert result.face_detected is True
            assert result.landmarks is not None
            assert len(result.landmarks) == 468
    
    def test_extract_facial_features(self):
        """Test facial feature extraction from landmarks"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Create mock landmarks array (468 landmarks with x, y, z)
            landmarks = np.random.rand(468, 3) * 100
            
            # Set specific landmark positions for testing
            landmarks[tracker.config.left_eye_idx] = [100, 150, 0]
            landmarks[tracker.config.right_eye_idx] = [200, 150, 0]
            landmarks[tracker.config.nose_tip_idx] = [150, 180, 0]
            landmarks[tracker.config.mouth_center_idx] = [150, 220, 0]
            
            features = tracker._extract_facial_features(landmarks)
            
            assert isinstance(features, FaceLandmarks)
            assert features.left_eye == (100.0, 150.0)
            assert features.right_eye == (200.0, 150.0)
            assert features.nose_tip == (150.0, 180.0)
            assert features.mouth_center == (150.0, 220.0)
    
    def test_calculate_nose_offset(self):
        """Test nose offset calculation"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Create facial features with known positions
            features = FaceLandmarks(
                left_eye=(100.0, 150.0),
                right_eye=(200.0, 150.0),
                nose_tip=(160.0, 180.0),  # 10 pixels right of center
                mouth_center=(150.0, 220.0)
            )
            
            offset = tracker._calculate_nose_offset(features)
            
            # Eye center is at (150, 150), nose is at (160, 180)
            # Horizontal offset should be 160 - 150 = 10
            assert offset == 10.0
    
    def test_calculate_gaze_deviation(self):
        """Test gaze deviation calculation"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Test with known values
            nose_offset = 20.0  # 20 pixels offset
            eye_distance = 100.0  # 100 pixels between eyes
            
            deviation = tracker._calculate_gaze_deviation(nose_offset, eye_distance)
            
            # Deviation should be |20| / 100 = 0.2 (20%)
            assert deviation == 0.2
    
    def test_calculate_gaze_deviation_zero_eye_distance(self):
        """Test gaze deviation calculation with zero eye distance"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            deviation = tracker._calculate_gaze_deviation(20.0, 0.0)
            
            # Should return 0 when eye distance is 0
            assert deviation == 0.0
    
    def test_calculate_gaze_score_no_face(self):
        """Test gaze score calculation with no face detected"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            result = FaceMeshResult(
                landmarks=None,
                nose_offset=0.0,
                eye_distance=0.0,
                gaze_deviation_percent=0.0,
                sustained_deviation=False,
                face_detected=False
            )
            
            score = tracker.calculate_gaze_score(result)
            assert score == tracker.config.base_score
    
    def test_calculate_gaze_score_normal_gaze(self):
        """Test gaze score calculation with normal gaze"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Deviation below threshold (22%)
            result = FaceMeshResult(
                landmarks=np.zeros((468, 3)),
                nose_offset=5.0,
                eye_distance=100.0,
                gaze_deviation_percent=0.1,  # 10% deviation
                sustained_deviation=False,
                face_detected=True
            )
            
            score = tracker.calculate_gaze_score(result)
            assert score == 0.0  # Should be 0 for normal gaze
    
    def test_calculate_gaze_score_high_deviation(self):
        """Test gaze score calculation with high gaze deviation"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # High deviation above threshold
            result = FaceMeshResult(
                landmarks=np.zeros((468, 3)),
                nose_offset=30.0,
                eye_distance=100.0,
                gaze_deviation_percent=0.4,  # 40% deviation
                sustained_deviation=False,
                face_detected=True
            )
            
            score = tracker.calculate_gaze_score(result)
            assert score > 0.0  # Should be positive for high deviation
    
    def test_calculate_gaze_score_sustained_deviation(self):
        """Test gaze score calculation with sustained deviation"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Sustained deviation
            result = FaceMeshResult(
                landmarks=np.zeros((468, 3)),
                nose_offset=25.0,
                eye_distance=100.0,
                gaze_deviation_percent=0.3,  # 30% deviation
                sustained_deviation=True,
                face_detected=True
            )
            
            score = tracker.calculate_gaze_score(result)
            
            # Should have higher score due to sustained deviation
            assert score >= 0.4  # Should include sustained boost
    
    def test_check_sustained_deviation(self):
        """Test sustained deviation detection logic"""
        config = GazeTrackingConfig(
            gaze_deviation_threshold=0.22,
            gaze_duration_threshold=3.0
        )
        
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker(config)
            
            # Test below threshold
            assert tracker._check_sustained_deviation(0.1) is False
            
            # Test above threshold but no duration
            tracker.sustained_deviation_duration = 1.0
            assert tracker._check_sustained_deviation(0.3) is False
            
            # Test above threshold with sufficient duration
            tracker.sustained_deviation_duration = 4.0
            assert tracker._check_sustained_deviation(0.3) is True
    
    def test_get_gaze_stats(self):
        """Test gaze statistics retrieval"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            stats = tracker.get_gaze_stats()
            
            assert 'is_initialized' in stats
            assert 'total_frames_processed' in stats
            assert 'total_faces_detected' in stats
            assert 'face_detection_rate_percent' in stats
            assert 'avg_inference_time_ms' in stats
            assert 'avg_gaze_deviation' in stats
            assert 'deviation_threshold' in stats
            assert 'duration_threshold' in stats
    
    def test_reset_deviation_tracking(self):
        """Test deviation tracking reset"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            # Add some tracking data
            tracker.deviation_history.extend([0.1, 0.2, 0.3])
            tracker.deviation_start_time = 12345.0
            tracker.sustained_deviation_duration = 2.0
            
            tracker.reset_deviation_tracking()
            
            assert len(tracker.deviation_history) == 0
            assert tracker.deviation_start_time is None
            assert tracker.sustained_deviation_duration == 0.0
    
    def test_calibrate_baseline(self):
        """Test baseline calibration"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            
            # Mock successful face detection for calibration
            with patch.object(tracker, 'detect_face_mesh') as mock_detect:
                mock_result = FaceMeshResult(
                    landmarks=np.zeros((468, 3)),
                    nose_offset=5.0,
                    eye_distance=100.0,
                    gaze_deviation_percent=0.1,
                    sustained_deviation=False,
                    face_detected=True
                )
                mock_detect.return_value = mock_result
                
                frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
                success = tracker.calibrate_baseline(frames)
                
                assert success is True
                assert tracker.baseline_nose_offset == 5.0
                assert tracker.baseline_eye_distance == 100.0
    
    def test_calibrate_baseline_insufficient_detections(self):
        """Test baseline calibration with insufficient face detections"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp') as mock_mp:
            mock_face_mesh = Mock()
            mock_mp.solutions.face_mesh.FaceMesh.return_value = mock_face_mesh
            
            tracker = GazeTracker()
            
            # Mock failed face detection for calibration
            with patch.object(tracker, 'detect_face_mesh') as mock_detect:
                mock_result = FaceMeshResult(
                    landmarks=None,
                    nose_offset=0.0,
                    eye_distance=0.0,
                    gaze_deviation_percent=0.0,
                    sustained_deviation=False,
                    face_detected=False
                )
                mock_detect.return_value = mock_result
                
                frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
                success = tracker.calibrate_baseline(frames)
                
                assert success is False
                assert tracker.baseline_nose_offset is None
                assert tracker.baseline_eye_distance is None
    
    def test_draw_face_mesh_no_face(self):
        """Test drawing face mesh with no face detected"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = FaceMeshResult(
                landmarks=None,
                nose_offset=0.0,
                eye_distance=0.0,
                gaze_deviation_percent=0.0,
                sustained_deviation=False,
                face_detected=False
            )
            
            output_frame = tracker.draw_face_mesh(frame, result)
            
            # Should return original frame unchanged
            assert np.array_equal(output_frame, frame)
    
    def test_draw_face_mesh_with_face(self):
        """Test drawing face mesh with face detected"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp'):
            tracker = GazeTracker()
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            landmarks = np.random.rand(468, 3) * 100
            
            result = FaceMeshResult(
                landmarks=landmarks,
                nose_offset=10.0,
                eye_distance=100.0,
                gaze_deviation_percent=0.3,
                sustained_deviation=True,
                face_detected=True
            )
            
            output_frame = tracker.draw_face_mesh(frame, result)
            
            # Frame should be modified (not equal to original)
            assert not np.array_equal(output_frame, frame)
    
    def test_mediapipe_not_available(self):
        """Test behavior when MediaPipe is not available"""
        with patch('anti_cheat_system.detectors.gaze_tracker.mp', None):
            tracker = GazeTracker()
            
            assert tracker.is_initialized is False
            
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = tracker.detect_face_mesh(frame)
            
            assert result.face_detected is False
            assert result.landmarks is None