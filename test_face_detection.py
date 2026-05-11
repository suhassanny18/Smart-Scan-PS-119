"""
Unit tests for multi-face detection engine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from anti_cheat_system.detectors.face_recognition import (
    MultiFaceDetector, FaceDetectionConfig, DetectionMetrics
)
from anti_cheat_system.models.data_models import FaceDetection, BoundingBox
from anti_cheat_system.models.enums import ModelType


class TestFaceDetectionConfig:
    """Test face detection configuration."""
    
    def test_config_creation(self):
        """Test creating face detection configuration."""
        config = FaceDetectionConfig(
            model_type=ModelType.YOLOV8_FACE,
            confidence_threshold=0.8,
            nms_threshold=0.4,
            max_detections=15,
            device="cpu"
        )
        
        assert config.model_type == ModelType.YOLOV8_FACE
        assert config.confidence_threshold == 0.8
        assert config.nms_threshold == 0.4
        assert config.max_detections == 15
        assert config.device == "cpu"
    
    def test_config_defaults(self):
        """Test configuration with default values."""
        config = FaceDetectionConfig()
        
        assert config.model_type == ModelType.YOLOV8_FACE
        assert config.confidence_threshold == 0.7
        assert config.nms_threshold == 0.45
        assert config.max_detections == 20
        assert config.device == "auto"
        assert config.input_size == (640, 640)


class TestDetectionMetrics:
    """Test detection metrics."""
    
    def test_metrics_creation(self):
        """Test creating detection metrics."""
        metrics = DetectionMetrics(
            total_detections=10,
            valid_detections=8,
            processing_time=0.05,
            fps=20.0
        )
        
        assert metrics.total_detections == 10
        assert metrics.valid_detections == 8
        assert metrics.processing_time == 0.05
        assert metrics.fps == 20.0
        assert isinstance(metrics.timestamp, datetime)


class TestMultiFaceDetector:
    """Test multi-face detector functionality."""
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Mock YOLO model for testing."""
        with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.overrides = {}
            mock_model.to.return_value = mock_model
            mock_model.model = Mock()  # Add model attribute for parameter counting
            mock_model.model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
            
            # Mock detection results
            mock_result = Mock()
            mock_result.boxes = Mock()
            mock_result.boxes.xyxy = Mock()
            mock_result.boxes.conf = Mock()
            
            # Create mock tensors
            mock_boxes = Mock()
            mock_boxes.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200], [300, 150, 400, 250]])
            mock_result.boxes.xyxy = mock_boxes
            
            mock_conf = Mock()
            mock_conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
            mock_result.boxes.conf = mock_conf
            
            # Mock model call to return results based on input
            def mock_model_call(*args, **kwargs):
                # Return multiple results for batch processing
                if isinstance(args[0], list) and len(args[0]) > 1:
                    return [mock_result for _ in args[0]]
                else:
                    return [mock_result]
            
            mock_model.side_effect = mock_model_call
            mock_yolo.return_value = mock_model
            
            yield mock_yolo, mock_model
    
    @pytest.fixture
    def face_detection_config(self):
        """Sample face detection configuration."""
        return FaceDetectionConfig(
            confidence_threshold=0.7,
            device="cpu",
            max_detections=10,
            batch_size=1
        )
    
    def test_detector_initialization(self, mock_yolo_model, face_detection_config):
        """Test face detector initialization."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        assert detector.config == face_detection_config
        assert detector.is_initialized is True
        assert detector.device == "cpu"
        assert mock_yolo.called
    
    def test_device_selection(self, mock_yolo_model):
        """Test automatic device selection."""
        mock_yolo, mock_model = mock_yolo_model
        
        # Test CPU selection
        config = FaceDetectionConfig(device="cpu")
        detector = MultiFaceDetector(config)
        assert detector.device == "cpu"
        
        # Test auto selection (should default to CPU in test environment)
        config = FaceDetectionConfig(device="auto")
        detector = MultiFaceDetector(config)
        assert detector.device in ["cpu", "cuda", "mps"]
    
    def test_single_frame_detection(self, mock_yolo_model, face_detection_config):
        """Test face detection on single frame."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect faces
        detections = detector.detect_faces(test_frame)
        
        # Verify results
        assert isinstance(detections, list)
        assert len(detections) == 2  # Based on mock data
        
        for detection in detections:
            assert isinstance(detection, FaceDetection)
            assert isinstance(detection.bbox, BoundingBox)
            assert 0.0 <= detection.confidence <= 1.0
            assert isinstance(detection.timestamp, datetime)
    
    def test_batch_detection(self, mock_yolo_model, face_detection_config):
        """Test batch face detection."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Create test frames
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        ]
        
        # Batch detect faces
        batch_detections = detector.batch_detect_faces(test_frames)
        
        # Verify results
        assert isinstance(batch_detections, list)
        assert len(batch_detections) == 2
        
        for detections in batch_detections:
            assert isinstance(detections, list)
            for detection in detections:
                assert isinstance(detection, FaceDetection)
    
    def test_face_crop_extraction(self, mock_yolo_model, face_detection_config):
        """Test face crop extraction."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Create test frame and detections
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_faces(test_frame)
        
        # Extract face crops
        face_crops = detector.extract_face_crops(test_frame, detections)
        
        # Verify results
        assert isinstance(face_crops, list)
        assert len(face_crops) == len(detections)
        
        for crop in face_crops:
            assert isinstance(crop, np.ndarray)
            assert crop.size > 0
    
    def test_detect_faces_with_crops(self, mock_yolo_model, face_detection_config):
        """Test combined detection and crop extraction."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect faces with crops
        detections, face_crops = detector.detect_faces_with_crops(test_frame)
        
        # Verify results
        assert isinstance(detections, list)
        assert isinstance(face_crops, list)
        assert len(detections) == len(face_crops)
    
    def test_face_validation(self, mock_yolo_model, face_detection_config):
        """Test face detection validation."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        frame_shape = (480, 640, 3)
        
        # Test valid face
        valid_bbox = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert detector._is_valid_face_detection(valid_bbox, frame_shape) is True
        
        # Test face outside frame bounds
        invalid_bbox = BoundingBox(x1=-10, y1=100, x2=200, y2=200)
        assert detector._is_valid_face_detection(invalid_bbox, frame_shape) is False
        
        # Test too small face
        small_bbox = BoundingBox(x1=100, y1=100, x2=110, y2=110)
        assert detector._is_valid_face_detection(small_bbox, frame_shape) is False
        
        # Test wrong aspect ratio
        wrong_ratio_bbox = BoundingBox(x1=100, y1=100, x2=400, y2=150)
        assert detector._is_valid_face_detection(wrong_ratio_bbox, frame_shape) is False
    
    def test_performance_metrics(self, mock_yolo_model, face_detection_config):
        """Test performance metrics collection."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Process some frames to generate metrics
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.detect_faces(test_frame)
        detector.detect_faces(test_frame)
        
        # Get metrics
        metrics = detector.get_performance_metrics()
        
        # Verify metrics structure
        assert isinstance(metrics, dict)
        assert "total_detections" in metrics
        assert "valid_detections" in metrics
        assert "current_fps" in metrics
        assert "avg_processing_time" in metrics
        assert "device" in metrics
        assert "model_type" in metrics
        assert "is_initialized" in metrics
        
        # Verify metrics values
        assert metrics["total_detections"] >= 0
        assert metrics["current_fps"] >= 0
        assert metrics["is_initialized"] is True
    
    def test_metrics_reset(self, mock_yolo_model, face_detection_config):
        """Test metrics reset functionality."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Process frame to generate metrics
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.detect_faces(test_frame)
        
        # Reset metrics
        detector.reset_metrics()
        
        # Verify reset
        metrics = detector.get_performance_metrics()
        assert metrics["total_detections"] == 0
        assert metrics["valid_detections"] == 0
    
    def test_config_update(self, mock_yolo_model, face_detection_config):
        """Test configuration update."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Update configuration
        new_config = FaceDetectionConfig(
            confidence_threshold=0.8,
            max_detections=15,
            device="cpu"
        )
        
        detector.update_config(new_config)
        
        # Verify update
        assert detector.config.confidence_threshold == 0.8
        assert detector.config.max_detections == 15
    
    def test_model_info(self, mock_yolo_model, face_detection_config):
        """Test model information retrieval."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Get model info
        model_info = detector.get_model_info()
        
        # Verify info structure
        assert isinstance(model_info, dict)
        assert "status" in model_info
        assert model_info["status"] == "initialized"
        assert "model_type" in model_info
        assert "device" in model_info
        assert "confidence_threshold" in model_info
    
    def test_error_handling(self, mock_yolo_model, face_detection_config):
        """Test error handling in face detection."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Test with None frame
        detections = detector.detect_faces(None)
        assert detections == []
        
        # Test with empty frame
        empty_frame = np.array([])
        detections = detector.detect_faces(empty_frame)
        assert detections == []
        
        # Test batch detection with empty list
        batch_detections = detector.batch_detect_faces([])
        assert batch_detections == []
        
        # Test batch detection with None frames
        batch_detections = detector.batch_detect_faces([None, None])
        assert len(batch_detections) == 2
        assert all(det == [] for det in batch_detections)
    
    def test_preprocessing(self, mock_yolo_model, face_detection_config):
        """Test frame preprocessing."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Test BGR to RGB conversion
        bgr_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed_frame = detector._preprocess_frame(bgr_frame)
        
        assert processed_frame is not None
        assert processed_frame.shape == bgr_frame.shape
        
        # Test with None frame
        processed_none = detector._preprocess_frame(None)
        assert processed_none is None
    
    def test_cleanup(self, mock_yolo_model, face_detection_config):
        """Test detector cleanup."""
        mock_yolo, mock_model = mock_yolo_model
        
        detector = MultiFaceDetector(face_detection_config)
        
        # Verify initialized state
        assert detector.is_initialized is True
        assert detector.model is not None
        
        # Cleanup
        detector.cleanup()
        
        # Verify cleanup
        assert detector.is_initialized is False
        assert detector.model is None


class TestIntegration:
    """Integration tests for face detection."""
    
    @pytest.fixture
    def mock_torch(self):
        """Mock torch for testing."""
        with patch('torch.cuda.is_available', return_value=False):
            yield
    
    def test_face_detection_pipeline(self, mock_torch):
        """Test complete face detection pipeline."""
        with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
            # Setup mock
            mock_model = Mock()
            mock_model.overrides = {}
            mock_model.to.return_value = mock_model
            mock_model.model = Mock()  # Add model attribute for parameter counting
            mock_model.model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
            
            # Mock successful detection
            mock_result = Mock()
            mock_result.boxes = Mock()
            
            mock_boxes = Mock()
            mock_boxes.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]])
            mock_result.boxes.xyxy = mock_boxes
            
            mock_conf = Mock()
            mock_conf.cpu.return_value.numpy.return_value = np.array([0.9])
            mock_result.boxes.conf = mock_conf
            
            # Mock model call to return results based on input
            def mock_model_call(*args, **kwargs):
                # Return multiple results for batch processing
                if isinstance(args[0], list) and len(args[0]) > 1:
                    return [mock_result for _ in args[0]]
                else:
                    return [mock_result]
            
            mock_model.side_effect = mock_model_call
            mock_yolo.return_value = mock_model
            
            # Create detector
            config = FaceDetectionConfig(device="cpu", confidence_threshold=0.7)
            detector = MultiFaceDetector(config)
            
            # Test detection pipeline
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Single detection
            detections = detector.detect_faces(test_frame)
            assert len(detections) == 1
            assert detections[0].confidence == 0.9
            
            # Detection with crops
            detections, crops = detector.detect_faces_with_crops(test_frame)
            assert len(detections) == 1
            assert len(crops) == 1
            
            # Batch detection
            batch_detections = detector.batch_detect_faces([test_frame, test_frame])
            assert len(batch_detections) == 2
            assert all(len(dets) == 1 for dets in batch_detections)
            
            # Verify metrics
            metrics = detector.get_performance_metrics()
            assert metrics["total_detections"] > 0
            assert metrics["is_initialized"] is True
            
            # Cleanup
            detector.cleanup()
    
    def test_performance_benchmark(self, mock_torch):
        """Test performance benchmarking."""
        with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
            # Setup mock for fast execution
            mock_model = Mock()
            mock_model.overrides = {}
            mock_model.to.return_value = mock_model
            
            mock_result = Mock()
            mock_result.boxes = None  # No detections for speed
            mock_model.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # Create detector
            config = FaceDetectionConfig(device="cpu")
            detector = MultiFaceDetector(config)
            
            # Run benchmark
            benchmark_results = detector.benchmark_performance(num_frames=10, frame_size=(320, 240))
            
            # Verify benchmark results
            assert isinstance(benchmark_results, dict)
            assert "single_frame" in benchmark_results
            assert "batch_processing" in benchmark_results
            assert "device" in benchmark_results
            
            single_frame_results = benchmark_results["single_frame"]
            assert "avg_time" in single_frame_results
            assert "fps" in single_frame_results
            assert single_frame_results["fps"] > 0
            
            # Cleanup
            detector.cleanup()