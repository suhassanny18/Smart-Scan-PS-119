"""
Manual test for multi-face detection engine to validate implementation.
"""

import numpy as np
from unittest.mock import Mock, patch
from anti_cheat_system.detectors.face_recognition import MultiFaceDetector, FaceDetectionConfig
from anti_cheat_system.models.enums import ModelType

def test_face_detection_config():
    """Test face detection configuration."""
    print("Testing face detection configuration...")
    
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
    
    print("✓ Face detection configuration test passed")

def test_multi_face_detector():
    """Test multi-face detector with mocked YOLO."""
    print("Testing multi-face detector...")
    
    with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
        # Setup mock YOLO model
        mock_model = Mock()
        mock_model.overrides = {}
        mock_model.to.return_value = mock_model
        
        # Mock detection results
        mock_result = Mock()
        mock_result.boxes = Mock()
        
        # Mock bounding boxes and confidences
        mock_boxes = Mock()
        mock_boxes.cpu.return_value.numpy.return_value = np.array([
            [100, 100, 200, 200],  # Face 1
            [300, 150, 400, 250]   # Face 2
        ])
        mock_result.boxes.xyxy = mock_boxes
        
        mock_conf = Mock()
        mock_conf.cpu.return_value.numpy.return_value = np.array([0.9, 0.8])
        mock_result.boxes.conf = mock_conf
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        # Create detector
        config = FaceDetectionConfig(
            confidence_threshold=0.7,
            device="cpu",
            max_detections=10
        )
        
        detector = MultiFaceDetector(config)
        
        # Test initialization
        assert detector.is_initialized is True
        assert detector.device == "cpu"
        
        # Test single frame detection
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_faces(test_frame)
        
        assert len(detections) == 2
        assert detections[0].confidence == 0.9
        assert detections[1].confidence == 0.8
        
        # Test face crop extraction
        face_crops = detector.extract_face_crops(test_frame, detections)
        assert len(face_crops) == 2
        
        # Test combined detection and crop extraction
        detections, crops = detector.detect_faces_with_crops(test_frame)
        assert len(detections) == 2
        assert len(crops) == 2
        
        # Test performance metrics
        metrics = detector.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "total_detections" in metrics
        assert "current_fps" in metrics
        assert metrics["is_initialized"] is True
        
        # Test model info
        model_info = detector.get_model_info()
        assert isinstance(model_info, dict)
        assert "device" in model_info
        
        # Test cleanup
        detector.cleanup()
        assert detector.is_initialized is False
    
    print("✓ Multi-face detector test passed")

def test_face_validation():
    """Test face detection validation logic."""
    print("Testing face validation...")
    
    with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
        # Setup minimal mock
        mock_model = Mock()
        mock_model.overrides = {}
        mock_model.to.return_value = mock_model
        mock_yolo.return_value = mock_model
        
        config = FaceDetectionConfig(device="cpu")
        detector = MultiFaceDetector(config)
        
        from anti_cheat_system.models.data_models import BoundingBox
        
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
        
        detector.cleanup()
    
    print("✓ Face validation test passed")

def test_error_handling():
    """Test error handling in face detection."""
    print("Testing error handling...")
    
    with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
        # Setup mock
        mock_model = Mock()
        mock_model.overrides = {}
        mock_model.to.return_value = mock_model
        mock_yolo.return_value = mock_model
        
        config = FaceDetectionConfig(device="cpu")
        detector = MultiFaceDetector(config)
        
        # Test with None frame
        detections = detector.detect_faces(None)
        assert detections == []
        
        # Test with empty frame
        empty_frame = np.array([])
        detections = detector.detect_faces(empty_frame)
        assert detections == []
        
        # Test preprocessing with None
        processed = detector._preprocess_frame(None)
        assert processed is None
        
        detector.cleanup()
    
    print("✓ Error handling test passed")

def test_performance_features():
    """Test performance monitoring features."""
    print("Testing performance features...")
    
    with patch('anti_cheat_system.detectors.face_recognition.YOLO') as mock_yolo:
        # Setup mock for no detections (faster)
        mock_model = Mock()
        mock_model.overrides = {}
        mock_model.to.return_value = mock_model
        
        mock_result = Mock()
        mock_result.boxes = None  # No detections
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        config = FaceDetectionConfig(device="cpu")
        detector = MultiFaceDetector(config)
        
        # Process a few frames to generate metrics
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detector.detect_faces(test_frame)
        detector.detect_faces(test_frame)
        
        # Test metrics
        metrics = detector.get_performance_metrics()
        assert metrics["total_detections"] >= 0
        assert metrics["current_fps"] >= 0
        
        # Test metrics reset
        detector.reset_metrics()
        metrics_after_reset = detector.get_performance_metrics()
        assert metrics_after_reset["total_detections"] == 0
        
        # Test config update
        new_config = FaceDetectionConfig(
            confidence_threshold=0.8,
            device="cpu"
        )
        detector.update_config(new_config)
        assert detector.config.confidence_threshold == 0.8
        
        detector.cleanup()
    
    print("✓ Performance features test passed")

if __name__ == "__main__":
    print("Running multi-face detection tests...")
    
    try:
        test_face_detection_config()
        test_multi_face_detector()
        test_face_validation()
        test_error_handling()
        test_performance_features()
        
        print("\n🎉 All multi-face detection tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()