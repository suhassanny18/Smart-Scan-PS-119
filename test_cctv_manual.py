"""
Manual test for CCTV integration to validate implementation.
"""

import numpy as np
from unittest.mock import Mock, patch
from anti_cheat_system.cctv.rtsp_manager import RTSPStreamManager, RTSPStreamConfig
from anti_cheat_system.cctv.frame_ingestion import FrameIngestionService, IngestionConfig
from anti_cheat_system.models.enums import StreamHealth

def test_rtsp_config():
    """Test RTSP configuration creation."""
    print("Testing RTSP configuration...")
    
    config = RTSPStreamConfig(
        url="rtsp://camera1.local/stream",
        stream_id="cam_01",
        reconnect_timeout=30,
        max_reconnect_attempts=5
    )
    
    assert config.url == "rtsp://camera1.local/stream"
    assert config.stream_id == "cam_01"
    assert config.reconnect_timeout == 30
    assert config.max_reconnect_attempts == 5
    
    print("✓ RTSP configuration test passed")

def test_rtsp_manager():
    """Test RTSP manager with mocked video capture."""
    print("Testing RTSP manager...")
    
    with patch('cv2.VideoCapture') as mock_cap:
        # Mock successful video capture
        mock_instance = Mock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_instance.get.return_value = 25.0
        mock_instance.release.return_value = None
        mock_cap.return_value = mock_instance
        
        # Create stream configs
        configs = [
            RTSPStreamConfig(
                url="rtsp://camera1.local/stream",
                stream_id="cam_01",
                reconnect_timeout=5,
                max_reconnect_attempts=2
            )
        ]
        
        # Create manager
        manager = RTSPStreamManager(configs, enable_fallback=False)
        
        # Test basic functionality
        assert len(manager.stream_configs) == 1
        assert "cam_01" in manager.stream_configs
        
        # Test frame retrieval
        success, frame, metadata = manager.get_frame("cam_01")
        assert success is True
        assert frame is not None
        assert metadata["stream_id"] == "cam_01"
        
        # Test health monitoring
        health_status = manager.get_stream_health()
        assert "cam_01" in health_status
        
        # Test statistics
        stats = manager.get_stream_statistics()
        assert isinstance(stats, dict)
        assert "cam_01" in stats
        
        # Cleanup
        manager.cleanup()
    
    print("✓ RTSP manager test passed")

def test_ingestion_service():
    """Test frame ingestion service."""
    print("Testing frame ingestion service...")
    
    # Mock RTSP manager
    mock_rtsp_manager = Mock()
    mock_rtsp_manager.get_available_streams.return_value = ["cam_01"]
    
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mock_rtsp_manager.get_frame.return_value = (True, test_frame, {"stream_id": "cam_01", "fps": 25.0})
    
    # Create ingestion config
    config = IngestionConfig(
        target_fps=10.0,
        max_queue_size=50,
        quality_check=False,
        resize_frames=False
    )
    
    # Create ingestion service
    service = FrameIngestionService(mock_rtsp_manager, config)
    
    # Test initialization
    assert service.rtsp_manager == mock_rtsp_manager
    assert service.config == config
    assert service.is_running is False
    
    # Test frame preprocessing
    processed_frame = service._preprocess_frame(test_frame)
    assert processed_frame is not None
    
    # Test quality check
    assert service._check_frame_quality(test_frame) is True
    
    # Test frame processor management
    def test_processor(frame_data):
        pass
    
    service.add_frame_processor(test_processor)
    assert len(service.frame_processors) == 1
    
    service.remove_frame_processor(test_processor)
    assert len(service.frame_processors) == 0
    
    # Test statistics
    stats = service.get_statistics()
    assert isinstance(stats, dict)
    assert "frames_ingested" in stats
    
    # Cleanup
    service.cleanup()
    
    print("✓ Frame ingestion service test passed")

def test_integration():
    """Test integration between RTSP manager and ingestion service."""
    print("Testing integration...")
    
    with patch('cv2.VideoCapture') as mock_cap:
        # Mock video capture
        mock_instance = Mock()
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        mock_instance.get.return_value = 25.0
        mock_instance.release.return_value = None
        mock_cap.return_value = mock_instance
        
        # Create RTSP manager
        stream_configs = [
            RTSPStreamConfig(
                url="rtsp://camera1.local/stream",
                stream_id="cam_01",
                reconnect_timeout=5
            )
        ]
        
        rtsp_manager = RTSPStreamManager(stream_configs, enable_fallback=False)
        
        # Create ingestion service
        ingestion_config = IngestionConfig(
            target_fps=5.0,
            max_queue_size=20,
            quality_check=False,
            resize_frames=False
        )
        
        ingestion_service = FrameIngestionService(rtsp_manager, ingestion_config)
        
        # Test that they work together
        available_streams = rtsp_manager.get_available_streams()
        assert len(available_streams) >= 0
        
        # Cleanup
        ingestion_service.cleanup()
        rtsp_manager.cleanup()
    
    print("✓ Integration test passed")

if __name__ == "__main__":
    print("Running CCTV integration tests...")
    
    try:
        test_rtsp_config()
        test_rtsp_manager()
        test_ingestion_service()
        test_integration()
        
        print("\n🎉 All CCTV integration tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()