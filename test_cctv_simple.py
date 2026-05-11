"""
Simple tests for CCTV integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from anti_cheat_system.cctv.rtsp_manager import RTSPStreamConfig, RTSPStreamManager
from anti_cheat_system.cctv.frame_ingestion import IngestionConfig, FrameIngestionService


def test_rtsp_config_creation():
    """Test creating RTSP stream configuration."""
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


def test_ingestion_config_creation():
    """Test creating ingestion configuration."""
    config = IngestionConfig(
        target_fps=20.0,
        max_queue_size=200,
        frame_skip_threshold=0.3
    )
    
    assert config.target_fps == 20.0
    assert config.max_queue_size == 200
    assert config.frame_skip_threshold == 0.3


@patch('cv2.VideoCapture')
def test_rtsp_manager_basic(mock_video_capture):
    """Test basic RTSP manager functionality."""
    # Mock successful video capture
    mock_instance = Mock()
    mock_instance.isOpened.return_value = True
    mock_instance.read.return_value = (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    mock_instance.get.return_value = 25.0
    mock_instance.release.return_value = None
    mock_video_capture.return_value = mock_instance
    
    # Create stream config
    configs = [
        RTSPStreamConfig(
            url="rtsp://camera1.local/stream",
            stream_id="cam_01"
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
    
    # Cleanup
    manager.cleanup()


def test_frame_ingestion_service_basic():
    """Test basic frame ingestion service functionality."""
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
    
    # Cleanup
    service.cleanup()