"""
Unit tests for video capture module
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from anti_cheat_system.video_capture import VideoCapture, FrameProcessor
from anti_cheat_system.models import VideoConfig


class TestVideoCapture:
    """Test VideoCapture class"""
    
    def test_video_capture_init_default_config(self):
        """Test VideoCapture initialization with default config"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            
            assert video_capture.config is not None
            assert video_capture.config.camera_index == 0
            assert video_capture.config.frame_width == 640
            assert video_capture.config.frame_height == 480
    
    def test_video_capture_init_custom_config(self):
        """Test VideoCapture initialization with custom config"""
        config = VideoConfig(
            camera_index=1,
            frame_width=1280,
            frame_height=720,
            fps=60
        )
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture(config)
            
            assert video_capture.config.camera_index == 1
            assert video_capture.config.frame_width == 1280
            assert video_capture.config.frame_height == 720
            assert video_capture.config.fps == 60
    
    def test_camera_initialization_success(self):
        """Test successful camera initialization"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap.get.return_value = 30.0  # FPS
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            
            assert video_capture.is_opened is True
            assert video_capture.is_camera_available() is True
            mock_cap.set.assert_called()  # Camera properties should be set
    
    def test_camera_initialization_failure(self):
        """Test camera initialization failure"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            
            assert video_capture.is_opened is False
            assert video_capture.is_camera_available() is False
    
    def test_get_frame_success(self):
        """Test successful frame capture"""
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, test_frame)
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            success, frame = video_capture.get_frame()
            
            assert success is True
            assert frame is not None
            assert frame.shape == test_frame.shape
            assert video_capture.frame_count == 1
    
    def test_get_frame_failure(self):
        """Test frame capture failure and recovery"""
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0  # Mock FPS value
            mock_cap.read.return_value = (True, test_frame)  # Initial success for setup
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            
            # Now change the mock to fail
            mock_cap.read.return_value = (False, None)
            
            # This should fail and attempt recovery
            success, frame = video_capture.get_frame()
            assert success is False
            assert frame is None
    
    def test_get_camera_info(self):
        """Test camera info retrieval"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS: 30.0
            }.get(prop, 0)
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            info = video_capture.get_camera_info()
            
            assert info["status"] == "opened"
            assert info["frame_width"] == 640
            assert info["frame_height"] == 480
            assert info["fps"] == 30.0
    
    def test_set_resolution(self):
        """Test resolution change"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            success = video_capture.set_resolution(1280, 720)
            
            assert success is True
            assert video_capture.config.frame_width == 1280
            assert video_capture.config.frame_height == 720
            mock_cap.set.assert_called()
    
    def test_release(self):
        """Test camera release"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap
            
            video_capture = VideoCapture()
            assert video_capture.is_opened is True
            
            video_capture.release()
            
            assert video_capture.is_opened is False
            mock_cap.release.assert_called_once()
    
    def test_context_manager(self):
        """Test VideoCapture as context manager"""
        with patch('cv2.VideoCapture') as mock_cv2:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cv2.return_value = mock_cap
            
            with VideoCapture() as video_capture:
                assert video_capture.is_opened is True
            
            # Should be released after context exit
            mock_cap.release.assert_called()


class TestFrameProcessor:
    """Test FrameProcessor utility class"""
    
    def test_resize_frame(self):
        """Test frame resizing"""
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Resize to different dimensions
        resized = FrameProcessor.resize_frame(frame, 320, 240)
        
        assert resized is not None
        assert resized.shape == (240, 320, 3)
    
    def test_resize_frame_none_input(self):
        """Test frame resizing with None input"""
        result = FrameProcessor.resize_frame(None, 320, 240)
        assert result is None
    
    def test_flip_frame_horizontal(self):
        """Test horizontal frame flipping"""
        # Create test frame with distinct pattern
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        frame[:, :100] = 255  # Left half white
        
        flipped = FrameProcessor.flip_frame(frame, horizontal=True)
        
        assert flipped is not None
        assert flipped.shape == frame.shape
        # Right half should now be white
        assert np.all(flipped[:, 100:] == 255)
        assert np.all(flipped[:, :100] == 0)
    
    def test_flip_frame_vertical(self):
        """Test vertical frame flipping"""
        # Create test frame with distinct pattern
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        frame[:100, :] = 255  # Top half white
        
        flipped = FrameProcessor.flip_frame(frame, horizontal=False)
        
        assert flipped is not None
        assert flipped.shape == frame.shape
        # Bottom half should now be white
        assert np.all(flipped[100:, :] == 255)
        assert np.all(flipped[:100, :] == 0)
    
    def test_flip_frame_none_input(self):
        """Test frame flipping with None input"""
        result = FrameProcessor.flip_frame(None)
        assert result is None
    
    def test_enhance_frame(self):
        """Test frame enhancement"""
        # Create test frame
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)  # Gray frame
        
        # Enhance brightness and contrast
        enhanced = FrameProcessor.enhance_frame(frame, brightness=50, contrast=1.5)
        
        assert enhanced is not None
        assert enhanced.shape == frame.shape
        # Enhanced frame should be different from original
        assert not np.array_equal(enhanced, frame)
    
    def test_enhance_frame_none_input(self):
        """Test frame enhancement with None input"""
        result = FrameProcessor.enhance_frame(None)
        assert result is None
    
    def test_add_timestamp(self):
        """Test timestamp addition"""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add timestamp
        timestamped = FrameProcessor.add_timestamp(frame, "2024-01-01 12:00:00")
        
        assert timestamped is not None
        assert timestamped.shape == frame.shape
        # Frame should be modified (not all zeros anymore)
        assert not np.array_equal(timestamped, frame)
    
    def test_add_timestamp_none_input(self):
        """Test timestamp addition with None input"""
        result = FrameProcessor.add_timestamp(None)
        assert result is None
    
    def test_add_timestamp_auto(self):
        """Test timestamp addition with automatic timestamp"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add automatic timestamp
        timestamped = FrameProcessor.add_timestamp(frame)
        
        assert timestamped is not None
        assert timestamped.shape == frame.shape
        # Frame should be modified
        assert not np.array_equal(timestamped, frame)