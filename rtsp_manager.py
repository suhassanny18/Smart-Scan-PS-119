"""
RTSP Stream Manager for handling multiple CCTV camera feeds.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from dataclasses import dataclass, field

from ..models.data_models import StreamStatus
from ..models.enums import StreamHealth

logger = logging.getLogger(__name__)


@dataclass
class RTSPStreamConfig:
    """Configuration for RTSP stream."""
    url: str
    stream_id: str
    reconnect_timeout: int = 30
    max_reconnect_attempts: int = 5
    frame_timeout: float = 5.0
    buffer_size: int = 1
    use_tcp: bool = True


class RTSPStreamManager:
    """
    Manages multiple RTSP camera streams with automatic reconnection and health monitoring.
    
    Features:
    - Multi-camera support with individual stream management
    - Automatic reconnection with exponential backoff
    - Stream health monitoring and failure detection
    - Thread-safe frame retrieval
    - Configurable timeouts and retry logic
    - Fallback to webcam support
    """
    
    def __init__(self, stream_configs: List[RTSPStreamConfig], enable_fallback: bool = True):
        """
        Initialize RTSP stream manager.
        
        Args:
            stream_configs: List of RTSP stream configurations
            enable_fallback: Enable webcam fallback if no RTSP streams available
        """
        self.stream_configs = {config.stream_id: config for config in stream_configs}
        self.enable_fallback = enable_fallback
        
        # Stream management
        self.streams: Dict[str, cv2.VideoCapture] = {}
        self.stream_status: Dict[str, StreamStatus] = {}
        self.stream_locks: Dict[str, threading.Lock] = {}
        self.reconnect_threads: Dict[str, threading.Thread] = {}
        
        # Health monitoring
        self.last_frame_times: Dict[str, datetime] = {}
        self.reconnect_attempts: Dict[str, int] = {}
        self.is_running = False
        
        # Initialize streams
        self._initialize_streams()
        
        logger.info(f"RTSP Stream Manager initialized with {len(self.stream_configs)} streams")
    
    def _initialize_streams(self) -> None:
        """Initialize all configured streams."""
        for stream_id, config in self.stream_configs.items():
            self.stream_locks[stream_id] = threading.Lock()
            self.reconnect_attempts[stream_id] = 0
            
            # Initialize stream status
            self.stream_status[stream_id] = StreamStatus(
                stream_id=stream_id,
                url=config.url,
                health=StreamHealth.FAILED,
                last_frame_time=datetime.now(),
                fps=0.0,
                reconnect_attempts=0
            )
            
            # Attempt initial connection
            self._connect_stream(stream_id)
        
        # Add webcam fallback if no streams are available and fallback is enabled
        if self.enable_fallback and not any(
            status.health == StreamHealth.HEALTHY 
            for status in self.stream_status.values()
        ):
            self._add_webcam_fallback()
    
    def _add_webcam_fallback(self) -> None:
        """Add webcam as fallback stream."""
        webcam_id = "webcam_0"
        webcam_config = RTSPStreamConfig(
            url="0",  # Default webcam
            stream_id=webcam_id,
            reconnect_timeout=10,
            max_reconnect_attempts=3
        )
        
        self.stream_configs[webcam_id] = webcam_config
        self.stream_locks[webcam_id] = threading.Lock()
        self.reconnect_attempts[webcam_id] = 0
        
        self.stream_status[webcam_id] = StreamStatus(
            stream_id=webcam_id,
            url="Webcam (Fallback)",
            health=StreamHealth.FAILED,
            last_frame_time=datetime.now(),
            fps=0.0,
            reconnect_attempts=0
        )
        
        logger.info("Added webcam fallback stream")
        self._connect_stream(webcam_id)
    
    def _connect_stream(self, stream_id: str) -> bool:
        """
        Connect to a specific stream.
        
        Args:
            stream_id: ID of the stream to connect
            
        Returns:
            True if connection successful, False otherwise
        """
        config = self.stream_configs.get(stream_id)
        if not config:
            logger.error(f"Stream config not found for {stream_id}")
            return False
        
        try:
            with self.stream_locks[stream_id]:
                # Close existing stream if any
                if stream_id in self.streams:
                    self.streams[stream_id].release()
                    del self.streams[stream_id]
                
                # Create new VideoCapture
                if config.url.isdigit():
                    # Webcam
                    cap = cv2.VideoCapture(int(config.url))
                else:
                    # RTSP stream
                    cap = cv2.VideoCapture(config.url)
                    
                    # Configure RTSP parameters
                    if config.use_tcp:
                        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.frame_timeout * 1000)
                        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.frame_timeout * 1000)
                    
                    # Set buffer size to reduce latency
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.buffer_size)
                
                # Test connection
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.streams[stream_id] = cap
                        self.last_frame_times[stream_id] = datetime.now()
                        self.reconnect_attempts[stream_id] = 0
                        
                        # Update stream status
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # Default to 25 FPS if unknown
                        self.stream_status[stream_id] = StreamStatus(
                            stream_id=stream_id,
                            url=config.url,
                            health=StreamHealth.HEALTHY,
                            last_frame_time=datetime.now(),
                            fps=fps,
                            reconnect_attempts=self.reconnect_attempts[stream_id]
                        )
                        
                        logger.info(f"Successfully connected to stream {stream_id} ({config.url})")
                        return True
                    else:
                        cap.release()
                        raise Exception("Failed to read initial frame")
                else:
                    cap.release()
                    raise Exception("Failed to open stream")
                    
        except Exception as e:
            error_msg = f"Failed to connect to stream {stream_id}: {e}"
            logger.error(error_msg)
            
            # Update stream status
            self.stream_status[stream_id] = StreamStatus(
                stream_id=stream_id,
                url=config.url,
                health=StreamHealth.FAILED,
                last_frame_time=self.last_frame_times.get(stream_id, datetime.now()),
                fps=0.0,
                reconnect_attempts=self.reconnect_attempts[stream_id],
                error_message=str(e)
            )
            
            return False
    
    def get_frame(self, stream_id: str) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Get the latest frame from a specific stream.
        
        Args:
            stream_id: ID of the stream to get frame from
            
        Returns:
            Tuple of (success, frame, metadata)
        """
        if stream_id not in self.stream_configs:
            return False, None, {"error": f"Stream {stream_id} not configured"}
        
        if stream_id not in self.streams:
            # Try to reconnect if not connected
            if not self._connect_stream(stream_id):
                return False, None, {"error": f"Stream {stream_id} not available"}
        
        try:
            with self.stream_locks[stream_id]:
                cap = self.streams[stream_id]
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Update last frame time
                    self.last_frame_times[stream_id] = datetime.now()
                    
                    # Update stream status
                    self.stream_status[stream_id].health = StreamHealth.HEALTHY
                    self.stream_status[stream_id].last_frame_time = datetime.now()
                    self.stream_status[stream_id].error_message = None
                    
                    # Prepare metadata
                    metadata = {
                        "stream_id": stream_id,
                        "timestamp": datetime.now(),
                        "frame_shape": frame.shape,
                        "fps": self.stream_status[stream_id].fps,
                        "health": StreamHealth.HEALTHY.value
                    }
                    
                    return True, frame, metadata
                else:
                    # Frame read failed
                    self._handle_stream_failure(stream_id, "Failed to read frame")
                    return False, None, {"error": "Failed to read frame", "stream_id": stream_id}
                    
        except Exception as e:
            error_msg = f"Error reading frame from stream {stream_id}: {e}"
            logger.error(error_msg)
            self._handle_stream_failure(stream_id, str(e))
            return False, None, {"error": error_msg, "stream_id": stream_id}
    
    def _handle_stream_failure(self, stream_id: str, error_message: str) -> None:
        """
        Handle stream failure and initiate reconnection if needed.
        
        Args:
            stream_id: ID of the failed stream
            error_message: Error message describing the failure
        """
        config = self.stream_configs[stream_id]
        self.reconnect_attempts[stream_id] += 1
        
        # Update stream status
        self.stream_status[stream_id].health = StreamHealth.FAILED
        self.stream_status[stream_id].error_message = error_message
        self.stream_status[stream_id].reconnect_attempts = self.reconnect_attempts[stream_id]
        
        logger.warning(f"Stream {stream_id} failed: {error_message} (attempt {self.reconnect_attempts[stream_id]})")
        
        # Clean up failed stream
        with self.stream_locks[stream_id]:
            if stream_id in self.streams:
                try:
                    self.streams[stream_id].release()
                except:
                    pass
                del self.streams[stream_id]
        
        # Start reconnection thread if not already running and within retry limits
        if (self.reconnect_attempts[stream_id] <= config.max_reconnect_attempts and 
            stream_id not in self.reconnect_threads):
            
            self._start_reconnection_thread(stream_id)
    
    def _start_reconnection_thread(self, stream_id: str) -> None:
        """
        Start a background thread to reconnect to a failed stream.
        
        Args:
            stream_id: ID of the stream to reconnect
        """
        def reconnect_worker():
            config = self.stream_configs[stream_id]
            
            # Exponential backoff
            wait_time = min(config.reconnect_timeout * (2 ** (self.reconnect_attempts[stream_id] - 1)), 300)
            
            logger.info(f"Attempting to reconnect to stream {stream_id} in {wait_time} seconds")
            time.sleep(wait_time)
            
            # Update status to reconnecting
            self.stream_status[stream_id].health = StreamHealth.RECONNECTING
            
            # Attempt reconnection
            if self._connect_stream(stream_id):
                logger.info(f"Successfully reconnected to stream {stream_id}")
            else:
                logger.error(f"Failed to reconnect to stream {stream_id}")
            
            # Remove from reconnect threads
            if stream_id in self.reconnect_threads:
                del self.reconnect_threads[stream_id]
        
        thread = threading.Thread(target=reconnect_worker, daemon=True)
        thread.start()
        self.reconnect_threads[stream_id] = thread
    
    def get_stream_health(self) -> Dict[str, StreamStatus]:
        """
        Get health status of all streams.
        
        Returns:
            Dictionary mapping stream IDs to their status
        """
        # Update health based on last frame time
        current_time = datetime.now()
        
        for stream_id, status in self.stream_status.items():
            if status.health == StreamHealth.HEALTHY:
                # Check if stream is stale
                time_since_last_frame = (current_time - status.last_frame_time).total_seconds()
                if time_since_last_frame > 10:  # 10 seconds threshold
                    status.health = StreamHealth.DEGRADED
                    status.error_message = f"No frames received for {time_since_last_frame:.1f} seconds"
        
        return self.stream_status.copy()
    
    def get_available_streams(self) -> List[str]:
        """
        Get list of currently available (healthy) streams.
        
        Returns:
            List of stream IDs that are currently healthy
        """
        return [
            stream_id for stream_id, status in self.stream_status.items()
            if status.health == StreamHealth.HEALTHY
        ]
    
    def force_reconnect(self, stream_id: str) -> bool:
        """
        Force reconnection of a specific stream.
        
        Args:
            stream_id: ID of the stream to reconnect
            
        Returns:
            True if reconnection successful, False otherwise
        """
        if stream_id not in self.stream_configs:
            logger.error(f"Stream {stream_id} not configured")
            return False
        
        logger.info(f"Forcing reconnection of stream {stream_id}")
        self.reconnect_attempts[stream_id] = 0  # Reset attempt counter
        return self._connect_stream(stream_id)
    
    def add_stream(self, config: RTSPStreamConfig) -> bool:
        """
        Add a new stream configuration and connect to it.
        
        Args:
            config: RTSP stream configuration
            
        Returns:
            True if stream added and connected successfully, False otherwise
        """
        if config.stream_id in self.stream_configs:
            logger.warning(f"Stream {config.stream_id} already exists, updating configuration")
        
        self.stream_configs[config.stream_id] = config
        self.stream_locks[config.stream_id] = threading.Lock()
        self.reconnect_attempts[config.stream_id] = 0
        
        # Initialize stream status
        self.stream_status[config.stream_id] = StreamStatus(
            stream_id=config.stream_id,
            url=config.url,
            health=StreamHealth.FAILED,
            last_frame_time=datetime.now(),
            fps=0.0,
            reconnect_attempts=0
        )
        
        # Attempt connection
        success = self._connect_stream(config.stream_id)
        
        if success:
            logger.info(f"Successfully added and connected to stream {config.stream_id}")
        else:
            logger.error(f"Failed to connect to newly added stream {config.stream_id}")
        
        return success
    
    def remove_stream(self, stream_id: str) -> bool:
        """
        Remove a stream and clean up its resources.
        
        Args:
            stream_id: ID of the stream to remove
            
        Returns:
            True if stream removed successfully, False otherwise
        """
        if stream_id not in self.stream_configs:
            logger.warning(f"Stream {stream_id} not found")
            return False
        
        # Stop reconnection thread if running
        if stream_id in self.reconnect_threads:
            # Note: We can't forcefully stop threads, but they're daemon threads
            del self.reconnect_threads[stream_id]
        
        # Clean up stream
        with self.stream_locks[stream_id]:
            if stream_id in self.streams:
                try:
                    self.streams[stream_id].release()
                except:
                    pass
                del self.streams[stream_id]
        
        # Clean up all references
        del self.stream_configs[stream_id]
        del self.stream_locks[stream_id]
        del self.stream_status[stream_id]
        del self.reconnect_attempts[stream_id]
        
        if stream_id in self.last_frame_times:
            del self.last_frame_times[stream_id]
        
        logger.info(f"Successfully removed stream {stream_id}")
        return True
    
    def get_stream_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for all streams.
        
        Returns:
            Dictionary with stream statistics
        """
        stats = {}
        current_time = datetime.now()
        
        for stream_id, status in self.stream_status.items():
            time_since_last_frame = (current_time - status.last_frame_time).total_seconds()
            
            stats[stream_id] = {
                "stream_id": stream_id,
                "url": status.url,
                "health": status.health.value,
                "fps": status.fps,
                "reconnect_attempts": status.reconnect_attempts,
                "time_since_last_frame": time_since_last_frame,
                "error_message": status.error_message,
                "is_connected": stream_id in self.streams,
                "is_reconnecting": stream_id in self.reconnect_threads
            }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up all streams and resources."""
        logger.info("Cleaning up RTSP Stream Manager")
        
        # Close all streams
        for stream_id in list(self.streams.keys()):
            with self.stream_locks[stream_id]:
                try:
                    self.streams[stream_id].release()
                except:
                    pass
        
        self.streams.clear()
        self.stream_status.clear()
        self.stream_locks.clear()
        self.reconnect_threads.clear()
        self.last_frame_times.clear()
        self.reconnect_attempts.clear()
        
        logger.info("RTSP Stream Manager cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass