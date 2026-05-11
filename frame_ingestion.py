"""
Frame Ingestion Service for processing CCTV frames with async queue management.
"""

import logging
import threading
import time
import queue
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import cv2
import numpy as np
from dataclasses import dataclass, field

from .rtsp_manager import RTSPStreamManager
from ..models.data_models import FrameMetadata

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for frame data and metadata."""
    frame: np.ndarray
    metadata: FrameMetadata
    stream_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    processing_start_time: Optional[datetime] = None


@dataclass
class IngestionConfig:
    """Configuration for frame ingestion."""
    target_fps: float = 15.0
    max_queue_size: int = 100
    frame_skip_threshold: float = 0.5  # Skip frames if processing is slow
    quality_check: bool = True
    resize_frames: bool = True
    target_width: int = 640
    target_height: int = 480
    enable_preprocessing: bool = True
    drop_frames_on_overflow: bool = True


class FrameIngestionService:
    """
    Service for ingesting frames from multiple CCTV streams with preprocessing and queue management.
    
    Features:
    - Multi-threaded frame ingestion from multiple streams
    - Async frame processing with configurable queue sizes
    - Frame preprocessing (resize, normalize, quality checks)
    - Adaptive frame skipping based on processing load
    - Frame drop detection and recovery
    - Performance monitoring and statistics
    """
    
    def __init__(self, rtsp_manager: RTSPStreamManager, config: IngestionConfig):
        """
        Initialize frame ingestion service.
        
        Args:
            rtsp_manager: RTSP stream manager instance
            config: Ingestion configuration
        """
        self.rtsp_manager = rtsp_manager
        self.config = config
        
        # Frame queues - separate queue for each stream
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.global_frame_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        
        # Threading
        self.ingestion_threads: Dict[str, threading.Thread] = {}
        self.is_running = False
        self.thread_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            "frames_ingested": 0,
            "frames_dropped": 0,
            "frames_processed": 0,
            "processing_errors": 0,
            "average_fps": 0.0,
            "average_processing_time": 0.0,
            "queue_sizes": {},
            "stream_stats": {}
        }
        
        # Performance monitoring
        self.last_stats_update = datetime.now()
        self.frame_times: List[float] = []
        self.processing_times: List[float] = []
        
        # Frame processors (callbacks)
        self.frame_processors: List[Callable[[FrameData], None]] = []
        
        logger.info("Frame Ingestion Service initialized")
    
    def start_ingestion(self) -> None:
        """Start frame ingestion from all available streams."""
        if self.is_running:
            logger.warning("Frame ingestion already running")
            return
        
        self.is_running = True
        available_streams = self.rtsp_manager.get_available_streams()
        
        if not available_streams:
            logger.error("No available streams for frame ingestion")
            return
        
        logger.info(f"Starting frame ingestion for {len(available_streams)} streams")
        
        # Create queues and start threads for each stream
        for stream_id in available_streams:
            self._start_stream_ingestion(stream_id)
        
        # Start global processing thread
        self._start_global_processor()
        
        logger.info("Frame ingestion started successfully")
    
    def stop_ingestion(self) -> None:
        """Stop frame ingestion and clean up threads."""
        if not self.is_running:
            return
        
        logger.info("Stopping frame ingestion")
        self.is_running = False
        
        # Wait for threads to finish (with timeout)
        with self.thread_lock:
            for stream_id, thread in self.ingestion_threads.items():
                if thread.is_alive():
                    thread.join(timeout=2.0)
                    if thread.is_alive():
                        logger.warning(f"Thread for stream {stream_id} did not stop gracefully")
        
        # Clear queues
        self._clear_queues()
        
        logger.info("Frame ingestion stopped")
    
    def _start_stream_ingestion(self, stream_id: str) -> None:
        """
        Start ingestion thread for a specific stream.
        
        Args:
            stream_id: ID of the stream to start ingestion for
        """
        # Create queue for this stream
        self.frame_queues[stream_id] = queue.Queue(maxsize=self.config.max_queue_size // 4)
        
        # Initialize stream stats
        self.stats["stream_stats"][stream_id] = {
            "frames_ingested": 0,
            "frames_dropped": 0,
            "last_frame_time": None,
            "fps": 0.0,
            "errors": 0
        }
        
        # Start ingestion thread
        thread = threading.Thread(
            target=self._stream_ingestion_worker,
            args=(stream_id,),
            daemon=True,
            name=f"FrameIngestion-{stream_id}"
        )
        
        with self.thread_lock:
            self.ingestion_threads[stream_id] = thread
        
        thread.start()
        logger.info(f"Started ingestion thread for stream {stream_id}")
    
    def _stream_ingestion_worker(self, stream_id: str) -> None:
        """
        Worker thread for ingesting frames from a specific stream.
        
        Args:
            stream_id: ID of the stream to ingest frames from
        """
        logger.info(f"Frame ingestion worker started for stream {stream_id}")
        
        frame_interval = 1.0 / self.config.target_fps
        last_frame_time = time.time()
        frame_count = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Control frame rate
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    time.sleep(frame_interval - time_since_last)
                    continue
                
                # Get frame from stream
                success, frame, metadata = self.rtsp_manager.get_frame(stream_id)
                
                if not success or frame is None:
                    self.stats["stream_stats"][stream_id]["errors"] += 1
                    time.sleep(0.1)  # Brief pause on error
                    continue
                
                # Preprocess frame
                processed_frame = self._preprocess_frame(frame)
                if processed_frame is None:
                    self.stats["stream_stats"][stream_id]["errors"] += 1
                    continue
                
                # Create frame data
                frame_metadata = FrameMetadata(
                    frame_id=f"{stream_id}_{frame_count:06d}",
                    timestamp=datetime.now(),
                    camera_id=stream_id,
                    frame_number=frame_count,
                    processing_time=0.0,  # Will be updated later
                    detections_count=0,   # Will be updated by processors
                    tracks_count=0        # Will be updated by processors
                )
                
                frame_data = FrameData(
                    frame=processed_frame,
                    metadata=frame_metadata,
                    stream_id=stream_id,
                    timestamp=datetime.now()
                )
                
                # Add to stream queue
                try:
                    self.frame_queues[stream_id].put_nowait(frame_data)
                    
                    # Also add to global queue for processing
                    try:
                        self.global_frame_queue.put_nowait(frame_data)
                        self.stats["frames_ingested"] += 1
                        self.stats["stream_stats"][stream_id]["frames_ingested"] += 1
                    except queue.Full:
                        if self.config.drop_frames_on_overflow:
                            # Drop oldest frame and add new one
                            try:
                                self.global_frame_queue.get_nowait()
                                self.global_frame_queue.put_nowait(frame_data)
                                self.stats["frames_dropped"] += 1
                            except queue.Empty:
                                pass
                        else:
                            self.stats["frames_dropped"] += 1
                            self.stats["stream_stats"][stream_id]["frames_dropped"] += 1
                
                except queue.Full:
                    self.stats["frames_dropped"] += 1
                    self.stats["stream_stats"][stream_id]["frames_dropped"] += 1
                
                # Update timing
                last_frame_time = current_time
                frame_count += 1
                
                # Update stream stats
                self.stats["stream_stats"][stream_id]["last_frame_time"] = datetime.now()
                
                # Calculate FPS
                if frame_count % 30 == 0:  # Update every 30 frames
                    elapsed = time.time() - (last_frame_time - (30 * frame_interval))
                    if elapsed > 0:
                        self.stats["stream_stats"][stream_id]["fps"] = 30 / elapsed
                
            except Exception as e:
                logger.error(f"Error in stream ingestion worker for {stream_id}: {e}")
                self.stats["stream_stats"][stream_id]["errors"] += 1
                time.sleep(1.0)  # Pause on error
        
        logger.info(f"Frame ingestion worker stopped for stream {stream_id}")
    
    def _start_global_processor(self) -> None:
        """Start global frame processing thread."""
        thread = threading.Thread(
            target=self._global_processing_worker,
            daemon=True,
            name="GlobalFrameProcessor"
        )
        thread.start()
        logger.info("Started global frame processing thread")
    
    def _global_processing_worker(self) -> None:
        """Worker thread for processing frames from the global queue."""
        logger.info("Global frame processing worker started")
        
        while self.is_running:
            try:
                # Get frame from global queue with timeout
                try:
                    frame_data = self.global_frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Mark processing start time
                frame_data.processing_start_time = datetime.now()
                
                # Process frame through all registered processors
                for processor in self.frame_processors:
                    try:
                        processor(frame_data)
                    except Exception as e:
                        logger.error(f"Error in frame processor: {e}")
                        self.stats["processing_errors"] += 1
                
                # Update processing time
                if frame_data.processing_start_time:
                    processing_time = (datetime.now() - frame_data.processing_start_time).total_seconds()
                    frame_data.metadata.processing_time = processing_time
                    self.processing_times.append(processing_time)
                    
                    # Keep only recent processing times
                    if len(self.processing_times) > 100:
                        self.processing_times = self.processing_times[-50:]
                
                self.stats["frames_processed"] += 1
                
                # Mark task as done
                self.global_frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in global frame processing worker: {e}")
                self.stats["processing_errors"] += 1
        
        logger.info("Global frame processing worker stopped")
    
    def _preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess frame with quality checks and normalization.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame or None if frame is invalid
        """
        if not self.config.enable_preprocessing:
            return frame
        
        try:
            # Quality check
            if self.config.quality_check:
                if not self._check_frame_quality(frame):
                    return None
            
            # Resize frame
            if self.config.resize_frames:
                frame = cv2.resize(
                    frame, 
                    (self.config.target_width, self.config.target_height),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Additional preprocessing can be added here
            # - Noise reduction
            # - Brightness/contrast adjustment
            # - Color space conversion
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None
    
    def _check_frame_quality(self, frame: np.ndarray) -> bool:
        """
        Check if frame meets quality requirements.
        
        Args:
            frame: Frame to check
            
        Returns:
            True if frame quality is acceptable, False otherwise
        """
        if frame is None or frame.size == 0:
            return False
        
        # Check frame dimensions
        height, width = frame.shape[:2]
        if height < 100 or width < 100:
            return False
        
        # Check if frame is too dark or too bright
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 10 or mean_brightness > 245:
            return False
        
        # Check for motion blur (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 50:  # Threshold for blur detection
            return False
        
        return True
    
    def add_frame_processor(self, processor: Callable[[FrameData], None]) -> None:
        """
        Add a frame processor callback.
        
        Args:
            processor: Callback function that processes frame data
        """
        self.frame_processors.append(processor)
        logger.info(f"Added frame processor: {processor.__name__}")
    
    def remove_frame_processor(self, processor: Callable[[FrameData], None]) -> None:
        """
        Remove a frame processor callback.
        
        Args:
            processor: Callback function to remove
        """
        if processor in self.frame_processors:
            self.frame_processors.remove(processor)
            logger.info(f"Removed frame processor: {processor.__name__}")
    
    def get_latest_frame(self, stream_id: str) -> Optional[FrameData]:
        """
        Get the latest frame from a specific stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Latest frame data or None if not available
        """
        if stream_id not in self.frame_queues:
            return None
        
        try:
            # Get the most recent frame (non-blocking)
            frame_data = None
            while True:
                try:
                    frame_data = self.frame_queues[stream_id].get_nowait()
                except queue.Empty:
                    break
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error getting latest frame for stream {stream_id}: {e}")
            return None
    
    def get_frame_queue_size(self, stream_id: str) -> int:
        """
        Get the current size of a stream's frame queue.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Queue size or -1 if stream not found
        """
        if stream_id not in self.frame_queues:
            return -1
        
        return self.frame_queues[stream_id].qsize()
    
    def get_global_queue_size(self) -> int:
        """Get the current size of the global frame queue."""
        return self.global_frame_queue.qsize()
    
    def _clear_queues(self) -> None:
        """Clear all frame queues."""
        # Clear stream queues
        for stream_id, frame_queue in self.frame_queues.items():
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    break
        
        # Clear global queue
        while not self.global_frame_queue.empty():
            try:
                self.global_frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("All frame queues cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive ingestion statistics.
        
        Returns:
            Dictionary with ingestion statistics
        """
        current_time = datetime.now()
        time_elapsed = (current_time - self.last_stats_update).total_seconds()
        
        # Calculate average FPS
        if time_elapsed > 0:
            self.stats["average_fps"] = self.stats["frames_processed"] / time_elapsed
        
        # Calculate average processing time
        if self.processing_times:
            self.stats["average_processing_time"] = np.mean(self.processing_times)
        
        # Update queue sizes
        self.stats["queue_sizes"] = {
            "global": self.global_frame_queue.qsize(),
            "streams": {
                stream_id: queue.qsize() 
                for stream_id, queue in self.frame_queues.items()
            }
        }
        
        # Add system health indicators
        self.stats["health"] = {
            "is_running": self.is_running,
            "active_streams": len([s for s in self.stats["stream_stats"].values() if s["fps"] > 0]),
            "total_streams": len(self.frame_queues),
            "queue_utilization": self.global_frame_queue.qsize() / self.config.max_queue_size,
            "processing_load": self.stats["average_processing_time"] * self.stats["average_fps"]
        }
        
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters."""
        self.stats = {
            "frames_ingested": 0,
            "frames_dropped": 0,
            "frames_processed": 0,
            "processing_errors": 0,
            "average_fps": 0.0,
            "average_processing_time": 0.0,
            "queue_sizes": {},
            "stream_stats": {
                stream_id: {
                    "frames_ingested": 0,
                    "frames_dropped": 0,
                    "last_frame_time": None,
                    "fps": 0.0,
                    "errors": 0
                }
                for stream_id in self.frame_queues.keys()
            }
        }
        
        self.last_stats_update = datetime.now()
        self.frame_times.clear()
        self.processing_times.clear()
        
        logger.info("Statistics reset")
    
    def handle_stream_failure(self, stream_id: str) -> None:
        """
        Handle failure of a specific stream.
        
        Args:
            stream_id: ID of the failed stream
        """
        logger.warning(f"Handling failure for stream {stream_id}")
        
        # Stop ingestion thread for this stream
        with self.thread_lock:
            if stream_id in self.ingestion_threads:
                # Thread will stop when is_running becomes False or stream fails
                pass
        
        # Clear stream queue
        if stream_id in self.frame_queues:
            while not self.frame_queues[stream_id].empty():
                try:
                    self.frame_queues[stream_id].get_nowait()
                except queue.Empty:
                    break
        
        # Update stats
        if stream_id in self.stats["stream_stats"]:
            self.stats["stream_stats"][stream_id]["errors"] += 1
    
    def add_new_stream(self, stream_id: str) -> None:
        """
        Add ingestion for a newly available stream.
        
        Args:
            stream_id: ID of the new stream
        """
        if not self.is_running:
            logger.warning("Cannot add stream - ingestion service not running")
            return
        
        if stream_id in self.frame_queues:
            logger.warning(f"Stream {stream_id} already being ingested")
            return
        
        logger.info(f"Adding ingestion for new stream {stream_id}")
        self._start_stream_ingestion(stream_id)
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up Frame Ingestion Service")
        
        self.stop_ingestion()
        self.frame_processors.clear()
        self.frame_queues.clear()
        self.ingestion_threads.clear()
        
        logger.info("Frame Ingestion Service cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass