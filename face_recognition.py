"""
Multi-Face Detection Engine using YOLOv8-face for detecting multiple students simultaneously.
"""

import logging
import time
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
from dataclasses import dataclass, field

from ..models.data_models import FaceDetection, BoundingBox
from ..models.enums import ModelType

logger = logging.getLogger(__name__)


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection."""
    model_type: ModelType = ModelType.YOLOV8_FACE
    model_path: str = "yolov8n-face.pt"
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.45
    max_detections: int = 20
    input_size: Tuple[int, int] = (640, 640)
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    half_precision: bool = True
    batch_size: int = 1
    enable_tracking: bool = False
    min_face_size: int = 30
    max_face_size: int = 500


@dataclass
class DetectionMetrics:
    """Metrics for face detection performance."""
    total_detections: int = 0
    valid_detections: int = 0
    processing_time: float = 0.0
    fps: float = 0.0
    model_inference_time: float = 0.0
    postprocessing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class MultiFaceDetector:
    """
    Multi-face detection engine using YOLOv8-face for simultaneous detection of multiple students.
    
    Features:
    - YOLOv8-face model for accurate face detection
    - Batch processing for performance optimization
    - GPU acceleration with automatic device selection
    - Configurable confidence and NMS thresholds
    - Face crop extraction with quality assessment
    - Performance monitoring and metrics collection
    - Support for multiple face detection models
    """
    
    def __init__(self, config: FaceDetectionConfig):
        """
        Initialize multi-face detector.
        
        Args:
            config: Face detection configuration
        """
        self.config = config
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # Performance tracking
        self.metrics = DetectionMetrics()
        self.recent_processing_times: List[float] = []
        
        # Face quality thresholds
        self.min_face_area = config.min_face_size ** 2
        self.max_face_area = config.max_face_size ** 2
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"MultiFaceDetector initialized with {config.model_type.value} on {self.device}")
    
    def _initialize_model(self) -> None:
        """Initialize the face detection model."""
        try:
            # Determine device
            self.device = self._get_device()
            
            # Load model based on type
            if self.config.model_type == ModelType.YOLOV8_FACE:
                self._load_yolov8_face_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Warm up model
            self._warmup_model()
            
            self.is_initialized = True
            logger.info("Face detection model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detection model: {e}")
            raise
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_yolov8_face_model(self) -> None:
        """Load YOLOv8-face model."""
        try:
            # Try to load custom face model first, fallback to regular YOLOv8
            try:
                self.model = YOLO(self.config.model_path)
            except Exception:
                logger.warning(f"Could not load {self.config.model_path}, using YOLOv8n")
                self.model = YOLO("yolov8n.pt")
            
            # Move model to device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Configure model settings
            self.model.overrides['conf'] = self.config.confidence_threshold
            self.model.overrides['iou'] = self.config.nms_threshold
            self.model.overrides['max_det'] = self.config.max_detections
            self.model.overrides['half'] = self.config.half_precision and self.device != "cpu"
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def _warmup_model(self) -> None:
        """Warm up the model with dummy input."""
        try:
            dummy_input = np.random.randint(0, 255, (*self.config.input_size, 3), dtype=np.uint8)
            self.detect_faces(dummy_input)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of face detections
        """
        if not self.is_initialized:
            logger.error("Face detector not initialized")
            return []
        
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Run inference
            inference_start = time.time()
            results = self.model(processed_frame, verbose=False)
            inference_time = time.time() - inference_start
            
            # Post-process results
            postprocess_start = time.time()
            detections = self._postprocess_results(results, frame.shape)
            postprocess_time = time.time() - postprocess_start
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(detections), processing_time, inference_time, postprocess_time)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def batch_detect_faces(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """
        Detect faces in multiple frames using batch processing.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one for each frame
        """
        if not self.is_initialized:
            logger.error("Face detector not initialized")
            return [[] for _ in frames]
        
        if not frames:
            return []
        
        start_time = time.time()
        
        try:
            # Preprocess all frames
            processed_frames = [self._preprocess_frame(frame) for frame in frames if frame is not None]
            
            if not processed_frames:
                return [[] for _ in frames]
            
            # Run batch inference
            inference_start = time.time()
            results = self.model(processed_frames, verbose=False)
            inference_time = time.time() - inference_start
            
            # Post-process results for each frame
            postprocess_start = time.time()
            all_detections = []
            
            for i, (result, original_frame) in enumerate(zip(results, frames)):
                if original_frame is not None:
                    detections = self._postprocess_results([result], original_frame.shape)
                    all_detections.append(detections)
                else:
                    all_detections.append([])
            
            postprocess_time = time.time() - postprocess_start
            
            # Update metrics
            total_detections = sum(len(dets) for dets in all_detections)
            processing_time = time.time() - start_time
            self._update_metrics(total_detections, processing_time, inference_time, postprocess_time)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Error in batch face detection: {e}")
            return [[] for _ in frames]
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for face detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        if frame is None:
            return None
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assume BGR format from OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def _postprocess_results(self, results, original_shape: Tuple[int, int, int]) -> List[FaceDetection]:
        """
        Post-process YOLO results to extract face detections.
        
        Args:
            results: YOLO inference results
            original_shape: Original frame shape (H, W, C)
            
        Returns:
            List of face detections
        """
        detections = []
        
        try:
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                
                # Filter by confidence
                valid_indices = confidences >= self.config.confidence_threshold
                boxes = boxes[valid_indices]
                confidences = confidences[valid_indices]
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = box
                    
                    # Create bounding box
                    bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                    
                    # Quality checks
                    if not self._is_valid_face_detection(bbox, original_shape):
                        continue
                    
                    # Create face detection
                    detection = FaceDetection(
                        bbox=bbox,
                        confidence=float(conf),
                        timestamp=datetime.now()
                    )
                    
                    detections.append(detection)
            
            # Sort by confidence (highest first)
            detections.sort(key=lambda x: x.confidence, reverse=True)
            
            # Limit number of detections
            if len(detections) > self.config.max_detections:
                detections = detections[:self.config.max_detections]
            
        except Exception as e:
            logger.error(f"Error post-processing face detection results: {e}")
        
        return detections
    
    def _is_valid_face_detection(self, bbox: BoundingBox, frame_shape: Tuple[int, int, int]) -> bool:
        """
        Validate face detection based on size and position.
        
        Args:
            bbox: Face bounding box
            frame_shape: Original frame shape (H, W, C)
            
        Returns:
            True if detection is valid, False otherwise
        """
        height, width = frame_shape[:2]
        
        # Check if bbox is within frame bounds
        if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > width or bbox.y2 > height:
            return False
        
        # Check face size
        face_area = bbox.area
        if face_area < self.min_face_area or face_area > self.max_face_area:
            return False
        
        # Check aspect ratio (faces should be roughly square)
        aspect_ratio = bbox.width / bbox.height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        return True
    
    def extract_face_crops(self, frame: np.ndarray, detections: List[FaceDetection]) -> List[np.ndarray]:
        """
        Extract face crops from frame based on detections.
        
        Args:
            frame: Original frame
            detections: List of face detections
            
        Returns:
            List of face crop images
        """
        if frame is None or not detections:
            return []
        
        face_crops = []
        
        try:
            for detection in detections:
                bbox = detection.bbox
                
                # Extract crop with some padding
                padding = 0.1  # 10% padding
                pad_x = int(bbox.width * padding)
                pad_y = int(bbox.height * padding)
                
                x1 = max(0, int(bbox.x1 - pad_x))
                y1 = max(0, int(bbox.y1 - pad_y))
                x2 = min(frame.shape[1], int(bbox.x2 + pad_x))
                y2 = min(frame.shape[0], int(bbox.y2 + pad_y))
                
                # Extract crop
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    # Update detection with face crop
                    detection.face_crop = face_crop.copy()
                    face_crops.append(face_crop)
                
        except Exception as e:
            logger.error(f"Error extracting face crops: {e}")
        
        return face_crops
    
    def detect_faces_with_crops(self, frame: np.ndarray) -> Tuple[List[FaceDetection], List[np.ndarray]]:
        """
        Detect faces and extract crops in one operation.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, face_crops)
        """
        detections = self.detect_faces(frame)
        face_crops = self.extract_face_crops(frame, detections)
        return detections, face_crops
    
    def _update_metrics(self, num_detections: int, processing_time: float, 
                       inference_time: float, postprocess_time: float) -> None:
        """
        Update performance metrics.
        
        Args:
            num_detections: Number of faces detected
            processing_time: Total processing time
            inference_time: Model inference time
            postprocess_time: Post-processing time
        """
        self.metrics.total_detections += num_detections
        self.metrics.valid_detections += num_detections
        self.metrics.processing_time = processing_time
        self.metrics.model_inference_time = inference_time
        self.metrics.postprocessing_time = postprocess_time
        self.metrics.timestamp = datetime.now()
        
        # Update FPS calculation
        self.recent_processing_times.append(processing_time)
        if len(self.recent_processing_times) > 30:  # Keep last 30 measurements
            self.recent_processing_times = self.recent_processing_times[-30:]
        
        if self.recent_processing_times:
            avg_processing_time = np.mean(self.recent_processing_times)
            self.metrics.fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_detections": self.metrics.total_detections,
            "valid_detections": self.metrics.valid_detections,
            "current_fps": self.metrics.fps,
            "avg_processing_time": self.metrics.processing_time,
            "avg_inference_time": self.metrics.model_inference_time,
            "avg_postprocessing_time": self.metrics.postprocessing_time,
            "device": self.device,
            "model_type": self.config.model_type.value,
            "confidence_threshold": self.config.confidence_threshold,
            "max_detections": self.config.max_detections,
            "is_initialized": self.is_initialized,
            "last_update": self.metrics.timestamp.isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = DetectionMetrics()
        self.recent_processing_times.clear()
        logger.info("Face detection metrics reset")
    
    def update_config(self, new_config: FaceDetectionConfig) -> None:
        """
        Update detector configuration.
        
        Args:
            new_config: New configuration
        """
        old_model_path = self.config.model_path
        old_device = self.config.device
        
        self.config = new_config
        
        # Reinitialize if model or device changed
        if (new_config.model_path != old_model_path or 
            new_config.device != old_device):
            logger.info("Reinitializing face detector due to config change")
            self._initialize_model()
        else:
            # Update model parameters
            if self.model:
                self.model.overrides['conf'] = new_config.confidence_threshold
                self.model.overrides['iou'] = new_config.nms_threshold
                self.model.overrides['max_det'] = new_config.max_detections
        
        logger.info("Face detector configuration updated")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_initialized or not self.model:
            return {"status": "not_initialized"}
        
        try:
            return {
                "status": "initialized",
                "model_type": self.config.model_type.value,
                "model_path": self.config.model_path,
                "device": self.device,
                "input_size": self.config.input_size,
                "confidence_threshold": self.config.confidence_threshold,
                "nms_threshold": self.config.nms_threshold,
                "max_detections": self.config.max_detections,
                "half_precision": self.config.half_precision,
                "parameters": sum(p.numel() for p in self.model.model.parameters()) if hasattr(self.model, 'model') else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e)}
    
    def benchmark_performance(self, num_frames: int = 100, frame_size: Tuple[int, int] = (640, 480)) -> Dict[str, Any]:
        """
        Benchmark face detection performance.
        
        Args:
            num_frames: Number of frames to process for benchmarking
            frame_size: Size of test frames (width, height)
            
        Returns:
            Benchmark results
        """
        if not self.is_initialized:
            return {"error": "Detector not initialized"}
        
        logger.info(f"Starting face detection benchmark with {num_frames} frames")
        
        # Generate test frames
        test_frames = []
        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (*frame_size[::-1], 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Benchmark single frame processing
        single_times = []
        for frame in test_frames[:min(50, num_frames)]:
            start_time = time.time()
            self.detect_faces(frame)
            single_times.append(time.time() - start_time)
        
        # Benchmark batch processing
        batch_sizes = [1, 2, 4, 8]
        batch_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_frames):
                continue
                
            batch_times = []
            for i in range(0, min(20 * batch_size, len(test_frames)), batch_size):
                batch = test_frames[i:i + batch_size]
                start_time = time.time()
                self.batch_detect_faces(batch)
                batch_times.append(time.time() - start_time)
            
            if batch_times:
                batch_results[batch_size] = {
                    "avg_time": np.mean(batch_times),
                    "fps": batch_size / np.mean(batch_times),
                    "frames_per_second_per_frame": 1.0 / (np.mean(batch_times) / batch_size)
                }
        
        results = {
            "single_frame": {
                "avg_time": np.mean(single_times),
                "min_time": np.min(single_times),
                "max_time": np.max(single_times),
                "fps": 1.0 / np.mean(single_times)
            },
            "batch_processing": batch_results,
            "device": self.device,
            "model_type": self.config.model_type.value,
            "frame_size": frame_size,
            "num_test_frames": num_frames
        }
        
        logger.info(f"Benchmark completed. Single frame FPS: {results['single_frame']['fps']:.2f}")
        return results
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up MultiFaceDetector")
        
        if self.model:
            try:
                # Clear CUDA cache if using GPU
                if self.device == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        
        self.model = None
        self.is_initialized = False
        self.recent_processing_times.clear()
        
        logger.info("MultiFaceDetector cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass