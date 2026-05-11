"""
Multi-Object Detection Engine for Anti-Cheat Detection System.
Detects suspicious items like phones, papers, chits, and other devices using YOLOv8n.
"""

import logging
import time
from typing import List, Optional, Tuple, Dict, Any, Set
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
from enum import Enum

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from ..models.data_models import ObjectDetection, BoundingBox
from ..models.enums import ModelType

logger = logging.getLogger(__name__)


class SuspiciousObjectType(Enum):
    """Types of suspicious objects for exam monitoring."""
    PHONE = "phone"
    PAPER = "paper"
    BOOK = "book"
    LAPTOP = "laptop"
    TABLET = "tablet"
    CALCULATOR = "calculator"
    UNKNOWN_DEVICE = "unknown_device"


@dataclass
class ObjectDetectionConfig:
    """Configuration for multi-object detection."""
    model_type: ModelType = ModelType.YOLOV8N
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.6
    nms_threshold: float = 0.45
    max_detections: int = 30
    input_size: Tuple[int, int] = (640, 640)
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    half_precision: bool = False
    batch_size: int = 1
    
    # Suspicious object filtering
    suspicious_classes: Set[int] = field(default_factory=lambda: {
        67,  # cell phone
        73,  # laptop
        76,  # keyboard
        77,  # mouse
        84,  # book
        # Add more COCO class IDs for suspicious items
    })
    
    # Temporal validation
    temporal_window: int = 10  # frames to consider for temporal validation
    min_detection_frames: int = 3  # minimum frames object must be detected
    confidence_decay: float = 0.95  # confidence decay per frame
    
    # Size filtering
    min_object_area: int = 100  # minimum object area in pixels
    max_object_area: int = 50000  # maximum object area in pixels
    min_aspect_ratio: float = 0.1
    max_aspect_ratio: float = 10.0


@dataclass
class DetectionMetrics:
    """Metrics for object detection performance."""
    total_detections: int = 0
    valid_detections: int = 0
    filtered_detections: int = 0
    processing_time: float = 0.0
    fps: float = 0.0
    model_inference_time: float = 0.0
    postprocessing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TemporalObject:
    """Tracks an object across multiple frames."""
    object_id: str
    object_type: SuspiciousObjectType
    detections: List[ObjectDetection] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    is_validated: bool = False
    
    def add_detection(self, detection: ObjectDetection) -> None:
        """Add a new detection for this object."""
        self.detections.append(detection)
        self.confidence_history.append(detection.confidence)
        self.last_seen = detection.timestamp
        
        # Keep only recent detections
        if len(self.detections) > 20:
            self.detections = self.detections[-20:]
            self.confidence_history = self.confidence_history[-20:]
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all detections."""
        return np.mean(self.confidence_history) if self.confidence_history else 0.0
    
    def get_detection_count(self) -> int:
        """Get number of detections for this object."""
        return len(self.detections)
    
    def is_temporally_valid(self, min_frames: int) -> bool:
        """Check if object has been detected for minimum required frames."""
        return len(self.detections) >= min_frames


class MultiObjectDetector:
    """
    Multi-object detection engine for detecting suspicious items in exam environment.
    
    Features:
    - YOLOv8n-based detection for common objects
    - Suspicious object classification and filtering
    - Temporal validation to reduce false positives
    - Object tracking across frames
    - Confidence scoring and validation
    - Performance optimization with GPU acceleration
    """
    
    def __init__(self, config: ObjectDetectionConfig):
        """
        Initialize multi-object detector.
        
        Args:
            config: Object detection configuration
        """
        self.config = config
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # Temporal tracking
        self.temporal_objects: Dict[str, TemporalObject] = {}
        self.next_object_id = 0
        
        # Performance tracking
        self.metrics = DetectionMetrics()
        self.recent_processing_times: List[float] = []
        
        # COCO class mapping to suspicious objects
        self.coco_to_suspicious = {
            67: SuspiciousObjectType.PHONE,      # cell phone
            73: SuspiciousObjectType.LAPTOP,     # laptop
            76: SuspiciousObjectType.UNKNOWN_DEVICE,  # keyboard
            77: SuspiciousObjectType.UNKNOWN_DEVICE,  # mouse
            84: SuspiciousObjectType.BOOK,       # book
            # Add more mappings as needed
        }
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"MultiObjectDetector initialized with {config.model_type.value}")
    
    def _initialize_model(self) -> None:
        """Initialize the object detection model."""
        try:
            if YOLO is None:
                raise ImportError("ultralytics package not available")
            
            # Determine device
            self.device = self._get_device()
            
            # Load model based on type
            if self.config.model_type == ModelType.YOLOV8N:
                self._load_yolov8_model()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Warm up model
            self._warmup_model()
            
            self.is_initialized = True
            logger.info("Object detection model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize object detection model: {e}")
            self.is_initialized = False
            raise
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        if self.config.device == "auto":
            # Auto-detect best available device
            import torch
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
    
    def _load_yolov8_model(self) -> None:
        """Load YOLOv8n model."""
        try:
            # Load YOLOv8n model
            self.model = YOLO(self.config.model_path)
            
            # Move model to device
            if self.device != "cpu":
                self.model.to(self.device)
            
            # Configure model settings
            self.model.overrides['conf'] = self.config.confidence_threshold
            self.model.overrides['iou'] = self.config.nms_threshold
            self.model.overrides['max_det'] = self.config.max_detections
            self.model.overrides['half'] = self.config.half_precision and self.device != "cpu"
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8n model: {e}")
            raise
    
    def _warmup_model(self) -> None:
        """Warm up the model with dummy input."""
        try:
            dummy_input = np.random.randint(0, 255, (*self.config.input_size, 3), dtype=np.uint8)
            self.detect_objects(dummy_input)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def detect_objects(self, frame: np.ndarray) -> List[ObjectDetection]:
        """
        Detect suspicious objects in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of object detections
        """
        if not self.is_initialized:
            logger.error("Object detector not initialized")
            return []
        
        if frame is None or frame.size == 0:
            return []
        
        start_time = time.time()
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Run inference
            inference_start = time.time()
            results = self.model(
                processed_frame,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                max_det=self.config.max_detections,
                device=self.device,
                half=self.config.half_precision,
                verbose=False
            )
            inference_time = time.time() - inference_start
            
            # Post-process results
            postprocess_start = time.time()
            detections = self._process_results(results, frame.shape)
            
            # Apply temporal validation
            validated_detections = self._apply_temporal_validation(detections)
            postprocess_time = time.time() - postprocess_start
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_metrics(len(validated_detections), processing_time, inference_time, postprocess_time)
            
            logger.debug(f"Detected {len(validated_detections)} suspicious objects in {processing_time:.3f}s")
            
            return validated_detections
            
        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return []
    
    def batch_detect_objects(self, frames: List[np.ndarray]) -> List[List[ObjectDetection]]:
        """
        Detect objects in multiple frames using batch processing.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one for each frame
        """
        if not self.is_initialized:
            logger.error("Object detector not initialized")
            return [[] for _ in frames]
        
        if not frames:
            return []
        
        start_time = time.time()
        
        try:
            # Process frames in batches
            batch_size = min(self.config.batch_size, len(frames))
            all_detections = []
            
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Preprocess batch
                processed_frames = [self._preprocess_frame(frame) for frame in batch_frames if frame is not None]
                
                if not processed_frames:
                    all_detections.extend([[] for _ in batch_frames])
                    continue
                
                # Run batch inference
                inference_start = time.time()
                results = self.model(
                    processed_frames,
                    conf=self.config.confidence_threshold,
                    iou=self.config.nms_threshold,
                    max_det=self.config.max_detections,
                    device=self.device,
                    half=self.config.half_precision,
                    verbose=False
                )
                inference_time = time.time() - inference_start
                
                # Process results for each frame in batch
                postprocess_start = time.time()
                for j, result in enumerate(results):
                    if j < len(batch_frames) and batch_frames[j] is not None:
                        detections = self._process_single_result(result, batch_frames[j].shape)
                        validated_detections = self._apply_temporal_validation(detections)
                        all_detections.append(validated_detections)
                    else:
                        all_detections.append([])
                postprocess_time = time.time() - postprocess_start
            
            # Update metrics
            total_detections = sum(len(dets) for dets in all_detections)
            processing_time = time.time() - start_time
            self._update_metrics(total_detections, processing_time, inference_time, postprocess_time)
            
            logger.debug(f"Batch detected {total_detections} objects in {len(frames)} frames ({processing_time:.3f}s)")
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch object detection failed: {e}")
            return [[] for _ in frames]
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for object detection.
        
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
    
    def _process_results(self, results, original_shape: Tuple[int, int, int]) -> List[ObjectDetection]:
        """
        Process YOLO detection results.
        
        Args:
            results: YOLO detection results
            original_shape: Original frame shape (H, W, C)
            
        Returns:
            List of object detections
        """
        if not results:
            return []
        
        # Handle single result
        if len(results) == 1:
            return self._process_single_result(results[0], original_shape)
        
        # Handle multiple results (shouldn't happen for single frame)
        all_detections = []
        for result in results:
            detections = self._process_single_result(result, original_shape)
            all_detections.extend(detections)
        
        return all_detections
    
    def _process_single_result(self, result, original_shape: Tuple[int, int, int]) -> List[ObjectDetection]:
        """
        Process a single YOLO result.
        
        Args:
            result: Single YOLO result
            original_shape: Original frame shape (H, W, C)
            
        Returns:
            List of object detections
        """
        detections = []
        
        if not hasattr(result, 'boxes') or result.boxes is None:
            return detections
        
        boxes = result.boxes
        
        # Extract detection data
        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
            coords = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
            class_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
            
            for i, coord in enumerate(coords):
                try:
                    x1, y1, x2, y2 = coord
                    confidence = confidences[i] if confidences is not None else 1.0
                    class_id = class_ids[i] if class_ids is not None else -1
                    
                    # Check if this is a suspicious object
                    if not self._is_suspicious_class(class_id):
                        continue
                    
                    # Validate detection
                    if not self._validate_detection(x1, y1, x2, y2, original_shape):
                        continue
                    
                    # Create bounding box
                    bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                    
                    # Determine object type
                    object_type = self._get_suspicious_object_type(class_id)
                    
                    # Create object detection
                    detection = ObjectDetection(
                        bbox=bbox,
                        confidence=float(confidence),
                        class_id=int(class_id),
                        class_name=self._get_class_name(class_id),
                        is_suspicious=True,
                        object_type=object_type.value,
                        timestamp=datetime.now()
                    )
                    
                    detections.append(detection)
                    
                except Exception as e:
                    logger.error(f"Error processing detection {i}: {e}")
                    continue
        
        return detections
    
    def _is_suspicious_class(self, class_id: int) -> bool:
        """
        Check if class ID corresponds to a suspicious object.
        
        Args:
            class_id: COCO class ID
            
        Returns:
            True if suspicious, False otherwise
        """
        return class_id in self.config.suspicious_classes
    
    def _get_suspicious_object_type(self, class_id: int) -> SuspiciousObjectType:
        """
        Map COCO class ID to suspicious object type.
        
        Args:
            class_id: COCO class ID
            
        Returns:
            Suspicious object type
        """
        return self.coco_to_suspicious.get(class_id, SuspiciousObjectType.UNKNOWN_DEVICE)
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Get class name from COCO class ID.
        
        Args:
            class_id: COCO class ID
            
        Returns:
            Class name
        """
        # COCO class names mapping (subset for suspicious objects)
        coco_names = {
            67: "cell phone",
            73: "laptop",
            76: "keyboard", 
            77: "mouse",
            84: "book"
        }
        return coco_names.get(class_id, f"class_{class_id}")
    
    def _validate_detection(self, x1: float, y1: float, x2: float, y2: float, frame_shape: Tuple[int, ...]) -> bool:
        """
        Validate object detection.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            frame_shape: Shape of the frame (H, W, C)
            
        Returns:
            True if detection is valid, False otherwise
        """
        h, w = frame_shape[:2]
        
        # Check if coordinates are within frame bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # Check if bounding box is valid
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Check object size constraints
        object_area = (x2 - x1) * (y2 - y1)
        if object_area < self.config.min_object_area or object_area > self.config.max_object_area:
            return False
        
        # Check aspect ratio
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio < self.config.min_aspect_ratio or aspect_ratio > self.config.max_aspect_ratio:
            return False
        
        return True
    
    def _apply_temporal_validation(self, detections: List[ObjectDetection]) -> List[ObjectDetection]:
        """
        Apply temporal validation to reduce false positives.
        
        Args:
            detections: Current frame detections
            
        Returns:
            Temporally validated detections
        """
        validated_detections = []
        
        # Update temporal objects with current detections
        for detection in detections:
            # Find matching temporal object or create new one
            matched_object = self._find_matching_temporal_object(detection)
            
            if matched_object:
                matched_object.add_detection(detection)
            else:
                # Create new temporal object
                object_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                
                temporal_obj = TemporalObject(
                    object_id=object_id,
                    object_type=detection.object_type
                )
                temporal_obj.add_detection(detection)
                self.temporal_objects[object_id] = temporal_obj
        
        # Clean up old temporal objects
        self._cleanup_temporal_objects()
        
        # Validate temporal objects and extract detections
        for temporal_obj in self.temporal_objects.values():
            if temporal_obj.is_temporally_valid(self.config.min_detection_frames):
                # Use the most recent detection from this temporal object
                if temporal_obj.detections:
                    latest_detection = temporal_obj.detections[-1]
                    # Update confidence based on temporal validation
                    latest_detection.confidence = temporal_obj.get_average_confidence()
                    validated_detections.append(latest_detection)
        
        return validated_detections
    
    def _find_matching_temporal_object(self, detection: ObjectDetection) -> Optional[TemporalObject]:
        """
        Find temporal object that matches the current detection.
        
        Args:
            detection: Current detection
            
        Returns:
            Matching temporal object or None
        """
        best_match = None
        best_iou = 0.0
        
        for temporal_obj in self.temporal_objects.values():
            if (temporal_obj.object_type == detection.object_type and 
                temporal_obj.detections):
                
                # Calculate IoU with most recent detection
                recent_detection = temporal_obj.detections[-1]
                iou = self._calculate_iou(detection.bbox, recent_detection.bbox)
                
                if iou > 0.3 and iou > best_iou:  # Minimum IoU threshold
                    best_match = temporal_obj
                    best_iou = iou
        
        return best_match
    
    def _calculate_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = bbox1.area
        area2 = bbox2.area
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _cleanup_temporal_objects(self) -> None:
        """Clean up old temporal objects that haven't been updated recently."""
        current_time = datetime.now()
        timeout_seconds = 5.0  # Remove objects not seen for 5 seconds
        
        objects_to_remove = []
        for obj_id, temporal_obj in self.temporal_objects.items():
            time_since_last_seen = (current_time - temporal_obj.last_seen).total_seconds()
            if time_since_last_seen > timeout_seconds:
                objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.temporal_objects[obj_id]
    
    def _update_metrics(self, num_detections: int, processing_time: float, 
                       inference_time: float, postprocess_time: float) -> None:
        """
        Update performance metrics.
        
        Args:
            num_detections: Number of objects detected
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
            "filtered_detections": self.metrics.filtered_detections,
            "current_fps": self.metrics.fps,
            "avg_processing_time": self.metrics.processing_time,
            "avg_inference_time": self.metrics.model_inference_time,
            "avg_postprocessing_time": self.metrics.postprocessing_time,
            "device": self.device,
            "model_type": self.config.model_type.value,
            "confidence_threshold": self.config.confidence_threshold,
            "max_detections": self.config.max_detections,
            "temporal_objects_count": len(self.temporal_objects),
            "suspicious_classes": list(self.config.suspicious_classes),
            "is_initialized": self.is_initialized,
            "last_update": self.metrics.timestamp.isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = DetectionMetrics()
        self.recent_processing_times.clear()
        logger.info("Object detection metrics reset")
    
    def reset_temporal_tracking(self) -> None:
        """Reset temporal object tracking."""
        self.temporal_objects.clear()
        self.next_object_id = 0
        logger.info("Temporal object tracking reset")
    
    def update_config(self, new_config: ObjectDetectionConfig) -> None:
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
            logger.info("Reinitializing object detector due to config change")
            self._initialize_model()
        else:
            # Update model parameters
            if self.model:
                self.model.overrides['conf'] = new_config.confidence_threshold
                self.model.overrides['iou'] = new_config.nms_threshold
                self.model.overrides['max_det'] = new_config.max_detections
        
        logger.info("Object detector configuration updated")
    
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
                "suspicious_classes": list(self.config.suspicious_classes),
                "temporal_window": self.config.temporal_window,
                "min_detection_frames": self.config.min_detection_frames,
                "parameters": sum(p.numel() for p in self.model.model.parameters()) if hasattr(self.model, 'model') else "unknown"
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up MultiObjectDetector")
        
        if self.model is not None:
            try:
                # Clear CUDA cache if using GPU
                if self.device == "cuda":
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error during cleanup: {e}")
        
        self.model = None
        self.is_initialized = False
        self.temporal_objects.clear()
        self.recent_processing_times.clear()
        
        logger.info("MultiObjectDetector cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass