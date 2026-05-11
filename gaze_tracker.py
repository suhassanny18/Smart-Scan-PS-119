"""
Gaze tracking engine for the Anti-Cheat Detection System
Uses MediaPipe Face Mesh to track facial landmarks and detect gaze deviation
"""

import cv2
import numpy as np
import logging
import time
import math
from typing import Optional, Tuple, List
from collections import deque

try:
    import mediapipe as mp
except ImportError:
    mp = None

from anti_cheat_system.models import (
    FaceMeshResult,
    FaceLandmarks,
    GazeTrackingConfig
)


class GazeTracker:
    """
    Gaze tracking engine using MediaPipe Face Mesh
    Detects face landmarks and calculates gaze deviation
    """
    
    def __init__(self, config: GazeTrackingConfig = None):
        """
        Initialize gaze tracker
        
        Args:
            config: GazeTrackingConfig with face mesh settings
        """
        self.config = config or GazeTrackingConfig()
        self.face_mesh = None
        self.is_initialized = False
        
        # Gaze deviation tracking
        self.deviation_history = deque(maxlen=100)  # Store last 100 measurements
        self.deviation_start_time = None
        self.sustained_deviation_duration = 0.0
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.total_faces_detected = 0
        self.total_frames_processed = 0
        
        # Calibration data
        self.baseline_nose_offset = None
        self.baseline_eye_distance = None
        self.calibration_samples = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize MediaPipe
        self._initialize_face_mesh()
    
    def _initialize_face_mesh(self) -> bool:
        """
        Initialize MediaPipe Face Mesh
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if mp is None:
                self.logger.error("MediaPipe not available. Please install mediapipe package.")
                return False
            
            self.logger.info("Initializing MediaPipe Face Mesh")
            
            # Initialize Face Mesh
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence
            )
            
            self.is_initialized = True
            self.logger.info("MediaPipe Face Mesh initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MediaPipe Face Mesh: {e}")
            return False
    
    def detect_face_mesh(self, frame: np.ndarray) -> FaceMeshResult:
        """
        Detect face mesh and calculate gaze metrics
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            FaceMeshResult: Face mesh detection results
        """
        if not self.is_initialized or self.face_mesh is None:
            return self._empty_face_mesh_result()
        
        try:
            start_time = time.time()
            self.total_frames_processed += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.face_mesh.process(rgb_frame)
            
            # Process results
            face_mesh_result = self._process_face_mesh_results(results, frame.shape)
            
            # Update deviation tracking
            self._update_deviation_tracking(face_mesh_result)
            
            # Track performance
            inference_time = (time.time() - start_time) * 1000
            self.inference_times.append(inference_time)
            
            if face_mesh_result.face_detected:
                self.total_faces_detected += 1
            
            return face_mesh_result
            
        except Exception as e:
            self.logger.error(f"Face mesh detection failed: {e}")
            return self._empty_face_mesh_result()
    
    def _process_face_mesh_results(self, results, frame_shape: Tuple[int, int, int]) -> FaceMeshResult:
        """
        Process MediaPipe face mesh results
        
        Args:
            results: MediaPipe face mesh results
            frame_shape: Shape of input frame (H, W, C)
            
        Returns:
            FaceMeshResult: Processed face mesh results
        """
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            return self._empty_face_mesh_result()
        
        # Get first face (we only track one face)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to numpy array
        landmarks = self._landmarks_to_numpy(face_landmarks, frame_shape)
        
        # Extract key facial features
        face_features = self._extract_facial_features(landmarks)
        
        # Calculate gaze metrics
        nose_offset = self._calculate_nose_offset(face_features)
        eye_distance = self._calculate_eye_distance(face_features)
        gaze_deviation_percent = self._calculate_gaze_deviation(nose_offset, eye_distance)
        
        # Check for sustained deviation
        sustained_deviation = self._check_sustained_deviation(gaze_deviation_percent)
        
        return FaceMeshResult(
            landmarks=landmarks,
            nose_offset=nose_offset,
            eye_distance=eye_distance,
            gaze_deviation_percent=gaze_deviation_percent,
            sustained_deviation=sustained_deviation,
            face_detected=True
        )
    
    def _landmarks_to_numpy(self, face_landmarks, frame_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array
        
        Args:
            face_landmarks: MediaPipe face landmarks
            frame_shape: Shape of input frame (H, W, C)
            
        Returns:
            np.ndarray: Landmarks as numpy array (N, 3) - x, y, z
        """
        height, width = frame_shape[:2]
        landmarks = []
        
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z  # Relative depth
            landmarks.append([x, y, z])
        
        return np.array(landmarks)
    
    def _extract_facial_features(self, landmarks: np.ndarray) -> FaceLandmarks:
        """
        Extract key facial features from landmarks
        
        Args:
            landmarks: Face landmarks array
            
        Returns:
            FaceLandmarks: Key facial feature points
        """
        # Extract key landmark points
        left_eye = landmarks[self.config.left_eye_idx][:2]  # Only x, y
        right_eye = landmarks[self.config.right_eye_idx][:2]
        nose_tip = landmarks[self.config.nose_tip_idx][:2]
        mouth_center = landmarks[self.config.mouth_center_idx][:2]
        
        return FaceLandmarks(
            left_eye=tuple(left_eye.astype(float)),
            right_eye=tuple(right_eye.astype(float)),
            nose_tip=tuple(nose_tip.astype(float)),
            mouth_center=tuple(mouth_center.astype(float))
        )
    
    def _calculate_nose_offset(self, face_features: FaceLandmarks) -> float:
        """
        Calculate nose offset from eye midpoint
        
        Args:
            face_features: Facial feature points
            
        Returns:
            float: Nose offset in pixels
        """
        eye_center = face_features.eye_center
        nose_tip = face_features.nose_tip
        
        # Calculate horizontal offset
        offset = nose_tip[0] - eye_center[0]
        return float(offset)
    
    def _calculate_eye_distance(self, face_features: FaceLandmarks) -> float:
        """
        Calculate distance between eyes
        
        Args:
            face_features: Facial feature points
            
        Returns:
            float: Eye distance in pixels
        """
        return face_features.eye_distance
    
    def _calculate_gaze_deviation(self, nose_offset: float, eye_distance: float) -> float:
        """
        Calculate gaze deviation percentage
        
        Args:
            nose_offset: Nose offset from eye center
            eye_distance: Distance between eyes
            
        Returns:
            float: Gaze deviation as percentage (0.0 to 1.0+)
        """
        if eye_distance == 0:
            return 0.0
        
        # Normalize offset by eye distance
        normalized_offset = abs(nose_offset) / eye_distance
        
        # Convert to percentage
        deviation_percent = normalized_offset
        
        return float(deviation_percent)
    
    def _update_deviation_tracking(self, face_mesh_result: FaceMeshResult):
        """
        Update gaze deviation tracking for sustained detection
        
        Args:
            face_mesh_result: Current face mesh result
        """
        current_time = time.time()
        
        if not face_mesh_result.face_detected:
            # Reset tracking if no face detected
            self.deviation_start_time = None
            self.sustained_deviation_duration = 0.0
            return
        
        deviation = face_mesh_result.gaze_deviation_percent
        self.deviation_history.append(deviation)
        
        # Check if deviation exceeds threshold
        if deviation > self.config.gaze_deviation_threshold:
            if self.deviation_start_time is None:
                self.deviation_start_time = current_time
            else:
                self.sustained_deviation_duration = current_time - self.deviation_start_time
        else:
            # Reset if deviation drops below threshold
            self.deviation_start_time = None
            self.sustained_deviation_duration = 0.0
    
    def _check_sustained_deviation(self, current_deviation: float) -> bool:
        """
        Check if sustained deviation condition is met
        
        Args:
            current_deviation: Current gaze deviation percentage
            
        Returns:
            bool: True if sustained deviation detected
        """
        return (current_deviation > self.config.gaze_deviation_threshold and 
                self.sustained_deviation_duration >= self.config.gaze_duration_threshold)
    
    def calculate_gaze_score(self, face_mesh_result: FaceMeshResult) -> float:
        """
        Calculate gaze tracking risk score
        
        Args:
            face_mesh_result: Face mesh result to score
            
        Returns:
            float: Risk score between 0.0 and 1.0
        """
        if not face_mesh_result.face_detected:
            return self.config.base_score
        
        deviation = face_mesh_result.gaze_deviation_percent
        
        # Base score from deviation amount
        if deviation <= self.config.gaze_deviation_threshold:
            base_score = 0.0
        else:
            # Scale deviation above threshold
            excess_deviation = deviation - self.config.gaze_deviation_threshold
            base_score = min(excess_deviation * 2.0, 0.6)  # Max 0.6 from deviation alone
        
        # Sustained deviation boost
        sustained_boost = 0.4 if face_mesh_result.sustained_deviation else 0.0
        
        # Calculate final score
        total_score = base_score + sustained_boost
        
        # Clamp to valid range
        return max(self.config.base_score, min(total_score, self.config.max_score))
    
    def _empty_face_mesh_result(self) -> FaceMeshResult:
        """
        Create empty face mesh result for error cases
        
        Returns:
            FaceMeshResult: Empty face mesh result
        """
        return FaceMeshResult(
            landmarks=None,
            nose_offset=0.0,
            eye_distance=0.0,
            gaze_deviation_percent=0.0,
            sustained_deviation=False,
            face_detected=False
        )
    
    def calibrate_baseline(self, frames: List[np.ndarray]) -> bool:
        """
        Calibrate baseline gaze measurements from sample frames
        
        Args:
            frames: List of calibration frames where user looks straight
            
        Returns:
            bool: True if calibration successful
        """
        if not self.is_initialized:
            return False
        
        self.logger.info(f"Starting gaze calibration with {len(frames)} frames")
        
        nose_offsets = []
        eye_distances = []
        
        for frame in frames:
            result = self.detect_face_mesh(frame)
            if result.face_detected:
                nose_offsets.append(result.nose_offset)
                eye_distances.append(result.eye_distance)
        
        if len(nose_offsets) < len(frames) * 0.5:  # Need at least 50% successful detections
            self.logger.warning("Insufficient face detections for calibration")
            return False
        
        # Calculate baseline values
        self.baseline_nose_offset = np.mean(nose_offsets)
        self.baseline_eye_distance = np.mean(eye_distances)
        
        self.logger.info(f"Calibration complete: nose_offset={self.baseline_nose_offset:.2f}, "
                        f"eye_distance={self.baseline_eye_distance:.2f}")
        return True
    
    def get_gaze_stats(self) -> dict:
        """
        Get gaze tracking statistics and performance metrics
        
        Returns:
            dict: Statistics dictionary
        """
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0
        avg_deviation = np.mean(self.deviation_history) if self.deviation_history else 0
        
        face_detection_rate = (self.total_faces_detected / max(self.total_frames_processed, 1)) * 100
        
        return {
            'is_initialized': self.is_initialized,
            'total_frames_processed': self.total_frames_processed,
            'total_faces_detected': self.total_faces_detected,
            'face_detection_rate_percent': round(face_detection_rate, 2),
            'avg_inference_time_ms': round(avg_inference_time, 2),
            'avg_gaze_deviation': round(avg_deviation, 4),
            'sustained_deviation_duration': round(self.sustained_deviation_duration, 2),
            'deviation_threshold': self.config.gaze_deviation_threshold,
            'duration_threshold': self.config.gaze_duration_threshold,
            'baseline_nose_offset': self.baseline_nose_offset,
            'baseline_eye_distance': self.baseline_eye_distance
        }
    
    def reset_deviation_tracking(self):
        """Reset gaze deviation tracking"""
        self.deviation_history.clear()
        self.deviation_start_time = None
        self.sustained_deviation_duration = 0.0
        self.logger.info("Gaze deviation tracking reset")
    
    def draw_face_mesh(self, frame: np.ndarray, face_mesh_result: FaceMeshResult) -> np.ndarray:
        """
        Draw face mesh landmarks and gaze indicators on frame
        
        Args:
            frame: Input frame
            face_mesh_result: Face mesh results to draw
            
        Returns:
            np.ndarray: Frame with drawn face mesh
        """
        if not face_mesh_result.face_detected or face_mesh_result.landmarks is None:
            return frame
        
        output_frame = frame.copy()
        landmarks = face_mesh_result.landmarks
        
        # Draw key landmarks
        key_points = [
            self.config.left_eye_idx,
            self.config.right_eye_idx,
            self.config.nose_tip_idx,
            self.config.mouth_center_idx
        ]
        
        for idx in key_points:
            if idx < len(landmarks):
                x, y = landmarks[idx][:2].astype(int)
                cv2.circle(output_frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw eye center and nose offset line
        if len(landmarks) > max(key_points):
            left_eye = landmarks[self.config.left_eye_idx][:2].astype(int)
            right_eye = landmarks[self.config.right_eye_idx][:2].astype(int)
            nose_tip = landmarks[self.config.nose_tip_idx][:2].astype(int)
            
            # Eye center
            eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            cv2.circle(output_frame, eye_center, 2, (255, 0, 0), -1)
            
            # Nose offset line
            cv2.line(output_frame, eye_center, tuple(nose_tip), (255, 0, 0), 2)
        
        # Draw gaze status
        deviation = face_mesh_result.gaze_deviation_percent
        if deviation > self.config.gaze_deviation_threshold:
            if face_mesh_result.sustained_deviation:
                status_text = f"GAZE DEVIATION: {deviation:.2f} [SUSTAINED]"
                status_color = (0, 0, 255)  # Red
            else:
                status_text = f"GAZE DEVIATION: {deviation:.2f}"
                status_color = (0, 165, 255)  # Orange
        else:
            status_text = f"GAZE OK: {deviation:.2f}"
            status_color = (0, 255, 0)  # Green
        
        cv2.putText(output_frame, status_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return output_frame
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                self.face_mesh.close()
        except:
            pass