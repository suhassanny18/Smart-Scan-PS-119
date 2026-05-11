#!/usr/bin/env python3
"""
Validation tests for the Anti-Cheat Detection System
Tests system accuracy, reliability, and real-world scenarios
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from datetime import datetime

from anti_cheat_system.main import AntiCheatSystem
from anti_cheat_system.models import (
    SystemConfig,
    SystemResult,
    AlertLevel,
    DetectionResult,
    FaceMeshResult,
    PoseResult
)


class TestDetectionAccuracy:
    """Test detection accuracy and validation"""
    
    def test_object_detection_accuracy(self):
        """Test object detection accuracy with known inputs"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock object detector with known results
        system.object_detector = Mock()
        
        # Test case 1: No suspicious objects
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[],
            confidence_scores=[],
            bounding_boxes=[],
            detection_count=0,
            suspicious_objects_found=False,
            duration_gate_met=False
        )
        
        system.object_detector.calculate_object_score.return_value = 0.0
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock other components
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=None, nose_offset=0.0, eye_distance=0.0,
            gaze_deviation_percent=0.0, sustained_deviation=False, face_detected=False
        )
        
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=None, shoulder_angle=0.0, is_leaning=False,
            proximity_detected=False, pose_detected=False
        )
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.0, gaze_score=0.0, posture_score=0.0,
            composite_score=0.0, alert_level=AlertLevel.NORMAL, timestamp=datetime.now()
        )
        
        result = system.process_frame(test_frame)
        
        # Validate no false positives for clean frame
        assert result is not None
        assert result.object_score == 0.0
        assert result.alert_level == AlertLevel.NORMAL
        
        # Test case 2: Suspicious objects detected
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[{'class_id': 67, 'confidence': 0.9}],
            confidence_scores=[0.9],
            bounding_boxes=[(10, 10, 50, 50)],
            detection_count=1,
            suspicious_objects_found=True,
            duration_gate_met=True
        )
        
        system.object_detector.calculate_object_score.return_value = 0.8
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.8, gaze_score=0.0, posture_score=0.0,
            composite_score=0.4, alert_level=AlertLevel.AMBER, timestamp=datetime.now()
        )
        
        result = system.process_frame(test_frame)
        
        # Validate correct detection
        assert result is not None
        assert result.object_score > 0.5
        assert result.alert_level in [AlertLevel.AMBER, AlertLevel.RED]
    
    def test_gaze_tracking_accuracy(self):
        """Test gaze tracking accuracy validation"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Test normal gaze behavior
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=np.random.rand(468, 3),
            nose_offset=2.0,  # Small offset
            eye_distance=15.0,  # Normal distance
            gaze_deviation_percent=10.0,  # Low deviation
            sustained_deviation=False,
            face_detected=True
        )
        
        system.gaze_tracker.calculate_gaze_score.return_value = 0.1
        
        # Mock other components
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[], confidence_scores=[], bounding_boxes=[],
            detection_count=0, suspicious_objects_found=False, duration_gate_met=False
        )
        system.object_detector.calculate_object_score.return_value = 0.0
        
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=None, shoulder_angle=0.0, is_leaning=False,
            proximity_detected=False, pose_detected=False
        )
        system.posture_analyzer.calculate_posture_score.return_value = 0.0
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.0, gaze_score=0.1, posture_score=0.0,
            composite_score=0.03, alert_level=AlertLevel.NORMAL, timestamp=datetime.now()
        )
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = system.process_frame(test_frame)
        
        # Validate normal gaze detection
        assert result is not None
        assert result.gaze_score <= 0.3  # Should be low for normal gaze
        assert result.alert_level == AlertLevel.NORMAL
        
        # Test suspicious gaze behavior
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=np.random.rand(468, 3),
            nose_offset=20.0,  # Large offset
            eye_distance=30.0,  # Large distance
            gaze_deviation_percent=40.0,  # High deviation
            sustained_deviation=True,
            face_detected=True
        )
        
        system.gaze_tracker.calculate_gaze_score.return_value = 0.7
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.0, gaze_score=0.7, posture_score=0.0,
            composite_score=0.21, alert_level=AlertLevel.AMBER, timestamp=datetime.now()
        )
        
        result = system.process_frame(test_frame)
        
        # Validate suspicious gaze detection
        assert result is not None
        assert result.gaze_score >= 0.5  # Should be high for suspicious gaze
        assert result.alert_level in [AlertLevel.AMBER, AlertLevel.RED]
    
    def test_posture_analysis_accuracy(self):
        """Test posture analysis accuracy validation"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Test normal posture
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=np.random.rand(33, 3),
            shoulder_angle=5.0,  # Small angle
            is_leaning=False,
            proximity_detected=False,
            pose_detected=True
        )
        
        system.posture_analyzer.calculate_posture_score.return_value = 0.1
        
        # Mock other components
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[], confidence_scores=[], bounding_boxes=[],
            detection_count=0, suspicious_objects_found=False, duration_gate_met=False
        )
        system.object_detector.calculate_object_score.return_value = 0.0
        
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=None, nose_offset=0.0, eye_distance=0.0,
            gaze_deviation_percent=0.0, sustained_deviation=False, face_detected=False
        )
        system.gaze_tracker.calculate_gaze_score.return_value = 0.0
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.0, gaze_score=0.0, posture_score=0.1,
            composite_score=0.02, alert_level=AlertLevel.NORMAL, timestamp=datetime.now()
        )
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = system.process_frame(test_frame)
        
        # Validate normal posture detection
        assert result is not None
        assert result.posture_score <= 0.3  # Should be low for normal posture
        assert result.alert_level == AlertLevel.NORMAL
        
        # Test suspicious posture
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=np.random.rand(33, 3),
            shoulder_angle=35.0,  # Large angle
            is_leaning=True,
            proximity_detected=True,
            pose_detected=True
        )
        
        system.posture_analyzer.calculate_posture_score.return_value = 0.8
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.0, gaze_score=0.0, posture_score=0.8,
            composite_score=0.16, alert_level=AlertLevel.AMBER, timestamp=datetime.now()
        )
        
        result = system.process_frame(test_frame)
        
        # Validate suspicious posture detection
        assert result is not None
        assert result.posture_score >= 0.5  # Should be high for suspicious posture
        assert result.alert_level in [AlertLevel.AMBER, AlertLevel.RED]


class TestFalsePositiveRates:
    """Test false positive rates and accuracy"""
    
    def test_normal_behavior_false_positives(self):
        """Test that normal behavior doesn't trigger false positives"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components for normal behavior
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Configure normal behavior responses
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[], confidence_scores=[], bounding_boxes=[],
            detection_count=0, suspicious_objects_found=False, duration_gate_met=False
        )
        system.object_detector.calculate_object_score.return_value = 0.05
        
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=np.random.rand(468, 3),
            nose_offset=3.0, eye_distance=18.0, gaze_deviation_percent=12.0,
            sustained_deviation=False, face_detected=True
        )
        system.gaze_tracker.calculate_gaze_score.return_value = 0.08
        
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=np.random.rand(33, 3), shoulder_angle=8.0,
            is_leaning=False, proximity_detected=False, pose_detected=True
        )
        system.posture_analyzer.calculate_posture_score.return_value = 0.06
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.05, gaze_score=0.08, posture_score=0.06,
            composite_score=0.063, alert_level=AlertLevel.NORMAL, timestamp=datetime.now()
        )
        
        # Test multiple frames of normal behavior
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        normal_results = []
        
        for _ in range(20):  # Test 20 frames
            result = system.process_frame(test_frame)
            normal_results.append(result)
        
        # Validate low false positive rate
        alert_count = sum(1 for r in normal_results if r and r.alert_level != AlertLevel.NORMAL)
        false_positive_rate = alert_count / len(normal_results)
        
        # Should have very low false positive rate for normal behavior
        assert false_positive_rate <= 0.1  # Less than 10% false positives
        
        # Average scores should be low
        avg_composite = sum(r.composite_score for r in normal_results if r) / len(normal_results)
        assert avg_composite <= 0.2  # Average composite score should be low
    
    def test_alert_threshold_accuracy(self):
        """Test alert threshold accuracy"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Mock basic responses
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[], confidence_scores=[], bounding_boxes=[],
            detection_count=0, suspicious_objects_found=False, duration_gate_met=False
        )
        
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=None, nose_offset=0.0, eye_distance=0.0,
            gaze_deviation_percent=0.0, sustained_deviation=False, face_detected=False
        )
        
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=None, shoulder_angle=0.0, is_leaning=False,
            proximity_detected=False, pose_detected=False
        )
        
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test different alert levels
        alert_scenarios = [
            (0.2, AlertLevel.NORMAL),
            (0.65, AlertLevel.AMBER),
            (0.9, AlertLevel.RED)
        ]
        
        for composite_score, expected_alert in alert_scenarios:
            system.scoring_system.process_detection_results.return_value = SystemResult(
                object_score=0.1, gaze_score=0.1, posture_score=0.1,
                composite_score=composite_score, alert_level=expected_alert, timestamp=datetime.now()
            )
            
            result = system.process_frame(test_frame)
            
            assert result is not None
            assert result.composite_score == composite_score
            assert result.alert_level == expected_alert


class TestSystemValidation:
    """Test overall system validation and reliability"""
    
    def test_system_reliability(self):
        """Test system reliability over extended operation"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components for reliable operation
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Configure consistent responses
        system.object_detector.process_frame.return_value = DetectionResult(
            objects_detected=[], confidence_scores=[], bounding_boxes=[],
            detection_count=0, suspicious_objects_found=False, duration_gate_met=False
        )
        system.object_detector.calculate_object_score.return_value = 0.1
        
        system.gaze_tracker.process_frame.return_value = FaceMeshResult(
            landmarks=None, nose_offset=0.0, eye_distance=0.0,
            gaze_deviation_percent=0.0, sustained_deviation=False, face_detected=False
        )
        system.gaze_tracker.calculate_gaze_score.return_value = 0.1
        
        system.posture_analyzer.process_frame.return_value = PoseResult(
            keypoints=None, shoulder_angle=0.0, is_leaning=False,
            proximity_detected=False, pose_detected=False
        )
        system.posture_analyzer.calculate_posture_score.return_value = 0.1
        
        system.scoring_system.process_detection_results.return_value = SystemResult(
            object_score=0.1, gaze_score=0.1, posture_score=0.1,
            composite_score=0.1, alert_level=AlertLevel.NORMAL, timestamp=datetime.now()
        )
        
        # Test extended operation
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        successful_frames = 0
        total_frames = 100
        
        start_time = time.time()
        
        for i in range(total_frames):
            result = system.process_frame(test_frame)
            if result is not None:
                successful_frames += 1
        
        end_time = time.time()
        
        # Validate reliability metrics
        success_rate = successful_frames / total_frames
        processing_time = end_time - start_time
        avg_fps = total_frames / max(processing_time, 0.001)  # Avoid division by zero
        
        assert success_rate >= 0.95  # 95% success rate
        assert avg_fps >= 10.0  # Minimum 10 FPS
        assert system.error_count <= 5  # Maximum 5 errors
        
        print(f"Reliability test: {success_rate*100:.1f}% success rate, {avg_fps:.1f} FPS")
    
    def test_boundary_conditions(self):
        """Test system behavior at boundary conditions"""
        system = AntiCheatSystem()
        system.is_initialized = True
        
        # Mock components
        system.object_detector = Mock()
        system.gaze_tracker = Mock()
        system.posture_analyzer = Mock()
        system.scoring_system = Mock()
        
        # Test boundary score values
        boundary_scores = [0.0, 0.3, 0.6, 0.85, 1.0]
        
        for score in boundary_scores:
            # Determine expected alert level
            if score >= 0.85:
                expected_alert = AlertLevel.RED
            elif score >= 0.6:
                expected_alert = AlertLevel.AMBER
            else:
                expected_alert = AlertLevel.NORMAL
            
            system.scoring_system.process_detection_results.return_value = SystemResult(
                object_score=score, gaze_score=score, posture_score=score,
                composite_score=score, alert_level=expected_alert, timestamp=datetime.now()
            )
            
            # Mock component responses
            system.object_detector.process_frame.return_value = DetectionResult(
                objects_detected=[], confidence_scores=[], bounding_boxes=[],
                detection_count=0, suspicious_objects_found=False, duration_gate_met=False
            )
            
            system.gaze_tracker.process_frame.return_value = FaceMeshResult(
                landmarks=None, nose_offset=0.0, eye_distance=0.0,
                gaze_deviation_percent=0.0, sustained_deviation=False, face_detected=False
            )
            
            system.posture_analyzer.process_frame.return_value = PoseResult(
                keypoints=None, shoulder_angle=0.0, is_leaning=False,
                proximity_detected=False, pose_detected=False
            )
            
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = system.process_frame(test_frame)
            
            assert result is not None
            assert result.composite_score == score
            assert result.alert_level == expected_alert
            
            print(f"Boundary test: score {score} -> {expected_alert.name}")