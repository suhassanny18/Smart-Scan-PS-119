"""
Unit tests for face recognition engine.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from anti_cheat_system.recognition.face_recognition_engine import (
    FaceRecognitionEngine, FaceRecognitionConfig, RecognitionResult
)
from anti_cheat_system.models.data_models import StudentIdentity


class TestFaceRecognitionConfig:
    """Test face recognition configuration."""
    
    def test_config_creation(self):
        """Test creating face recognition configuration."""
        config = FaceRecognitionConfig(
            model_type="facenet",
            recognition_threshold=0.7,
            high_confidence_threshold=0.9,
            device="cuda"
        )
        
        assert config.model_type == "facenet"
        assert config.recognition_threshold == 0.7
        assert config.high_confidence_threshold == 0.9
        assert config.device == "cuda"
    
    def test_config_defaults(self):
        """Test configuration with default values."""
        config = FaceRecognitionConfig()
        
        assert config.model_type == "facenet"
        assert config.recognition_threshold == 0.6
        assert config.high_confidence_threshold == 0.8
        assert config.verification_threshold == 0.4
        assert config.face_size == (160, 160)
        assert config.use_faiss == True


class TestRecognitionResult:
    """Test recognition result."""
    
    def test_result_creation(self):
        """Test creating recognition result."""
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        
        result = RecognitionResult(
            student_identity=student_identity,
            confidence=0.85,
            similarity_score=0.82,
            processing_time=0.05
        )
        
        assert result.student_identity == student_identity
        assert result.confidence == 0.85
        assert result.similarity_score == 0.82
        assert result.processing_time == 0.05
        assert result.is_recognized == True
    
    def test_result_not_recognized(self):
        """Test recognition result when not recognized."""
        result = RecognitionResult()
        
        assert result.student_identity is None
        assert result.confidence == 0.0
        assert result.is_recognized == False


class TestFaceRecognitionEngine:
    """Test face recognition engine."""
    
    @pytest.fixture
    def recognition_config(self):
        """Sample recognition configuration."""
        return FaceRecognitionConfig(
            model_type="facenet",
            recognition_threshold=0.6,
            device="cpu",  # Use CPU for testing
            use_faiss=False  # Disable FAISS for testing
        )
    
    @pytest.fixture
    def mock_face_image(self):
        """Mock face image."""
        return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_student_identity(self):
        """Sample student identity."""
        return StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_engine_initialization_fallback(self, recognition_config):
        """Test engine initialization with fallback model."""
        engine = FaceRecognitionEngine(recognition_config)
        
        assert engine.config == recognition_config
        assert engine.device is not None
        assert engine.model is not None
        assert engine.faiss_index is None  # FAISS disabled
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_extract_embedding(self, recognition_config, mock_face_image):
        """Test embedding extraction."""
        engine = FaceRecognitionEngine(recognition_config)
        
        embedding = engine.extract_embedding(mock_face_image)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # Should be flattened
        assert embedding.shape[0] > 0  # Should have some dimensions
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_batch_extract_embeddings(self, recognition_config):
        """Test batch embedding extraction."""
        engine = FaceRecognitionEngine(recognition_config)
        
        # Create multiple face images
        face_images = [
            np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        embeddings = engine.batch_extract_embeddings(face_images)
        
        assert len(embeddings) == 3
        for embedding in embeddings:
            if embedding is not None:
                assert isinstance(embedding, np.ndarray)
                assert len(embedding.shape) == 1
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_add_student_to_database(self, recognition_config, sample_student_identity):
        """Test adding student to recognition database."""
        engine = FaceRecognitionEngine(recognition_config)
        
        # Create mock embedding
        embedding = np.random.randn(512)
        
        success = engine.add_student_to_database(sample_student_identity, embedding)
        
        assert success == True
        assert len(engine.embedding_database) == 1
        assert len(engine.identity_database) == 1
        assert engine.identity_database[0] == sample_student_identity
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_search_similar_face(self, recognition_config, sample_student_identity):
        """Test searching for similar face."""
        engine = FaceRecognitionEngine(recognition_config)
        
        # Add student to database
        embedding = np.random.randn(512)
        engine.add_student_to_database(sample_student_identity, embedding)
        
        # Search with similar embedding
        similar_embedding = embedding + np.random.randn(512) * 0.1  # Add small noise
        identity, confidence, similarity = engine._search_similar_face(similar_embedding)
        
        # Should find the student with reasonable confidence
        assert identity is not None
        assert identity.roll_number == sample_student_identity.roll_number
        assert confidence > 0.0
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_verify_identity(self, recognition_config, sample_student_identity):
        """Test identity verification."""
        engine = FaceRecognitionEngine(recognition_config)
        
        # Add student to database
        original_embedding = np.random.randn(512)
        engine.add_student_to_database(sample_student_identity, original_embedding)
        
        # Verify with same embedding
        is_verified, confidence = engine.verify_identity(original_embedding, sample_student_identity)
        
        assert is_verified == True
        assert confidence > engine.config.verification_threshold
        
        # Verify with different embedding
        different_embedding = np.random.randn(512)
        is_verified, confidence = engine.verify_identity(different_embedding, sample_student_identity)
        
        assert is_verified == False or confidence < engine.config.verification_threshold
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_performance_metrics(self, recognition_config):
        """Test performance metrics collection."""
        engine = FaceRecognitionEngine(recognition_config)
        
        metrics = engine.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_recognitions" in metrics
        assert "successful_recognitions" in metrics
        assert "recognition_rate" in metrics
        assert "database_size" in metrics
        assert "device" in metrics
        assert "model_type" in metrics
    
    @patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False)
    def test_database_operations(self, recognition_config, sample_student_identity):
        """Test database save/load operations."""
        engine = FaceRecognitionEngine(recognition_config)
        
        # Add student
        embedding = np.random.randn(512)
        engine.add_student_to_database(sample_student_identity, embedding)
        
        # Save database
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name
        
        success = engine.save_database_to_file(temp_path)
        assert success == True
        
        # Create new engine and load database
        new_engine = FaceRecognitionEngine(recognition_config)
        success = new_engine.load_database_from_file(temp_path)
        
        assert success == True
        assert len(new_engine.embedding_database) == 1
        assert len(new_engine.identity_database) == 1
        assert new_engine.identity_database[0].roll_number == sample_student_identity.roll_number
        
        # Cleanup
        import os
        os.unlink(temp_path)