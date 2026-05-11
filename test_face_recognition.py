"""
Unit tests for face recognition engine and identity mapping.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from anti_cheat_system.recognition.face_recognition_engine import (
    FaceRecognitionEngine, FaceRecognitionConfig, RecognitionResult
)
from anti_cheat_system.recognition.identity_mapper import (
    IdentityMapper, IdentityMappingConfig, IdentityMapping, IdentityState
)
from anti_cheat_system.recognition.student_database import StudentDatabaseManager
from anti_cheat_system.models.data_models import StudentIdentity, StudentTrack, BoundingBox


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


class TestIdentityMappingConfig:
    """Test identity mapping configuration."""
    
    def test_config_creation(self):
        """Test creating identity mapping configuration."""
        config = IdentityMappingConfig(
            recognition_interval_frames=20,
            tentative_confidence_threshold=0.7,
            confirmation_confidence_threshold=0.9
        )
        
        assert config.recognition_interval_frames == 20
        assert config.tentative_confidence_threshold == 0.7
        assert config.confirmation_confidence_threshold == 0.9
    
    def test_config_defaults(self):
        """Test configuration with default values."""
        config = IdentityMappingConfig()
        
        assert config.recognition_interval_frames == 30
        assert config.fast_recognition_interval == 5
        assert config.tentative_confidence_threshold == 0.6
        assert config.confirmation_confidence_threshold == 0.8
        assert config.enable_conflict_resolution == True


class TestIdentityMapping:
    """Test identity mapping data structure."""
    
    def test_mapping_creation(self):
        """Test creating identity mapping."""
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        
        mapping = IdentityMapping(
            track_id=1,
            student_identity=student_identity,
            confidence=0.85,
            state=IdentityState.TENTATIVE,
            first_recognized=datetime.now(),
            last_confirmed=datetime.now()
        )
        
        assert mapping.track_id == 1
        assert mapping.student_identity == student_identity
        assert mapping.confidence == 0.85
        assert mapping.state == IdentityState.TENTATIVE
        assert isinstance(mapping.first_recognized, datetime)


class TestIdentityMapper:
    """Test identity mapper."""
    
    @pytest.fixture
    def mapping_config(self):
        """Sample mapping configuration."""
        return IdentityMappingConfig(
            recognition_interval_frames=10,
            fast_recognition_interval=3,
            tentative_confidence_threshold=0.6,
            confirmation_confidence_threshold=0.8
        )
    
    @pytest.fixture
    def sample_tracks(self):
        """Sample student tracks."""
        return [
            StudentTrack(
                track_id=1,
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
                confidence=0.8,
                age=10,
                last_seen=datetime.now(),
                is_confirmed=True
            ),
            StudentTrack(
                track_id=2,
                bbox=BoundingBox(x1=300, y1=150, x2=400, y2=250),
                confidence=0.9,
                age=5,
                last_seen=datetime.now(),
                is_confirmed=True
            )
        ]
    
    def test_mapper_initialization(self, mapping_config):
        """Test identity mapper initialization."""
        mapper = IdentityMapper(mapping_config)
        
        assert mapper.config == mapping_config
        assert len(mapper.identity_mappings) == 0
        assert len(mapper.roll_number_mappings) == 0
        assert len(mapper.recognition_queue) == 0
    
    def test_update_with_tracks(self, mapping_config, sample_tracks):
        """Test updating mapper with tracks."""
        mapper = IdentityMapper(mapping_config)
        
        # Update with tracks
        identity_mappings = mapper.update(sample_tracks, frame_number=1)
        
        assert isinstance(identity_mappings, dict)
        # Initially no mappings since recognition is simulated
    
    def test_should_recognize_track(self, mapping_config, sample_tracks):
        """Test track recognition scheduling."""
        mapper = IdentityMapper(mapping_config)
        
        track = sample_tracks[0]
        
        # New track should be recognized quickly
        should_recognize = mapper._should_recognize_track(track)
        assert should_recognize == True
        
        # After recognition, should wait for interval
        mapper.last_recognition_frame[track.track_id] = 1
        mapper.frame_count = 2  # Only 1 frame passed
        should_recognize = mapper._should_recognize_track(track)
        assert should_recognize == False
        
        # After interval, should recognize again
        mapper.frame_count = 15  # More than interval
        should_recognize = mapper._should_recognize_track(track)
        assert should_recognize == True
    
    def test_force_identity_mapping(self, mapping_config):
        """Test forcing identity mapping."""
        mapper = IdentityMapper(mapping_config)
        
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        
        success = mapper.force_identity_mapping(1, student_identity)
        
        assert success == True
        assert 1 in mapper.identity_mappings
        assert mapper.identity_mappings[1].state == IdentityState.CONFIRMED
        assert mapper.identity_mappings[1].confidence == 1.0
        assert mapper.roll_number_mappings["ROLL001"] == 1
    
    def test_conflict_detection(self, mapping_config):
        """Test identity conflict detection."""
        mapper = IdentityMapper(mapping_config)
        
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        
        # Force mapping for track 1
        mapper.force_identity_mapping(1, student_identity)
        
        # Simulate recognition result for track 2 with same identity
        track = StudentTrack(
            track_id=2,
            bbox=BoundingBox(x1=300, y1=150, x2=400, y2=250),
            confidence=0.9,
            age=5,
            last_seen=datetime.now(),
            is_confirmed=True
        )
        
        result = {
            'is_recognized': True,
            'student_identity': student_identity,
            'confidence': 0.85
        }
        
        # This should detect a conflict
        mapper._process_recognition_result(track, result)
        
        # Check that conflict was detected
        assert "ROLL001" in mapper.active_conflicts
        assert len(mapper.active_conflicts["ROLL001"]) == 2
        assert 1 in mapper.active_conflicts["ROLL001"]
        assert 2 in mapper.active_conflicts["ROLL001"]
    
    def test_performance_metrics(self, mapping_config):
        """Test performance metrics collection."""
        mapper = IdentityMapper(mapping_config)
        
        metrics = mapper.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_mappings_created" in metrics
        assert "confirmed_mappings" in metrics
        assert "tentative_mappings" in metrics
        assert "recognition_success_rate" in metrics
        assert "conflicts_detected" in metrics
        assert "active_mappings" in metrics
    
    def test_identity_statistics(self, mapping_config):
        """Test identity statistics."""
        mapper = IdentityMapper(mapping_config)
        
        # Add some mappings
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        mapper.force_identity_mapping(1, student_identity)
        
        stats = mapper.get_identity_statistics()
        
        assert isinstance(stats, dict)
        assert "state_distribution" in stats
        assert "avg_confidence" in stats
        assert "total_students_recognized" in stats
        assert stats["total_students_recognized"] == 1
    
    def test_mapping_retrieval(self, mapping_config):
        """Test mapping retrieval methods."""
        mapper = IdentityMapper(mapping_config)
        
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        mapper.force_identity_mapping(1, student_identity)
        
        # Get by track ID
        mapping = mapper.get_mapping_by_track_id(1)
        assert mapping is not None
        assert mapping.track_id == 1
        assert mapping.student_identity.roll_number == "ROLL001"
        
        # Get by roll number
        mapping = mapper.get_mapping_by_roll_number("ROLL001")
        assert mapping is not None
        assert mapping.track_id == 1
        
        # Get non-existent mapping
        mapping = mapper.get_mapping_by_track_id(999)
        assert mapping is None
    
    def test_cleanup(self, mapping_config):
        """Test mapper cleanup."""
        mapper = IdentityMapper(mapping_config)
        
        # Add some data
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        mapper.force_identity_mapping(1, student_identity)
        
        # Cleanup
        mapper.cleanup()
        
        # Verify cleanup
        assert len(mapper.identity_mappings) == 0
        assert len(mapper.roll_number_mappings) == 0
        assert len(mapper.recognition_queue) == 0
        assert len(mapper.active_conflicts) == 0


class TestStudentDatabaseManager:
    """Test student database manager."""
    
    @pytest.fixture
    def mock_db_service(self):
        """Mock database service."""
        db_service = Mock()
        session = Mock()
        db_service.get_session.return_value.__enter__.return_value = session
        db_service.get_session.return_value.__exit__.return_value = None
        return db_service, session
    
    @pytest.fixture
    def sample_student_identity(self):
        """Sample student identity."""
        return StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
    
    def test_manager_initialization(self, mock_db_service):
        """Test database manager initialization."""
        db_service, _ = mock_db_service
        manager = StudentDatabaseManager(db_service)
        
        assert manager.db_service == db_service
    
    def test_add_student(self, mock_db_service, sample_student_identity):
        """Test adding student to database."""
        db_service, session = mock_db_service
        manager = StudentDatabaseManager(db_service)
        
        # Mock no existing student
        session.query.return_value.filter.return_value.first.return_value = None
        
        # Add student
        embedding = np.random.randn(512)
        success = manager.add_student(sample_student_identity, embedding)
        
        assert success == True
        session.add.assert_called_once()
        session.commit.assert_called_once()
    
    def test_get_database_statistics(self, mock_db_service):
        """Test getting database statistics."""
        db_service, session = mock_db_service
        manager = StudentDatabaseManager(db_service)
        
        # Mock query results
        session.query.return_value.filter.return_value.count.return_value = 10
        session.query.return_value.filter.return_value.distinct.return_value.all.return_value = [("CS101",), ("CS102",)]
        
        stats = manager.get_database_statistics()
        
        assert isinstance(stats, dict)
        assert "total_students" in stats
        assert "students_with_embeddings" in stats
        assert "embedding_coverage" in stats


class TestIntegration:
    """Integration tests for face recognition and identity mapping."""
    
    def test_recognition_to_mapping_pipeline(self):
        """Test complete pipeline from recognition to identity mapping."""
        # Create components
        recognition_config = FaceRecognitionConfig(
            model_type="facenet",
            device="cpu",
            use_faiss=False
        )
        mapping_config = IdentityMappingConfig()
        
        with patch('anti_cheat_system.recognition.face_recognition_engine.FACENET_AVAILABLE', False):
            recognition_engine = FaceRecognitionEngine(recognition_config)
            identity_mapper = IdentityMapper(mapping_config)
            
            # Add student to recognition database
            student_identity = StudentIdentity(
                roll_number="ROLL001",
                name="Test Student",
                class_id="CS101",
                confidence=0.85
            )
            embedding = np.random.randn(512)
            recognition_engine.add_student_to_database(student_identity, embedding)
            
            # Create student tracks
            tracks = [
                StudentTrack(
                    track_id=1,
                    bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
                    confidence=0.8,
                    age=10,
                    last_seen=datetime.now(),
                    is_confirmed=True
                )
            ]
            
            # Update identity mapper
            identity_mappings = identity_mapper.update(tracks, frame_number=1)
            
            # Verify pipeline works
            assert isinstance(identity_mappings, dict)
            
            # Get metrics
            recognition_metrics = recognition_engine.get_performance_metrics()
            mapping_metrics = identity_mapper.get_performance_metrics()
            
            assert isinstance(recognition_metrics, dict)
            assert isinstance(mapping_metrics, dict)
    
    def test_conflict_resolution_pipeline(self):
        """Test conflict resolution in identity mapping."""
        mapping_config = IdentityMappingConfig(
            duplicate_identity_handling="highest_confidence"
        )
        mapper = IdentityMapper(mapping_config)
        
        # Create student identity
        student_identity = StudentIdentity(
            roll_number="ROLL001",
            name="Test Student",
            class_id="CS101",
            confidence=0.85
        )
        
        # Force mapping for track 1 with lower confidence
        mapping1 = IdentityMapping(
            track_id=1,
            student_identity=student_identity,
            confidence=0.7,
            state=IdentityState.CONFIRMED,
            first_recognized=datetime.now(),
            last_confirmed=datetime.now(),
            max_confidence=0.7
        )
        mapper.identity_mappings[1] = mapping1
        mapper.roll_number_mappings["ROLL001"] = 1
        
        # Simulate conflict with track 2 (higher confidence)
        mapper._handle_identity_conflict(2, 1, student_identity, 0.9)
        
        # Add mapping for track 2
        mapping2 = IdentityMapping(
            track_id=2,
            student_identity=student_identity,
            confidence=0.9,
            state=IdentityState.CONFLICTED,
            first_recognized=datetime.now(),
            last_confirmed=datetime.now(),
            max_confidence=0.9
        )
        mapper.identity_mappings[2] = mapping2
        
        # Resolve conflicts
        mapper._resolve_conflicts()
        
        # Track 2 should win due to higher confidence
        assert 2 in mapper.identity_mappings
        assert mapper.identity_mappings[2].state == IdentityState.CONFIRMED
        assert mapper.roll_number_mappings["ROLL001"] == 2
        
        # Track 1 should be removed
        assert 1 not in mapper.identity_mappings