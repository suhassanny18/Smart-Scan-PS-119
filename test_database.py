"""
Unit tests for database operations in the Anti-Cheat Detection System.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import tempfile
import os

from anti_cheat_system.database.models import Base, Student, TrackingSession, BehaviorEvent, Alert
from anti_cheat_system.database.db_service import DatabaseService
from anti_cheat_system.models.data_models import BehaviorEvent as BehaviorEventData, Alert as AlertData
from anti_cheat_system.models.enums import BehaviorEventType, AlertLevel, AttendanceStatus


@pytest.fixture
def test_db():
    """Create a test database."""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    db_service = DatabaseService("sqlite:///:memory:")
    db_service.engine = engine
    db_service.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return db_service


@pytest.fixture
def sample_student_data():
    """Sample student data for testing."""
    return {
        "roll_number": "CS2021001",
        "name": "John Doe",
        "class_id": "CS2021",
        "email": "john.doe@school.edu"
    }


@pytest.fixture
def sample_face_embedding():
    """Sample face embedding for testing."""
    return np.random.rand(512).astype(np.float32)


class TestDatabaseService:
    """Test DatabaseService functionality."""
    
    def test_database_initialization(self, test_db):
        """Test database service initialization."""
        assert test_db.database_url == "sqlite:///:memory:"
        assert test_db.engine is not None
        assert test_db.SessionLocal is not None
    
    def test_health_check_healthy(self, test_db):
        """Test database health check when healthy."""
        health = test_db.health_check()
        
        assert health["status"] == "healthy"
        assert "database_url" in health
        assert "tables" in health
        assert "timestamp" in health
    
    def test_health_check_unhealthy(self):
        """Test database health check when unhealthy."""
        # Create service with invalid URL
        db_service = DatabaseService("invalid://url")
        health = db_service.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health


class TestStudentManagement:
    """Test student management operations."""
    
    def test_create_student(self, test_db, sample_student_data):
        """Test creating a new student."""
        student = test_db.create_student(**sample_student_data)
        
        assert student.roll_number == sample_student_data["roll_number"]
        assert student.name == sample_student_data["name"]
        assert student.class_id == sample_student_data["class_id"]
        assert student.email == sample_student_data["email"]
        assert student.is_active is True
        assert student.created_at is not None
    
    def test_create_student_with_embedding(self, test_db, sample_student_data, sample_face_embedding):
        """Test creating a student with face embedding."""
        student = test_db.create_student(
            face_embedding=sample_face_embedding,
            **sample_student_data
        )
        
        assert student.face_embedding is not None
        
        # Retrieve and verify embedding
        retrieved_embedding = test_db.get_student_embedding(sample_student_data["roll_number"])
        assert retrieved_embedding is not None
        np.testing.assert_array_equal(retrieved_embedding, sample_face_embedding)
    
    def test_create_duplicate_student(self, test_db, sample_student_data):
        """Test creating a duplicate student."""
        test_db.create_student(**sample_student_data)
        
        with pytest.raises(ValueError, match="Student with roll number .* already exists"):
            test_db.create_student(**sample_student_data)
    
    def test_get_student_by_roll_number(self, test_db, sample_student_data):
        """Test retrieving student by roll number."""
        created_student = test_db.create_student(**sample_student_data)
        retrieved_student = test_db.get_student_by_roll_number(sample_student_data["roll_number"])
        
        assert retrieved_student is not None
        assert retrieved_student.id == created_student.id
        assert retrieved_student.roll_number == sample_student_data["roll_number"]
    
    def test_get_nonexistent_student(self, test_db):
        """Test retrieving non-existent student."""
        student = test_db.get_student_by_roll_number("NONEXISTENT")
        assert student is None
    
    def test_get_students_by_class(self, test_db):
        """Test retrieving students by class."""
        # Create multiple students in same class
        for i in range(3):
            test_db.create_student(
                roll_number=f"CS2021{i:03d}",
                name=f"Student {i}",
                class_id="CS2021",
                email=f"student{i}@school.edu"
            )
        
        # Create student in different class
        test_db.create_student(
            roll_number="EE2021001",
            name="EE Student",
            class_id="EE2021",
            email="ee.student@school.edu"
        )
        
        cs_students = test_db.get_students_by_class("CS2021")
        ee_students = test_db.get_students_by_class("EE2021")
        
        assert len(cs_students) == 3
        assert len(ee_students) == 1
        assert all(s.class_id == "CS2021" for s in cs_students)
        assert ee_students[0].class_id == "EE2021"
    
    def test_update_student_embedding(self, test_db, sample_student_data, sample_face_embedding):
        """Test updating student face embedding."""
        test_db.create_student(**sample_student_data)
        
        # Update embedding
        success = test_db.update_student_embedding(
            sample_student_data["roll_number"], 
            sample_face_embedding
        )
        assert success is True
        
        # Verify embedding was updated
        retrieved_embedding = test_db.get_student_embedding(sample_student_data["roll_number"])
        np.testing.assert_array_equal(retrieved_embedding, sample_face_embedding)
    
    def test_update_nonexistent_student_embedding(self, test_db, sample_face_embedding):
        """Test updating embedding for non-existent student."""
        success = test_db.update_student_embedding("NONEXISTENT", sample_face_embedding)
        assert success is False
    
    def test_get_nonexistent_student_embedding(self, test_db):
        """Test getting embedding for non-existent student."""
        embedding = test_db.get_student_embedding("NONEXISTENT")
        assert embedding is None


class TestTrackingSessionManagement:
    """Test tracking session management operations."""
    
    def test_create_tracking_session(self, test_db, sample_student_data):
        """Test creating a tracking session."""
        student = test_db.create_student(**sample_student_data)
        
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        assert session.session_id == "session_001"
        assert session.student_id == student.id
        assert session.camera_id == "cam_01"
        assert session.start_time is not None
        assert session.end_time is None
    
    def test_create_session_nonexistent_student(self, test_db):
        """Test creating session for non-existent student."""
        with pytest.raises(ValueError, match="Student with roll number .* not found"):
            test_db.create_tracking_session(
                session_id="session_001",
                student_roll_number="NONEXISTENT",
                camera_id="cam_01"
            )
    
    def test_end_tracking_session(self, test_db, sample_student_data):
        """Test ending a tracking session."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        success = test_db.end_tracking_session(
            session_id="session_001",
            final_suspicion_score=75.5,
            max_suspicion_score=95.0,
            alert_count=2
        )
        
        assert success is True
        
        # Verify session was updated
        with test_db.get_session() as db_session:
            updated_session = db_session.query(TrackingSession).filter(
                TrackingSession.session_id == "session_001"
            ).first()
            
            assert updated_session.end_time is not None
            assert updated_session.duration_seconds is not None
            assert updated_session.final_suspicion_score == 75.5
            assert updated_session.max_suspicion_score == 95.0
            assert updated_session.alert_count == 2
    
    def test_end_nonexistent_session(self, test_db):
        """Test ending non-existent session."""
        success = test_db.end_tracking_session(
            session_id="NONEXISTENT",
            final_suspicion_score=0.0,
            max_suspicion_score=0.0,
            alert_count=0
        )
        assert success is False


class TestBehaviorEventLogging:
    """Test behavior event logging operations."""
    
    def test_log_behavior_event(self, test_db, sample_student_data):
        """Test logging a behavior event."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        event_data = BehaviorEventData(
            event_type=BehaviorEventType.PHONE_DETECTED,
            confidence=0.95,
            timestamp=datetime.now(),
            duration=2.5,
            evidence_path="/evidence/phone_001.jpg",
            metadata={"frame_number": 1500}
        )
        
        logged_event = test_db.log_behavior_event(
            event=event_data,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01",
            frame_number=1500
        )
        
        assert logged_event.student_id == student.id
        assert logged_event.session_id == session.id
        assert logged_event.event_type == BehaviorEventType.PHONE_DETECTED.value
        assert logged_event.confidence == 0.95
        assert logged_event.duration_seconds == 2.5
        assert logged_event.evidence_path == "/evidence/phone_001.jpg"
        assert logged_event.frame_number == 1500
        assert logged_event.camera_id == "cam_01"
    
    def test_log_event_nonexistent_student(self, test_db):
        """Test logging event for non-existent student."""
        event_data = BehaviorEventData(
            event_type=BehaviorEventType.TALKING,
            confidence=0.8,
            timestamp=datetime.now(),
            duration=1.0
        )
        
        with pytest.raises(ValueError, match="Student with roll number .* not found"):
            test_db.log_behavior_event(
                event=event_data,
                student_roll_number="NONEXISTENT",
                session_id="session_001",
                camera_id="cam_01"
            )
    
    def test_get_behavior_events(self, test_db, sample_student_data):
        """Test retrieving behavior events."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        # Log multiple events
        event_types = [BehaviorEventType.PHONE_DETECTED, BehaviorEventType.TALKING, BehaviorEventType.LOOKING_AWAY]
        for i, event_type in enumerate(event_types):
            event_data = BehaviorEventData(
                event_type=event_type,
                confidence=0.8 + i * 0.05,
                timestamp=datetime.now() + timedelta(seconds=i),
                duration=1.0 + i * 0.5
            )
            test_db.log_behavior_event(
                event=event_data,
                student_roll_number=student.roll_number,
                session_id="session_001",
                camera_id="cam_01"
            )
        
        # Get all events
        all_events = test_db.get_behavior_events(student.roll_number)
        assert len(all_events) == 3
        
        # Get events by type
        phone_events = test_db.get_behavior_events(
            student.roll_number,
            event_types=[BehaviorEventType.PHONE_DETECTED]
        )
        assert len(phone_events) == 1
        assert phone_events[0].event_type == BehaviorEventType.PHONE_DETECTED.value
    
    def test_get_events_nonexistent_student(self, test_db):
        """Test getting events for non-existent student."""
        events = test_db.get_behavior_events("NONEXISTENT")
        assert len(events) == 0


class TestAlertManagement:
    """Test alert management operations."""
    
    def test_log_alert(self, test_db, sample_student_data):
        """Test logging an alert."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        alert_data = AlertData(
            alert_id="alert_001",
            roll_number=student.roll_number,
            student_name=student.name,
            alert_level=AlertLevel.CRITICAL,
            alert_type="phone_detection",
            composite_score=95.5,
            contributing_behaviors=["phone_detected", "looking_away"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=datetime.now(),
            email_recipients=["admin@school.edu"]
        )
        
        logged_alert = test_db.log_alert(
            alert=alert_data,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01"
        )
        
        assert logged_alert.alert_id == "alert_001"
        assert logged_alert.student_id == student.id
        assert logged_alert.session_id == session.id
        assert logged_alert.alert_level == AlertLevel.CRITICAL.value
        assert logged_alert.alert_type == "phone_detection"
        assert logged_alert.composite_score == 95.5
        assert logged_alert.contributing_behaviors == ["phone_detected", "looking_away"]
        assert logged_alert.evidence_screenshot == "/evidence/alert_001.jpg"
        assert logged_alert.email_recipients == ["admin@school.edu"]
        assert logged_alert.camera_id == "cam_01"
    
    def test_update_alert_email_status(self, test_db, sample_student_data):
        """Test updating alert email status."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        alert_data = AlertData(
            alert_id="alert_001",
            roll_number=student.roll_number,
            student_name=student.name,
            alert_level=AlertLevel.WARNING,
            alert_type="talking",
            composite_score=65.0,
            contributing_behaviors=["talking"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=datetime.now()
        )
        
        test_db.log_alert(
            alert=alert_data,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01"
        )
        
        # Update email status
        email_time = datetime.now()
        success = test_db.update_alert_email_status(
            alert_id="alert_001",
            email_sent=True,
            email_sent_at=email_time
        )
        
        assert success is True
        
        # Verify update
        with test_db.get_session() as db_session:
            alert = db_session.query(Alert).filter(Alert.alert_id == "alert_001").first()
            assert alert.email_sent is True
            assert alert.email_sent_at == email_time
    
    def test_get_recent_alerts(self, test_db, sample_student_data):
        """Test retrieving recent alerts."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        # Create alerts with different levels and times
        alert_levels = [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.WARNING]
        for i, level in enumerate(alert_levels):
            alert_data = AlertData(
                alert_id=f"alert_{i:03d}",
                roll_number=student.roll_number,
                student_name=student.name,
                alert_level=level,
                alert_type="test",
                composite_score=60.0 + i * 10,
                contributing_behaviors=["test"],
                evidence_screenshot=f"/evidence/alert_{i:03d}.jpg",
                timestamp=datetime.now() - timedelta(hours=i)
            )
            test_db.log_alert(
                alert=alert_data,
                student_roll_number=student.roll_number,
                session_id="session_001",
                camera_id="cam_01"
            )
        
        # Get all recent alerts
        all_alerts = test_db.get_recent_alerts(hours=24)
        assert len(all_alerts) == 3
        
        # Get only critical alerts
        critical_alerts = test_db.get_recent_alerts(hours=24, alert_level=AlertLevel.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].alert_level == AlertLevel.CRITICAL.value


class TestSystemMetrics:
    """Test system metrics operations."""
    
    def test_log_system_metrics(self, test_db):
        """Test logging system metrics."""
        camera_status = {"cam_01": "healthy", "cam_02": "degraded"}
        
        metrics = test_db.log_system_metrics(
            cpu_usage=65.5,
            memory_usage=78.2,
            gpu_usage=45.0,
            fps=18.5,
            processing_latency=0.15,
            active_tracks=12,
            active_students=8,
            alerts_generated=3,
            camera_status=camera_status
        )
        
        assert metrics.cpu_usage == 65.5
        assert metrics.memory_usage == 78.2
        assert metrics.gpu_usage == 45.0
        assert metrics.fps == 18.5
        assert metrics.processing_latency == 0.15
        assert metrics.active_tracks == 12
        assert metrics.active_students == 8
        assert metrics.alerts_generated == 3
        assert metrics.camera_status == camera_status
    
    def test_get_system_metrics_history(self, test_db):
        """Test retrieving system metrics history."""
        # Log metrics at different times
        for i in range(5):
            test_db.log_system_metrics(
                cpu_usage=50.0 + i * 5,
                memory_usage=60.0 + i * 3,
                gpu_usage=None,
                fps=15.0 + i,
                processing_latency=0.1 + i * 0.01,
                active_tracks=10 + i,
                active_students=8 + i,
                alerts_generated=i
            )
        
        history = test_db.get_system_metrics_history(hours=24)
        assert len(history) == 5
        
        # Verify ordering (most recent first)
        assert history[0].cpu_usage == 70.0  # Last logged
        assert history[-1].cpu_usage == 50.0  # First logged


class TestStatistics:
    """Test statistics and analytics operations."""
    
    def test_get_student_statistics(self, test_db, sample_student_data):
        """Test getting comprehensive student statistics."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        # Log some behavior events
        event_types = [BehaviorEventType.PHONE_DETECTED, BehaviorEventType.TALKING, BehaviorEventType.TALKING]
        for event_type in event_types:
            event_data = BehaviorEventData(
                event_type=event_type,
                confidence=0.8,
                timestamp=datetime.now(),
                duration=1.0
            )
            test_db.log_behavior_event(
                event=event_data,
                student_roll_number=student.roll_number,
                session_id="session_001",
                camera_id="cam_01"
            )
        
        # Log an alert
        alert_data = AlertData(
            alert_id="alert_001",
            roll_number=student.roll_number,
            student_name=student.name,
            alert_level=AlertLevel.WARNING,
            alert_type="talking",
            composite_score=65.0,
            contributing_behaviors=["talking"],
            evidence_screenshot="/evidence/alert_001.jpg",
            timestamp=datetime.now()
        )
        test_db.log_alert(
            alert=alert_data,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01"
        )
        
        # End session
        test_db.end_tracking_session(
            session_id="session_001",
            final_suspicion_score=65.0,
            max_suspicion_score=80.0,
            alert_count=1
        )
        
        stats = test_db.get_student_statistics(student.roll_number, days=7)
        
        assert stats["student_info"]["roll_number"] == student.roll_number
        assert stats["student_info"]["name"] == student.name
        assert stats["behavior_events"]["phone_detected"] == 1
        assert stats["behavior_events"]["talking"] == 2
        assert stats["alerts"]["warning"] == 1
        assert stats["sessions"]["count"] == 1
        assert stats["sessions"]["avg_suspicion_score"] == 65.0
        assert stats["sessions"]["max_suspicion_score"] == 80.0
    
    def test_get_class_statistics(self, test_db):
        """Test getting class statistics."""
        # Create multiple students in same class
        students = []
        for i in range(3):
            student = test_db.create_student(
                roll_number=f"CS2021{i:03d}",
                name=f"Student {i}",
                class_id="CS2021",
                email=f"student{i}@school.edu"
            )
            students.append(student)
        
        # Create sessions and events for each student
        for i, student in enumerate(students):
            session = test_db.create_tracking_session(
                session_id=f"session_{i:03d}",
                student_roll_number=student.roll_number,
                camera_id="cam_01"
            )
            
            # Log behavior event
            event_data = BehaviorEventData(
                event_type=BehaviorEventType.LOOKING_AWAY,
                confidence=0.8,
                timestamp=datetime.now(),
                duration=1.0
            )
            test_db.log_behavior_event(
                event=event_data,
                student_roll_number=student.roll_number,
                session_id=f"session_{i:03d}",
                camera_id="cam_01"
            )
            
            # End session
            test_db.end_tracking_session(
                session_id=f"session_{i:03d}",
                final_suspicion_score=50.0 + i * 10,
                max_suspicion_score=60.0 + i * 10,
                alert_count=0
            )
        
        stats = test_db.get_class_statistics("CS2021", days=7)
        
        assert stats["class_id"] == "CS2021"
        assert stats["student_count"] == 3
        assert stats["total_behavior_events"] == 3
        assert stats["total_alerts"] == 0
        assert stats["average_suspicion_score"] == 60.0  # (50 + 60 + 70) / 3


class TestDataCleanup:
    """Test data cleanup operations."""
    
    def test_cleanup_old_data(self, test_db, sample_student_data):
        """Test cleaning up old data."""
        student = test_db.create_student(**sample_student_data)
        session = test_db.create_tracking_session(
            session_id="session_001",
            student_roll_number=student.roll_number,
            camera_id="cam_01"
        )
        
        # Create old behavior event
        old_event = BehaviorEventData(
            event_type=BehaviorEventType.TALKING,
            confidence=0.8,
            timestamp=datetime.now() - timedelta(days=35),  # Older than 30 days
            duration=1.0
        )
        test_db.log_behavior_event(
            event=old_event,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01"
        )
        
        # Create recent behavior event
        recent_event = BehaviorEventData(
            event_type=BehaviorEventType.LOOKING_AWAY,
            confidence=0.8,
            timestamp=datetime.now() - timedelta(days=5),  # Recent
            duration=1.0
        )
        test_db.log_behavior_event(
            event=recent_event,
            student_roll_number=student.roll_number,
            session_id="session_001",
            camera_id="cam_01"
        )
        
        # Verify both events exist
        all_events = test_db.get_behavior_events(student.roll_number)
        assert len(all_events) == 2
        
        # Cleanup old data (keep 30 days)
        test_db.cleanup_old_data(days_to_keep=30)
        
        # Verify only recent event remains
        remaining_events = test_db.get_behavior_events(student.roll_number)
        assert len(remaining_events) == 1
        assert remaining_events[0].event_type == BehaviorEventType.LOOKING_AWAY.value