"""
Database service layer for the Anti-Cheat Detection System.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import create_engine, and_, or_, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import numpy as np
import pickle

from .models import (
    Base, Student, TrackingSession, BehaviorEvent, Alert, 
    EmailLog, SystemMetrics, Configuration, AuditLog
)
from ..models.data_models import BehaviorEvent as BehaviorEventData, Alert as AlertData
from ..models.enums import AttendanceStatus, AlertLevel, BehaviorEventType

logger = logging.getLogger(__name__)


class DatabaseService:
    """Database service for managing all database operations."""
    
    def __init__(self, database_url: str):
        """Initialize database service."""
        self.database_url = database_url
        
        # Configure engine parameters based on database type
        if database_url.startswith("sqlite"):
            # SQLite doesn't support pool_size, max_overflow, etc.
            self.engine = create_engine(
                database_url,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        else:
            # PostgreSQL and other databases
            self.engine = create_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database service initialized")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    # Student Management
    def create_student(self, roll_number: str, name: str, class_id: str, 
                      email: Optional[str] = None, face_embedding: Optional[np.ndarray] = None) -> Student:
        """Create a new student record."""
        with self.get_session() as session:
            # Check if student already exists
            existing = session.query(Student).filter(Student.roll_number == roll_number).first()
            if existing:
                raise ValueError(f"Student with roll number {roll_number} already exists")
            
            # Serialize face embedding if provided
            embedding_data = None
            if face_embedding is not None:
                embedding_data = pickle.dumps(face_embedding)
            
            student = Student(
                roll_number=roll_number,
                name=name,
                class_id=class_id,
                email=email,
                face_embedding=embedding_data
            )
            
            session.add(student)
            session.flush()
            
            self._log_audit_action(session, "CREATE_STUDENT", "Student", str(student.id), 
                                 new_values={"roll_number": roll_number, "name": name, "class_id": class_id})
            
            # Refresh to get all attributes loaded
            session.refresh(student)
            
            # Detach from session so it can be used outside
            session.expunge(student)
            
            return student
    
    def get_student_by_roll_number(self, roll_number: str) -> Optional[Student]:
        """Get student by roll number."""
        with self.get_session() as session:
            return session.query(Student).filter(Student.roll_number == roll_number).first()
    
    def get_students_by_class(self, class_id: str) -> List[Student]:
        """Get all students in a class."""
        with self.get_session() as session:
            return session.query(Student).filter(Student.class_id == class_id).all()
    
    def update_student_embedding(self, roll_number: str, face_embedding: np.ndarray) -> bool:
        """Update student's face embedding."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == roll_number).first()
            if not student:
                return False
            
            old_embedding = student.face_embedding
            student.face_embedding = pickle.dumps(face_embedding)
            student.updated_at = datetime.now()
            
            self._log_audit_action(session, "UPDATE_EMBEDDING", "Student", str(student.id),
                                 old_values={"has_embedding": old_embedding is not None},
                                 new_values={"has_embedding": True})
            
            return True
    
    def get_student_embedding(self, roll_number: str) -> Optional[np.ndarray]:
        """Get student's face embedding."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == roll_number).first()
            if student and student.face_embedding:
                return pickle.loads(student.face_embedding)
            return None
    
    # Tracking Session Management
    def create_tracking_session(self, session_id: str, student_roll_number: str, camera_id: str) -> TrackingSession:
        """Create a new tracking session."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == student_roll_number).first()
            if not student:
                raise ValueError(f"Student with roll number {student_roll_number} not found")
            
            tracking_session = TrackingSession(
                session_id=session_id,
                student_id=student.id,
                camera_id=camera_id,
                start_time=datetime.now()
            )
            
            session.add(tracking_session)
            session.flush()
            
            self._log_audit_action(session, "CREATE_SESSION", "TrackingSession", str(tracking_session.id),
                                 new_values={"session_id": session_id, "student_roll": student_roll_number})
            
            return tracking_session
    
    def end_tracking_session(self, session_id: str, final_suspicion_score: float, 
                           max_suspicion_score: float, alert_count: int) -> bool:
        """End a tracking session."""
        with self.get_session() as session:
            tracking_session = session.query(TrackingSession).filter(
                TrackingSession.session_id == session_id
            ).first()
            
            if not tracking_session:
                return False
            
            end_time = datetime.now()
            duration = (end_time - tracking_session.start_time).total_seconds()
            
            tracking_session.end_time = end_time
            tracking_session.duration_seconds = int(duration)
            tracking_session.final_suspicion_score = final_suspicion_score
            tracking_session.max_suspicion_score = max_suspicion_score
            tracking_session.alert_count = alert_count
            
            self._log_audit_action(session, "END_SESSION", "TrackingSession", str(tracking_session.id),
                                 new_values={"duration": duration, "final_score": final_suspicion_score})
            
            return True
    
    # Behavior Event Logging
    def log_behavior_event(self, event: BehaviorEventData, student_roll_number: str, 
                          session_id: str, camera_id: str, frame_number: Optional[int] = None) -> BehaviorEvent:
        """Log a behavior event."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == student_roll_number).first()
            if not student:
                raise ValueError(f"Student with roll number {student_roll_number} not found")
            
            tracking_session = session.query(TrackingSession).filter(
                TrackingSession.session_id == session_id
            ).first()
            if not tracking_session:
                raise ValueError(f"Tracking session {session_id} not found")
            
            behavior_event = BehaviorEvent(
                event_id=f"{session_id}_{event.event_type.value}_{int(event.timestamp.timestamp())}",
                student_id=student.id,
                session_id=tracking_session.id,
                event_type=event.event_type.value,
                confidence=event.confidence,
                duration_seconds=event.duration,
                timestamp=event.timestamp,
                evidence_path=event.evidence_path,
                event_metadata=event.metadata,
                frame_number=frame_number,
                camera_id=camera_id
            )
            
            session.add(behavior_event)
            session.flush()
            
            return behavior_event
    
    def get_behavior_events(self, student_roll_number: str, 
                           time_range: Optional[Tuple[datetime, datetime]] = None,
                           event_types: Optional[List[BehaviorEventType]] = None) -> List[BehaviorEvent]:
        """Get behavior events for a student."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == student_roll_number).first()
            if not student:
                return []
            
            query = session.query(BehaviorEvent).filter(BehaviorEvent.student_id == student.id)
            
            if time_range:
                start_time, end_time = time_range
                query = query.filter(and_(
                    BehaviorEvent.timestamp >= start_time,
                    BehaviorEvent.timestamp <= end_time
                ))
            
            if event_types:
                event_type_values = [et.value for et in event_types]
                query = query.filter(BehaviorEvent.event_type.in_(event_type_values))
            
            return query.order_by(desc(BehaviorEvent.timestamp)).all()
    
    # Alert Management
    def log_alert(self, alert: AlertData, student_roll_number: str, session_id: str, camera_id: str) -> Alert:
        """Log an alert."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == student_roll_number).first()
            if not student:
                raise ValueError(f"Student with roll number {student_roll_number} not found")
            
            tracking_session = session.query(TrackingSession).filter(
                TrackingSession.session_id == session_id
            ).first()
            if not tracking_session:
                raise ValueError(f"Tracking session {session_id} not found")
            
            alert_record = Alert(
                alert_id=alert.alert_id,
                student_id=student.id,
                session_id=tracking_session.id,
                alert_level=alert.alert_level.value,
                alert_type=alert.alert_type,
                composite_score=alert.composite_score,
                contributing_behaviors=alert.contributing_behaviors,
                evidence_screenshot=alert.evidence_screenshot,
                timestamp=alert.timestamp,
                email_recipients=alert.email_recipients,
                camera_id=camera_id
            )
            
            session.add(alert_record)
            session.flush()
            
            self._log_audit_action(session, "CREATE_ALERT", "Alert", str(alert_record.id),
                                 new_values={"alert_level": alert.alert_level.value, 
                                           "composite_score": alert.composite_score})
            
            return alert_record
    
    def update_alert_email_status(self, alert_id: str, email_sent: bool, 
                                 email_sent_at: Optional[datetime] = None) -> bool:
        """Update alert email status."""
        with self.get_session() as session:
            alert = session.query(Alert).filter(Alert.alert_id == alert_id).first()
            if not alert:
                return False
            
            alert.email_sent = email_sent
            alert.email_sent_at = email_sent_at or datetime.now()
            
            return True
    
    def get_recent_alerts(self, hours: int = 24, alert_level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get recent alerts."""
        with self.get_session() as session:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            query = session.query(Alert).filter(Alert.timestamp >= cutoff_time)
            
            if alert_level:
                query = query.filter(Alert.alert_level == alert_level.value)
            
            return query.order_by(desc(Alert.timestamp)).all()
    
    # Email Logging
    def log_email_attempt(self, alert_id: int, recipient: str, subject: str, 
                         success: bool, error_message: Optional[str] = None,
                         retry_count: int = 0) -> EmailLog:
        """Log email sending attempt."""
        with self.get_session() as session:
            email_log = EmailLog(
                alert_id=alert_id,
                recipient=recipient,
                subject=subject,
                success=success,
                error_message=error_message,
                retry_count=retry_count
            )
            
            session.add(email_log)
            session.flush()
            
            return email_log
    
    # System Metrics
    def log_system_metrics(self, cpu_usage: float, memory_usage: float, 
                          gpu_usage: Optional[float], fps: float, processing_latency: float,
                          active_tracks: int, active_students: int, alerts_generated: int,
                          camera_status: Optional[Dict[str, Any]] = None) -> SystemMetrics:
        """Log system performance metrics."""
        with self.get_session() as session:
            metrics = SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                fps=fps,
                processing_latency=processing_latency,
                active_tracks=active_tracks,
                active_students=active_students,
                alerts_generated=alerts_generated,
                camera_status=camera_status
            )
            
            session.add(metrics)
            session.flush()
            
            return metrics
    
    def get_system_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get system metrics history."""
        with self.get_session() as session:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return session.query(SystemMetrics).filter(
                SystemMetrics.timestamp >= cutoff_time
            ).order_by(desc(SystemMetrics.timestamp)).all()
    
    # Configuration Management
    def save_configuration(self, config_key: str, config_value: Any, created_by: str = "system") -> Configuration:
        """Save configuration setting."""
        with self.get_session() as session:
            # Deactivate old configuration
            session.query(Configuration).filter(
                and_(Configuration.config_key == config_key, Configuration.is_active == True)
            ).update({"is_active": False})
            
            # Create new configuration
            config = Configuration(
                config_key=config_key,
                config_value=config_value,
                created_by=created_by
            )
            
            session.add(config)
            session.flush()
            
            self._log_audit_action(session, "UPDATE_CONFIG", "Configuration", str(config.id),
                                 new_values={"config_key": config_key})
            
            return config
    
    def get_configuration(self, config_key: str) -> Optional[Any]:
        """Get active configuration value."""
        with self.get_session() as session:
            config = session.query(Configuration).filter(
                and_(Configuration.config_key == config_key, Configuration.is_active == True)
            ).first()
            
            return config.config_value if config else None
    
    # Statistics and Analytics
    def get_student_statistics(self, student_roll_number: str, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive statistics for a student."""
        with self.get_session() as session:
            student = session.query(Student).filter(Student.roll_number == student_roll_number).first()
            if not student:
                return {}
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Get behavior event counts
            behavior_counts = session.query(
                BehaviorEvent.event_type,
                func.count(BehaviorEvent.id).label('count')
            ).filter(
                and_(BehaviorEvent.student_id == student.id, BehaviorEvent.timestamp >= cutoff_time)
            ).group_by(BehaviorEvent.event_type).all()
            
            # Get alert counts
            alert_counts = session.query(
                Alert.alert_level,
                func.count(Alert.id).label('count')
            ).filter(
                and_(Alert.student_id == student.id, Alert.timestamp >= cutoff_time)
            ).group_by(Alert.alert_level).all()
            
            # Get session statistics
            session_stats = session.query(
                func.count(TrackingSession.id).label('session_count'),
                func.avg(TrackingSession.final_suspicion_score).label('avg_score'),
                func.max(TrackingSession.max_suspicion_score).label('max_score'),
                func.sum(TrackingSession.duration_seconds).label('total_duration')
            ).filter(
                and_(TrackingSession.student_id == student.id, TrackingSession.start_time >= cutoff_time)
            ).first()
            
            return {
                "student_info": {
                    "roll_number": student.roll_number,
                    "name": student.name,
                    "class_id": student.class_id
                },
                "behavior_events": {row.event_type: row.count for row in behavior_counts},
                "alerts": {row.alert_level: row.count for row in alert_counts},
                "sessions": {
                    "count": session_stats.session_count or 0,
                    "avg_suspicion_score": float(session_stats.avg_score or 0),
                    "max_suspicion_score": float(session_stats.max_score or 0),
                    "total_duration_hours": (session_stats.total_duration or 0) / 3600
                },
                "period_days": days
            }
    
    def get_class_statistics(self, class_id: str, days: int = 7) -> Dict[str, Any]:
        """Get statistics for an entire class."""
        with self.get_session() as session:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Get students in class
            students = session.query(Student).filter(Student.class_id == class_id).all()
            student_ids = [s.id for s in students]
            
            if not student_ids:
                return {"class_id": class_id, "student_count": 0}
            
            # Get aggregate statistics
            total_alerts = session.query(func.count(Alert.id)).filter(
                and_(Alert.student_id.in_(student_ids), Alert.timestamp >= cutoff_time)
            ).scalar()
            
            total_events = session.query(func.count(BehaviorEvent.id)).filter(
                and_(BehaviorEvent.student_id.in_(student_ids), BehaviorEvent.timestamp >= cutoff_time)
            ).scalar()
            
            avg_suspicion = session.query(func.avg(TrackingSession.final_suspicion_score)).filter(
                and_(TrackingSession.student_id.in_(student_ids), TrackingSession.start_time >= cutoff_time)
            ).scalar()
            
            return {
                "class_id": class_id,
                "student_count": len(students),
                "total_alerts": total_alerts or 0,
                "total_behavior_events": total_events or 0,
                "average_suspicion_score": float(avg_suspicion or 0),
                "period_days": days
            }
    
    # Utility Methods
    def _log_audit_action(self, session: Session, action: str, entity_type: str, 
                         entity_id: str, old_values: Optional[Dict] = None,
                         new_values: Optional[Dict] = None, user_id: str = "system"):
        """Log audit action."""
        audit_log = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            old_values=old_values,
            new_values=new_values
        )
        session.add(audit_log)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data beyond retention period."""
        with self.get_session() as session:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old behavior events
            deleted_events = session.query(BehaviorEvent).filter(
                BehaviorEvent.timestamp < cutoff_time
            ).delete()
            
            # Clean up old system metrics
            deleted_metrics = session.query(SystemMetrics).filter(
                SystemMetrics.timestamp < cutoff_time
            ).delete()
            
            # Clean up old email logs
            deleted_emails = session.query(EmailLog).filter(
                EmailLog.sent_at < cutoff_time
            ).delete()
            
            logger.info(f"Cleaned up {deleted_events} behavior events, "
                       f"{deleted_metrics} metrics, {deleted_emails} email logs")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            with self.get_session() as session:
                # Test basic connectivity
                session.execute("SELECT 1")
                
                # Get table counts
                student_count = session.query(func.count(Student.id)).scalar()
                session_count = session.query(func.count(TrackingSession.id)).scalar()
                alert_count = session.query(func.count(Alert.id)).scalar()
                
                return {
                    "status": "healthy",
                    "database_url": self.database_url.split('@')[0] + "@***",  # Hide credentials
                    "tables": {
                        "students": student_count,
                        "tracking_sessions": session_count,
                        "alerts": alert_count
                    },
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }