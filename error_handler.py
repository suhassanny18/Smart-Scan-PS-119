"""
Error handling and logging system for the Anti-Cheat Detection System
"""

import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Represents an error event in the system"""
    timestamp: datetime
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    traceback_info: str = ""
    context: Dict[str, Any] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


class ErrorHandler:
    """Basic error handling and recovery system"""
    
    def __init__(self, config=None):
        self.error_history: List[ErrorEvent] = []
        self.component_error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Setup basic logging
        self.logger = logging.getLogger("anti_cheat_system")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def register_recovery_strategy(self, component: str, strategy: Callable):
        """Register a recovery strategy for a component"""
        self.recovery_strategies[component] = strategy
    
    def handle_error(self, component: str, error: Exception, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Dict[str, Any] = None,
                    attempt_recovery: bool = True) -> bool:
        """Handle an error with logging and recovery"""
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            component=component,
            error_type=type(error).__name__,
            severity=severity,
            message=str(error),
            context=context or {}
        )
        
        # Update counters
        self.error_history.append(error_event)
        self.component_error_counts[component] = self.component_error_counts.get(component, 0) + 1
        
        # Log the error
        error_msg = f"[{component}] {error_event.error_type}: {error_event.message}"
        
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg)
        else:
            self.logger.info(error_msg)
        
        return True


def setup_error_handling(config=None) -> ErrorHandler:
    """Setup error handling for the system"""
    return ErrorHandler(config)
