#!/usr/bin/env python3
"""
Test import of error handler components
"""

# Test individual imports
try:
    import logging
    print("✓ logging imported")
except ImportError as e:
    print(f"✗ logging failed: {e}")

try:
    import logging.handlers
    print("✓ logging.handlers imported")
except ImportError as e:
    print(f"✗ logging.handlers failed: {e}")

try:
    import psutil
    print("✓ psutil imported")
except ImportError as e:
    print(f"✗ psutil failed: {e}")

try:
    from anti_cheat_system.models.config import LoggingConfig
    print("✓ LoggingConfig imported")
except ImportError as e:
    print(f"✗ LoggingConfig failed: {e}")

try:
    from anti_cheat_system.models.enums import SystemState
    print("✓ SystemState imported")
except ImportError as e:
    print(f"✗ SystemState failed: {e}")

# Test if we can create the classes manually
try:
    from dataclasses import dataclass
    from enum import Enum
    from datetime import datetime
    from typing import Dict, List, Optional, Any, Callable
    
    class ErrorSeverity(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    @dataclass
    class ErrorEvent:
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
    
    print("✓ ErrorEvent created successfully")
    
    # Test creating an instance
    event = ErrorEvent(
        timestamp=datetime.now(),
        component="test",
        error_type="TestError",
        severity=ErrorSeverity.LOW,
        message="Test message"
    )
    print(f"✓ ErrorEvent instance created: {event}")
    
except Exception as e:
    print(f"✗ ErrorEvent creation failed: {e}")
    import traceback
    traceback.print_exc()