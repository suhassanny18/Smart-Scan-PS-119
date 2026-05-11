#!/usr/bin/env python3
"""
Debug script to test error handler import
"""

import traceback
import sys

try:
    print("Testing models import...")
    from anti_cheat_system.models.config import LoggingConfig
    from anti_cheat_system.models.enums import SystemState
    print("Models imported successfully!")
    
    print("Testing error_handler module execution...")
    with open('anti_cheat_system/error_handler.py', 'r') as f:
        content = f.read()
        exec(content)
    print("Module executed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()