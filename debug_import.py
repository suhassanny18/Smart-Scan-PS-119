#!/usr/bin/env python3
"""
Debug import issues
"""

import ast
import inspect

# Read the file and parse it
with open('anti_cheat_system/error_handler.py', 'r') as f:
    content = f.read()

# Parse the AST
tree = ast.parse(content)

# Find all class definitions
classes = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        classes.append(node.name)

print(f"Classes found in AST: {classes}")

# Try to import the module and inspect it
try:
    import anti_cheat_system.error_handler as eh
    print(f"Module imported successfully")
    print(f"Module attributes: {[x for x in dir(eh) if not x.startswith('_')]}")
    
    # Check if ErrorEvent is defined but not accessible
    if hasattr(eh, 'ErrorEvent'):
        print("ErrorEvent is accessible via hasattr")
    else:
        print("ErrorEvent is NOT accessible via hasattr")
    
    # Check the module's globals
    print(f"Module globals containing 'Error': {[k for k in eh.__dict__.keys() if 'Error' in k]}")
    
except Exception as e:
    print(f"Import failed: {e}")