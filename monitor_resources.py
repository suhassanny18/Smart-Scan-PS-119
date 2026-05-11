#!/usr/bin/env python3
"""
Resource monitoring script for Anti-Cheat Detection System
"""

import time
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

try:
    from anti_cheat_system.error_handler import ResourceMonitor, setup_error_handling
    
    def main():
        error_handler = setup_error_handling()
        monitor = ResourceMonitor(error_handler)
        
        print('Starting resource monitoring...')
        while True:
            stats = monitor.check_resources()
            print(f'CPU: {stats["cpu_percent"]:.1f}% | Memory: {stats["memory_percent"]:.1f}% | FPS: {stats["fps"]:.1f}')
            time.sleep(5)
    
    if __name__ == '__main__':
        main()
except ImportError:
    print('Resource monitoring not available - system components not found')
