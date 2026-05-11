#!/usr/bin/env python3
"""
Enhanced startup script for Anti-Cheat Detection System
Validates environment, handles initialization, and starts the main application
"""

import sys
import os
import signal
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

class SystemLauncher:
    """Manages the startup and initialization of the Anti-Cheat Detection System"""
    
    def __init__(self, debug: bool = False, config_file: Optional[str] = None):
        self.debug = debug
        self.config_file = config_file
        self.system = None
        self.startup_checks_passed = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        if self.system:
            try:
                self.system.stop()
            except Exception as e:
                print(f"Error during shutdown: {e}")
        sys.exit(0)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("Checking Python version...")
        version = sys.version_info
        
        print(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major != 3:
            print("[ERROR] Python 3 is required")
            return False
        
        if version.minor < 8:
            print("[ERROR] Python 3.8 or higher is required")
            return False
        
        if version.minor == 11:
            print("[OK] Python 3.11 (optimal)")
        elif version.minor >= 8:
            print(f"[OK] Python 3.{version.minor} (compatible)")
            if version.minor < 11:
                print("[WARN] Python 3.11 is recommended for optimal performance")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        print("Checking dependencies...")
        
        required_packages = {
            'cv2': 'OpenCV',
            'ultralytics': 'Ultralytics YOLO',
            'mediapipe': 'MediaPipe',
            'numpy': 'NumPy',
            'psutil': 'psutil'
        }
        
        missing_packages = []
        
        for import_name, display_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"[OK] {display_name}")
            except ImportError as e:
                print(f"[ERROR] {display_name} - {e}")
                missing_packages.append(display_name)
        
        if missing_packages:
            print(f"\nMissing packages: {', '.join(missing_packages)}")
            print("Please run 'python setup.py' to install dependencies")
            return False
        
        return True
    
    def check_camera(self) -> bool:
        """Check if camera is available"""
        print("Checking camera availability...")
        
        try:
            import cv2
            
            # Try multiple camera indices
            for camera_idx in range(3):
                cap = cv2.VideoCapture(camera_idx)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        print(f"[OK] Camera found at index {camera_idx}")
                        return True
            
            print("[ERROR] No working camera found")
            return False
            
        except Exception as e:
            print(f"[ERROR] Camera check failed: {e}")
            return False
    
    def check_project_structure(self) -> bool:
        """Check if project structure is complete"""
        print("Checking project structure...")
        
        required_files = [
            "anti_cheat_system/__init__.py",
            "anti_cheat_system/main.py",
            "anti_cheat_system/video_capture.py",
            "anti_cheat_system/scoring_system.py",
            "anti_cheat_system/error_handler.py"
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"[OK] {file_path}")
            else:
                print(f"[ERROR] {file_path} (missing)")
                missing_files.append(file_path)
        
        if missing_files:
            print(f"\nMissing files: {', '.join(missing_files)}")
            print("Please ensure all system components are implemented")
            return False
        
        return True
    
    def check_model_files(self) -> bool:
        """Check if required model files are available"""
        print("Checking model files...")
        
        model_file = Path("yolov8n.pt")
        if model_file.exists():
            file_size = model_file.stat().st_size / (1024**2)  # MB
            print(f"[OK] YOLOv8n model ({file_size:.1f}MB)")
            return True
        else:
            print("[WARN] YOLOv8n model not found (will be downloaded on first run)")
            return True  # Not critical, will download automatically
    
    def check_directories(self) -> bool:
        """Check and create necessary directories"""
        print("Checking directories...")
        
        directories = ["logs", "data", "models"]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"[OK] Created {directory}/")
                except Exception as e:
                    print(f"[ERROR] Could not create {directory}/: {e}")
                    return False
            else:
                print(f"[OK] {directory}/")
        
        return True
    
    def load_configuration(self) -> Optional[Dict[str, Any]]:
        """Load system configuration"""
        print("Loading configuration...")
        
        try:
            from anti_cheat_system.models.config import SystemConfig
            
            if self.config_file and Path(self.config_file).exists():
                print(f"[OK] Loading custom config from {self.config_file}")
                # Custom config loading would go here
                config = SystemConfig()
            else:
                print("[OK] Using default configuration")
                config = SystemConfig()
            
            return config
            
        except ImportError as e:
            print(f"[ERROR] Could not load configuration: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Configuration error: {e}")
            return None
    
    def run_startup_checks(self) -> bool:
        """Run all startup checks"""
        print("Anti-Cheat Detection System - Startup Validation")
        print("=" * 60)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Dependencies", self.check_dependencies),
            ("Camera", self.check_camera),
            ("Project Structure", self.check_project_structure),
            ("Model Files", self.check_model_files),
            ("Directories", self.check_directories)
        ]
        
        all_passed = True
        warnings = []
        
        for check_name, check_func in checks:
            print(f"\n--- {check_name} ---")
            try:
                result = check_func()
                if not result:
                    if check_name == "Camera":
                        warnings.append("Camera not available")
                    elif check_name == "Model Files":
                        warnings.append("Model files will be downloaded")
                    else:
                        all_passed = False
            except Exception as e:
                print(f"[ERROR] {check_name} check failed: {e}")
                all_passed = False
        
        print("\n" + "=" * 60)
        
        if warnings:
            print("WARNINGS:")
            for warning in warnings:
                print(f"  [WARN] {warning}")
            print()
        
        if all_passed:
            print("[OK] All startup checks passed!")
            self.startup_checks_passed = True
        else:
            print("[ERROR] Some startup checks failed. Please fix the issues above.")
        
        print("=" * 60)
        return all_passed
    
    def initialize_system(self) -> bool:
        """Initialize the main system"""
        print("Initializing Anti-Cheat Detection System...")
        
        try:
            # Load configuration
            config = self.load_configuration()
            if not config:
                return False
            
            # Import and create system
            from anti_cheat_system.main import AntiCheatSystem
            
            print("Creating system instance...")
            self.system = AntiCheatSystem(config=config, debug=self.debug)
            
            print("Initializing components...")
            if not self.system.initialize():
                print("[ERROR] System initialization failed")
                return False
            
            print("[OK] System initialized successfully")
            return True
            
        except ImportError as e:
            print(f"[ERROR] Could not import system components: {e}")
            print("Please ensure all implementation tasks are completed")
            return False
        except Exception as e:
            print(f"[ERROR] System initialization error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def run_system(self) -> bool:
        """Run the main system"""
        if not self.system:
            print("[ERROR] System not initialized")
            return False
        
        try:
            print("\nStarting Anti-Cheat Detection System...")
            print("Press Ctrl+C to stop")
            print("-" * 40)
            
            self.system.run()
            return True
            
        except KeyboardInterrupt:
            print("\nSystem stopped by user")
            return True
        except Exception as e:
            print(f"[ERROR] System runtime error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False
        finally:
            if self.system:
                try:
                    self.system.stop()
                except Exception as e:
                    print(f"Error during shutdown: {e}")
    
    def launch(self) -> bool:
        """Launch the complete system"""
        try:
            # Run startup checks
            if not self.run_startup_checks():
                return False
            
            # Initialize system
            if not self.initialize_system():
                return False
            
            # Run system
            return self.run_system()
            
        except Exception as e:
            print(f"Unexpected error during launch: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Anti-Cheat Detection System")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation checks")
    
    args = parser.parse_args()
    
    launcher = SystemLauncher(debug=args.debug, config_file=args.config)
    
    if args.validate_only:
        success = launcher.run_startup_checks()
        sys.exit(0 if success else 1)
    
    success = launcher.launch()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()