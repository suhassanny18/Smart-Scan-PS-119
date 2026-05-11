#!/usr/bin/env python3
"""
Configuration validation script for Anti-Cheat Detection System
Validates system requirements, dependencies, and hardware compatibility
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class SystemValidator:
    """Validates system requirements and configuration"""
    
    def __init__(self):
        self.results = {
            'python_version': False,
            'dependencies': False,
            'camera': False,
            'gpu': False,
            'disk_space': False,
            'memory': False,
            'model_files': False
        }
        self.warnings = []
        self.errors = []
    
    def validate_python_version(self) -> bool:
        """Validate Python version"""
        print("Checking Python version...")
        version = sys.version_info
        
        if version.major == 3 and version.minor == 11:
            print(f"[OK] Python {version.major}.{version.minor}.{version.micro} (recommended)")
            self.results['python_version'] = True
            return True
        elif version.major == 3 and version.minor >= 8:
            print(f"[WARN] Python {version.major}.{version.minor}.{version.micro} (compatible but not optimal)")
            self.warnings.append(f"Python 3.11 is recommended, you have {version.major}.{version.minor}")
            self.results['python_version'] = True
            return True
        else:
            print(f"[ERROR] Python {version.major}.{version.minor}.{version.micro} (incompatible)")
            self.errors.append(f"Python 3.8+ required, you have {version.major}.{version.minor}")
            return False
    
    def validate_dependencies(self) -> bool:
        """Validate required dependencies"""
        print("Checking dependencies...")
        
        required_packages = {
            'cv2': 'opencv-python',
            'ultralytics': 'ultralytics',
            'mediapipe': 'mediapipe',
            'numpy': 'numpy',
            'psutil': 'psutil'
        }
        
        missing_packages = []
        
        for import_name, package_name in required_packages.items():
            try:
                __import__(import_name)
                print(f"[OK] {package_name}")
            except ImportError:
                print(f"[ERROR] {package_name} (missing)")
                missing_packages.append(package_name)
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        self.results['dependencies'] = True
        return True
    
    def validate_camera(self) -> bool:
        """Validate camera availability"""
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
                        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                        self.results['camera'] = True
                        return True
            
            print("[ERROR] No working camera found")
            self.errors.append("No camera detected. System requires a working camera.")
            return False
            
        except Exception as e:
            print(f"[ERROR] Camera check failed: {e}")
            self.errors.append(f"Camera validation error: {e}")
            return False
    
    def validate_gpu_support(self) -> bool:
        """Validate GPU support (optional)"""
        print("Checking GPU support...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[OK] CUDA GPU available: {gpu_name} ({gpu_count} device(s))")
                self.results['gpu'] = True
                return True
            else:
                print("[WARN] No CUDA GPU detected (CPU mode will be used)")
                self.warnings.append("GPU acceleration not available, performance may be reduced")
                return True
                
        except ImportError:
            print("[WARN] PyTorch not available for GPU check")
            self.warnings.append("Cannot check GPU support without PyTorch")
            return True
        except Exception as e:
            print(f"[WARN] GPU check failed: {e}")
            self.warnings.append(f"GPU validation error: {e}")
            return True
    
    def validate_disk_space(self) -> bool:
        """Validate available disk space"""
        print("Checking disk space...")
        
        try:
            import psutil
            
            # Check current directory disk usage
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            required_gb = 2.0  # Minimum 2GB free space
            
            if free_gb >= required_gb:
                print(f"[OK] Disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB total")
                self.results['disk_space'] = True
                return True
            else:
                print(f"[ERROR] Insufficient disk space: {free_gb:.1f}GB free (minimum {required_gb}GB required)")
                self.errors.append(f"Need at least {required_gb}GB free disk space")
                return False
                
        except Exception as e:
            print(f"[WARN] Disk space check failed: {e}")
            self.warnings.append(f"Could not check disk space: {e}")
            return True
    
    def validate_memory(self) -> bool:
        """Validate available memory"""
        print("Checking system memory...")
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            required_gb = 4.0  # Minimum 4GB total memory
            
            if total_gb >= required_gb:
                print(f"[OK] System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
                self.results['memory'] = True
                return True
            else:
                print(f"[WARN] Low system memory: {total_gb:.1f}GB total (recommended {required_gb}GB+)")
                self.warnings.append(f"System has {total_gb:.1f}GB RAM, {required_gb}GB+ recommended")
                self.results['memory'] = True  # Don't fail for this
                return True
                
        except Exception as e:
            print(f"[WARN] Memory check failed: {e}")
            self.warnings.append(f"Could not check system memory: {e}")
            return True
    
    def validate_model_files(self) -> bool:
        """Validate model files"""
        print("Checking model files...")
        
        model_files = ['yolov8n.pt']
        missing_files = []
        
        for model_file in model_files:
            if Path(model_file).exists():
                file_size = Path(model_file).stat().st_size / (1024**2)  # MB
                print(f"[OK] {model_file} ({file_size:.1f}MB)")
            else:
                print(f"[WARN] {model_file} (will be downloaded on first run)")
                missing_files.append(model_file)
        
        if missing_files:
            self.warnings.append(f"Model files will be downloaded: {', '.join(missing_files)}")
        
        self.results['model_files'] = True
        return True
    
    def validate_project_structure(self) -> bool:
        """Validate project directory structure"""
        print("Checking project structure...")
        
        required_dirs = [
            'anti_cheat_system',
            'anti_cheat_system/models',
            'anti_cheat_system/detectors',
            'tests',
            'logs'
        ]
        
        required_files = [
            'anti_cheat_system/__init__.py',
            'anti_cheat_system/main.py',
            'anti_cheat_system/video_capture.py',
            'anti_cheat_system/scoring_system.py',
            'requirements.txt'
        ]
        
        missing_items = []
        
        # Check directories
        for dir_path in required_dirs:
            if Path(dir_path).is_dir():
                print(f"[OK] {dir_path}/")
            else:
                print(f"[ERROR] {dir_path}/ (missing)")
                missing_items.append(f"directory: {dir_path}")
        
        # Check files
        for file_path in required_files:
            if Path(file_path).is_file():
                print(f"[OK] {file_path}")
            else:
                print(f"[ERROR] {file_path} (missing)")
                missing_items.append(f"file: {file_path}")
        
        if missing_items:
            self.errors.append(f"Missing project components: {', '.join(missing_items)}")
            return False
        
        return True
    
    def run_validation(self) -> bool:
        """Run complete validation"""
        print("Anti-Cheat Detection System - Configuration Validation")
        print("=" * 60)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print("=" * 60)
        
        validation_steps = [
            ("Python Version", self.validate_python_version),
            ("Dependencies", self.validate_dependencies),
            ("Camera", self.validate_camera),
            ("GPU Support", self.validate_gpu_support),
            ("Disk Space", self.validate_disk_space),
            ("System Memory", self.validate_memory),
            ("Model Files", self.validate_model_files),
            ("Project Structure", self.validate_project_structure)
        ]
        
        all_passed = True
        
        for step_name, validation_func in validation_steps:
            print(f"\n--- {step_name} ---")
            try:
                if not validation_func():
                    all_passed = False
            except Exception as e:
                print(f"[ERROR] Validation failed: {e}")
                self.errors.append(f"{step_name} validation error: {e}")
                all_passed = False
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  [WARN] {warning}")
            print()
        
        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  [ERROR] {error}")
            print()
        
        if all_passed and not self.errors:
            print("[OK] All validations passed! System is ready to run.")
            if self.warnings:
                print("  Note: Some warnings were found but won't prevent operation.")
        else:
            print("[ERROR] Validation failed. Please fix the errors above before running the system.")
        
        print("=" * 60)
        return all_passed and not self.errors


def main():
    """Main validation function"""
    validator = SystemValidator()
    success = validator.run_validation()
    
    if not success:
        print("\nTo fix common issues:")
        print("1. Run setup.py to install dependencies")
        print("2. Ensure a camera is connected and working")
        print("3. Check that you have sufficient disk space and memory")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()