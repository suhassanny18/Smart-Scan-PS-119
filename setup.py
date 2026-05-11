#!/usr/bin/env python3
"""
Enhanced setup script for Anti-Cheat Detection System
Creates virtual environment, installs dependencies, and validates installation
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path
from typing import Optional, List

class SetupManager:
    """Manages the setup process for the Anti-Cheat Detection System"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.is_windows = self.platform == 'windows'
        self.venv_path = Path("venv")
        self.python_exe = self._get_python_executable()
        
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable path"""
        if self.venv_path.exists():
            if self.is_windows:
                return str(self.venv_path / "Scripts" / "python.exe")
            else:
                return str(self.venv_path / "bin" / "python")
        return sys.executable
    
    def run_command(self, command: str, check: bool = True, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result"""
        print(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if check and result.returncode != 0:
                print(f"Error: {result.stderr}")
                if not capture_output:
                    print("Command failed. Check output above.")
                raise subprocess.CalledProcessError(result.returncode, command)
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            raise
        except Exception as e:
            print(f"Command failed: {e}")
            raise
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        print("Checking Python version...")
        version = sys.version_info
        
        print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version.major != 3:
            print("✗ Python 3 is required")
            return False
        
        if version.minor < 8:
            print("✗ Python 3.8 or higher is required")
            return False
        
        if version.minor == 11:
            print("✓ Python 3.11 (optimal)")
        elif version.minor >= 8:
            print(f"✓ Python 3.{version.minor} (compatible)")
        
        return True
    
    def find_python_command(self) -> Optional[str]:
        """Find the best Python command to use"""
        python_commands = []
        
        if self.is_windows:
            python_commands = ["py -3.11", "py -3", "python3.11", "python3", "python"]
        else:
            python_commands = ["python3.11", "python3", "python"]
        
        for cmd in python_commands:
            try:
                result = subprocess.run(
                    f"{cmd} --version", 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version_str = result.stdout.strip()
                    print(f"Found: {cmd} -> {version_str}")
                    return cmd
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                continue
        
        return None
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment"""
        if self.venv_path.exists():
            print("Virtual environment already exists")
            return True
        
        print("Creating virtual environment...")
        
        # Find Python command
        python_cmd = self.find_python_command()
        if not python_cmd:
            print("✗ Could not find suitable Python installation")
            return False
        
        try:
            self.run_command(f"{python_cmd} -m venv venv")
            print("✓ Virtual environment created successfully")
            
            # Update python executable path
            self.python_exe = self._get_python_executable()
            return True
            
        except Exception as e:
            print(f"✗ Failed to create virtual environment: {e}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install required dependencies"""
        print("Installing dependencies...")
        
        if not Path("requirements.txt").exists():
            print("✗ requirements.txt not found")
            return False
        
        try:
            # Upgrade pip first
            print("Upgrading pip...")
            self.run_command(f'"{self.python_exe}" -m pip install --upgrade pip')
            
            # Install requirements
            print("Installing packages from requirements.txt...")
            self.run_command(f'"{self.python_exe}" -m pip install -r requirements.txt')
            
            print("✓ Dependencies installed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to install dependencies: {e}")
            return False
    
    def verify_installation(self) -> bool:
        """Verify that all required packages are installed"""
        print("Verifying installation...")
        
        test_imports = [
            ("cv2", "OpenCV"),
            ("ultralytics", "Ultralytics YOLO"),
            ("mediapipe", "MediaPipe"),
            ("numpy", "NumPy"),
            ("psutil", "psutil")
        ]
        
        failed_imports = []
        
        for import_name, display_name in test_imports:
            try:
                result = self.run_command(
                    f'"{self.python_exe}" -c "import {import_name}; print(\'✓ {display_name}\')"',
                    capture_output=False
                )
                if result.returncode != 0:
                    failed_imports.append(display_name)
            except Exception:
                failed_imports.append(display_name)
        
        if failed_imports:
            print(f"✗ Failed to import: {', '.join(failed_imports)}")
            return False
        
        print("✓ All packages verified successfully")
        return True
    
    def create_project_directories(self) -> bool:
        """Create necessary project directories"""
        print("Creating project directories...")
        
        directories = [
            "logs",
            "data",
            "models"
        ]
        
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created {directory}/")
            else:
                print(f"✓ {directory}/ already exists")
        
        return True
    
    def download_models(self) -> bool:
        """Download required model files"""
        print("Checking model files...")
        
        model_file = Path("yolov8n.pt")
        if model_file.exists():
            print("✓ YOLOv8n model already exists")
            return True
        
        print("YOLOv8n model will be downloaded on first run")
        return True
    
    def create_activation_scripts(self) -> bool:
        """Create convenient activation scripts"""
        print("Creating activation scripts...")
        
        try:
            if self.is_windows:
                # Windows batch file
                activate_script = Path("activate.bat")
                with open(activate_script, 'w') as f:
                    f.write("@echo off\n")
                    f.write("call venv\\Scripts\\activate.bat\n")
                    f.write("echo Anti-Cheat Detection System environment activated\n")
                    f.write("echo Run 'python run_system.py' to start the system\n")
                print("✓ Created activate.bat")
                
                # Windows PowerShell script
                ps_script = Path("activate.ps1")
                with open(ps_script, 'w') as f:
                    f.write("& .\\venv\\Scripts\\Activate.ps1\n")
                    f.write("Write-Host 'Anti-Cheat Detection System environment activated'\n")
                    f.write("Write-Host 'Run python run_system.py to start the system'\n")
                print("✓ Created activate.ps1")
                
            else:
                # Unix shell script
                activate_script = Path("activate.sh")
                with open(activate_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("source venv/bin/activate\n")
                    f.write("echo 'Anti-Cheat Detection System environment activated'\n")
                    f.write("echo 'Run python run_system.py to start the system'\n")
                
                # Make executable
                os.chmod(activate_script, 0o755)
                print("✓ Created activate.sh")
            
            return True
            
        except Exception as e:
            print(f"⚠ Could not create activation scripts: {e}")
            return True  # Not critical
    
    def run_setup(self) -> bool:
        """Run the complete setup process"""
        print("Anti-Cheat Detection System - Setup")
        print("=" * 50)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Architecture: {platform.machine()}")
        print("=" * 50)
        
        setup_steps = [
            ("Python Version Check", self.check_python_version),
            ("Virtual Environment", self.create_virtual_environment),
            ("Dependencies Installation", self.install_dependencies),
            ("Installation Verification", self.verify_installation),
            ("Project Directories", self.create_project_directories),
            ("Model Files", self.download_models),
            ("Activation Scripts", self.create_activation_scripts)
        ]
        
        for step_name, step_func in setup_steps:
            print(f"\n--- {step_name} ---")
            try:
                if not step_func():
                    print(f"✗ {step_name} failed")
                    return False
            except KeyboardInterrupt:
                print("\nSetup interrupted by user")
                return False
            except Exception as e:
                print(f"✗ {step_name} failed with error: {e}")
                return False
        
        # Print success message
        print("\n" + "=" * 50)
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Next steps:")
        print("1. Activate the virtual environment:")
        
        if self.is_windows:
            print("   - Command Prompt: activate.bat")
            print("   - PowerShell: .\\activate.ps1")
            print("   - Manual: .\\venv\\Scripts\\activate")
        else:
            print("   - ./activate.sh")
            print("   - Manual: source venv/bin/activate")
        
        print("2. Validate configuration: python validate_config.py")
        print("3. Start the system: python run_system.py")
        print("=" * 50)
        
        return True


def main():
    """Main setup function"""
    try:
        setup_manager = SetupManager()
        success = setup_manager.run_setup()
        
        if not success:
            print("\nSetup failed. Please check the errors above and try again.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()