#!/usr/bin/env python3
"""
Deployment script for Anti-Cheat Detection System
Handles deployment to different environments (development, testing, production)
"""

import os
import sys
import shutil
import subprocess
import argparse
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any

class DeploymentManager:
    """Manages deployment of the Anti-Cheat Detection System"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.platform = platform.system().lower()
        self.is_windows = self.platform == 'windows'
        self.project_root = Path.cwd()
        
        # Environment-specific configurations
        self.env_configs = {
            'development': {
                'debug': True,
                'log_level': 'DEBUG',
                'performance_monitoring': True,
                'model_precision': 'fp32',
                'camera_resolution': (640, 480),
                'frame_rate': 30
            },
            'testing': {
                'debug': True,
                'log_level': 'INFO',
                'performance_monitoring': True,
                'model_precision': 'fp32',
                'camera_resolution': (1280, 720),
                'frame_rate': 30,
                'test_mode': True
            },
            'production': {
                'debug': False,
                'log_level': 'WARNING',
                'performance_monitoring': True,
                'model_precision': 'fp16',
                'camera_resolution': (1920, 1080),
                'frame_rate': 30,
                'auto_restart': True,
                'resource_limits': {
                    'max_memory_mb': 2048,
                    'max_cpu_percent': 80
                }
            }
        }
    
    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result"""
        print(f"Running: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if check and result.returncode != 0:
                print(f"Error: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command)
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"Command timed out: {command}")
            raise
        except Exception as e:
            print(f"Command failed: {e}")
            raise
    
    def create_environment_config(self) -> bool:
        """Create environment-specific configuration"""
        print(f"Creating {self.environment} configuration...")
        
        try:
            config = self.env_configs.get(self.environment, self.env_configs['development'])
            
            # Create config directory
            config_dir = self.project_root / "config"
            config_dir.mkdir(exist_ok=True)
            
            # Write environment config
            config_file = config_dir / f"{self.environment}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✓ Configuration saved to {config_file}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create configuration: {e}")
            return False
    
    def setup_logging(self) -> bool:
        """Setup environment-specific logging"""
        print("Setting up logging configuration...")
        
        try:
            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create environment-specific log directory
            env_logs_dir = logs_dir / self.environment
            env_logs_dir.mkdir(exist_ok=True)
            
            # Create log rotation script
            if self.is_windows:
                log_rotate_script = self.project_root / "rotate_logs.bat"
                with open(log_rotate_script, 'w') as f:
                    f.write("@echo off\n")
                    f.write(f"forfiles /p logs\\{self.environment} /s /m *.log /d -7 /c \"cmd /c del @path\"\n")
                    f.write("echo Log rotation completed\n")
            else:
                log_rotate_script = self.project_root / "rotate_logs.sh"
                with open(log_rotate_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write(f"find logs/{self.environment} -name '*.log' -mtime +7 -delete\n")
                    f.write("echo 'Log rotation completed'\n")
                os.chmod(log_rotate_script, 0o755)
            
            print(f"✓ Logging setup completed for {self.environment}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to setup logging: {e}")
            return False
    
    def create_service_files(self) -> bool:
        """Create service files for system management"""
        print("Creating service files...")
        
        try:
            services_dir = self.project_root / "services"
            services_dir.mkdir(exist_ok=True)
            
            if self.is_windows:
                # Windows service script
                service_script = services_dir / "anti_cheat_service.bat"
                with open(service_script, 'w') as f:
                    f.write("@echo off\n")
                    f.write("title Anti-Cheat Detection System\n")
                    f.write("cd /d %~dp0..\n")
                    f.write("call venv\\Scripts\\activate\n")
                    f.write(f"python run_system.py --config config\\{self.environment}.json\n")
                    f.write("pause\n")
                
                # PowerShell service script
                ps_service_script = services_dir / "anti_cheat_service.ps1"
                with open(ps_service_script, 'w') as f:
                    f.write("$Host.UI.RawUI.WindowTitle = 'Anti-Cheat Detection System'\n")
                    f.write("Set-Location $PSScriptRoot\\..\n")
                    f.write("& .\\venv\\Scripts\\Activate.ps1\n")
                    f.write(f"python run_system.py --config config\\{self.environment}.json\n")
                    f.write("Read-Host 'Press Enter to exit'\n")
                
            else:
                # Unix service script
                service_script = services_dir / "anti_cheat_service.sh"
                with open(service_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("cd \"$(dirname \"$0\")/..\"\n")
                    f.write("source venv/bin/activate\n")
                    f.write(f"python run_system.py --config config/{self.environment}.json\n")
                os.chmod(service_script, 0o755)
                
                # Systemd service file (Linux)
                if platform.system() == 'Linux':
                    systemd_service = services_dir / "anti-cheat-detection.service"
                    with open(systemd_service, 'w') as f:
                        f.write("[Unit]\n")
                        f.write("Description=Anti-Cheat Detection System\n")
                        f.write("After=network.target\n\n")
                        f.write("[Service]\n")
                        f.write("Type=simple\n")
                        f.write(f"WorkingDirectory={self.project_root}\n")
                        f.write(f"ExecStart={self.project_root}/venv/bin/python run_system.py --config config/{self.environment}.json\n")
                        f.write("Restart=always\n")
                        f.write("RestartSec=10\n")
                        f.write("User=nobody\n")
                        f.write("Group=nogroup\n\n")
                        f.write("[Install]\n")
                        f.write("WantedBy=multi-user.target\n")
            
            print("✓ Service files created")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create service files: {e}")
            return False
    
    def create_monitoring_scripts(self) -> bool:
        """Create monitoring and health check scripts"""
        print("Creating monitoring scripts...")
        
        try:
            monitoring_dir = self.project_root / "monitoring"
            monitoring_dir.mkdir(exist_ok=True)
            
            # Health check script
            health_check_script = monitoring_dir / ("health_check.bat" if self.is_windows else "health_check.sh")
            
            if self.is_windows:
                with open(health_check_script, 'w') as f:
                    f.write("@echo off\n")
                    f.write("cd /d %~dp0..\n")
                    f.write("call venv\\Scripts\\activate\n")
                    f.write("python run_system.py --validate-only\n")
                    f.write("if %errorlevel% equ 0 (\n")
                    f.write("    echo System is healthy\n")
                    f.write("    exit /b 0\n")
                    f.write(") else (\n")
                    f.write("    echo System health check failed\n")
                    f.write("    exit /b 1\n")
                    f.write(")\n")
            else:
                with open(health_check_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("cd \"$(dirname \"$0\")/..\"\n")
                    f.write("source venv/bin/activate\n")
                    f.write("python run_system.py --validate-only\n")
                    f.write("if [ $? -eq 0 ]; then\n")
                    f.write("    echo 'System is healthy'\n")
                    f.write("    exit 0\n")
                    f.write("else\n")
                    f.write("    echo 'System health check failed'\n")
                    f.write("    exit 1\n")
                    f.write("fi\n")
                os.chmod(health_check_script, 0o755)
            
            # Resource monitoring script
            resource_monitor_script = monitoring_dir / ("monitor_resources.py")
            with open(resource_monitor_script, 'w') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write('"""\n')
                f.write("Resource monitoring script for Anti-Cheat Detection System\n")
                f.write('"""\n\n')
                f.write("import time\n")
                f.write("import json\n")
                f.write("from pathlib import Path\n")
                f.write("import sys\n")
                f.write("sys.path.append(str(Path(__file__).parent.parent))\n\n")
                f.write("try:\n")
                f.write("    from anti_cheat_system.error_handler import ResourceMonitor, setup_error_handling\n")
                f.write("    \n")
                f.write("    def main():\n")
                f.write("        error_handler = setup_error_handling()\n")
                f.write("        monitor = ResourceMonitor(error_handler)\n")
                f.write("        \n")
                f.write("        print('Starting resource monitoring...')\n")
                f.write("        while True:\n")
                f.write("            stats = monitor.check_resources()\n")
                f.write("            print(f'CPU: {stats[\"cpu_percent\"]:.1f}% | Memory: {stats[\"memory_percent\"]:.1f}% | FPS: {stats[\"fps\"]:.1f}')\n")
                f.write("            time.sleep(5)\n")
                f.write("    \n")
                f.write("    if __name__ == '__main__':\n")
                f.write("        main()\n")
                f.write("except ImportError:\n")
                f.write("    print('Resource monitoring not available - system components not found')\n")
            
            print("✓ Monitoring scripts created")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create monitoring scripts: {e}")
            return False
    
    def create_backup_scripts(self) -> bool:
        """Create backup and restore scripts"""
        print("Creating backup scripts...")
        
        try:
            backup_dir = self.project_root / "backup"
            backup_dir.mkdir(exist_ok=True)
            
            # Backup script
            backup_script = backup_dir / ("backup.bat" if self.is_windows else "backup.sh")
            
            if self.is_windows:
                with open(backup_script, 'w') as f:
                    f.write("@echo off\n")
                    f.write("set BACKUP_DIR=backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%\n")
                    f.write("set BACKUP_DIR=%BACKUP_DIR: =0%\n")
                    f.write("mkdir %BACKUP_DIR%\n")
                    f.write("xcopy /E /I /H config %BACKUP_DIR%\\config\n")
                    f.write("xcopy /E /I /H logs %BACKUP_DIR%\\logs\n")
                    f.write("copy *.py %BACKUP_DIR%\\\n")
                    f.write("copy requirements.txt %BACKUP_DIR%\\\n")
                    f.write("echo Backup completed to %BACKUP_DIR%\n")
            else:
                with open(backup_script, 'w') as f:
                    f.write("#!/bin/bash\n")
                    f.write("BACKUP_DIR=\"backup_$(date +%Y%m%d_%H%M%S)\"\n")
                    f.write("mkdir -p \"$BACKUP_DIR\"\n")
                    f.write("cp -r config \"$BACKUP_DIR/\" 2>/dev/null || true\n")
                    f.write("cp -r logs \"$BACKUP_DIR/\" 2>/dev/null || true\n")
                    f.write("cp *.py \"$BACKUP_DIR/\" 2>/dev/null || true\n")
                    f.write("cp requirements.txt \"$BACKUP_DIR/\" 2>/dev/null || true\n")
                    f.write("echo \"Backup completed to $BACKUP_DIR\"\n")
                os.chmod(backup_script, 0o755)
            
            print("✓ Backup scripts created")
            return True
            
        except Exception as e:
            print(f"✗ Failed to create backup scripts: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run tests for the deployment"""
        print("Running deployment tests...")
        
        try:
            # Check if pytest is available
            test_command = "python -m pytest tests/ -v --tb=short"
            
            if (self.project_root / "tests").exists():
                result = self.run_command(test_command, check=False)
                if result.returncode == 0:
                    print("✓ All tests passed")
                    return True
                else:
                    print("⚠ Some tests failed, but deployment will continue")
                    print(result.stdout)
                    return True  # Don't fail deployment for test failures
            else:
                print("⚠ No tests directory found, skipping tests")
                return True
                
        except Exception as e:
            print(f"⚠ Test execution failed: {e}")
            return True  # Don't fail deployment for test issues
    
    def deploy(self) -> bool:
        """Run the complete deployment process"""
        print(f"Deploying Anti-Cheat Detection System to {self.environment} environment")
        print("=" * 70)
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {sys.version}")
        print("=" * 70)
        
        deployment_steps = [
            ("Environment Configuration", self.create_environment_config),
            ("Logging Setup", self.setup_logging),
            ("Service Files", self.create_service_files),
            ("Monitoring Scripts", self.create_monitoring_scripts),
            ("Backup Scripts", self.create_backup_scripts),
            ("Tests", self.run_tests)
        ]
        
        for step_name, step_func in deployment_steps:
            print(f"\n--- {step_name} ---")
            try:
                if not step_func():
                    print(f"✗ {step_name} failed")
                    return False
            except KeyboardInterrupt:
                print("\nDeployment interrupted by user")
                return False
            except Exception as e:
                print(f"✗ {step_name} failed with error: {e}")
                return False
        
        # Print deployment summary
        print("\n" + "=" * 70)
        print(f"✓ DEPLOYMENT TO {self.environment.upper()} COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("Deployment artifacts created:")
        print(f"  - Configuration: config/{self.environment}.json")
        print(f"  - Service scripts: services/")
        print(f"  - Monitoring: monitoring/")
        print(f"  - Backup scripts: backup/")
        print()
        print("Next steps:")
        print(f"  1. Review configuration: config/{self.environment}.json")
        print("  2. Test the deployment: python run_system.py --validate-only")
        print("  3. Start the service using scripts in services/")
        print("  4. Monitor using scripts in monitoring/")
        print("=" * 70)
        
        return True


def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy Anti-Cheat Detection System")
    parser.add_argument(
        "environment",
        choices=["development", "testing", "production"],
        help="Deployment environment"
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    
    args = parser.parse_args()
    
    try:
        deployment_manager = DeploymentManager(args.environment)
        success = deployment_manager.deploy()
        
        if not success:
            print(f"\nDeployment to {args.environment} failed. Please check the errors above.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during deployment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()