# Anti-Cheat Detection System - Installation Guide

## Overview

The Anti-Cheat Detection System is a real-time computer vision application that monitors students during online examinations. This guide provides step-by-step instructions for installing and setting up the system.

## System Requirements

### Hardware Requirements
- **Camera**: USB webcam or built-in camera (minimum 720p resolution)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better recommended)
- **RAM**: Minimum 4GB, recommended 8GB for optimal performance
- **Storage**: At least 2GB free disk space
- **GPU**: Optional CUDA-compatible GPU for accelerated inference

### Software Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher (Python 3.11 recommended)
- **Internet Connection**: Required for initial model download

## Installation Methods

### Method 1: Automated Setup (Recommended)

1. **Download or clone the project**
   ```bash
   git clone <repository-url>
   cd anti-cheat-detection-system
   ```

2. **Run the setup script**
   ```bash
   python setup.py
   ```
   
   The setup script will:
   - Check Python version compatibility
   - Create a virtual environment
   - Install all required dependencies
   - Verify the installation
   - Create activation scripts

3. **Activate the virtual environment**
   
   **Windows (Command Prompt):**
   ```cmd
   activate.bat
   ```
   
   **Windows (PowerShell):**
   ```powershell
   .\activate.ps1
   ```
   
   **Linux/macOS:**
   ```bash
   ./activate.sh
   ```
   
   **Manual activation:**
   - Windows: `.\venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

### Method 2: Manual Installation

1. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

2. **Activate virtual environment**
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Validation and Testing

### Configuration Validation

Run the configuration validator to check your system:

```bash
python validate_config.py
```

This will check:
- Python version compatibility
- Required dependencies
- Camera availability
- GPU support (optional)
- Disk space and memory
- Project structure
- Model files

### Startup Validation

Test the system startup without running the full application:

```bash
python run_system.py --validate-only
```

## Running the System

### Basic Usage

1. **Activate the virtual environment** (if not already active)
2. **Start the system**:
   ```bash
   python run_system.py
   ```

### Advanced Usage

**Debug mode** (shows detailed error information):
```bash
python run_system.py --debug
```

**Custom configuration** (if you have a custom config file):
```bash
python run_system.py --config path/to/config.json
```

**Validation only** (check system without starting):
```bash
python run_system.py --validate-only
```

## Troubleshooting

### Common Issues

#### 1. Python Version Issues
**Problem**: "Python 3.8+ required" or compatibility warnings
**Solution**: 
- Install Python 3.11 from [python.org](https://python.org)
- Use `py -3.11` on Windows or `python3.11` on Linux/macOS

#### 2. Camera Not Detected
**Problem**: "No working camera found"
**Solutions**:
- Ensure camera is connected and not used by other applications
- Check camera permissions in system settings
- Try different camera indices (the system will auto-detect)
- Test camera with other applications first

#### 3. Dependency Installation Failures
**Problem**: Package installation errors
**Solutions**:
- Update pip: `python -m pip install --upgrade pip`
- Install Visual Studio Build Tools (Windows)
- Install system dependencies (Linux): `sudo apt-get install python3-dev`
- Use conda instead of pip if issues persist

#### 4. GPU/CUDA Issues
**Problem**: GPU not detected or CUDA errors
**Solutions**:
- GPU acceleration is optional - system will use CPU
- Install CUDA toolkit if you want GPU acceleration
- Ensure PyTorch CUDA version matches your CUDA installation

#### 5. Model Download Issues
**Problem**: YOLOv8n model download fails
**Solutions**:
- Check internet connection
- Download manually from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Place `yolov8n.pt` in the project root directory

#### 6. Permission Errors
**Problem**: File permission or access errors
**Solutions**:
- Run as administrator (Windows) or with sudo (Linux/macOS)
- Check file permissions in project directory
- Ensure antivirus isn't blocking files

### Getting Help

1. **Check the logs**: Look in the `logs/` directory for detailed error information
2. **Run validation**: Use `python validate_config.py` to identify issues
3. **Debug mode**: Run with `--debug` flag for detailed error traces
4. **System information**: Note your OS, Python version, and hardware specs

## Performance Optimization

### For Better Performance
- Use Python 3.11 for optimal performance
- Enable GPU acceleration if available
- Ensure adequate RAM (8GB+ recommended)
- Close unnecessary applications while running
- Use SSD storage for better I/O performance

### Resource Monitoring
The system includes built-in resource monitoring that will:
- Track CPU and memory usage
- Monitor frame processing rates
- Log performance metrics
- Alert on resource constraints

## Security Considerations

- The system processes video locally - no data is sent externally
- Camera access is required for operation
- Log files may contain system information
- Ensure proper file permissions in production environments

## Uninstallation

To remove the system:

1. **Deactivate virtual environment**: `deactivate`
2. **Remove project directory**: Delete the entire project folder
3. **Remove Python packages** (if installed globally): 
   ```bash
   pip uninstall opencv-python ultralytics mediapipe numpy psutil
   ```

## Support and Updates

- Check for updates in the project repository
- Review the changelog for new features and fixes
- Report issues with detailed system information and logs
- Contribute improvements via pull requests

## Next Steps

After successful installation:
1. Review the [User Guide](USER_GUIDE.md) for operation instructions
2. Check the [Configuration Guide](CONFIG_GUIDE.md) for customization options
3. Read the [Developer Guide](DEVELOPER_GUIDE.md) if you plan to modify the system