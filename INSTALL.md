# Installation Guide - MANUS

Detailed step-by-step installation instructions for all platforms.

## Table of Contents

1. [Windows](#windows)
2. [Linux (Ubuntu/Debian)](#linux-ubuntudebian)
3. [macOS](#macos)
4. [Troubleshooting](#troubleshooting)
5. [Virtual Environments](#virtual-environments)

---

## Windows

### Prerequisites Check

1. **Python 3.8+** - [Download](https://www.python.org/downloads/windows/)
   ```cmd
   python --version
   ```

2. **Git** - [Download](https://git-scm.com/download/win) (optional but recommended)

3. **Camera drivers** - Most Windows 10+ systems have built-in support

### Step-by-Step Installation

**1. Clone repository (or download ZIP)**

Using Git (recommended):
```cmd
git clone https://github.com/Yash_12711/MANUS.git
cd MANUS
```

Or download ZIP from GitHub and extract.

**2. Create virtual environment**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` prefix in your terminal.

**3. Install dependencies**

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Run the application**

```cmd
python main.py
```

### Troubleshooting Windows

**"Python not found"**
- Ensure Python is added to PATH during installation
- Restart terminal after installation
- Use full path: `C:\Python311\python main.py`

**"Camera not working"**
- Update camera drivers from Device Manager
- Disable any privacy settings blocking camera
- Try unplugging/replugging USB camera

**"Permission denied"**
- Run Command Prompt as Administrator
- Delete `.venv` folder and recreate it

---

## Linux (Ubuntu/Debian)

### Prerequisites

1. **Python 3.8+**
   ```bash
   python3 --version
   ```
   If not installed:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. **Camera permissions**
   ```bash
   sudo usermod -a -G video $USER
   ```
   Log out and back in for permissions to take effect.

3. **Required system packages**
   ```bash
   sudo apt install libatlas-base-dev libjasper-dev libtiff-dev libjpeg-dev \
                    zlib1g-dev libharfbuzz0b libwebp6 libtiff5
   ```

### Step-by-Step Installation

**1. Clone repository**

```bash
cd ~
git clone https://github.com/Yash_12711/MANUS.git
cd MANUS
```

**2. Create virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` prefix in your terminal.

**3. Install dependencies**

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**4. Test camera access**

```bash
ls -la /dev/video*
```

Should show at least `/dev/video0`.

**5. Run application**

```bash
python3 main.py
```

### Troubleshooting Linux

**"Permission denied" for camera**
```bash
sudo usermod -a -G video $USER
# Log out and back in
```

**"OpenCV not working" or import errors**
```bash
pip install --upgrade opencv-python mediapipe
```

**Camera still not found**
```bash
# Check if camera is detected
v4l2-ctl --list-devices

# Manually enable camera in privacy settings (GNOME)
gsettings set org.gnome.desktop.privacy disable-camera false
```

**pip install fails**
```bash
# Use binary wheels
pip install --only-binary=:all: -r requirements.txt
```

---

## macOS

### Prerequisites

1. **Python 3.8+** (via Homebrew recommended)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python@3.11
   ```

2. **Git**
   ```bash
   brew install git
   ```

3. **Camera permissions** - Grant in System Preferences → Security & Privacy → Camera

### Step-by-Step Installation

**1. Clone repository**

```bash
git clone https://github.com/Yash_12711/MANUS.git
cd MANUS
```

**2. Create virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Grant camera permissions**

```bash
# Ensure Terminal/iTerm has camera permission in System Preferences
# Security & Privacy → Camera → Allow Terminal
```

**5. Run application**

```bash
python3 main.py
```

### Troubleshooting macOS

**"Camera permission denied"**
- Go to System Preferences → Security & Privacy → Camera
- Check that Terminal (or iTerm) has permission enabled

**"OpenCV build fails"**
```bash
pip install opencv-python --no-cache-dir
```

**"Python command not found"**
```bash
# Use explicit version
python3 main.py

# Or set alias
echo 'alias python="python3"' >> ~/.zshrc
source ~/.zshrc
```

**Slow performance**
- Ensure no other camera-using apps are running (Zoom, FaceTime, etc.)
- Lower camera resolution in Settings GUI

---

## Virtual Environments

Why use virtual environments?

✅ **Isolated dependencies** - Project won't interfere with system Python
✅ **Reproducible setup** - Same environment on all machines
✅ **Easy to remove** - Just delete the `.venv` folder
✅ **Professional standard** - Always used in production

### Creating New Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Activating Existing Environment

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### Deactivating Environment

```bash
deactivate
```

### Creating requirements.txt from environment

```bash
pip freeze > requirements.txt
```

---

## Docker Installation (Optional)

For consistent environment across machines:

**1. Install Docker** - [Download](https://www.docker.com/products/docker-desktop)

**2. Create Dockerfile**

```dockerfile
FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**3. Build and run**

```bash
docker build -t manus .
docker run --device /dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY manus
```

---

## Conda Installation (Optional)

For users who prefer Anaconda/Miniconda:

**1. Install Miniconda** - [Download](https://docs.conda.io/en/latest/miniconda.html)

**2. Create environment**

```bash
conda create -n manus python=3.11
conda activate manus
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
# Or use conda packages
conda install -c conda-forge mediapipe opencv numpy pyautogui
```

**4. Run**

```bash
python main.py
```

---

## Verifying Installation

After completing installation, verify everything works:

```bash
# Check Python version
python --version  # Should be 3.8+

# Check virtual environment is active
which python  # Should show .venv path

# Test imports
python -c "import cv2; import mediapipe; import pyautogui; print('✓ All imports successful')"

# Check camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print(f'Camera working: {cap.isOpened()}')"

# Run main application
python main.py
```

---

## Common Installation Issues

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Ensure virtual environment is activated |
| Camera not found | Check USB connection and drivers |
| Permission denied | Run with `sudo` or check group membership |
| Import errors | Run `pip install --upgrade` for specific package |
| Out of memory | Lower resolution: `CAMERA_RESOLUTION_WIDTH = 320` |

---

## Getting Help

1. **Check troubleshooting** above
2. **Enable debug mode**: `DEBUG_MODE = True` in `config.py`
3. **Check logs**: `tail -f data/hand_tracking_log.txt`
4. **GitHub Issues**: [Report problem](https://github.com/Yash_12711/MANUS/issues)

---

## Next Steps

After successful installation:

1. Read [README.md](README.md) for usage guide
2. Review [CODE_STRUCTURE.md](CODE_STRUCTURE.md) for technical details
3. Try different gestures and configurations
4. Check [Performance Tuning](README.md#performance-tuning) if needed

Enjoy! 🎮
