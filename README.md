# ADAS3 Server
Version: 0.5 Alpha

<div align="left">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
  </a>
  <a href="https://www.gnu.org/software/bash/">
    <img src="https://img.shields.io/badge/Shell-Bash-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white" alt="Bash">
  </a>
  <a href="https://github.com/zarkentroska/ADAS3-Server/releases">
    <img src="https://img.shields.io/github/downloads/zarkentroska/ADAS3-Server/total?style=flat-square&labelColor=6B7280&color=22C55E&label=Downloads" alt="Downloads">
  </a>
</div>

ADAS3 Server is a real-time drone detection system that combines video analysis, audio processing, and RF spectrum monitoring. It works in conjunction with the ADAS3 Android Client to provide a complete solution for detecting and monitoring unmanned aerial vehicles (UAVs) in real-time.

This application is part of a client-server architecture project that works together with ADAS3 Android Client. The ultimate goal of this combined project is drone detection for various applications, providing a complete solution for monitoring and detecting unmanned aerial vehicles (UAVs) in real-time.

<div align="center">
  <img src="https://raw.githubusercontent.com/zarkentroska/ADAS3-Server/main/UI-ADAS3Server.png" alt="ADAS3 Server Interface" width="800">
  <p><em>Screenshot of the ADAS3 Server user interface</em></p>
</div>

## 🎯 Main features

### 🎥 Video analysis
- **Real-time video streaming** from Android devices via HTTP
- **YOLO-based drone detection** with customizable models
- **Multiple YOLO model slots** (up to 10 configurable models)
- **Dynamic model switching** during runtime
- **Bounding box visualization** with confidence scores
- **Configurable detection thresholds**
- **Model path management** with automatic resource location

### 🔊 Audio detection
- **Real-time audio streaming** from Android devices
- **TensorFlow-based audio analysis** for drone sound detection
- **PCM audio processing** with configurable sample rates
- **Audio normalization** using pre-computed statistics
- **Quick enable/disable** via UI button
- **Visual feedback** with volume/mute icons

### 📡 RF spectrum analysis (TinySA integration)
- **TinySA Ultra device integration** via USB or HTTP (through Android)
- **Automatic detection**: TinySA is detected on-the-fly when connected
- **Real-time spectrum visualization** overlaid on video
- **Multiple scanning modes**:
  - **2.4 GHz band**: Optimized for drone detection at 2.4 GHz
  - **5.8 GHz band**: Optimized for drone detection at 5.8 GHz
  - **Mixed sweeps**: Alternates between 2.4 GHz and 5.8 GHz every 10 sweeps
  - **Custom frequency range**: User-defined frequency ranges
  - **Advanced intervals**: Custom configurations with multiple frequency intervals
- **Automatic frequency range detection**
- **Graph overlay** with frequency and dBm scales
- **Configurable scanning intervals**

### 🌐 Network integration
- **Tailscale VPN integration** with automatic service detection
- **One-click Tailscale activation/deactivation**
- **Automatic Tailscale installation** for Windows and Linux
- **OAuth-based authentication** flow
- **Service status monitoring**
- **Cross-platform support** (Windows/Linux)
- **ADB support**: ADB drivers are included and automatically detect wired Android connections when devices are connected via USB

### 🎨 User interface
- **OpenCV-based GUI** with real-time video display
- **Interactive controls** for all features
- **Status indicators** for:
  - YOLO detection status
  - Audio streaming status
  - TinySA connection status
  - Tailscale connection status
- **Settings dialogs** for configuration
- **Model selection interface**

### 🌍 Multi-language support
Full support for 5 languages:

- 🇪🇸 **Spanish** (default)
- 🇬🇧 **English**
- 🇫🇷 **French** (Français)
- 🇮🇹 **Italian** (Italiano)
- 🇵🇹 **Portuguese** (Português)

### ⚙️ Configuration management
- **Persistent configuration** for:
  - Camera settings
  - YOLO models and slots
  - Language preferences
  - Tailscale settings
  - TinySA intervals
- **JSON-based configuration files**
- **Automatic resource path resolution** (works in compiled executables)

## 📋 Requirements

### System requirements
- **Windows 10/11** or **Linux** (Ubuntu/Debian recommended)
- **Python 3.8+** (for development)
- **Minimum 4GB RAM** (8GB+ recommended)
- **GPU support** (optional, for faster YOLO inference)

### Dependencies
- OpenCV (cv2)
- TensorFlow
- PyTorch / Ultralytics YOLO
- NumPy
- Librosa
- PyAudio
- Tkinter
- Requests
- Matplotlib
- Serial (pyserial)

### Hardware (optional)
- **TinySA Ultra** - RF spectrum analyzer (USB or via Android)
- **Android device** - Running ADAS3 Android Client

## 🚀 Installation

### Option 1: precompiled packages

<div align="center">
  <a href="https://github.com/zarkentroska/ADAS3-Server/releases/latest">
    <img src="https://user-images.githubusercontent.com/69304392/148696068-0cfea65d-b18f-4685-82b5-329a330b1c0d.png" alt="Get it on GitHub" width="200">
  </a>
</div>

#### Windows
1. Download `ADAS3-Server-0.5Alpha-win-x64.exe` from [Releases](https://github.com/zarkentroska/ADAS3-Server/releases)
2. Run the installer
3. Launch the application

#### Linux (Debian/Ubuntu)
1. Download `adas3-server-0.5alpha_amd64.deb` from [Releases](https://github.com/zarkentroska/ADAS3-Server/releases)
2. Install the package:
   ```bash
   sudo dpkg -i adas3-server-0.5alpha_amd64.deb
   ```
3. Launch from terminal: `adas3` or from applications menu

### Option 2: build from source

#### Prerequisites
```bash
# Install Python 3.8+
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# Install system dependencies (Linux)
sudo apt-get install python3-tk portaudio19-dev libasound2-dev
```

#### Setup
```bash
# Clone the repository
git clone https://github.com/zarkentroska/ADAS3-Server.git
cd ADAS3-Server

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python testcam.py
```

#### Building executables

**Windows (.exe):**
```bash
pyinstaller --noconfirm pyinstaller.spec
# Output: dist/DetectorDrones.exe
```

**Linux (.deb):**
```bash
./build_deb.sh
# Output: adas3-server-0.5alpha_amd64.deb
```

## 🎮 Usage

### Initial setup

1. **Start the application**
   - On Windows: Double-click the `.exe` file
   - On Linux: Run `adas3` from terminal or applications menu

2. **Connect to Android Client**
   - Ensure ADAS3 Android Client is running on your Android device
   - Enter the Android device's IP address in the application
   - The video stream should start automatically

3. **Configure YOLO Models** (Optional)
   - Click the YOLO settings icon (⚙️)
   - Add or select YOLO model files (.pt format)
   - Configure up to 10 model slots

### Basic operation

1. **Video detection**
   - Video stream displays automatically when connected
   - Drones are detected and highlighted with bounding boxes
   - Confidence scores are shown for each detection

2. **Audio detection**
   - Click the audio button (🎤) to enable/disable audio streaming
   - Audio analysis runs in the background
   - Detection results are displayed in the console

3. **TinySA integration**
   - Connect TinySA via USB or configure via Android
   - Click the TinySA button to start scanning
   - Select frequency range and scanning mode
   - Spectrum graph appears overlaid on video

4. **Tailscale integration**
   - Click the Tailscale ON/OFF button
   - If not installed, use "Install Service" option
   - Follow OAuth authentication in browser
   - Status indicator shows connection state

### Advanced configuration

#### YOLO models
- Models are stored in `yolo_models_config.json`
- Default model: `best.pt`
- Custom models can be added via settings dialog
- Model paths are automatically resolved in compiled executables

#### Audio settings
- Audio model: `drone_audio_model.h5`
- Normalization files: `audio_mean.npy`, `audio_std.npy`
- Sample rate: Configurable (default: 44100 Hz)

#### TinySA configuration
- Advanced intervals stored in `tinysa_advanced_intervals.json`
- Custom frequency ranges can be configured
- Scanning intervals are configurable

## 🔧 Configuration files

- `config_camara.json` - Camera and general settings
- `yolo_models_config.json` - YOLO model configuration
- `language_config.json` - Language preference
- `tailscale_config.json` - Tailscale settings
- `tinysa_advanced_intervals.json` - TinySA frequency intervals

## 🛠️ Development

### Project structure
```
ADAS3-Server/
├── testcam.py                 # Main application script
├── pyinstaller.spec          # PyInstaller configuration
├── build_deb.sh              # Linux .deb build script
├── BUILD_DEB.md              # Build documentation
├── *.pt                      # YOLO model files
├── *.h5                      # TensorFlow audio model
├── *.npy                     # Audio normalization files
├── *.json                    # Configuration files
└── *.png, *.ico              # UI resources
```

### Technologies used
- **Python 3.8+** - Programming language
- **OpenCV** - Video processing and GUI
- **TensorFlow** - Audio analysis
- **PyTorch / Ultralytics** - YOLO object detection
- **Tkinter** - Configuration dialogs
- **PyInstaller** - Executable packaging
- **Librosa** - Audio processing
- **NumPy** - Numerical computations

## 🔒 Security

- **Tailscale VPN** - Secure remote connections
- **OAuth authentication** - Secure Tailscale login
- **Local configuration** - All settings stored locally

## 📝 License

See LICENSE file for details.

## 🔄 Version history

### v0.5 Alpha
- Initial release with all main features
- YOLO-based drone detection
- TensorFlow audio analysis
- TinySA Ultra integration
- Tailscale VPN support
- Multi-language support (5 languages)
- Windows and Linux executables
- Configurable YOLO model slots
- Real-time video and audio streaming

## 📚 Related projects

- **[ADAS3 Android Client](https://github.com/zarkentroska/ADAS3-Client)** - Android streaming client
- **[ADAS3 Releases](https://github.com/zarkentroska/adas3)** - Compiled binaries repository
