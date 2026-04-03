# MediaPipe Setup Guide

## Overview

MANUS uses MediaPipe's Hand Landmarker for real-time hand gesture detection. Starting with MediaPipe 0.10.0, the API migrated from the Solutions API to the Tasks API, and model files are no longer bundled with the pip package.

## What You Need

1. **MediaPipe Tasks API** (0.10+) - Installed via `pip install -r requirements.txt`
2. **Hand Landmarker Model** (.task file) - Must be downloaded separately

## Model Files

Two model options are available:

| Model | File | Size | Speed | Accuracy |
|-------|------|------|-------|----------|
| **Lite** | `hand_landmarker_lite.task` | ~3 MB | 🚀 Faster | Normal |
| **Full** | `hand_landmarker_full.task` | ~25 MB | 🐢 Slower | 🎯 Better |

### Recommended Choice

- **Most users**: Use **lite model** (faster, smaller)
- **High accuracy needed**: Use **full model**

## Installation

### Option 1: Automatic Download (Recommended)

```bash
# Download both models
python download_models.py

# Or just the lite model
python download_models.py --lite

# To re-download existing models
python download_models.py --force
```

Models are cached in `~/.cache/mediapipe/` for future runs.

### Option 2: Manual Download

1. Visit: https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker
2. Download the model file (.task)
3. Create cache directory:
   ```bash
   mkdir -p ~/.cache/mediapipe
   ```
4. Move downloaded file:
   ```bash
   mv hand_landmarker_lite.task ~/.cache/mediapipe/
   ```

### Option 3: Custom Model Location

Edit `config.py` to specify a custom model path:

```python
# config.py
MP_MODEL_PATH = "/path/to/your/hand_landmarker_lite.task"
```

Then update `mediapipe_compat.py` to use this path.

## Verification

After downloading models, verify setup:

```bash
python -c "
import os
cache_dir = os.path.expanduser('~/.cache/mediapipe')
models = [f for f in os.listdir(cache_dir) if f.endswith('.task')]
print(f'Found models: {models}')
"
```

## Troubleshooting

### Error: "MODEL FILES NOT FOUND"

```
[ERROR] MODEL FILES NOT FOUND
[ERROR] The MediaPipe hand landmarker model files are required but not found.
```

**Solution:**
```bash
python download_models.py
```

### Error: Network timeout during download

If download times out:

1. Try with `--lite` flag:
   ```bash
   python download_models.py --lite
   ```

2. Or manually download from: https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker

3. Place in: `~/.cache/mediapipe/`

### Error: Permission denied

Ensure cache directory is writable:

```bash
mkdir -p ~/.cache/mediapipe
chmod 755 ~/.cache/mediapipe
```

## MediaPipe API Changes

### From Solutions API (pre-0.10) to Tasks API (0.10+)

**Old (Solutions API):**
```python
import mediapipe as mp
hands = mp.solutions.hands.Hands()
results = hands.process(image_rgb)
```

**New (Tasks API):**
```python
from mediapipe.tasks.python.vision import HandLandmarker
landmarker = HandLandmarker.create_from_options(options)
results = landmarker.detect(image)
```

MANUS uses a compatibility layer (`mediapipe_compat.py`) to maintain the old interface while using the new Tasks API under the hood.

## Performance Tips

### If detection is slow:

1. Use **lite model** instead of full:
   ```bash
   python download_models.py --lite --force
   ```

2. Lower camera resolution in `config.py`:
   ```python
   CAMERA_RESOLUTION_WIDTH = 320
   CAMERA_RESOLUTION_HEIGHT = 240
   ```

3. Reduce detection frequency in main loop

### If detection is inaccurate:

1. Use **full model** instead of lite:
   ```bash
   python download_models.py --force  # Re-downloads --full by default
   ```

2. Improve lighting conditions

3. Adjust detection confidence in `config.py`:
   ```python
   MP_DETECTION_CONFIDENCE = 0.5
   MP_TRACKING_CONFIDENCE = 0.5
   ```

## FAQ

**Q: Do I need both model files?**  
A: No. You only need one. The lite model is recommended for most use cases.

**Q: Can I use an older version of MediaPipe?**  
A: MediaPipe 0.9.x used the old Solutions API, but compatibility with Python 3.14 is unclear. Recommend using 0.10.33 with the Tasks API.

**Q: Where do models get cached?**  
A: `~/.cache/mediapipe/` (Linux/macOS) or `C:\Users\<user>\AppData\Local\Temp\mediapipe` (Windows)

**Q: Can I use different MediaPipe models?**  
A: This version is specifically optimized for Hand Landmarker. Other models (pose, face) are not currently supported.

**Q: Does the application work without models?**  
A: No. Hand landmark models are required. We don't have a fallback hand detection algorithm.

## References

- [MediaPipe Hand Landmarker](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker)
- [MediaPipe Tasks Documentation](https://ai.google.dev/mediapipe)
- [Model Download Instructions](https://ai.google.dev/mediapipe/solutions/vision/hand_landmarker)
