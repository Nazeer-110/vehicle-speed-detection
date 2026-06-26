# 🚗 AI Vehicle Speed Detection System

YOLOv8 + custom IOU tracker based vehicle detection, tracking, and speed estimation with a Streamlit web interface.

## Features

- **Vehicle Detection** — YOLOv8 detects cars, motorcycles, buses, and trucks
- **Vehicle Tracking** — Custom IOU-based tracker assigns unique IDs to each vehicle
- **Speed Estimation** — Calculates speed in km/h from pixel displacement across frames
- **Web Interface** — Streamlit GUI for upload, processing, and download
- **Browser-Compatible Output** — H.264 MP4 video plays directly in Chrome/Edge
- **CSV Report** — Download speed data for each detected vehicle

## Try It

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/nazeer-ahmed-cs/vehicle-speed-detection)

## Requirements

- Python 3.8+
- Ultralytics YOLOv8
- OpenCV
- Streamlit

## Installation

```bash
git clone https://github.com/nazeer-ahmed-cs/vehicle-speed-detection.git
cd vehicle-speed-detection
pip install -r requirements.txt
```

The YOLOv8 model (`yolov8n.pt`) is downloaded automatically on first run.

## Usage

```bash
streamlit run app.py
```

1. Upload a traffic video (MP4, AVI, or MOV)
2. Click **"Start Detection"**
3. Wait for processing to complete
4. View the annotated output video
5. Download the speed report CSV

## How It Works

1. **Detection** — Each frame is passed through YOLOv8 to detect vehicles
2. **Tracking** — A custom IOU-based tracker matches detections across frames using bounding box overlap
3. **Speed Calculation** — Speed = (pixel_distance / calibration_factor) / time_difference × 3.6
4. **Encoding** — Frames are piped through FFmpeg `libx264` for browser-compatible H.264 output

## Project Structure

```
├── app.py              # Streamlit web interface
├── main.py             # Detection, tracking, speed processing
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages (FFmpeg for Streamlit Cloud)
└── .devcontainer/      # VS Code dev container config
```

## Deployment

### Streamlit Cloud

The app is ready for Streamlit Cloud deployment:

1. Push to GitHub
2. Connect repo on [Streamlit Cloud](https://streamlit.io/cloud)
3. Set main file to `app.py`
4. Deploy

`packages.txt` ensures FFmpeg is installed for H.264 encoding.

### Local

```bash
streamlit run app.py
```

## License

MIT
