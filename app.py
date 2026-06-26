import streamlit as st
import os
import cv2
from pathlib import Path
from main import process_video, validate_video

st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>
    video {
        max-width: 700px !important;
        width: 100% !important;
        height: auto !important;
        display: block !important;
        margin: 1rem auto !important;
        border-radius: 8px;
    }
    .stVideo {
        max-width: 700px;
        margin: 0 auto;
        display: block;
    }
    .block-container {
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 AI Vehicle Speed Detection System")
st.write("YOLOv8 based vehicle detection, tracking and speed estimation.")

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)


def validate_uploaded_video(path):
    info = validate_video(path)
    if not info["valid"]:
        st.error(f"Invalid video: {info.get('error')}")
        return None
    if info["frame_count"] < 10:
        st.error("Video is too short (less than 10 frames).")
        return None
    if info["fps"] <= 0:
        st.error("Could not determine video FPS.")
        return None
    return info


def centered_video(path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        abs_path = os.path.abspath(path)
        st.video(abs_path)


uploaded_file = st.file_uploader(
    "Upload Traffic Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:
    input_path = os.path.join("uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Video uploaded successfully")
    st.subheader("Original Video")

    video_info = validate_uploaded_video(input_path)
    if video_info:
        centered_video(input_path)

        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
        col2.metric("FPS", f"{video_info['fps']:.2f}")
        col3.metric("Frames", video_info["frame_count"])

        if st.button("🚀 Start Detection"):
            try:
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()

                status_text.info("Processing video with YOLOv8...")
                progress_bar.progress(20, text="Running detection...")

                output_video, csv_file, used_codec = process_video(
                    input_video=input_path, target_width=640
                )

                progress_bar.progress(80, text="Finalizing output...")

                st.success("✅ Processing Completed!")
                progress_bar.progress(100, text="Done!")
                progress_bar.empty()
                status_text.empty()

                st.subheader("Processed Video")
                if os.path.exists(output_video):
                    file_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                    st.info(f"Output: {file_size_mb:.2f} MB | Codec: {used_codec}")

                    output_info = validate_video(output_video)
                    if output_info["valid"]:
                        st.caption(
                            f"{output_info['width']}x{output_info['height']} "
                            f"| {output_info['fps']:.2f} FPS "
                            f"| {output_info['frame_count']} frames"
                        )

                    centered_video(output_video)

                    with open(output_video, "rb") as f:
                        st.download_button(
                            label="📥 Download Output Video",
                            data=f,
                            file_name="processed_traffic.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.error("Output video not found!")

                if os.path.exists(csv_file):
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            label="📥 Download Speed Report CSV",
                            data=f,
                            file_name="speed_log.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error: {e}")
