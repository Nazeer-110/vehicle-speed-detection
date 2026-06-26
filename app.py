import streamlit as st
import os
from main import process_video, validate_video

st.set_page_config(
    page_title="AI Traffic Monitoring",
    page_icon="🚗",
    layout="wide"
)

st.markdown(
    """
    <style>
    .video-container {
        max-width: 700px;
        margin: 0 auto;
    }
    .video-container video {
        width: 100%;
        height: auto;
        border-radius: 8px;
    }
    .stVideo {
        max-width: 700px;
        margin: 0 auto;
        display: block;
    }
    video {
        max-width: 700px !important;
        display: block !important;
        margin: 0 auto !important;
        height: auto !important;
    }
    .block-container {
        max-width: 900px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 AI Vehicle Speed Detection System")
st.write("YOLOv8 + ByteTrack based vehicle detection, tracking and speed estimation.")

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)


def get_video_bytes(path):
    with open(path, "rb") as f:
        return f.read()


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


def centered_video(video_bytes):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.video(video_bytes, format="video/mp4")


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
        preview_bytes = get_video_bytes(input_path)
        centered_video(preview_bytes)

        col1, col2, col3 = st.columns(3)
        col1.metric("Resolution", f"{video_info['width']}x{video_info['height']}")
        col2.metric("FPS", f"{video_info['fps']:.2f}")
        col3.metric("Frames", video_info["frame_count"])

        if st.button("🚀 Start Detection"):
            try:
                progress_bar = st.progress(0, text="Initializing...")
                status_text = st.empty()

                status_text.info("Processing video with YOLOv8 + ByteTrack...")
                progress_bar.progress(20, text="Running detection...")

                output_video, csv_file = process_video(
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
                    st.info(f"Output size: {file_size_mb:.2f} MB | Codec: H.264")

                    output_info = validate_video(output_video)
                    if output_info["valid"]:
                        st.caption(
                            f"{output_info['width']}x{output_info['height']} "
                            f"| {output_info['fps']:.2f} FPS "
                            f"| {output_info['frame_count']} frames"
                        )

                    output_bytes = get_video_bytes(output_video)
                    centered_video(output_bytes)
                else:
                    st.error("Output video not found!")

                if os.path.exists(csv_file):
                    csv_bytes = get_video_bytes(csv_file)
                    st.download_button(
                        label="📥 Download Speed Report CSV",
                        data=csv_bytes,
                        file_name="speed_log.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error: {e}")
