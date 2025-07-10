import cv2
import streamlit as st
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch
from PoseModule import poseDetector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from tensorflow.keras.models import load_model
# Theme Colors
PRIMARY_COLOR = "#2d3e50"
SUCCESS_COLOR = "#28a745"
DANGER_COLOR = "#e74c3c"
BACKGROUND_COLOR = "#f9f9f9"
HEADER_COLOR = "#34495e"
TEXT_COLOR = "#2c3e50"
BUTTON_COLOR = "#3498db"
FONT_FAMILY = "'Roboto', sans-serif"

# Inject Custom CSS for Modernized UI
st.markdown(f"""
    <style>
    /* Global Styles */
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: {FONT_FAMILY};
    }}

    /* Header Styling */
    .favicon-header-container {{
        text-align: center;
        margin-bottom: 20px;
    }}
    .favicon-header-container img {{
        width: 100px;
        border-radius: 50%;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
        margin-bottom: 10px;
    }}
    .favicon-header-container img:hover {{
        transform: scale(1.1);
    }}
    .favicon-header-container h1 {{
        color: white;
        font-size: 2.5rem;
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, {HEADER_COLOR}, {PRIMARY_COLOR});
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2);
        display: inline-block;
        text-align: center;
    }}

    /* Sidebar Styling */
    .sidebar .sidebar-content {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border-radius: 15px;
        padding: 20px;
    }}

    /* Button Styling */
    .css-1aumxhk {{
        background-color: {BUTTON_COLOR};
        color: white;
        border-radius: 10px;
        padding: 12px 25px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.2);
    }}
    .css-1aumxhk:hover {{
        background-color: {SUCCESS_COLOR};
        transform: scale(1.05);
    }}

    /* Prediction Styling */
    .prediction-box {{
        background-color: {SUCCESS_COLOR};
        border-radius: 15px;
        color: white;
        padding: 15px;
        text-align: center;
        font-size: 1.2rem;
        margin: 20px 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }}

    /* Card-style containers for content */
    .card {{
        background-color: grey;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}
    </style>
    """, unsafe_allow_html=True)

# Header Section with Favicon
st.markdown(f"""
    <div class="favicon-header-container">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQD6nqDuIW0j847wP-NVHbC7of_WAcimz_t9g&s" alt="Favicon">
        <h1>Cerebral Palsy Detection via Pose Estimation</h1>
    </div>
    """, unsafe_allow_html=True)

# Instructions Section
st.markdown("""
    <div class="card">
        <p><strong>Instructions:</strong> Upload a video or use the webcam to analyze joint angles and predict potential signs of cerebral palsy. Adjust the settings in the sidebar as needed.</p>
    </div>
    """, unsafe_allow_html=True)


# Sidebar for additional user control
st.sidebar.title("Settings")
input_method = st.sidebar.radio("Choose input method", ('Upload Video', 'Use Webcam'))
tolerance = st.sidebar.slider("Tolerance for Joint Angle Match (°)", 5, 30, 10)
st.sidebar.write(f"Tolerance set to: {tolerance}°")

# Initialize the pose detector
detector = poseDetector()

# Responsive layout: columns for video and bar graph side by side
col1, col2 = st.columns([1, 1])  # Ensure equal size for responsiveness
video_placeholder = col1.empty()
chart_placeholder = col2.empty()

# Placeholders for dynamic content
status_placeholder = st.empty()
progress_placeholder = st.empty()
prediction_placeholder = st.empty()  # Prediction text placeholder

# Load models
def load_models():
    ann_model = load_model('ann_model.h5')
    cnn_model = load_model('cnn_model.h5')
    scaler = joblib.load('scaler.pkl')
    return ann_model, cnn_model, scaler

ann_model, cnn_model, scaler = load_models()

# Function to preprocess input data
def preprocess_input(data):
    scaled_data = scaler.transform(data)
    return scaled_data


# Function to update the prediction and bar chart together
def display_prediction(predicted_class):
    # Update video prediction label
    prediction_placeholder.empty()  # Clear the previous prediction

    # Set background color based on the predicted class
    if predicted_class == 'Normal':
        background_color = "#28a745"  # Green for Normal
    elif predicted_class == 'Cerebral Palsy':
        background_color = "#dc3545"  # Red for Cerebral Palsy
    else:
        background_color = "#ffc107"  # Yellow for Unknown Disease

    # Display prediction class below the video dynamically
    prediction_html = f"""
    <div style="text-align:center; font-size: 24px; font-weight: bold; padding: 10px; color: white; background-color: {background_color}; border-radius: 10px;">
        {predicted_class}
    </div>
    """
    prediction_placeholder.markdown(prediction_html, unsafe_allow_html=True)

# Function to update the seaborn bar chart comparing extracted and normal angles
def update_bar_chart(angles):
    # Reordering the angles dictionary to follow the desired sequence
    ordered_joints = ['Left_Shoulder', 'Right_Shoulder',
                      'Left_Elbow', 'Right_Elbow',
                      'Left_Hip', 'Right_Hip',
                      'Left_Knee', 'Right_Knee']
    
    # Create a dictionary with the angles in the new order
    ordered_angles = {joint: angles[joint] for joint in ordered_joints}

    num_cols = 2  # Two gauges per row
    num_rows = math.ceil(len(ordered_angles) / num_cols)
    
    # Create subplots with two columns
    fig = make_subplots(
        rows=num_rows, cols=num_cols, 
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}] for _ in range(num_rows)],
        horizontal_spacing=0.5,  # Reduced horizontal space between columns
        vertical_spacing=0.1     # Reduced vertical space between rows
    )

    for i, (joint, angle) in enumerate(ordered_angles.items(), start=1):
        row = (i - 1) // num_cols + 1
        col = (i - 1) % num_cols + 1

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=angle,
            title={'text': joint},
            gauge={
                'axis': {'range': [0, 200]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 100], 'color': "lightgray"},
                    {'range': [100, 150], 'color': "lightblue"},
                    {'range': [150, 200], 'color': "lightgreen"}
                ],
            }
        ), row=row, col=col)

    fig.update_layout(
        height=500,  # Decrease height to make rows closer
        width=800,   # Keep the width constant
        margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins
    )
    
    chart_placeholder.plotly_chart(fig)


# List to store angle history for iterative difference calculation
angle_history = []

# Function to compute differences between consecutive frames
def iterative_difference(history):
    iterative_differences = pd.DataFrame()
    
    if len(history) < 2:
        return pd.DataFrame()

    previous_angles = history[-2]
    current_angles = history[-1]

    for feature in previous_angles.index:
        difference = current_angles[feature] - previous_angles[feature]
        iterative_differences[f'{feature}_iterative_difference'] = [difference]
    
    return iterative_differences

# Function to process frames and calculate angles
def process_frame(frame):
    resized = cv2.resize(frame, (300, 400))  # Resize for consistency in display
    resized = detector.findPose(resized)
    lmList = detector.findPosition(resized, draw=False)

    if lmList:
        left_elbow = detector.findAngle(resized, 11, 13, 15)
        right_elbow = detector.findAngle(resized, 12, 14, 16)
        left_knee = detector.findAngle(resized, 23, 25, 27)
        right_knee = detector.findAngle(resized, 24, 26, 28)
        left_shoulder = detector.findAngle(resized, 12, 11, 13)
        right_shoulder = detector.findAngle(resized, 11, 12, 14)
        left_hip = detector.findAngle(resized, 11, 23, 25)
        right_hip = detector.findAngle(resized, 12, 24, 26)

        angles = {
            "Left_Elbow": left_elbow,
            "Right_Elbow": right_elbow,
            "Left_Knee": left_knee,
            "Right_Knee": right_knee,
            "Left_Shoulder": left_shoulder,
            "Right_Shoulder": right_shoulder,
            "Left_Hip": left_hip,
            "Right_Hip": right_hip
        }

        return angles, resized
    return None, None

# Processing video input

def process_video_input(video_source):
    cap = cv2.VideoCapture(video_source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if input_method == 'Upload Video' else 0
    target_fps = 30  # Target FPS for uploaded videos
    frame_interval = 1 / target_fps

    frame_count = 0
    progress = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if input_method == 'Upload Video':
            progress.progress(frame_count / total_frames)

        angles, resized_frame = process_frame(frame)

        if angles:
            fields = ['Left_Elbow', 'Right_Elbow', 'Left_Knee', 'Right_Knee', 'Left_Shoulder', 'Right_Shoulder', 'Left_Hip', 'Right_Hip']
            df = pd.DataFrame([angles], columns=fields)

            angle_history.append(df.iloc[0])

            if len(angle_history) > 1:
                iterative_differences_df = iterative_difference(angle_history)
                iterative_differences_df.reset_index(drop=True, inplace=True)

                df.reset_index(drop=True, inplace=True)
                merged_df = pd.concat([df, iterative_differences_df], axis=1)

                if not merged_df.empty:
                    preprocessed_data = preprocess_input(merged_df)

                    ann_pred = ann_model.predict(preprocessed_data)

                    cnn_input = ann_pred.reshape(1, 3, 1, 1)
                    cnn_pred = cnn_model.predict(cnn_input)

                    # Extracting the highest score and the corresponding label
                    prediction_labels = ['Normal', 'Cerebral Palsy', 'Unknown Disease']
                    predicted_class = prediction_labels[cnn_pred.argmax()]  # Get the label with the highest probability
                    predicted_probability = cnn_pred.max()  # Get the highest probability value

                    # Overwrite the prediction after each frame
                    prediction_placeholder.subheader('Prediction Results')

                    prediction_placeholder.write(f"**Prediction Probability:** {predicted_probability:.2f}")
                    prediction_placeholder.write(f" {predicted_class}")


                    # Call the display_prediction function to update the output with color
                    display_prediction(predicted_class)

                    # Update bar graph comparing normal and extracted angles
                    update_bar_chart(angles)
            else:
                st.write("Not enough frames to compute differences.")

            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            for i, (joint, angle) in enumerate(angles.items()):
                cv2.putText(resized_frame, f"{joint}: {int(angle)}", (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display the video frame
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True, caption="Processed Video")

        time.sleep(frame_interval)

    cap.release()
    cv2.destroyAllWindows()


# If a file is uploaded or webcam is selected, process it
if input_method == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_file is not None:
        temp_video_file = uploaded_file.name
        with open(temp_video_file, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        process_video_input(temp_video_file)

elif input_method == 'Use Webcam':
    process_video_input(1)