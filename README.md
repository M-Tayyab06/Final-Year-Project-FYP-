# Final-Year-Project-FYP: Cerebral Palsy Detection via Pose Estimation

**by [@M-Tayyab06](https://github.com/M-Tayyab06)**

A final-year AI-based healthcare project that detects *Cerebral Palsy (CP)* using real-time human pose estimation and deep learning. This Streamlit-powered web application analyzes joint angles from video or webcam input, processes them using custom-trained ANN and CNN models, and classifies whether the movement pattern indicates **Normal**, **Cerebral Palsy**, or **Unknown Disease**.
*An interactive interface for real-time pose-based diagnosis.*

---

## 🧠 Project Highlights

* 🧍 **Pose Estimation** with [MediaPipe](https://mediapipe.dev) to track human joint coordinates.
* 🥮 **Joint Angle Analysis**: Real-time computation of 8 critical joint angles.
* 🧪 **ML Pipeline**:

  * ANN for feature transformation
  * CNN for classification
* 📈 **Visual Feedback** with dynamic gauge charts for every joint.
* 📹 **Dual Input Mode**: Upload your own video or use webcam live feed.
* 🪄 **Custom Scaler** to normalize joint angles before prediction.
* 💨 **Modern UI** with custom CSS styling via Streamlit.

---

## 🔬 Use Case

Cerebral Palsy (CP) affects posture and movement due to abnormal brain development. This tool provides a **non-invasive, AI-driven screening system** that detects abnormal movement patterns from video data.

---

## 📸 Application Demo
![Dashboard Demo](https://github.com/user-attachments/assets/caf1a047-0691-4665-983e-af303d6d4e9a)
> Trained on labeled pose datasets using extracted joint-angle differences.
---

## 🗂️ Project Structure

```
Final-Year-Project-FYP/
│
├── app.py                  # Streamlit main app
├── PoseModule.py           # Pose detector using MediaPipe
├── ann_model.h5            # Trained ANN model
├── cnn_model.h5            # Trained CNN classifier
├── scaler.pkl              # Standard scaler for feature preprocessing
├── requirements.txt        # All Python dependencies
├── assets/
│   ├── confusion_matrix.png
│   └── dashboard_preview.png
└── README.md               # This file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/M-Tayyab06/Final-Year-Project-FYP.git
cd Final-Year-Project-FYP
```

### 2. Create Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate     # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Pose Estimation**

   * Using MediaPipe's `Pose` module to detect body landmarks.
2. **Joint Angle Calculation**

   * Angles are computed for elbows, shoulders, hips, and knees.
3. **Iterative Frame Analysis**

   * Difference in joint angles over frames is extracted.
4. **Feature Scaling**

   * Scaled using pre-trained `scaler.pkl`.
5. **Model Prediction**

   * ANN → intermediate representation
   * CNN → final prediction (Normal / CP / Unknown)
6. **Live UI Update**

   * Bar charts + color-coded predictions rendered via Plotly & Streamlit.

---

## 🧪 Model Performance
![Confusion Matrix](https://github.com/user-attachments/assets/26a2856b-d6ae-4136-8d1a-34624a0a1b63)
*Confusion Matrix can be found in the `/assets` folder.*

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for full list. Major ones include:

* `mediapipe`
* `opencv-python`
* `streamlit`
* `tensorflow`
* `plotly`
* `joblib`
* `pandas`, `matplotlib`, `seaborn`

---

## 🙌 Acknowledgements

* [MediaPipe](https://mediapipe.dev/) – For robust pose tracking.
* [Streamlit](https://streamlit.io/) – For the interactive dashboard.
* My mentors and peers for their support during this final-year journey.

---

## 🡩‍💼 Author

**Muhammad Tayyab Shafique**
📧 [tayyab.shafique06@gmail.com](mailto:tayyab.shafique06@gmail.com)
🔗 [GitHub](https://github.com/M-Tayyab06) | [LinkedIn](https://www.linkedin.com/in/muhammad-tayyab06)

---

## ⭐ Star the Repository

If you found this helpful or inspiring, consider giving it a ⭐ on [GitHub](https://github.com/M-Tayyab06/Final-Year-Project-FYP) — it helps more people find this work!
