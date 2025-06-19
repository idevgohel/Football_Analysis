
# ⚽ Football Analysis - Tactical Analysis through Object Detection and Tracking

- This repository showcases a project applying computer vision and machine learning to extract advanced tactical and spatial insights from football match footage.

---

## 📌 Description

- The notebook presents an end-to-end pipeline that processes raw match videos to deliver intelligent insights about player dynamics, team formations, ball possession trends, and more. It incorporates object detection, tracking, and spatial transformation techniques to project the game onto a top-down pitch for better analysis.

---

## 🧠 Challenges

- Accurate detection and tracking under frequent occlusions and rapid motion
- Differentiating teams based on jersey colors under variable lighting
- Mapping dynamic camera angles to a fixed top-down perspective
- Real-time processing constraints when running inference on videos

---

## 🗃 Dataset Link

- Ball, Players, and Referees detection dataset: [Roboflow Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- Pitch Keypoints detection dataset: [Roboflow Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)

---

## 📁 File Description

| File/Folder                 | Description |
|----------------------------|-------------|
| `football_analysis.ipynb` | Main notebook containing all code for preprocessing, detection, tracking, homography, and visualization. |
| `train_ball_players_referees.ipynb` | Notebook for training an object detection model to identify footballs, players, and referees. |
| `train_pitch_keypoints.ipynb` | Notebook for training a keypoint detection model to identify pitch features such as field lines and corners. |

---

## 📹 Output Videos

The notebook generates annotated video outputs showing:

- Detected players, ball, goalkeepers, and referees
- Unique player IDs tracked across frames
- Team coloring and top-down mapped positions
- Optional: Voronoi diagrams and trajectory plots

---

## 🔄 Project Flow

- Data Acquisition: Download `.mp4` match footage via `gdown`.
- Preprocessing: Extract frames and resize for consistency.
- Object Detection & Tracking: Use a Roboflow-trained model with `inference-gpu` and DeepSORT for real-time detection and ID tracking.
- Team Classification: Perform K-Means clustering on jersey colors to group players into teams.
- Pitch Homography: Apply manual reference point mapping using OpenCV’s `cv2.getPerspectiveTransform`.
- Spatial & Temporal Analysis: Generate Voronoi diagrams showing influence zones, Track and plot ball trajectories, Visualize in a broadcast or top-down perspective

---

## 🔧 Setup Instructions

- API Keys:  
   - [HuggingFace Access Token](https://huggingface.co/settings) → `HF_TOKEN`  
   - [Roboflow API Key](https://app.roboflow.com/settings/api) → `ROBOFLOW_API_KEY`
- Colab Settings:
   - Use GPU Runtime: `Runtime > Change runtime type > GPU`
- Dependencies:
   - Install essential packages like `inference-gpu`, `opencv-python`, `matplotlib`, and `scikit-learn`.
