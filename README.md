Integrated Object & Pose Detection Pipeline

An end-to-end computer vision pipeline that combines YOLO object detection, MediaPipe human pose estimation, scene reasoning, and a local LLM to transform video into structured scene understanding and natural-language responses.

The project is being developed incrementally, from low-level visual detection to scene narration and video question answering.

✨ Features

🎯 Object Detection using Ultralytics YOLO

🧍 Human Pose Estimation using MediaPipe

🧩 Structured Scene Representation from sampled video frames

🧠 Temporal Scene Reasoning over multi-second video segments

📝 Natural-Language Scene Narration using a local LLM

💬 Video Q&A using the generated scene representation

🔒 Local LLM inference through Ollama

⚡ Designed to work without requiring a cloud AI API

🏗️ Pipeline Architecture

                    Input Video
                         │
                         ▼
              ┌─────────────────────┐
              │ YOLO Object         │
              │ Detection           │
              └──────────┬──────────┘
                         │
                         │
              ┌──────────▼──────────┐
              │ MediaPipe Pose      │
              │ Estimation           │
              └──────────┬──────────┘
                         │
                         ▼
              Structured Scene Data
                         │
                         ▼
              ┌─────────────────────┐
              │ Scene Reasoner      │
              │ Temporal Analysis   │
              └──────────┬──────────┘
                         │
                         ▼
                scene_reasoning.json
                         │
                         ▼
              ┌─────────────────────┐
              │ Local LLM           │
              │ Qwen3 4B Instruct   │
              │ via Ollama           │
              └──────────┬──────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Scene Narration             Video Q&A

📂 Project Structure

Integrated-Object-Pose-Detection-Pipeline/
│
├── IntegratedObjectPoseDetectionPipeline.py
├── IntegratedObjectPoseDetectionPipeline_v2.py
│
├── SceneReasoner_v3.py
├── SceneNarrator_v4.py
├── VideoQA_v5.py
│
├── requirements.txt
├── README.md
├── LICENSE
│
└── .gitignore

Generated files such as processed videos, intermediate JSON files, model weights, and the Python virtual environment are intentionally excluded from the repository.

🔧 Technologies Used

Technology

Purpose

Python

Core programming language

OpenCV

Video processing and frame handling

Ultralytics YOLO

Object detection

MediaPipe

Human pose estimation

NumPy

Numerical processing

Requests

Communication with the local Ollama API

Ollama

Local LLM runtime

Qwen3 4B Instruct

Scene narration / question answering

🚀 Installation

1. Clone the repository

git clone https://github.com/YOUR_USERNAME/Integrated-Object-Pose-Detection-Pipeline.git
cd Integrated-Object-Pose-Detection-Pipeline

2. Create a virtual environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\Activate.ps1

3. Install Python dependencies

pip install -r requirements.txt

🤖 Install Ollama

The scene narration and Video Q&A components use a local LLM through Ollama.

Install Ollama for your operating system, then download the model:

ollama run qwen3:4b-instruct

Verify that Ollama is available:

ollama list

The Python application communicates with the local Ollama API.

▶️ Usage

Step 1 — Run the object + pose pipeline

python IntegratedObjectPoseDetectionPipeline_v2.py

This processes the input video and generates structured detection information.

Step 2 — Run scene reasoning

python SceneReasoner_v3.py

This groups observations into temporal segments and produces:

scene_reasoning.json

Step 3 — Generate a natural-language description

Make sure Ollama is running and the Qwen model is available:

ollama run qwen3:4b-instruct

Then run:

python SceneNarrator_v4.py

The generated narration is saved to:

scene_narration.txt

Step 4 — Video Q&A

Run:

python VideoQA_v5.py

The system uses the structured scene information to answer questions about the processed video.

🧪 Current Output

The pipeline currently extracts information such as:

Approximate number of people visible in different time segments

Objects observed across the video

Objects that persist across segments

Human pose presence

Temporal changes in scene observations

Natural-language summaries generated from the structured information

Example:

OVERALL SCENE:
A scene with people and several persistent objects, including
backpacks, handbags, traffic lights, and cars, is observed
over a period of time.

TIMELINE:
The observed number of people changes across the video,
while several objects remain visible during different
time segments.

IMPORTANT OBSERVATIONS:
- Multiple people are visible throughout the video.
- Traffic lights persist across multiple segments.
- Other objects appear or disappear over time.

⚠️ Current Limitations

This project is still under active development.

Person counting

The current system reports observed people per sampled frame, not unique identities.

For example:

10 people detected

does not mean that exactly 10 unique people appeared in the entire video.

Multi-person pose estimation

The current MediaPipe configuration provides pose landmarks for the detected pose representation but does not yet establish reliable individual pose tracking for every YOLO-detected person.

Object persistence

Objects are currently analyzed using temporal observations rather than robust identity tracking.

Scene understanding

The local LLM is constrained to the structured observations supplied to it. It should not be treated as a replacement for direct frame-level visual reasoning.

🛣️ Roadmap

Planned improvements include:

Multi-object tracking

Persistent person/object IDs

Person-object association

Better temporal event detection

Scene graph generation

Improved action recognition

Direct video-to-Q&A interaction

Better long-video memory

Web-based user interface

Real-time processing

More efficient CPU inference

Optional vision-language model integration

📌 Development Versions

The project is developed in stages:

V1 → Basic YOLO + MediaPipe integration
V2 → Structured scene data extraction
V3 → Temporal scene reasoning
V4 → Local LLM scene narration
V5 → Video Question Answering

The versioned scripts are intentionally kept in the repository so the evolution of the system can be tracked.

🔐 Privacy

The LLM component is designed to run locally through Ollama. Video data and scene information do not need to be sent to a cloud LLM API for the current implementation.

📄 License

This project is licensed under the Apache License 2.0.

See the LICENSE file for the full license text.

👤 Author

Avinash Raj

B.Tech Information Technology

GitHub: https://github.com/RajAvinash7

⭐ If you find the project useful, feel free to star the repository.
