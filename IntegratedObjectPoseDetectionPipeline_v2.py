import cv2
import mediapipe as mp
from ultralytics import YOLO
import json
import os
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

input_video = "videoplayback (2) (online-video-cutter.com).mp4"

output_video = "output_scene_v2.mp4"
scene_json = "scene_data.json"
scene_summary = "scene_summary.txt"

USE_MEDIAPIPE = True
USE_YOLO = True

# Analyze the scene every N frames
SCENE_INTERVAL = 15

# YOLO confidence
YOLO_CONFIDENCE = 0.40


# ============================================================
# MEDIAPIPE
# ============================================================

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) if USE_MEDIAPIPE else None


# ============================================================
# YOLO
# ============================================================

yolo_model = YOLO("yolov8n.pt") if USE_YOLO else None


# ============================================================
# VIDEO
# ============================================================

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print(f"❌ ERROR: Could not open video: {input_video}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 60)
print("VIDEO SCENE INTELLIGENCE - V2")
print("=" * 60)
print(f"Input:       {input_video}")
print(f"Resolution:  {frame_width} x {frame_height}")
print(f"FPS:         {fps:.2f}")
print(f"Frames:      {frame_count}")
print("=" * 60)


# ============================================================
# OUTPUT VIDEO
# ============================================================

out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)


# ============================================================
# SCENE STORAGE
# ============================================================

scene_data = []

frame_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    ret, original_frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Keep original frame clean for AI models
    # --------------------------------------------------------

    clean_frame = original_frame.copy()

    # Output frame is where we draw everything
    frame = original_frame.copy()


    # ========================================================
    # YOLO OBJECT DETECTION
    # ========================================================

    detected_objects = []

    if yolo_model:

        yolo_results = yolo_model(
            clean_frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        result = yolo_results[0]

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = yolo_model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detected_objects.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2]
                })

        # Draw YOLO detections
        frame = result.plot()


    # ========================================================
    # MEDIAPIPE POSE
    # ========================================================

    pose_detected = False
    pose_landmarks = []

    if pose:

        # IMPORTANT:
        # MediaPipe receives CLEAN frame, not YOLO-drawn frame
        rgb_frame = cv2.cvtColor(
            clean_frame,
            cv2.COLOR_BGR2RGB
        )

        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:

            pose_detected = True

            landmarks = pose_results.pose_landmarks.landmark

            for landmark in landmarks:

                pose_landmarks.append({
                    "x": round(landmark.x, 4),
                    "y": round(landmark.y, 4),
                    "z": round(landmark.z, 4),
                    "visibility": round(
                        landmark.visibility,
                        4
                    )
                })

            # Draw pose
            landmark_spec = mp_draw.DrawingSpec(
                color=(255, 255, 255),
                thickness=2,
                circle_radius=2
            )

            connection_spec = mp_draw.DrawingSpec(
                color=(255, 255, 255),
                thickness=2
            )

            mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_spec,
                connection_spec
            )


    # ========================================================
    # SCENE ANALYSIS
    # ========================================================

    if frame_number % SCENE_INTERVAL == 0:

        object_counter = Counter(
            obj["class"]
            for obj in detected_objects
        )

        timestamp = frame_number / fps

        scene = {
            "timestamp": round(timestamp, 2),

            "frame": frame_number,

            "objects": dict(object_counter),

            "detections": detected_objects,

            "pose_detected": pose_detected,

            "pose_landmarks": pose_landmarks
        }

        scene_data.append(scene)

        print(
            f"[{timestamp:7.2f}s] "
            f"Objects: {dict(object_counter)} "
            f"| Pose: {pose_detected}"
        )


    # ========================================================
    # WRITE OUTPUT VIDEO
    # ========================================================

    out.write(frame)


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

if pose:
    pose.close()


# ============================================================
# SAVE SCENE DATA
# ============================================================

with open(scene_json, "w", encoding="utf-8") as f:

    json.dump(
        scene_data,
        f,
        indent=4
    )


# ============================================================
# CREATE BASIC NATURAL LANGUAGE SUMMARY
# ============================================================

all_objects = Counter()

pose_frames = 0

for scene in scene_data:

    for object_name, count in scene["objects"].items():

        all_objects[object_name] += count

    if scene["pose_detected"]:
        pose_frames += 1


summary_parts = []

if all_objects:

    object_text = ", ".join(
        f"{count} {name}"
        for name, count in all_objects.items()
    )

    summary_parts.append(
        f"The video contains detections of {object_text}."
    )

if pose_frames > 0:

    summary_parts.append(
        "Human pose landmarks were detected in multiple "
        "parts of the video."
    )

if not summary_parts:

    summary_parts.append(
        "No significant objects or human poses were detected."
    )


final_summary = " ".join(summary_parts)


with open(scene_summary, "w", encoding="utf-8") as f:

    f.write(final_summary)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("✅ PROCESSING COMPLETE")
print("=" * 60)

print(f"Video output : {output_video}")
print(f"Scene data   : {scene_json}")
print(f"Summary      : {scene_summary}")

print()
print("🧠 SCENE SUMMARY")
print(final_summary)

print("=" * 60)