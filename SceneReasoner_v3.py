import json
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "scene_data.json"
OUTPUT_FILE = "scene_reasoning.json"

SEGMENT_DURATION = 5.0


# ============================================================
# LOAD SCENE DATA
# ============================================================

try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        scene_data = json.load(f)

except FileNotFoundError:
    print(f"❌ ERROR: {INPUT_FILE} not found.")
    exit()

except json.JSONDecodeError:
    print(f"❌ ERROR: {INPUT_FILE} is not valid JSON.")
    exit()


if not scene_data:
    print("❌ ERROR: Scene data is empty.")
    exit()


print("=" * 65)
print("SCENE REASONER V3")
print("=" * 65)

print(f"Loaded {len(scene_data)} scene samples.")


# ============================================================
# DETERMINE VIDEO DURATION
# ============================================================

video_duration = max(
    scene["timestamp"]
    for scene in scene_data
)


# ============================================================
# CREATE SEGMENTS
# ============================================================

segments = []

segment_start = 0.0

while segment_start < video_duration:

    segment_end = min(
        segment_start + SEGMENT_DURATION,
        video_duration
    )

    segment_frames = [
        scene
        for scene in scene_data
        if segment_start <= scene["timestamp"] < segment_end
    ]

    if segment_frames:

        segments.append({
            "start": round(segment_start, 2),
            "end": round(segment_end, 2),
            "frames": segment_frames
        })

    segment_start += SEGMENT_DURATION


# ============================================================
# ANALYZE EACH SEGMENT
# ============================================================

processed_segments = []


for segment in segments:

    frames = segment["frames"]

    # --------------------------------------------------------
    # PEOPLE COUNT
    # --------------------------------------------------------

    people_counts = []

    for scene in frames:

        count = scene.get(
            "objects", {}
        ).get("person", 0)

        people_counts.append(count)


    min_people = min(people_counts)
    max_people = max(people_counts)
    avg_people = sum(people_counts) / len(people_counts)


    # --------------------------------------------------------
    # OBJECT ANALYSIS
    # --------------------------------------------------------

    object_counts = defaultdict(list)

    for scene in frames:

        objects = scene.get("objects", {})

        for object_name, count in objects.items():

            if object_name == "person":
                continue

            object_counts[object_name].append(count)


    object_summary = {}


    for object_name, counts in object_counts.items():

        object_summary[object_name] = {
            "max_count": max(counts),
            "min_count": min(counts),
            "average_count": round(
                sum(counts) / len(counts),
                2
            ),

            # Number of sampled frames in which
            # this object was detected
            "frames_detected": len(counts),

            # Fraction of segment samples
            # containing this object
            "persistence": round(
                len(counts) / len(frames),
                2
            )
        }


    # --------------------------------------------------------
    # PERSISTENT OBJECTS
    # --------------------------------------------------------

    persistent_objects = [
        object_name
        for object_name, info
        in object_summary.items()
        if info["persistence"] >= 0.5
    ]


    # --------------------------------------------------------
    # POSE ANALYSIS
    # --------------------------------------------------------

    pose_frames = sum(
        1
        for scene in frames
        if scene.get("pose_detected", False)
    )

    pose_persistence = pose_frames / len(frames)


    # --------------------------------------------------------
    # CREATE SEGMENT DESCRIPTION
    # --------------------------------------------------------

    description_parts = []


    # People
    if min_people == max_people:

        description_parts.append(
            f"Approximately {max_people} people "
            f"were detected."
        )

    else:

        description_parts.append(
            f"Approximately {min_people} to "
            f"{max_people} people were detected."
        )


    # Objects
    if persistent_objects:

        readable_objects = ", ".join(
            persistent_objects
        )

        description_parts.append(
            f"Persistent objects included "
            f"{readable_objects}."
        )


    # Pose
    if pose_persistence >= 0.5:

        description_parts.append(
            "Human pose landmarks were detected "
            "throughout most of this segment."
        )

    elif pose_frames > 0:

        description_parts.append(
            "Human pose landmarks were detected "
            "intermittently."
        )


    segment_description = " ".join(
        description_parts
    )


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    processed_segments.append({

        "start": segment["start"],

        "end": segment["end"],

        "people": {
            "minimum": min_people,
            "maximum": max_people,
            "average": round(avg_people, 2)
        },

        "objects": object_summary,

        "persistent_objects": persistent_objects,

        "pose": {
            "frames_detected": pose_frames,
            "total_samples": len(frames),
            "persistence": round(
                pose_persistence,
                2
            )
        },

        "description": segment_description
    })


# ============================================================
# GLOBAL VIDEO ANALYSIS
# ============================================================

global_object_presence = Counter()

global_object_maximums = defaultdict(int)

global_people_counts = []


for scene in scene_data:

    objects = scene.get(
        "objects", {}
    )

    # People
    global_people_counts.append(
        objects.get("person", 0)
    )

    # Other objects
    for object_name, count in objects.items():

        if object_name == "person":
            continue

        global_object_presence[object_name] += 1

        global_object_maximums[object_name] = max(
            global_object_maximums[object_name],
            count
        )


# ============================================================
# GLOBAL OBJECT SUMMARY
# ============================================================

global_objects = {}

total_samples = len(scene_data)


for object_name in global_object_presence:

    presence = (
        global_object_presence[object_name]
        / total_samples
    )

    global_objects[object_name] = {

        "sample_presence": round(
            presence,
            2
        ),

        "maximum_count": global_object_maximums[
            object_name
        ]
    }


# ============================================================
# GLOBAL VIDEO SUMMARY
# ============================================================

overall_min_people = min(
    global_people_counts
)

overall_max_people = max(
    global_people_counts
)

overall_average_people = (
    sum(global_people_counts)
    / len(global_people_counts)
)


result = {

    "video": {

        "duration_seconds": round(
            video_duration,
            2
        ),

        "sample_count": len(scene_data),

        "segment_duration": SEGMENT_DURATION
    },


    "overall_scene": {

        "people": {

            "minimum_observed": overall_min_people,

            "maximum_observed": overall_max_people,

            "average_observed": round(
                overall_average_people,
                2
            )
        },

        "objects": global_objects
    },


    "segments": processed_segments
}


# ============================================================
# SAVE RESULT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 65)
print("VIDEO OVERVIEW")
print("=" * 65)

print(
    f"Duration: {video_duration:.2f} seconds"
)

print(
    f"Observed people range: "
    f"{overall_min_people} - "
    f"{overall_max_people}"
)

print(
    f"Average people per sample: "
    f"{overall_average_people:.2f}"
)


print()
print("=" * 65)
print("SEGMENT ANALYSIS")
print("=" * 65)


for segment in processed_segments:

    print()
    print(
        f"[{segment['start']:.1f}s - "
        f"{segment['end']:.1f}s]"
    )

    print(
        f"People: "
        f"{segment['people']['minimum']} - "
        f"{segment['people']['maximum']} "
        f"(avg "
        f"{segment['people']['average']})"
    )

    print(
        "Persistent objects:",
        segment["persistent_objects"]
    )

    print(
        "Pose persistence:",
        segment["pose"]["persistence"]
    )

    print(
        "Description:",
        segment["description"]
    )


print()
print("=" * 65)
print("✅ SCENE REASONING COMPLETE")
print("=" * 65)

print(
    f"Saved LLM-ready scene data to: "
    f"{OUTPUT_FILE}"
)

print("=" * 65)