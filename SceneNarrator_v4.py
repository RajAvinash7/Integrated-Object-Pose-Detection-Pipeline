import json
import requests
import time


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "scene_reasoning.json"
OUTPUT_FILE = "scene_narration.txt"

OLLAMA_URL = "http://localhost:11434/api/chat"

# New instruct model
MODEL = "qwen3:4b-instruct"

# Maximum time to wait for Ollama
TIMEOUT = 600


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SCENE NARRATOR V4.3")
print("=" * 70)


# ============================================================
# LOAD SCENE REASONING
# ============================================================

print(f"\nLoading: {INPUT_FILE}")

try:

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        scene_data = json.load(file)

except FileNotFoundError:

    print(f"\n❌ ERROR: {INPUT_FILE} was not found.")
    print("Run SceneReasoner_v3.py first.")
    exit()

except json.JSONDecodeError as error:

    print("\n❌ ERROR: Invalid JSON file.")
    print(error)
    exit()


print("Scene reasoning loaded successfully.")


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_first(dictionary, keys, default=None):
    """
    Return the first existing key from a dictionary.
    """

    for key in keys:

        if key in dictionary:
            return dictionary[key]

    return default


# ============================================================
# EXTRACT VIDEO OVERVIEW
# ============================================================

video_overview = {}


# Possible overview keys
if isinstance(scene_data, dict):

    for key in [
        "overview",
        "video_overview",
        "video_summary",
        "global_summary"
    ]:

        if key in scene_data:

            video_overview = scene_data[key]

            break


# ============================================================
# EXTRACT SEGMENTS
# ============================================================

segments = []


if isinstance(scene_data, dict):

    possible_segment_keys = [
        "segments",
        "processed_segments",
        "scene_segments"
    ]

    for key in possible_segment_keys:

        if key in scene_data:

            if isinstance(scene_data[key], list):

                segments = scene_data[key]

            break


# ============================================================
# BUILD COMPACT REPRESENTATION
# ============================================================

compact_segments = []


for index, segment in enumerate(segments, start=1):

    if not isinstance(segment, dict):
        continue


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    start_time = get_first(
        segment,
        [
            "start",
            "start_time",
            "start_seconds",
            "start_timestamp"
        ],
        "unknown"
    )


    end_time = get_first(
        segment,
        [
            "end",
            "end_time",
            "end_seconds",
            "end_timestamp"
        ],
        "unknown"
    )


    # --------------------------------------------------------
    # PEOPLE
    # --------------------------------------------------------

    people_data = get_first(
        segment,
        [
            "people",
            "person",
            "people_count"
        ],
        {}
    )


    # --------------------------------------------------------
    # PERSISTENT OBJECTS
    # --------------------------------------------------------

    persistent_objects = get_first(
        segment,
        [
            "persistent_objects",
            "persistent_object",
            "objects"
        ],
        []
    )


    # --------------------------------------------------------
    # POSE
    # --------------------------------------------------------

    pose_data = get_first(
        segment,
        [
            "pose",
            "pose_data"
        ],
        {}
    )


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    description = get_first(
        segment,
        [
            "description",
            "scene_description",
            "summary"
        ],
        ""
    )


    # --------------------------------------------------------
    # STORE COMPACT SEGMENT
    # --------------------------------------------------------

    compact_segment = {

        "segment": index,

        "time": f"{start_time} - {end_time}",

        "people": people_data,

        "persistent_objects": persistent_objects,

        "pose": pose_data,

        "description": description
    }


    compact_segments.append(compact_segment)


# ============================================================
# BUILD COMPACT SCENE DATA
# ============================================================

compact_scene = {

    "video_overview": video_overview,

    "segments": compact_segments
}


# ============================================================
# CONVERT TO JSON TEXT
# ============================================================

scene_text = json.dumps(
    compact_scene,
    indent=2,
    ensure_ascii=False
)


print(
    f"\nCompact scene information: "
    f"{len(scene_text):,} characters"
)

print(
    f"Segments supplied to LLM: "
    f"{len(compact_segments)}"
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

system_prompt = """
You are a video scene description generator.

Your job is to convert structured observations from a video
into a short, factual natural-language description.

IMPORTANT:

OUTPUT ONLY THE FINAL DESCRIPTION.

DO NOT show reasoning.

DO NOT explain your process.

DO NOT say:
- "We are given..."
- "Let's analyze..."
- "Let's break this down..."
- "Step 1..."
- "We need to..."
- "The data shows that we need..."
- "I will..."
- "Based on the instructions..."

Do not repeat the instructions.

Do not output analysis.

Use exactly these three sections:

OVERALL SCENE:
Write one short paragraph describing the overall scene.

TIMELINE:
Write one short paragraph describing the important changes
between the video segments.

IMPORTANT OBSERVATIONS:
- Write one factual observation.
- Write one factual observation.
- Write one factual observation.

STRICT FACTUAL RULES:

1. Use ONLY information present in the supplied data.

2. Never invent objects, people, actions, locations,
   identities, relationships, emotions, or events.

3. Do NOT claim an exact number of unique people.
   The people numbers are observations from sampled frames.

4. Use approximate wording for people counts:
   "approximately 7 to 15 people"
   rather than claiming there are exactly 15 unique people.

5. Persistent objects may be mentioned when supported
   by the supplied data.

6. Do not claim that an object is present throughout the
   entire video unless the data supports that.

7. Do not infer actions.
   For example, detecting a person does NOT mean that the
   person is walking, running, standing, talking, etc.

8. Do not invent locations such as "street", "road",
   "city", "park", "building", etc. unless the data explicitly
   provides that information.

9. Do not invent object behavior such as "pulsing",
   "moving", "approaching", or "leaving".

10. Do not mention:
    YOLO
    MediaPipe
    JSON
    AI
    machine learning
    computer vision
    neural networks
    models
    algorithms
    detection systems

11. Keep the entire answer under 150 words.

12. Be factual rather than creative.
"""


# ============================================================
# USER PROMPT
# ============================================================

user_prompt = f"""
/no_think

Generate the final video description now.

OUTPUT ONLY:

OVERALL SCENE:
...

TIMELINE:
...

IMPORTANT OBSERVATIONS:
- ...
- ...
- ...

Do not explain your reasoning.

Here are the structured observations:

{scene_text}
"""


# ============================================================
# OLLAMA PAYLOAD
# ============================================================

payload = {

    "model": MODEL,

    "messages": [

        {
            "role": "system",
            "content": system_prompt
        },

        {
            "role": "user",
            "content": user_prompt
        }

    ],

    # Return one complete response
    "stream": False,

    # Explicitly disable Qwen thinking
    "think": False,

    "options": {

        # Low randomness
        "temperature": 0.1,

        # Enough for our short narration
        "num_predict": 250
    }
}


# ============================================================
# SEND REQUEST TO OLLAMA
# ============================================================

print("\nSending scene information to Qwen3 4B Instruct...")
print("Thinking mode: OFF")
print("Generating final narration...\n")


start_time = time.time()


try:

    response = requests.post(

        OLLAMA_URL,

        json=payload,

        timeout=TIMEOUT
    )


    response.raise_for_status()


except requests.exceptions.ConnectionError:

    print("\n❌ ERROR: Could not connect to Ollama.")

    print("\nMake sure Ollama is running.")

    print("\nRun:")

    print("    ollama run qwen3:4b-instruct")

    exit()


except requests.exceptions.Timeout:

    print("\n❌ ERROR: Ollama timed out.")

    print(
        f"Ollama did not respond within "
        f"{TIMEOUT} seconds."
    )

    exit()


except requests.exceptions.RequestException as error:

    print("\n❌ ERROR communicating with Ollama:")

    print(error)

    exit()


# ============================================================
# RESPONSE TIME
# ============================================================

elapsed_time = time.time() - start_time

print(
    f"Response received in "
    f"{elapsed_time:.1f} seconds."
)


# ============================================================
# PARSE RESPONSE
# ============================================================

try:

    result = response.json()

except json.JSONDecodeError:

    print("\n❌ ERROR: Ollama returned invalid JSON.")

    print("\nRaw response:")

    print(response.text)

    exit()


# ============================================================
# EXTRACT MESSAGE
# ============================================================

message = result.get(
    "message",
    {}
)


narration = message.get(
    "content",
    ""
).strip()


# ============================================================
# EMPTY RESPONSE CHECK
# ============================================================

if not narration:

    print(
        "\n❌ ERROR: Ollama returned "
        "an empty response."
    )

    print("\nFull Ollama response:")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    exit()


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n")

print("=" * 70)

print("AI GENERATED SCENE DESCRIPTION")

print("=" * 70)

print()

print(narration)

print()

print("=" * 70)


# ============================================================
# SAVE RESULT
# ============================================================

try:

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(narration)

except Exception as error:

    print(
        "\n❌ ERROR saving narration:"
    )

    print(error)

    exit()


# ============================================================
# COMPLETE
# ============================================================

print(
    f"✅ Saved narration to: "
    f"{OUTPUT_FILE}"
)

print("=" * 70)