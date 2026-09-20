import json
import requests
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "scene_reasoning.json"

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "qwen3:4b-instruct"

TIMEOUT = 600


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("VIDEO Q&A V5")
print("=" * 70)


# ============================================================
# LOAD SCENE REASONING
# ============================================================

try:

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        scene_data = json.load(file)

except FileNotFoundError:

    print(
        f"\n❌ ERROR: {INPUT_FILE} not found."
    )

    print(
        "Run SceneReasoner_v3.py first."
    )

    exit()

except json.JSONDecodeError as error:

    print("\n❌ Invalid JSON file.")

    print(error)

    exit()


print(
    f"\nLoaded: {INPUT_FILE}"
)


# ============================================================
# EXTRACT SEGMENTS
# ============================================================

if isinstance(scene_data, dict):

    segments = (
        scene_data.get("segments")
        or scene_data.get("processed_segments")
        or scene_data.get("scene_segments")
        or []
    )

else:

    segments = []


print(
    f"Loaded {len(segments)} video segments."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_value(dictionary, keys, default=None):

    """
    Return the first available key.
    """

    if not isinstance(dictionary, dict):
        return default

    for key in keys:

        if key in dictionary:

            return dictionary[key]

    return default


def convert_to_float(value):

    """
    Safely convert a value to float.
    """

    try:

        if isinstance(value, (int, float)):

            return float(value)

        match = re.search(
            r"-?\d+(?:\.\d+)?",
            str(value)
        )

        if match:

            return float(match.group())

    except Exception:

        pass

    return None


def get_segment_time(segment):

    """
    Extract start/end time from a segment.
    """

    start = get_value(
        segment,
        [
            "start",
            "start_time",
            "start_seconds",
            "start_timestamp"
        ],
        0
    )

    end = get_value(
        segment,
        [
            "end",
            "end_time",
            "end_seconds",
            "end_timestamp"
        ],
        0
    )

    start = convert_to_float(start)
    end = convert_to_float(end)

    if start is None:
        start = 0

    if end is None:
        end = start

    return start, end


def get_people_data(segment):

    """
    Extract people statistics.
    """

    people = get_value(
        segment,
        [
            "people",
            "person",
            "people_count"
        ],
        {}
    )

    if not isinstance(people, dict):

        return {
            "raw": people
        }

    return {

        "minimum": get_value(
            people,
            [
                "min",
                "minimum",
                "min_people"
            ]
        ),

        "maximum": get_value(
            people,
            [
                "max",
                "maximum",
                "max_people"
            ]
        ),

        "average": get_value(
            people,
            [
                "avg",
                "average",
                "average_people"
            ]
        )
    }


def get_objects(segment):

    """
    Extract persistent objects.
    """

    objects = get_value(
        segment,
        [
            "persistent_objects",
            "persistent_object",
            "objects"
        ],
        []
    )

    if isinstance(objects, list):

        return objects

    if isinstance(objects, str):

        return [objects]

    return []


def get_description(segment):

    return get_value(
        segment,
        [
            "description",
            "scene_description",
            "summary"
        ],
        ""
    )


# ============================================================
# NORMALIZE ALL SEGMENTS
# ============================================================

normalized_segments = []


for index, segment in enumerate(
    segments,
    start=1
):

    if not isinstance(segment, dict):

        continue

    start, end = get_segment_time(
        segment
    )

    people = get_people_data(
        segment
    )

    objects = get_objects(
        segment
    )

    description = get_description(
        segment
    )

    normalized_segments.append({

        "segment_number": index,

        "start": start,

        "end": end,

        "people": people,

        "objects": objects,

        "description": description
    })


# ============================================================
# DISPLAY SEGMENT INFORMATION
# ============================================================

print("\nVideo timeline:")

for segment in normalized_segments:

    print(
        f"  Segment {segment['segment_number']}: "
        f"{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s"
    )


# ============================================================
# QUESTION TIME EXTRACTION
# ============================================================

def extract_time_range(question):

    """
    Detect simple time references such as:

    5 to 10 seconds
    between 5 and 10 seconds
    from 5-10 seconds
    at 10 seconds
    """

    question_lower = question.lower()

    # --------------------------------------------------------
    # "between X and Y"
    # --------------------------------------------------------

    match = re.search(
        r"(?:between|from)\s+"
        r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?"
        r"\s*(?:and|to|-)\s*"
        r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?",
        question_lower
    )

    if match:

        return (
            float(match.group(1)),
            float(match.group(2))
        )


    # --------------------------------------------------------
    # "X to Y seconds"
    # --------------------------------------------------------

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:to|-)\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:s|sec|seconds)",
        question_lower
    )

    if match:

        return (
            float(match.group(1)),
            float(match.group(2))
        )


    # --------------------------------------------------------
    # "at X seconds"
    # --------------------------------------------------------

    match = re.search(
        r"(?:at|around)\s+"
        r"(\d+(?:\.\d+)?)\s*"
        r"(?:s|sec|seconds)",
        question_lower
    )

    if match:

        time_point = float(
            match.group(1)
        )

        return (
            time_point,
            time_point
        )


    return None


# ============================================================
# SELECT RELEVANT SEGMENTS
# ============================================================

def retrieve_segments(question):

    """
    Retrieve segments relevant to the user's question.
    """

    time_range = extract_time_range(
        question
    )


    # --------------------------------------------------------
    # TIME-BASED RETRIEVAL
    # --------------------------------------------------------

    if time_range:

        query_start, query_end = time_range

        selected = []

        for segment in normalized_segments:

            segment_start = segment["start"]
            segment_end = segment["end"]

            # Check whether ranges overlap
            if (
                segment_end >= query_start
                and
                segment_start <= query_end
            ):

                selected.append(segment)


        if selected:

            return selected


    # --------------------------------------------------------
    # KEYWORD RETRIEVAL
    # --------------------------------------------------------

    question_lower = question.lower()


    keywords = [

        "people",
        "person",
        "human",

        "backpack",
        "handbag",
        "traffic light",
        "car",
        "skateboard",
        "cup",

        "object",
        "objects",

        "pose",
        "landmark"
    ]


    matching_keywords = [

        keyword
        for keyword in keywords
        if keyword in question_lower
    ]


    if matching_keywords:

        selected = []

        for segment in normalized_segments:

            segment_text = json.dumps(
                segment
            ).lower()

            if any(
                keyword in segment_text
                for keyword in matching_keywords
            ):

                selected.append(
                    segment
                )


        if selected:

            return selected


    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return normalized_segments


# ============================================================
# BUILD FACT EVIDENCE
# ============================================================

def build_evidence(question):

    relevant_segments = retrieve_segments(
        question
    )

    evidence = {

        "question": question,

        "relevant_segments": relevant_segments
    }

    return evidence


# ============================================================
# OLLAMA FUNCTION
# ============================================================

def ask_qwen(question, evidence):

    """
    Send retrieved evidence to Qwen.
    """

    system_prompt = """
You are a factual Video Question Answering assistant.

Answer the user's question ONLY from the supplied video evidence.

The evidence comes from structured observations.

IMPORTANT RULES:

1. Never invent information.

2. Never assume an action unless the evidence explicitly
   states that action.

3. Do not claim exact unique person counts.
   People counts are observations from sampled frames.

4. If the evidence does not contain enough information,
   say:

   "The available video data does not provide enough
   information to determine that."

5. Do not use outside knowledge.

6. Do not mention YOLO, MediaPipe, JSON, AI,
   machine learning, models, or computer vision.

7. Answer directly.

8. Keep the answer concise.

9. If the question asks about a time range, use only
   the relevant segments.

10. Do not make assumptions about location, identity,
    movement, emotion, or relationships.
"""


    user_prompt = f"""
/no_think

USER QUESTION:

{question}


VIDEO EVIDENCE:

{json.dumps(
    evidence,
    indent=2,
    ensure_ascii=False
)}


Answer the user's question directly.

Do not explain your reasoning.
"""


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

        "stream": False,

        "think": False,

        "options": {

            "temperature": 0.1,

            "num_predict": 180
        }
    }


    start = time.time()


    try:

        response = requests.post(

            OLLAMA_URL,

            json=payload,

            timeout=TIMEOUT
        )

        response.raise_for_status()


    except requests.exceptions.ConnectionError:

        print(
            "\n❌ Cannot connect to Ollama."
        )

        print(
            "Run: ollama run qwen3:4b-instruct"
        )

        return None


    except requests.exceptions.Timeout:

        print(
            "\n❌ Ollama request timed out."
        )

        return None


    except requests.exceptions.RequestException as error:

        print(
            "\n❌ Ollama error:"
        )

        print(error)

        return None


    elapsed = time.time() - start


    print(
        f"\nResponse received in "
        f"{elapsed:.1f} seconds."
    )


    try:

        result = response.json()

    except Exception:

        print(
            "\n❌ Invalid Ollama response."
        )

        return None


    message = result.get(
        "message",
        {}
    )


    answer = message.get(
        "content",
        ""
    ).strip()


    if not answer:

        print(
            "\n❌ Empty answer from Qwen."
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        return None


    return answer


# ============================================================
# ASK QUESTION
# ============================================================

def process_question(question):

    print("\n" + "-" * 70)

    print(
        "QUESTION:"
    )

    print(question)


    # --------------------------------------------------------
    # RETRIEVE EVIDENCE
    # --------------------------------------------------------

    evidence = build_evidence(
        question
    )


    relevant_segments = evidence[
        "relevant_segments"
    ]


    print(
        f"\nRetrieved "
        f"{len(relevant_segments)} relevant segment(s)."
    )


    # --------------------------------------------------------
    # ASK QWEN
    # --------------------------------------------------------

    answer = ask_qwen(
        question,
        evidence
    )


    if answer is None:

        return


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "VIDEO ANSWER"
    )

    print("=" * 70)

    print()

    print(answer)

    print("=" * 70)


# ============================================================
# INTERACTIVE LOOP
# ============================================================

print("\n" + "=" * 70)

print(
    "VIDEO Q&A READY"
)

print("=" * 70)

print(
    "\nAsk questions about the processed video."
)

print(
    "Type 'exit' or 'quit' to stop."
)

print("\nExample questions:")

print(
    '  What objects were present between 5 and 15 seconds?'
)

print(
    '  How many people were observed in the first segment?'
)

print(
    '  Which objects persisted in the video?'
)

print(
    '  What changed between 10 and 15 seconds?'
)

print(
    '  Was a car observed in the video?'
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    try:

        question = input(
            "\nYou: "
        ).strip()

    except KeyboardInterrupt:

        print(
            "\n\nExiting..."
        )

        break


    if not question:

        continue


    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print(
            "\nVideo Q&A stopped."
        )

        break


    process_question(
        question
    )