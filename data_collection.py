import os
import time
import shutil
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ==========================================
# 1. Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACTIONS = np.array(["Hi", "Thank You", "I Love You", "Namaste", "Yes", "No"])
DATA_PATH = os.path.join(BASE_DIR, "ISL_Data")

NO_SEQUENCES = 30      # clips per gesture
SEQUENCE_LENGTH = 30   # frames per clip
POSE_DIM = 132         # first 132 values of a frame are pose, the rest are hands

# A clip is redone automatically if the hands were visible in fewer frames than this.
# Set MIN_HAND_VISIBILITY = 0 to turn the check off.
MIN_HAND_VISIBILITY = 0.6
MAX_RETRIES = 3        # after this many automatic redos the clip is saved anyway

POSE_MODEL = os.path.join(BASE_DIR, "pose_landmarker_full.task")
HAND_MODEL = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URLS = {
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

WINDOW = "ISL Data Collector"

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
RED = (0, 0, 255)


def ensure_models():
    """Download the MediaPipe model files once if they are missing."""
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)} ...")
            urllib.request.urlretrieve(url, path)
    print("Model files ready.")


def make_folders():
    """Not used by the menu any more (folders are created per clip), kept for other scripts."""
    for action in ACTIONS:
        for sequence in range(NO_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


# ==========================================
# 2. MediaPipe Tasks helpers
# ==========================================
_last_ts = 0


def next_timestamp_ms():
    """Timestamps passed to detect_for_video must strictly increase."""
    global _last_ts
    ts = int(time.monotonic() * 1000)
    if ts <= _last_ts:
        ts = _last_ts + 1
    _last_ts = ts
    return ts


def make_detectors():
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
    )
    return pose, hands


def mediapipe_detection(image, detectors):
    """Run pose + hand detection on a BGR frame. Returns (pose_result, hand_result)."""
    pose_det, hand_det = detectors
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = next_timestamp_ms()
    return pose_det.detect_for_video(mp_img, ts), hand_det.detect_for_video(mp_img, ts)


def draw_styled_landmarks(image, results):
    """Draw pose and hand landmarks as dots on the frame."""
    pose_res, hand_res = results
    h, w = image.shape[:2]
    if pose_res.pose_landmarks:
        for lm in pose_res.pose_landmarks[0]:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (80, 44, 121), -1)
    for hand in hand_res.hand_landmarks:
        for lm in hand:
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (245, 117, 66), -1)


def extract_keypoints(results):
    """
    258 values per frame: pose 33*4 (x, y, z, visibility) + left hand 21*3 + right hand 21*3.
    Hand labels come from the camera's point of view. That is fine as long as the
    same extractor is used for collection and for live prediction.
    """
    pose_res, hand_res = results

    pose = np.zeros(33 * 4)
    if pose_res.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_res.pose_landmarks[0]]
        ).flatten()

    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    for landmarks, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        if handedness[0].category_name == "Left":
            lh = arr
        else:
            rh = arr

    return np.concatenate([pose, lh, rh])


def put_text(image, text, org, scale, color, thickness=2):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow is more reliable on Windows
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    return cap


def sequence_done(action, sequence):
    folder = os.path.join(DATA_PATH, action, str(sequence))
    return all(os.path.exists(os.path.join(folder, f"{i}.npy")) for i in range(SEQUENCE_LENGTH))


def clip_counts():
    """Number of complete clips (0..NO_SEQUENCES-1) per sign, in ACTIONS order."""
    return [sum(sequence_done(a, s) for s in range(NO_SEQUENCES)) for a in ACTIONS]


# ==========================================
# 3. On-screen menu
# ==========================================
def draw_panel(frame, lines):
    """Darken the top-left area of the camera view and write colored text lines on it."""
    height = 14 + 30 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (frame.shape[1] - 8, 8 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for i, (text, color) in enumerate(lines):
        put_text(frame, text, (18, 36 + 30 * i), 0.55, color, 1)


def choose_signs(cap):
    """
    Menu shown in the video window.
    Returns (list_of_sign_indexes, overwrite) or None to quit.
    Keys: 1-6 pick/unpick a sign, A all/none, ENTER next, then R = re-record, M = only missing, B = back, Q = quit.
    """
    counts = clip_counts()
    selected = set()
    stage = "pick"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            return None

        if stage == "pick":
            lines = [("SELECT SIGNS TO RECORD", YELLOW)]
            for i, action in enumerate(ACTIONS):
                mark = "[x]" if i in selected else "[ ]"
                lines.append((f"{mark} {i + 1}  {action}   ({counts[i]}/{NO_SEQUENCES} clips)",
                              GREEN if i in selected else WHITE))
            lines.append(("Keys: 1-%d pick | A all | ENTER next | Q quit" % len(ACTIONS), YELLOW))
        else:
            names = ", ".join(str(ACTIONS[i]) for i in sorted(selected))
            lines = [
                (f"Signs: {names}", YELLOW),
                (f"R = RE-RECORD all {NO_SEQUENCES} clips (replaces old ones)", GREEN),
                ("M = record only MISSING clips (keeps existing)", GREEN),
                ("B = back | Q = quit", WHITE),
            ]

        draw_panel(frame, lines)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 255:
            continue
        if key == ord("q"):
            return None

        if stage == "pick":
            if ord("1") <= key <= ord("0") + len(ACTIONS):
                i = key - ord("1")
                selected.symmetric_difference_update({i})
            elif key == ord("a"):
                selected = set() if len(selected) == len(ACTIONS) else set(range(len(ACTIONS)))
            elif key in (13, 10) and selected:
                stage = "mode"
        else:
            if key == ord("r"):
                return sorted(selected), True
            if key == ord("m"):
                return sorted(selected), False
            if key == ord("b"):
                stage = "pick"


def wait_for_continue(cap, finished_action, next_action):
    """Pause between signs. 'p' continues, 'q' goes back to the menu. Returns True to continue."""
    print(f"'{finished_action}' finished. Press 'p' to continue with '{next_action}', or 'q' for the menu.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            return False
        draw_panel(frame, [
            (f"'{finished_action}' DONE - PAUSED", YELLOW),
            (f"Next: '{next_action}' | 'p' = CONTINUE | 'q' = MENU", GREEN),
        ])
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("p"):
            for _ in range(5):  # flush buffered frames
                cap.read()
            return True
        if key == ord("q"):
            return False


# ==========================================
# 4. Recording
# ==========================================
def record_clip(cap, detectors, action, sequence, n, total, min_visibility):
    """
    Records one clip in memory and writes it to disk only when all 30 frames were captured.
    Returns "ok", "redo" (hands not visible enough), "quit" (user pressed q) or "camera" (feed lost).
    """
    ret, frame = cap.read()
    if not ret:
        print("Camera feed lost.")
        return "camera"
    put_text(frame, "GET READY...", (180, 200), 1.2, RED, 3)
    put_text(frame, f"Recording '{action}' | Clip {n}/{total} (#{sequence})", (15, 30), 0.7, YELLOW)
    cv2.imshow(WINDOW, frame)
    if cv2.waitKey(2000) & 0xFF == ord("q"):
        return "quit"

    for _ in range(5):  # flush buffered frames from the pause
        cap.read()

    keypoints = []
    for frame_num in range(SEQUENCE_LENGTH):
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            return "camera"

        results = mediapipe_detection(frame, detectors)
        draw_styled_landmarks(frame, results)
        put_text(
            frame,
            f"Recording '{action}' | Clip {n}/{total} | Frame {frame_num + 1}/{SEQUENCE_LENGTH}",
            (15, 30), 0.6, GREEN,
        )
        cv2.imshow(WINDOW, frame)
        keypoints.append(extract_keypoints(results))

        if cv2.waitKey(10) & 0xFF == ord("q"):
            return "quit"

    clip = np.array(keypoints)
    visible = (np.abs(clip[:, POSE_DIM:]).sum(axis=1) > 0).mean()
    if visible < min_visibility:
        put_text(frame, f"Hands lost ({visible:.0%} visible) - redoing this clip", (15, 60), 0.7, RED)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1200)
        return "redo"

    folder = os.path.join(DATA_PATH, action, str(sequence))
    if os.path.isdir(folder):
        shutil.rmtree(folder)  # replace the old clip completely
    os.makedirs(folder)
    for i, kp in enumerate(clip):
        np.save(os.path.join(folder, f"{i}.npy"), kp)
    return "ok"


def run_recording(cap, detectors, selected, overwrite):
    """Records the chosen signs. Returns "menu" to go back to the menu or "exit" to stop."""
    for pos, idx in enumerate(selected):
        action = ACTIONS[idx]
        todo = [s for s in range(NO_SEQUENCES) if overwrite or not sequence_done(action, s)]
        if not todo:
            print(f"'{action}': all {NO_SEQUENCES} clips already exist, nothing to record.")
            continue

        for n, sequence in enumerate(todo, 1):
            attempt = 0
            while True:
                min_vis = MIN_HAND_VISIBILITY if attempt < MAX_RETRIES else 0.0
                status = record_clip(cap, detectors, action, sequence, n, len(todo), min_vis)
                if status == "redo":
                    attempt += 1
                    continue
                break
            if status == "quit":
                print("Stopped. Back to the menu.")
                return "menu"
            if status == "camera":
                return "exit"

        print(f"'{action}' recorded ({len(todo)} clips).")
        if pos < len(selected) - 1:
            if not wait_for_continue(cap, action, ACTIONS[selected[pos + 1]]):
                return "menu"

    print("Done. Run convert_data.py, then train.py to use the new clips.")
    return "menu"


# ==========================================
# 5. Main
# ==========================================
def collect_data():
    ensure_models()

    cap = open_camera()
    if not cap.isOpened():
        print("Error: Could not access the webcam. Close Zoom/Teams/Camera app and retry.")
        return

    detectors = make_detectors()
    print("Webcam initialized. Click the video window and use the on-screen menu.")

    try:
        while True:
            choice = choose_signs(cap)
            if choice is None:
                break
            selected, overwrite = choice
            if run_recording(cap, detectors, selected, overwrite) == "exit":
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detectors[0].close()
        detectors[1].close()


if __name__ == "__main__":
    collect_data()