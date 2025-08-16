import hashlib


EPSILON = 1e-6
NR_OPENPOSE_KEYPOINTS = 25
MAX_NR_KEYPOINTS = 150
DYN_MASK_DILATION_KERNEL_SIZE = 33
PIX_INTERP_WINDOW_SIZE = 3
STRENGTH_GRAY_MASK = 0.05
SPECIAL_OPENPOSE_VERTEX_IDS = {
    "nose": 9120,
    "reye": 9929,
    "leye": 9448,
    "rear": 616,
    "lear": 6,
    "rthumb": 8079,
    "rindex": 7669,
    "rmiddle": 7794,
    "rring": 7905,
    "rpinky": 8022,
    "lthumb": 5361,
    "lindex": 4933,
    "lmiddle": 5058,
    "lring": 5169,
    "lpinky": 5286,
    "LBigToe": 5770,
    "LSmallToe": 5780,
    "LHeel": 8846,
    "RBigToe": 8463,
    "RSmallToe": 8474,
    "RHeel": 8635,
}

OPENPOSE_BODY25_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]
NOT_BONES_VERTEX_NAME_MAP = {
    "nose": "nose",
    "right_eye": "reye",
    "left_eye": "leye",
    "right_ear": "rear",
    "left_ear": "lear",
    "right_thumb": "rthumb",
    "right_index": "rindex",
    "right_middle": "rmiddle",
    "right_ring": "rring",
    "right_pinky": "rpinky",
    "left_thumb": "lthumb",
    "left_index": "lindex",
    "left_middle": "lmiddle",
    "left_ring": "lring",
    "left_pinky": "lpinky",
    "left_big_toe": "LBigToe",
    "left_small_toe": "LSmallToe",
    "left_heel": "LHeel",
    "right_big_toe": "RBigToe",
    "right_small_toe": "RSmallToe",
    "right_heel": "RHeel",
}
# Official OpenPose limb sequence (0-based indices)
OPENPOSE_BODY25_PAIRS = [
    (2, 3),  # right shoulder -> right elbow
    (1, 0),  # right shoulder -> left shoulder
    (6, 5),
    (3, 4),  # right elbow -> right wrist
    (6, 7),  # left shoulder -> left elbow
    (1, 9),  # right shoulder -> right hip
    (9, 10),  # right hip -> right knee
    (10, 11),  # right knee -> right ankle
    (1, 12),  # right shoulder -> left hip
    (12, 13),  # left hip -> left knee
    (13, 14),  # left knee -> left ankle
    (2, 1),  # right shoulder -> neck
    (1, 5),
    (0, 15),  # neck -> right eye
    (15, 17),  # right eye -> right ear
    (0, 16),  # neck -> left eye
    (16, 18),  # left eye -> left ear
]
# Standard palette (12 distinct colors)
PALETTE = [
    (255, 0, 0),  # 1 Red
    (0, 255, 0),  # 2 Green
    (0, 0, 255),  # 3 Blue
    (255, 255, 0),  # 4 Yellow
    (255, 0, 255),  # 5 Magenta
    (0, 255, 255),  # 6 Cyan
    (128, 0, 128),  # 7 Purple
    (255, 165, 0),  # 8 Orange
    (0, 128, 0),  # 9 Dark Green
    (128, 0, 0),  # 10 Dark Red
    (0, 0, 128),  # 11 Dark Blue
    (128, 128, 0),  # 12 Olive
]
# Body part color mapping for segmentation visualization
# Each body part ID maps to a specific color (BGR format for OpenCV)
# All colors are bright enough to avoid transparency issues
BODY_PART_COLORS = {
    1: (255, 0, 0),  # Blue - head
    2: (0, 255, 0),  # Green - torso
    3: (0, 0, 255),  # Red - left arm
    4: (255, 255, 0),  # Cyan - right arm
    5: (255, 0, 255),  # Magenta - left leg
    6: (0, 255, 255),  # Yellow - right leg
    7: (180, 0, 0),  # Bright blue - left hand
    8: (0, 180, 0),  # Bright green - right hand
    9: (0, 0, 180),  # Bright red - left foot
    10: (180, 180, 0),  # Bright cyan - right foot
    11: (255, 165, 0),  # Orange - spine
    12: (180, 0, 180),  # Bright purple - shoulders
    13: (200, 200, 200),  # Bright grey - for unknown/fallback body parts
    # Additional body part IDs that might be used by the model
    14: (255, 100, 0),  # Dark orange - additional body part
    15: (100, 255, 100),  # Light green - additional body part
    16: (100, 100, 255),  # Light blue - additional body part
    17: (255, 100, 255),  # Light magenta - additional body part
    18: (100, 255, 255),  # Light cyan - additional body part
    19: (255, 255, 100),  # Light yellow - additional body part
    20: (150, 150, 255),  # Light purple - additional body part
    21: (255, 150, 150),  # Light pink - additional body part
    22: (150, 255, 150),  # Mint green - additional body part
    23: (255, 200, 100),  # Peach - additional body part
    24: (100, 200, 255),  # Sky blue - additional body part
}

# --------------------------- Color utility ---------------------------


def _generate_hashed_color(part_id: int, min_brightness: int = 100) -> tuple[int, int, int]:
    """Generate a deterministic bright BGR color for *part_id* using a hash.
    Ensures each channel is at least *min_brightness* to avoid transparency."""
    hash_val = int(hashlib.md5(str(part_id).encode()).hexdigest()[:6], 16)
    r = (hash_val >> 16) & 0xFF
    g = (hash_val >> 8) & 0xFF
    b = hash_val & 0xFF
    r = max(r, min_brightness)
    g = max(g, min_brightness)
    b = max(b, min_brightness)

    return (b, g, r)  # BGR


def get_body_part_bgr(part_id: int) -> tuple[int, int, int]:
    """Return the BGR color for a body-part *part_id*.

    1. If the ID exists in the fixed BODY_PART_COLORS dict, use that.
    2. Otherwise generate a bright deterministic color via hashing (stable across runs).
    """
    if part_id in BODY_PART_COLORS:
        return BODY_PART_COLORS[part_id]

    return _generate_hashed_color(part_id)


# Export a palette lookup for external modules wanting a dict view
DEFAULT_BODY_PART_PALETTE = {pid: col for pid, col in BODY_PART_COLORS.items()}
