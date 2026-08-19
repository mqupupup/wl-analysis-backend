from app.domain.enums import MovementPattern, ExercisePhase

EXERCISE_ZH_MAP = {
    "Squat": "深蹲",
    "Bench Press": "卧推",
    "Deadlift": "硬拉",
    "Overhead Press": "过头推举",
    "Unknown": "未知动作"
}

PATTERN_TO_EXERCISE = {
    MovementPattern.LOWER_BODY_SQUAT: "Squat",
    MovementPattern.LOWER_BODY_HINGE: "Deadlift",
    MovementPattern.UPPER_BODY_HORIZONTAL_PUSH: "Bench Press",
    MovementPattern.UPPER_BODY_VERTICAL_PUSH: "Overhead Press",
}

FSM_DEBOUNCE_FRAMES = 5

# ✅ 完整 FSM 状态转移表（替代 ...）
FSM_TRANSITIONS = {
    "Squat": {
        ExercisePhase.IDLE: [ExercisePhase.SETUP],
        ExercisePhase.SETUP: [ExercisePhase.ECCENTRIC],
        ExercisePhase.ECCENTRIC: [ExercisePhase.BOTTOM],
        ExercisePhase.BOTTOM: [ExercisePhase.CONCENTRIC],
        ExercisePhase.CONCENTRIC: [ExercisePhase.LOCKOUT],
        ExercisePhase.LOCKOUT: [ExercisePhase.FINISH, ExercisePhase.ECCENTRIC],
        ExercisePhase.FINISH: [ExercisePhase.IDLE],
    },
    "Bench Press": {
        ExercisePhase.IDLE: [ExercisePhase.SETUP],
        ExercisePhase.SETUP: [ExercisePhase.ECCENTRIC],
        ExercisePhase.ECCENTRIC: [ExercisePhase.BOTTOM],
        ExercisePhase.BOTTOM: [ExercisePhase.CONCENTRIC],
        ExercisePhase.CONCENTRIC: [ExercisePhase.LOCKOUT],
        ExercisePhase.LOCKOUT: [ExercisePhase.FINISH, ExercisePhase.ECCENTRIC],
        ExercisePhase.FINISH: [ExercisePhase.IDLE],
    },
    "Deadlift": {
        ExercisePhase.IDLE: [ExercisePhase.SETUP],
        ExercisePhase.SETUP: [ExercisePhase.CONCENTRIC],
        ExercisePhase.CONCENTRIC: [ExercisePhase.LOCKOUT],
        ExercisePhase.LOCKOUT: [ExercisePhase.FINISH, ExercisePhase.ECCENTRIC],
        ExercisePhase.ECCENTRIC: [ExercisePhase.BOTTOM],
        ExercisePhase.BOTTOM: [ExercisePhase.SETUP],
        ExercisePhase.FINISH: [ExercisePhase.IDLE],
    },
    "Overhead Press": {
        ExercisePhase.IDLE: [ExercisePhase.SETUP],
        ExercisePhase.SETUP: [ExercisePhase.ECCENTRIC],
        ExercisePhase.ECCENTRIC: [ExercisePhase.BOTTOM],
        ExercisePhase.BOTTOM: [ExercisePhase.CONCENTRIC],
        ExercisePhase.CONCENTRIC: [ExercisePhase.LOCKOUT],
        ExercisePhase.LOCKOUT: [ExercisePhase.FINISH, ExercisePhase.ECCENTRIC],
        ExercisePhase.FINISH: [ExercisePhase.IDLE],
    },
}

#  完整错误知识库
ERROR_KNOWLEDGE_BASE = {
    "squat_knee_valgus": {
        "name": "膝盖内扣",
        "pattern": [MovementPattern.LOWER_BODY_SQUAT],
        "severity": "high",
        "condition": {"metric": "knee_min_angle", "threshold": 110, "direction": "below"},
        "risk": "ACL/MCL韧带损伤风险极高",
        "feedback": "下蹲时膝盖指向脚尖方向，可在膝上方套弹力带激活臀中肌",
        "drills": ["蚌式开合", "弹力带侧向行走", "高脚杯深蹲"],
    },
    "squat_depth_insufficient": {
        "name": "下蹲深度不足",
        "pattern": [MovementPattern.LOWER_BODY_SQUAT],
        "severity": "medium",
        "condition": {"metric": "knee_min_angle", "threshold": 90, "direction": "above"},
        "risk": "股四头肌刺激不充分，长期可能导致肌力不平衡",
        "feedback": "尝试蹲至大腿平行或低于水平面，可箱式深蹲辅助找深度感",
        "drills": ["箱式深蹲", "暂停深蹲", "踝关节灵活性训练"],
    },
    "bench_elbow_flare": {
        "name": "肘部过度外展",
        "pattern": [MovementPattern.UPPER_BODY_HORIZONTAL_PUSH],
        "severity": "high",
        "condition": {"metric": "elbow_min_angle", "threshold": 110, "direction": "below"},
        "risk": "肩峰撞击综合征、肩袖损伤风险",
        "feedback": "大臂与躯干夹角控制在45-60°，想象把杠铃掰弯以激活背阔肌稳定肩胛",
        "drills": ["哑铃地板卧推", "弹力带面拉", "肩胛骨俯卧撑"],
    },
    "deadlift_round_back": {
        "name": "龟背硬拉",
        "pattern": [MovementPattern.LOWER_BODY_HINGE],
        "severity": "high",
        "condition": {"metric": "torso_max_angle", "threshold": 45, "direction": "above"},
        "risk": "腰椎间盘突出、竖脊肌拉伤风险极高",
        "feedback": "起杠前收紧核心、挺胸沉肩，保持脊柱中立位全程不变",
        "drills": ["罗马尼亚硬拉", "早安式体前屈", "鸟狗式"],
    },
}

# 完整 MediaPipe Pose 33点骨骼连接（替代 ...）
SKELETON_CONNECTIONS = [
    # 面部
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10],
    # 躯干
    [11, 12],
    [11, 13], [13, 15], [15, 17], [15, 19], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [18, 20],
    [11, 23], [12, 24], [23, 24],
    # 下肢
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
]


class ValidationStatus(str, Enum):
    VALID = "valid"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"
    TRANSITION = "transition"
    INSUFFICIENT_SIGNAL = "insufficient_signal"

class MetricStatus(str, Enum):
    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"

class ErrorStatus(str, Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    INSUFFICIENT_DATA = "insufficient_data"

class ErrorSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

class SignalSource(str, Enum):
    BILATERAL = "bilateral"
    LEFT = "left"
    RIGHT = "right"
    RAW = "raw"