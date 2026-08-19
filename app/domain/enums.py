from enum import Enum


class MovementPattern(str, Enum):
    LOWER_BODY_SQUAT = "lower_body_squat"
    LOWER_BODY_HINGE = "lower_body_hinge"
    UPPER_BODY_HORIZONTAL_PUSH = "upper_body_horizontal_push"
    UPPER_BODY_VERTICAL_PUSH = "upper_body_vertical_push"
    UNKNOWN = "unknown"


class ExercisePhase(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    ECCENTRIC = "eccentric"
    BOTTOM = "bottom"
    CONCENTRIC = "concentric"
    LOCKOUT = "lockout"
    FINISH = "finish"


class ValidationStatus(str, Enum):
    VALID = "valid"
    INCOMPLETE = "incomplete"
    TRANSITION = "transition"
    REJECTED = "rejected"
    INSUFFICIENT_SIGNAL = "insufficient_signal"
    SHORT_ROM = "short_rom"
    REJECTED_ROM = "rejected_rom"
    INVALID_DURATION = "invalid_duration"


class SignalSource(str, Enum):
    BILATERAL = "bilateral"
    LEFT = "left"
    RIGHT = "right"
    RAW = "raw"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"


class MetricStatus(str, Enum):
    VALID = "valid"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


class ErrorStatus(str, Enum):
    DETECTED = "detected"
    SUSPECTED = "suspected"
    NOT_DETECTED = "not_detected"
    INSUFFICIENT_DATA = "insufficient_data"


class ErrorSeverity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"