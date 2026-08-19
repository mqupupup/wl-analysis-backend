# app/core/config.py

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent.parent

# 上传目录（使用绝对路径）
UPLOADS_DIR = BASE_DIR / "uploads"

# 其他配置
# MAX_FRAMES_TO_PROCESS = 300
MAX_FRAMES_TO_PROCESS = 10000
TARGET_SKELETON_FRAMES = 50
NORMALIZE_TARGET_LENGTH = 100