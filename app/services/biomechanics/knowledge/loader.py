"""知识库加载器"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

KNOWLEDGE_DIR = Path(__file__).parent / "exercises"

_exercise_cache: Dict[str, Dict] = {}

# exercise name → json filename 映射
EXERCISE_FILE_MAP = {
    "Squat": "squat.json",
    "Bench Press": "bench_press.json",
    "Deadlift": "deadlift.json",
    "Overhead Press": "overhead_press.json",
    "Bicep Curl": "bicep_curl.json",
}


def load_exercise(exercise_name: str) -> Optional[Dict[str, Any]]:
    """加载动作知识库"""
    if exercise_name in _exercise_cache:
        return _exercise_cache[exercise_name]

    filename = EXERCISE_FILE_MAP.get(exercise_name)
    if not filename:
        print(f"⚠️ 未找到 {exercise_name} 的知识库文件")
        return None

    filepath = KNOWLEDGE_DIR / filename
    if not filepath.exists():
        print(f"⚠️ 知识库文件不存在: {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    _exercise_cache[exercise_name] = data
    print(f"📚 加载知识库: {exercise_name} ({len(data['errors'])} 个错误检测)")
    return data


def get_errors_for_exercise(exercise_name: str) -> list:
    """获取动作的错误检测列表"""
    data = load_exercise(exercise_name)
    return data['errors'] if data else []


def get_key_points_for_exercise(exercise_name: str) -> list:
    """获取动作的评分维度"""
    data = load_exercise(exercise_name)
    return data['key_points'] if data else []


def get_phases_for_exercise(exercise_name: str) -> Optional[Dict]:
    """获取动作的阶段定义"""
    data = load_exercise(exercise_name)
    return data['phases'] if data else None


def get_rep_detection_rules(exercise_name: str) -> Optional[Dict]:
    """获取 rep 检测规则"""
    data = load_exercise(exercise_name)
    return data['rep_detection'] if data else None


def list_available_exercises() -> list:
    """列出所有可用动作"""
    return list(EXERCISE_FILE_MAP.keys())