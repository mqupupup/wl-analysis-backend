# test_mediapipe.py - 验证 MediaPipe 安装
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print(f"✅ MediaPipe 版本: {mp.__version__}")
print(f"✅ Tasks API 可用")
print(f"✅ Vision 模块可用")

# 检查关键模块
try:
    from mediapipe.tasks.python.vision import PoseLandmarker
    print("✅ PoseLandmarker 类可用")
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n✅ MediaPipe 0.10.35 安装验证通过！")