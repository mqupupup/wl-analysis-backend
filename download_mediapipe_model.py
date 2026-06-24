# download_mediapipe_model.py - 下载 MediaPipe 模型
import urllib.request
import os
import ssl
from pathlib import Path

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

def download_model():
    """下载 MediaPipe Pose Landmarker 模型"""
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    model_path = Path(__file__).parent / "pose_landmarker.task"
    
    if model_path.exists():
        file_size = model_path.stat().st_size
        if file_size > 1 * 1024 * 1024:  # 大于1MB
            print(f"✅ 模型已存在: {model_path} ({file_size / 1024 / 1024:.1f} MB)")
            return True
        else:
            print("⚠️  模型文件太小，可能是下载不完整，重新下载...")
            model_path.unlink()
    
    print("="*60)
    print("📥 正在下载 MediaPipe Pose 模型")
    print("="*60)
    print(f"URL: {model_url}")
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        file_size = model_path.stat().st_size
        print(f"✅ 模型下载成功: {model_path}")
        print(f"   文件大小: {file_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n💡 如果下载失败，请手动下载:")
        print(f"   1. 访问: {model_url}")
        print(f"   2. 下载 pose_landmarker_heavy.task 文件")
        print(f"   3. 重命名为 pose_landmarker.task")
        print(f"   4. 放在项目根目录: {Path(__file__).parent}")
        return False

if __name__ == "__main__":
    download_model()