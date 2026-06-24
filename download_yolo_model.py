# download_yolo_model.py - 下载 YOLOv8 模型（支持多个镜像源）
import urllib.request
import os
import ssl
from pathlib import Path
import requests

# 禁用 SSL 验证（某些镜像站可能需要）
ssl._create_default_https_context = ssl._create_unverified_context

def download_with_requests(url, save_path, description=""):
    """使用 requests 下载文件"""
    try:
        print(f"📥 {description}")
        print(f"   URL: {url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   进度: {percent:.1f}%", end='\r')
        
        print(f"\n✅ {description} 下载成功")
        return True
        
    except Exception as e:
        print(f"❌ {description} 下载失败: {e}")
        return False

def download_model():
    """下载 YOLOv8 模型 - 尝试多个镜像源"""
    model_path = Path(__file__).parent / "yolov8n.pt"
    
    # 如果模型已存在，直接返回
    if model_path.exists():
        file_size = model_path.stat().st_size
        if file_size > 5 * 1024 * 1024:  # 大于5MB
            print(f"✅ 模型已存在: {model_path} ({file_size / 1024 / 1024:.1f} MB)")
            return True
        else:
            print("⚠️  模型文件太小，可能是下载不完整，重新下载...")
            model_path.unlink()
    
    print("="*60)
    print("📥 正在下载 YOLOv8 模型 (yolov8n.pt)")
    print("="*60)
    
    # 多个镜像源（按推荐顺序）
    mirror_sources = [
        {
            'name': '镜像源1: kkgithub',
            'url': 'https://download.kkgithub.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt'
        },
        {
            'name': '镜像源2: fgit',
            'url': 'https://hub.fgit.cf/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt'
        },
        {
            'name': '镜像源3: ghproxy',
            'url': 'https://ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt'
        },
        {
            'name': '官方源 (可能较慢)',
            'url': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt'
        }
    ]
    
    # 尝试每个镜像源
    for source in mirror_sources:
        if download_with_requests(source['url'], model_path, source['name']):
            file_size = model_path.stat().st_size
            print(f"✅ 模型下载完成: {model_path}")
            print(f"   文件大小: {file_size / 1024 / 1024:.1f} MB")
            return True
    
    # 如果所有镜像都失败，提供百度网盘选项
    print("\n" + "="*60)
    print("❌ 所有镜像源下载失败")
    print("="*60)
    print("\n💡 请手动下载模型文件:")
    print("\n【百度网盘下载】")
    print("   链接1: https://pan.baidu.com/s/1rjAXMnFGPva1UfkWGXUHBw")
    print("   提取码: b57y")
    print("\n   链接2: https://pan.baidu.com/s/1NnArcAuIKl7SPu2IaAFQ9A")
    print("   提取码: 644w")
    print("\n【下载步骤】")
    print("   1. 点击上面的百度网盘链接")
    print("   2. 输入提取码")
    print("   3. 下载 yolov8n.pt 文件")
    print("   4. 将文件放在项目根目录: " + str(Path(__file__).parent))
    print("\n" + "="*60)
    
    return False

if __name__ == "__main__":
    download_model()