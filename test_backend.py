# test_backend.py - 测试后端
import requests

print("🧪 测试 WL Analysis AI Backend")
print("="*60)

# 测试健康检查
print("\n1. 测试健康检查...")
try:
    response = requests.get("http://localhost:8001/health")
    print(f"   ✅ 状态: {response.status_code}")
    print(f"   内容: {response.json()}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

# 测试信息接口
print("\n2. 测试信息接口...")
try:
    response = requests.get("http://localhost:8001/info")
    print(f"   ✅ 状态: {response.status_code}")
    info = response.json()
    print(f"   服务器: {info['server']}")
    print(f"   版本: {info['version']}")
    print(f"   MediaPipe: {info['models']['pose_estimation']}")
    print(f"   YOLO: {info['models']['barbell_detection']}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

print("\n" + "="*60)
print("✅ 测试完成！")