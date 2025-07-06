#!/usr/bin/env python3
"""
测试脚本：上传本地音频文件并调用转录API
"""

import requests
import json
import os
import sys
from pathlib import Path


class TranscriptionAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def test_health(self):
        """测试健康检查端点"""
        print("🔍 测试健康检查...")
        try:
            response = requests.get(f"{self.api_base}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 健康检查成功: {data}")
                return True
            else:
                print(f"❌ 健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 健康检查异常: {e}")
            return False
    
    def test_supported_formats(self):
        """测试获取支持格式端点"""
        print("\n📋 获取支持的音频格式...")
        try:
            response = requests.get(f"{self.api_base}/supported-formats")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 支持格式: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"❌ 获取格式失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 获取格式异常: {e}")
            return None
    
    def transcribe_file(self, file_path, language="zh", model="whisper-1", response_format="verbose_json"):
        """
        上传音频文件进行转录
        
        Args:
            file_path: 音频文件路径
            language: 语言代码 (zh=中文, en=英文, ja=日文等)
            model: Whisper模型
            response_format: 响应格式
        """
        print(f"\n🎵 开始转录文件: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return None
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"📏 文件大小: {file_size_mb:.2f}MB")
        
        try:
            # 准备文件上传
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'audio/mpeg')}
                data = {
                    'language': language,
                    'model': model,
                    'response_format': response_format,
                    'temperature': 0.0
                }
                
                print(f"📤 上传文件到: {self.api_base}/transcribe")
                print(f"🔧 参数: {data}")
                
                # 发送请求
                response = requests.post(
                    f"{self.api_base}/transcribe",
                    files=files,
                    data=data,
                    timeout=300  # 5分钟超时
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 转录成功!")
                print(f"📝 转录文本: {result.get('text', 'N/A')}")
                print(f"🌍 检测语言: {result.get('language', 'N/A')}")
                print(f"⏱️ 音频时长: {result.get('duration', 'N/A')}秒")
                
                if result.get('segments'):
                    print(f"📊 分段数量: {len(result['segments'])}")
                    # 显示前3个分段
                    for i, segment in enumerate(result['segments'][:3]):
                        print(f"  分段 {i+1}: [{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['text']}")
                
                return result
            else:
                print(f"❌ 转录失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时，音频文件可能太大或处理时间过长")
            return None
        except Exception as e:
            print(f"❌ 转录异常: {e}")
            return None
    
    def transcribe_from_url(self, url, language="zh", model="whisper-1"):
        """从URL转录音频文件"""
        print(f"\n🌐 从URL转录: {url}")
        
        try:
            data = {
                'url': url,
                'language': language,
                'model': model,
                'response_format': 'verbose_json',
                'temperature': 0.0
            }
            
            response = requests.post(
                f"{self.api_base}/transcribe/url",
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ URL转录成功!")
                print(f"📝 转录文本: {result.get('text', 'N/A')}")
                return result
            else:
                print(f"❌ URL转录失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ URL转录异常: {e}")
            return None


def main():
    """主函数"""
    print("🚀 MemRingAI 转录API测试脚本")
    print("=" * 50)
    
    # 创建测试器实例
    tester = TranscriptionAPITester()
    
    # 测试健康检查
    if not tester.test_health():
        print("❌ 服务不可用，请检查服务是否启动")
        return
    
    # 获取支持格式
    formats = tester.test_supported_formats()
    if not formats:
        print("❌ 无法获取支持格式")
        return
    
    # 测试文件转录
    print("\n" + "=" * 50)
    print("🎵 文件转录测试")
    print("=" * 50)
    
    # 示例音频文件路径（请根据实际情况修改）
    test_files = [
        # 添加您的音频文件路径
        # "path/to/your/audio1.mp3",
        # "path/to/your/audio2.wav",
    ]
    
    if test_files:
        for file_path in test_files:
            if os.path.exists(file_path):
                result = tester.transcribe_file(file_path, language="zh")
                if result:
                    # 保存结果到文件
                    output_file = f"{Path(file_path).stem}_transcription.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"💾 转录结果已保存到: {output_file}")
            else:
                print(f"⚠️ 文件不存在: {file_path}")
    else:
        print("⚠️ 请在脚本中添加测试音频文件路径")
        print("示例:")
        print("test_files = [")
        print("    'audio/sample1.mp3',")
        print("    'audio/sample2.wav',")
        print("]")
    
    # URL转录测试
    print("\n" + "=" * 50)
    print("🌐 URL转录测试")
    print("=" * 50)
    
    # 示例URL（请根据实际情况修改）
    test_urls = [
        # "https://example.com/audio.mp3",
    ]
    
    if test_urls:
        for url in test_urls:
            result = tester.transcribe_from_url(url, language="zh")
    else:
        print("⚠️ 请在脚本中添加测试URL")
        print("示例:")
        print("test_urls = [")
        print("    'https://example.com/audio.mp3',")
        print("]")


if __name__ == "__main__":
    main() 