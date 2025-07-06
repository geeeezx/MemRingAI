#!/usr/bin/env python3
"""
测试脚本：测试多个ASR提供商（OpenAI和豆包）
使用方法：python test_asr_providers.py <音频文件路径>
"""

import requests
import json
import os
import sys
from pathlib import Path


class ASRProviderTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
    
    def test_health(self):
        """测试健康检查"""
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
    
    def test_providers(self):
        """测试可用的ASR提供商"""
        print("\n🔧 获取可用的ASR提供商...")
        try:
            response = requests.get(f"{self.api_base}/providers")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 提供商信息: {json.dumps(data, indent=2, ensure_ascii=False)}")
                return data
            else:
                print(f"❌ 获取提供商失败: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 获取提供商异常: {e}")
            return None
    
    def transcribe_with_provider(self, file_path, provider="auto", language="zh"):
        """
        使用指定提供商转录音频文件
        
        Args:
            file_path: 音频文件路径
            provider: ASR提供商 (openai, volcengine, auto)
            language: 语言代码
        """
        print(f"\n🎵 使用 {provider} 转录文件: {file_path}")
        
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
                    'model': 'whisper-1',
                    'response_format': 'verbose_json',
                    'temperature': 0.0,
                    'provider': provider
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
    
    def compare_providers(self, file_path, language="zh"):
        """
        比较不同提供商的转录结果
        
        Args:
            file_path: 音频文件路径
            language: 语言代码
        """
        print(f"\n🔄 比较不同提供商的转录结果")
        print("=" * 60)
        
        providers = ["openai", "volcengine", "auto"]
        results = {}
        
        for provider in providers:
            print(f"\n📋 测试提供商: {provider}")
            result = self.transcribe_with_provider(file_path, provider, language)
            if result:
                results[provider] = result
                # 保存结果到文件
                output_file = f"{Path(file_path).stem}_{provider}_transcription.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 转录结果已保存到: {output_file}")
            else:
                print(f"❌ {provider} 转录失败")
        
        # 比较结果
        if len(results) > 1:
            print(f"\n📊 转录结果比较:")
            print("=" * 60)
            for provider, result in results.items():
                text = result.get('text', '')
                duration = result.get('duration', 0)
                segments = len(result.get('segments', []))
                print(f"{provider.upper():12} | 时长: {duration:.1f}s | 分段: {segments} | 文本: {text[:50]}...")


def main():
    """主函数"""
    print("🚀 MemRingAI 多ASR提供商测试脚本")
    print("=" * 50)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_asr_providers.py <音频文件路径>")
        print("示例: python test_asr_providers.py audio.mp3")
        return
    
    file_path = sys.argv[1]
    
    # 创建测试器实例
    tester = ASRProviderTester()
    
    # 测试健康检查
    if not tester.test_health():
        print("❌ 服务不可用，请确保服务已启动")
        return
    
    # 测试提供商
    providers_info = tester.test_providers()
    if not providers_info:
        print("❌ 无法获取提供商信息")
        return
    
    # 比较不同提供商的转录结果
    tester.compare_providers(file_path)
    
    print("\n🎉 测试完成!")


if __name__ == "__main__":
    main() 