#!/usr/bin/env python3
"""
简单测试脚本：上传单个音频文件进行转录
使用方法：python test_simple.py <音频文件路径>
"""

import requests
import json
import os
import sys
from pathlib import Path


def transcribe_audio_file(file_path, language="zh", model="whisper-1"):
    """
    上传音频文件进行转录
    
    Args:
        file_path: 音频文件路径
        language: 语言代码 (zh=中文, en=英文, ja=日文等)
        model: 模型名称 (whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe等)
    """
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe"
    
    print(f"🎵 开始转录文件: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📏 文件大小: {file_size_mb:.2f}MB")
    
    # 根据模型选择响应格式
    is_new_model = model.startswith(('gpt-4o-transcribe', 'gpt-4o-mini-transcribe'))
    response_format = 'json' if is_new_model else 'verbose_json'
    
    try:
        # 准备文件上传
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/mpeg')}
            data = {
                'language': language,
                'model': model,
                'response_format': response_format,
                'temperature': 0.0,
                'provider': 'openai'  # 强制使用OpenAI
            }
            
            print(f"📤 上传文件到: {api_url}")
            print(f"🔧 参数: language={language}, model={model}, response_format={response_format}, provider=openai")
            
            # 发送请求
            response = requests.post(
                api_url,
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
            
            # 保存结果到文件
            output_file = f"{Path(file_path).stem}_transcription.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 转录结果已保存到: {output_file}")
            
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


def test_health():
    """测试健康检查"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code == 200:
            print("✅ 服务健康检查通过")
            return True
        else:
            print(f"❌ 服务健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        return False


def main():
    """主函数"""
    print("🚀 MemRingAI 转录API简单测试 (OpenAI专用)")
    print("=" * 40)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_simple.py <音频文件路径> [模型名称] [语言]")
        print("示例: python test_simple.py audio.mp3")
        print("示例: python test_simple.py audio.mp3 gpt-4o-transcribe")
        print("示例: python test_simple.py audio.mp3 gpt-4o-mini-transcribe zh")
        print("\n可用模型:")
        print("  - whisper-1 (默认)")
        print("  - gpt-4o-transcribe")
        print("  - gpt-4o-mini-transcribe")
        return
    
    file_path = sys.argv[1]
    
    # 获取可选参数
    model = sys.argv[2] if len(sys.argv) > 2 else "whisper-1"
    language = sys.argv[3] if len(sys.argv) > 3 else "zh"
    
    print(f"📁 文件路径: {file_path}")
    print(f"🤖 使用模型: {model}")
    print(f"🌍 语言设置: {language}")
    print()
    
    # 测试健康检查
    if not test_health():
        print("❌ 服务不可用，请确保服务已启动")
        return
    
    # 转录文件
    result = transcribe_audio_file(file_path, language, model)
    
    if result:
        print("\n🎉 测试完成!")
    else:
        print("\n❌ 测试失败!")


if __name__ == "__main__":
    main() 