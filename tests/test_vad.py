#!/usr/bin/env python3
"""
VAD测试脚本：测试语音活动检测功能
使用方法：python test_vad.py <音频文件路径>
"""

import requests
import json
import os
import sys
from pathlib import Path


def test_vad_processing(file_path, enable_vad=True, language="zh"):
    """
    测试VAD处理功能
    
    Args:
        file_path: 音频文件路径
        enable_vad: 是否启用VAD
        language: 语言代码
    """
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe"
    
    print(f"🎵 开始VAD测试: {file_path}")
    print(f"🔧 VAD启用状态: {enable_vad}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📏 文件大小: {file_size_mb:.2f}MB")
    
    # 根据文件扩展名选择响应格式
    file_ext = Path(file_path).suffix.lower()
    is_new_model = True  # 使用新模型进行测试
    model = "gpt-4o-transcribe"
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
                'provider': 'openai',
                'enable_vad': enable_vad
            }
            
            print(f"📤 上传文件到: {api_url}")
            print(f"🔧 参数: language={language}, model={model}, enable_vad={enable_vad}")
            
            # 发送请求
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300  # 5分钟超时
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ VAD测试成功!")
            print(f"📝 转录文本: {result.get('text', 'N/A')}")
            print(f"🌍 检测语言: {result.get('language', 'N/A')}")
            print(f"⏱️ 音频时长: {result.get('duration', 'N/A')}秒")
            
            # 显示VAD信息
            vad_info = result.get('vad_info')
            if vad_info:
                print(f"🎤 VAD信息:")
                print(f"   - 语音段数量: {vad_info.get('segment_count', 'N/A')}")
                print(f"   - 语音比例: {vad_info.get('speech_ratio', 'N/A'):.2%}")
                print(f"   - 总语音时长: {vad_info.get('total_speech_duration', 'N/A'):.2f}秒")
                
                # 显示前几个语音段
                speech_segments = vad_info.get('speech_segments', [])
                if speech_segments:
                    print(f"   - 语音段详情:")
                    for i, segment in enumerate(speech_segments[:3]):
                        print(f"     段 {i+1}: [{segment['start']:.1f}s-{segment['end']:.1f}s]")
            else:
                print("⚠️ 未返回VAD信息")
            
            # 保存结果到文件
            output_file = f"{Path(file_path).stem}_vad_test.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 测试结果已保存到: {output_file}")
            
            return result
        else:
            print(f"❌ VAD测试失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ 请求超时，音频文件可能太大或处理时间过长")
        return None
    except Exception as e:
        print(f"❌ VAD测试异常: {e}")
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
    print("🚀 MemRingAI VAD功能测试")
    print("=" * 40)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_vad.py <音频文件路径> [enable_vad] [语言]")
        print("示例: python test_vad.py audio.wav")
        print("示例: python test_vad.py audio.opus true zh")
        print("示例: python test_vad.py audio.mp3 false en")
        return
    
    file_path = sys.argv[1]
    
    # 获取可选参数
    enable_vad = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
    language = sys.argv[3] if len(sys.argv) > 3 else "zh"
    
    print(f"📁 文件路径: {file_path}")
    print(f"🎤 VAD启用: {enable_vad}")
    print(f"🌍 语言设置: {language}")
    print()
    
    # 测试健康检查
    if not test_health():
        print("❌ 服务不可用，请确保服务已启动")
        return
    
    # 测试VAD处理
    result = test_vad_processing(file_path, enable_vad, language)
    
    if result:
        print("\n🎉 VAD测试完成!")
    else:
        print("\n❌ VAD测试失败!")


if __name__ == "__main__":
    main() 