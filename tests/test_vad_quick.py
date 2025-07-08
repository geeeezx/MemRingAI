#!/usr/bin/env python3
"""
VAD快速测试脚本：快速测试audio_test文件夹中的音频文件
使用方法: python test_vad_quick.py [file_name]
"""

import requests
import json
import os
import sys
from pathlib import Path


def test_vad_quick(file_name=None, enable_vad=True, model="gpt-4o-transcribe"):
    """
    快速VAD测试
    
    Args:
        file_name: 指定文件名（可选）
        enable_vad: 是否启用VAD
        model: 使用的模型
    """
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe"
    audio_dir = Path("tests/audio_test")
    
    print("🚀 VAD快速测试")
    print("=" * 30)
    
    # 检查服务健康状态
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=5)
        if response.status_code != 200:
            print("❌ 服务不可用")
            return
        print("✅ 服务正常")
    except:
        print("❌ 无法连接到服务")
        return
    
    # 获取音频文件
    if file_name:
        file_path = audio_dir / file_name
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            return
        audio_files = [file_path]
    else:
        # 自动选择第一个音频文件
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.opus"))
        if not audio_files:
            print("❌ 没有找到音频文件")
            return
        audio_files = [audio_files[0]]  # 只测试第一个文件
    
    # 测试文件
    for file_path in audio_files:
        print(f"\n🎵 测试文件: {file_path.name}")
        print(f"🔧 VAD={enable_vad}, 模型={model}")
        
        # 检查文件大小
        file_size = file_path.stat().st_size / (1024 * 1024)
        print(f"📏 文件大小: {file_size:.2f}MB")
        
        # 根据模型选择响应格式
        response_format = 'json' if model.startswith(('gpt-4o-transcribe', 'gpt-4o-mini-transcribe')) else 'verbose_json'
        
        try:
            # 准备文件上传
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'audio/mpeg')}
                data = {
                    'language': 'zh',
                    'model': model,
                    'response_format': response_format,
                    'temperature': 0.0,
                    'provider': 'openai',
                    'enable_vad': enable_vad
                }
                
                print("📤 上传中...")
                
                # 发送请求
                response = requests.post(
                    api_url,
                    files=files,
                    data=data,
                    timeout=120  # 2分钟超时
                )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 测试成功!")
                
                # 显示基本信息
                text = result.get('text', '')
                print(f"📝 转录文本: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"🌍 语言: {result.get('language', 'N/A')}")
                print(f"⏱️ 时长: {result.get('duration', 'N/A')}秒")
                
                # 显示时间信息
                timing_info = result.get('timing_info')
                if timing_info:
                    print(f"⏱️ 时间统计:")
                    print(f"   - 文件保存: {timing_info.get('file_save', 0):.2f}s")
                    print(f"   - VAD处理: {timing_info.get('vad_processing', 0):.2f}s")
                    print(f"   - 请求准备: {timing_info.get('request_preparation', 0):.2f}s")
                    print(f"   - 服务设置: {timing_info.get('service_setup', 0):.2f}s")
                    print(f"   - 转录: {timing_info.get('transcription', 0):.2f}s")
                    print(f"   - 响应准备: {timing_info.get('response_preparation', 0):.2f}s")
                    print(f"   - 总时间: {timing_info.get('total_time', 0):.2f}s")
                    
                    # 显示VAD详细时间
                    if timing_info.get('vad_processing', 0) > 0:
                        vad_info = result.get('vad_info')
                        if vad_info and vad_info.get('timing_info'):
                            vad_timing = vad_info['timing_info']
                            print(f"   🎤 VAD详细时间:")
                            print(f"     - 格式检查: {vad_timing.get('format_check', 0):.2f}s")
                            print(f"     - 音频转换: {vad_timing.get('audio_conversion', 0):.2f}s")
                            print(f"     - 文件读取: {vad_timing.get('file_reading', 0):.2f}s")
                            print(f"     - VAD检测: {vad_timing.get('vad_detection', 0):.2f}s")
                            print(f"     - VAD总时间: {vad_timing.get('total_vad_time', 0):.2f}s")
                            
                            # 分析哪个步骤最慢
                            vad_steps = {
                                '格式检查': vad_timing.get('format_check', 0),
                                '音频转换': vad_timing.get('audio_conversion', 0),
                                '文件读取': vad_timing.get('file_reading', 0),
                                'VAD检测': vad_timing.get('vad_detection', 0)
                            }
                            
                            if vad_steps:
                                max_step = max(vad_steps.items(), key=lambda x: x[1])
                                total_vad = vad_timing.get('total_vad_time', 0)
                                if total_vad > 0:
                                    percentage = max_step[1] / total_vad * 100
                                    print(f"   🔍 VAD瓶颈分析: {max_step[0]} 最慢 ({max_step[1]:.2f}s, {percentage:.1f}%)")
                        else:
                            print(f"   ⚠️ 无VAD详细时间信息")
                
                # 显示VAD信息
                vad_info = result.get('vad_info')
                if vad_info:
                    print(f"🎤 VAD信息:")
                    print(f"   - 语音段: {vad_info.get('segment_count', 'N/A')}")
                    print(f"   - 语音比例: {vad_info.get('speech_ratio', 'N/A'):.1%}")
                    print(f"   - 语音时长: {vad_info.get('total_speech_duration', 'N/A'):.1f}秒")
                    
                    # 显示语音段
                    segments = vad_info.get('speech_segments', [])
                    if segments:
                        print(f"   - 语音段详情:")
                        for i, seg in enumerate(segments[:3]):
                            print(f"     [{seg['start']:.1f}s-{seg['end']:.1f}s]")
                        if len(segments) > 3:
                            print(f"     ... 还有 {len(segments) - 3} 个段")
                else:
                    print("⚠️ 无VAD信息")
                
                # 保存结果
                output_file = f"{file_path.stem}_vad_quick.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 结果已保存: {output_file}")
                
            else:
                print(f"❌ 失败: {response.status_code}")
                print(f"错误: {response.text}")
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
        except Exception as e:
            print(f"❌ 错误: {e}")


def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("使用方法: python test_vad_quick.py [文件名] [enable_vad] [model]")
        print("示例:")
        print("  python test_vad_quick.py")
        print("  python test_vad_quick.py ref_lbw.WAV")
        print("  python test_vad_quick.py lbw2.opus true gpt-4o-transcribe")
        print("  python test_vad_quick.py ref_lbw.WAV false whisper-1")
        return
    
    # 解析参数
    file_name = sys.argv[1] if len(sys.argv) > 1 else None
    enable_vad = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-4o-transcribe"
    
    test_vad_quick(file_name, enable_vad, model)


if __name__ == "__main__":
    main() 