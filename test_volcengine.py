#!/usr/bin/env python3
"""
豆包ASR API测试脚本
使用方法: python test_volcengine.py <音频文件路径>
"""

import requests
import json
import os
import sys
from pathlib import Path


def test_volcengine_transcription(file_path, language="zh"):
    """
    测试豆包ASR转录
    
    Args:
        file_path: 音频文件路径
        language: 语言代码
    """
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe"
    
    print(f"🎵 使用豆包ASR转录文件: {file_path}")
    
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
                'model': 'bigmodel',
                'response_format': 'verbose_json',
                'temperature': 0.0,
                'provider': 'volcengine'  # 指定使用豆包
            }
            
            print(f"📤 上传文件到: {api_url}")
            print(f"🔧 参数: {data}")
            
            # 发送请求
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=300  # 5分钟超时
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 豆包ASR转录成功!")
            print(f"📝 转录文本: {result.get('text', 'N/A')}")
            print(f"🌍 检测语言: {result.get('language', 'N/A')}")
            print(f"⏱️ 音频时长: {result.get('duration', 'N/A')}秒")
            
            if result.get('segments'):
                print(f"📊 分段数量: {len(result['segments'])}")
                # 显示前3个分段
                for i, segment in enumerate(result['segments'][:3]):
                    print(f"  分段 {i+1}: [{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['text']}")
            
            # 保存结果到文件
            output_file = f"{Path(file_path).stem}_volcengine_transcription.json"
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


def test_providers():
    """测试可用的ASR提供商"""
    print("🔧 获取可用的ASR提供商...")
    try:
        response = requests.get("http://localhost:8000/api/v1/providers")
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


def main():
    """主函数"""
    print("🚀 豆包ASR API测试脚本")
    print("=" * 40)
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("使用方法: python test_volcengine.py <音频文件路径>")
        print("示例: python test_volcengine.py audio.mp3")
        return
    
    file_path = sys.argv[1]
    
    # 测试提供商
    providers_info = test_providers()
    if not providers_info:
        print("❌ 无法获取提供商信息")
        return
    
    # 检查豆包是否可用
    volcengine_info = providers_info.get('providers', {}).get('volcengine', {})
    if not volcengine_info.get('configured', False):
        print("❌ 豆包ASR未配置，请检查环境变量")
        print("需要设置: VOLCENGINE_APP_ID, VOLCENGINE_ACCESS_TOKEN")
        return
    
    if not volcengine_info.get('valid', False):
        print("❌ 豆包ASR配置无效，请检查凭据")
        return
    
    print("✅ 豆包ASR配置有效")
    
    # 测试转录
    result = test_volcengine_transcription(file_path)
    
    if result:
        print("\n🎉 豆包ASR测试完成!")
    else:
        print("\n❌ 豆包ASR测试失败!")


if __name__ == "__main__":
    main() 