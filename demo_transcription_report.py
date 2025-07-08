#!/usr/bin/env python3
"""
Demo script for combined transcription and report generation
Transcribes audio files and generates comprehensive business reports
使用方法：python demo_transcription_report.py <音频文件路径>
"""

import requests
import json
import os
import sys
from pathlib import Path
import time


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_transcription_results(transcription: dict):
    """Print formatted transcription results"""
    print("\n📝 TRANSCRIPTION RESULTS:")
    print("-" * 40)
    print(f"🗣️  Text: {transcription.get('text', 'N/A')}")
    print(f"🌍 Language: {transcription.get('language', 'N/A')}")
    print(f"⏱️  Duration: {transcription.get('duration', 'N/A')} seconds")
    
    # Show VAD information if available
    if transcription.get('vad_info'):
        vad_info = transcription['vad_info']
        print(f"\n🎤 VOICE ACTIVITY DETECTION:")
        print(f"   Speech Ratio: {vad_info.get('speech_ratio', 0):.1%}")
        print(f"   Speech Duration: {vad_info.get('total_speech_duration', 0):.1f}s")
        print(f"   Speech Segments: {vad_info.get('segment_count', 0)}")
        
        if vad_info.get('acceleration_info'):
            acc_info = vad_info['acceleration_info']
            print(f"   Audio Acceleration: {acc_info.get('factor', 1.0):.1f}x speed")
            print(f"   Time Saved: {acc_info.get('original_duration', 0) - acc_info.get('accelerated_duration', 0):.1f}s")
    
    # Show timing information if available
    if transcription.get('timing_info'):
        timing = transcription['timing_info']
        print(f"\n⏱️  PROCESSING TIMING:")
        print(f"   Total Time: {timing.get('total_time', 0):.2f}s")
        if timing.get('vad_processing'):
            print(f"   VAD Processing: {timing['vad_processing']:.2f}s")
        if timing.get('transcription'):
            print(f"   Transcription: {timing['transcription']:.2f}s")
    
    if transcription.get('segments'):
        print(f"\n📊 TRANSCRIPTION SEGMENTS:")
        print(f"   Total Segments: {len(transcription['segments'])}")
        # Show first few segments
        for i, segment in enumerate(transcription['segments'][:3]):
            print(f"   [{segment['start']:.1f}s-{segment['end']:.1f}s] {segment['text']}")
        if len(transcription['segments']) > 3:
            print(f"   ... and {len(transcription['segments']) - 3} more segments")


def print_report_results(report: dict):
    """Print formatted report results"""
    print("\n📊 BUSINESS ANALYSIS REPORT:")
    print("-" * 40)
    
    print(f"\n💡 IDEA SUMMARY:")
    print(f"   {report['idea_summary']}")
    
    print(f"\n📈 MARKET ANALYSIS:")
    print(f"   {report['market_analysis']}")
    
    print(f"\n🔧 TECHNICAL FEASIBILITY:")
    print(f"   {report['technical_feasibility']}")
    
    print(f"\n📋 IMPLEMENTATION STEPS:")
    for i, step in enumerate(report['implementation_steps'], 1):
        print(f"   {i}. {step}")
    
    print(f"\n⚠️ POTENTIAL CHALLENGES:")
    for i, challenge in enumerate(report['potential_challenges'], 1):
        print(f"   {i}. {challenge}")
    
    print(f"\n🎯 SUCCESS FACTORS:")
    for i, factor in enumerate(report['success_factors'], 1):
        print(f"   {i}. {factor}")
    
    print(f"\n⏰ ESTIMATED TIMELINE:")
    print(f"   {report['estimated_timeline']}")
    
    print(f"\n🚀 NEXT ACTIONS:")
    for i, action in enumerate(report['next_actions'], 1):
        print(f"   {i}. {action}")


def transcribe_and_analyze_audio(file_path: str, language="auto", report_focus=None):
    """
    Upload audio file for transcription and report generation
    
    Args:
        file_path: Path to audio file
        language: Language code (auto, zh, en, etc.)
        report_focus: Optional focus area for the report
    """
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe-and-report"
    
    print(f"🎵 Starting combined transcription and analysis for: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    # Check file size
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"📏 File size: {file_size_mb:.2f}MB")
    
    try:
        # Prepare file upload
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/mpeg')}
            data = {
                'language': language if language != 'auto' else None,
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'temperature': 0.0,
                'provider': 'auto',
                'generate_report': True
            }
            
            if report_focus:
                data['report_focus'] = report_focus
            
            print(f"📤 Uploading to: {api_url}")
            print(f"🔧 Parameters: language={language}, report_focus={report_focus}")
            print("🎤 VAD (Voice Activity Detection): ENABLED")
            print("⏳ Processing with VAD + Transcription + Report Generation...")
            
            # Send request
            start_time = time.time()
            response = requests.post(
                api_url,
                files=files,
                data=data,
                timeout=600  # 10 minutes timeout for combined processing
            )
            processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Combined processing successful!")
            print(f"⏱️ Total processing time: {processing_time:.2f}s")
            
            # Print transcription results
            if result.get('transcription'):
                print_transcription_results(result['transcription'])
            
            # Print report results
            if result.get('report_generation_success') and result.get('report'):
                print_report_results(result['report'])
                print(f"\n📊 ANALYSIS METRICS:")
                print(f"   Report tokens used: {result.get('total_tokens_used', 'N/A')}")
                print(f"   Server processing time: {result.get('processing_time_seconds', 0):.2f}s")
                
                # Show enhanced insights if VAD data was used
                if result.get('transcription', {}).get('vad_info'):
                    vad_info = result['transcription']['vad_info']
                    speech_ratio = vad_info.get('speech_ratio', 0)
                    print(f"\n🎯 PRESENTATION QUALITY INSIGHTS:")
                    if speech_ratio > 0.7:
                        print(f"   ✅ High confidence presentation ({speech_ratio:.0%} speech content)")
                    elif speech_ratio > 0.4:
                        print(f"   ⚡ Moderate presentation quality ({speech_ratio:.0%} speech content)")
                    else:
                        print(f"   ⚠️  Consider more structured delivery ({speech_ratio:.0%} speech content)")
                        
                    if vad_info.get('segment_count', 0) > 5:
                        print(f"   📈 Well-structured idea ({vad_info['segment_count']} segments)")
                    else:
                        print(f"   💬 Concise presentation ({vad_info['segment_count']} segments)")
            elif result.get('report_error'):
                print(f"\n❌ Report generation failed: {result['report_error']}")
            
            # Save results to file
            output_file = f"{Path(file_path).stem}_transcription_report.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Complete results saved to: {output_file}")
            
            return result
        else:
            print(f"❌ Processing failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - processing took too long")
        return None
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return None


def transcribe_only_demo(file_path: str):
    """Demo transcription only (original functionality)"""
    print_header("TRANSCRIPTION ONLY DEMO")
    
    base_url = "http://localhost:8000"
    api_url = f"{base_url}/api/v1/transcribe"
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/mpeg')}
            data = {
                'language': 'auto',
                'model': 'whisper-1',
                'response_format': 'verbose_json',
                'temperature': 0.0
            }
            
            response = requests.post(api_url, files=files, data=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Transcription successful!")
            print_transcription_results(result)
            return result
        else:
            print(f"❌ Transcription failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_health():
    """Test service health"""
    try:
        response = requests.get("http://localhost:8000/api/v1/health")
        if response.status_code == 200:
            print("✅ Service health check passed")
            return True
        else:
            print(f"❌ Service health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return False


def interactive_demo():
    """Interactive demo for testing different features"""
    print_header("INTERACTIVE DEMO")
    
    print("Choose a demo mode:")
    print("1. Combined transcription + report generation")
    print("2. Transcription only")
    print("3. Combined with custom report focus")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "4":
                print("👋 Goodbye!")
                break
                
            if choice in ["1", "2", "3"]:
                file_path = input("Enter audio file path: ").strip()
                
                if not file_path or not os.path.exists(file_path):
                    print("❌ Invalid file path")
                    continue
                
                if choice == "1":
                    # Combined demo
                    transcribe_and_analyze_audio(file_path)
                elif choice == "2":
                    # Transcription only
                    transcribe_only_demo(file_path)
                elif choice == "3":
                    # Combined with focus
                    focus = input("Enter report focus (technical/market/implementation): ").strip()
                    transcribe_and_analyze_audio(file_path, report_focus=focus if focus else None)
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function"""
    print_header("MemRingAI Combined Transcription & Report Demo")
    
    # Test health first
    if not test_health():
        print("❌ Service not available. Please ensure the service is running:")
        print("   uv run python -m app.main")
        return
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\nUsage options:")
        print("1. python demo_transcription_report.py <audio_file_path>")
        print("2. python demo_transcription_report.py --interactive")
        print("\nExamples:")
        print("   python demo_transcription_report.py audio.mp3")
        print("   python demo_transcription_report.py tests/audio_test/ref_lbw.WAV")
        print("   python demo_transcription_report.py --interactive")
        return
    
    if sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        file_path = sys.argv[1]
        
        # Demo 1: Combined transcription and report
        print_header("COMBINED TRANSCRIPTION + REPORT DEMO")
        result = transcribe_and_analyze_audio(file_path)
        
        if result:
            print("\n🎉 Combined demo completed successfully!")
            
            # Optional: Show transcription-only for comparison
            if input("\nWould you like to see transcription-only results for comparison? (y/n): ").lower().startswith('y'):
                transcribe_only_demo(file_path)
        else:
            print("\n❌ Combined demo failed!")


if __name__ == "__main__":
    main() 