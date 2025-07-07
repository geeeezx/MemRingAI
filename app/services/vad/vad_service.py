"""Voice Activity Detection (VAD) service for audio preprocessing."""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import wave
import struct

import onnxruntime as onnx
from pydub import AudioSegment
import librosa

from app.config import get_settings

logger = logging.getLogger(__name__)


class VADService:
    """Voice Activity Detection service using Silero VAD model."""
    
    def __init__(self, model_path: str = 'model/silero_vad.onnx'):
        """
        Initialize VAD service.
        
        Args:
            model_path: Path to the Silero VAD ONNX model
        """
        # Convert relative path to absolute path based on current file location
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, model_path)
        else:
            self.model_path = model_path
            
        self.vad_session = None
        self._load_model()
        
        # VAD parameters
        self.sample_rate = 16000
        self.threshold = 0.5
        self.min_speech_duration_ms = 250
        self.max_speech_duration_s = float('inf')
        self.min_silence_duration_ms = 100
        self.window_size_samples = 1024
        self.speech_pad_ms = 400
        
        # Audio acceleration parameters - load from config
        settings = get_settings()
        self.enable_acceleration = settings.enable_audio_acceleration
        self.acceleration_factor = settings.audio_acceleration_factor
        self.min_segment_duration_for_acceleration = settings.audio_min_segment_duration_for_acceleration
        self.max_acceleration_factor = settings.audio_max_acceleration_factor
        
    def _load_model(self):
        """Load the Silero VAD model."""
        try:
            logger.info(f"Attempting to load VAD model from: {self.model_path}")
            logger.info(f"Current working directory: {os.getcwd()}")
            logger.info(f"File exists: {os.path.exists(self.model_path)}")
            
            if os.path.exists(self.model_path):
                self.vad_session = onnx.InferenceSession(self.model_path)
                logger.info(f"VAD model loaded successfully from {self.model_path}")
            else:
                logger.warning(f"VAD model not found at {self.model_path}, VAD will be disabled")
                # Try to find the model file in common locations
                current_dir = os.path.dirname(os.path.abspath(__file__))
                possible_paths = [
                    os.path.join(current_dir, "model", "silero_vad.onnx"),
                    os.path.join(current_dir, "..", "model", "silero_vad.onnx"),
                    os.path.join(current_dir, "..", "..", "model", "silero_vad.onnx"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Found model at alternative location: {path}")
                        break
                self.vad_session = None
        except Exception as e:
            logger.error(f"Failed to load VAD model: {str(e)}")
            self.vad_session = None
    
    def convert_opus_to_wav(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert OPUS audio file to WAV format.
        
        Args:
            input_path: Path to the input OPUS file
            output_path: Path for the output WAV file (optional)
            
        Returns:
            Path to the converted WAV file
        """
        try:
            if output_path is None:
                # Create temporary WAV file in the same directory as input
                input_dir = Path(input_path).parent
                output_path = input_dir / f"converted_{Path(input_path).stem}.wav"
            
            # Check if FFmpeg is available
            try:
                from pydub.utils import which
                ffmpeg_path = which("ffmpeg")
                if not ffmpeg_path:
                    logger.warning("FFmpeg not found. OPUS conversion may fail.")
                    logger.info("Please install FFmpeg: https://ffmpeg.org/download.html")
            except ImportError:
                logger.warning("Could not check for FFmpeg availability")
            
            # Load OPUS file and convert to WAV
            try:
                audio = AudioSegment.from_file(input_path, format="opus")
                audio.export(output_path, format="wav")
                
                logger.info(f"Converted OPUS to WAV: {input_path} -> {output_path}")
                return str(output_path)
                
            except Exception as conversion_error:
                logger.error(f"OPUS conversion failed: {str(conversion_error)}")
                
                # Try alternative approach using ffmpeg directly
                try:
                    import subprocess
                    logger.info("Attempting alternative conversion method...")
                    
                    cmd = [
                        "ffmpeg", "-i", input_path, 
                        "-acodec", "pcm_s16le", 
                        "-ar", "16000", 
                        "-ac", "1", 
                        str(output_path), 
                        "-y"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        logger.info(f"Alternative conversion successful: {input_path} -> {output_path}")
                        return str(output_path)
                    else:
                        logger.error(f"Alternative conversion failed: {result.stderr}")
                        raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                        
                except FileNotFoundError:
                    raise Exception("FFmpeg not found. Please install FFmpeg to convert OPUS files.")
                except Exception as alt_error:
                    raise Exception(f"All conversion methods failed. Original error: {conversion_error}, Alternative error: {alt_error}")
            
        except Exception as e:
            logger.error(f"Failed to convert OPUS to WAV: {str(e)}")
            raise
    
    def convert_audio_to_wav(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Convert any supported audio format to WAV.
        
        Args:
            input_path: Path to the input audio file
            output_path: Path for the output WAV file (optional)
            
        Returns:
            Path to the converted WAV file
        """
        try:
            if output_path is None:
                # Create temporary WAV file in the same directory as input
                input_dir = Path(input_path).parent
                output_path = input_dir / f"converted_{Path(input_path).stem}.wav"
            
            # Load audio file and convert to WAV
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            
            logger.info(f"Converted audio to WAV: {input_path} -> {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to convert audio to WAV: {str(e)}")
            raise
    
    def read_wav_file(self, wav_path: str) -> Tuple[np.ndarray, int]:
        """
        Read WAV file and return audio data and sample rate.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Try using wave module first (faster for WAV files)
            if wav_path.lower().endswith('.wav'):
                try:
                    with wave.open(wav_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        audio_data = wav_file.readframes(n_frames)
                        
                        # Convert bytes to numpy array
                        if wav_file.getsampwidth() == 2:  # 16-bit
                            audio_data = np.frombuffer(audio_data, dtype=np.int16)
                        elif wav_file.getsampwidth() == 4:  # 32-bit
                            audio_data = np.frombuffer(audio_data, dtype=np.int32)
                        else:
                            raise ValueError(f"Unsupported sample width: {wav_file.getsampwidth()}")
                        
                        # Convert to float32 and normalize
                        audio_data = audio_data.astype(np.float32)
                        if np.max(np.abs(audio_data)) > 0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        
                        # Resample if needed
                        if sample_rate != self.sample_rate:
                            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
                            sample_rate = self.sample_rate
                        
                        return audio_data, sample_rate
                        
                except Exception as wave_error:
                    logger.warning(f"Failed to read with wave module: {wave_error}, falling back to librosa")
            
            # Fallback to librosa for non-WAV files or if wave fails
            audio_data, sample_rate = librosa.load(wav_path, sr=self.sample_rate)
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to read WAV file: {str(e)}")
            raise
    
    def get_speech_timestamps(self, audio_data: np.ndarray, sample_rate: int) -> List[dict]:
        """
        Get speech timestamps using Silero VAD.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of speech segments with start and end timestamps
        """
        if self.vad_session is None:
            logger.warning("VAD model not loaded, returning full audio as speech")
            return [{
                'start': 0.0,
                'end': len(audio_data) / sample_rate,
                'confidence': 1.0
            }]
        
        try:
            # Prepare audio for VAD
            audio_length = len(audio_data)
            
            # Get speech timestamps
            speech_timestamps = self._get_speech_timestamps_silero(
                audio_data, 
                sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                max_speech_duration_s=self.max_speech_duration_s,
                min_silence_duration_ms=self.min_silence_duration_ms,
                window_size_samples=self.window_size_samples,
                speech_pad_ms=self.speech_pad_ms
            )
            
            # Apply acceleration if enabled
            if self.enable_acceleration:
                acceleration_result = self._merge_segments_with_acceleration(
                    speech_timestamps, audio_data, sample_rate
                )
                
                # Convert to list of dictionaries with acceleration info
                speech_segments = []
                for start_sample, end_sample in acceleration_result['segments']:
                    start_time = start_sample / sample_rate
                    end_time = end_sample / sample_rate
                    speech_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'confidence': 1.0,
                        'accelerated': True
                    })
                
                # Add acceleration info to the first segment
                if speech_segments:
                    speech_segments[0]['acceleration_info'] = {
                        'applied': acceleration_result['acceleration_applied'],
                        'original_duration': acceleration_result['original_duration'],
                        'accelerated_duration': acceleration_result['accelerated_duration'],
                        'factor': acceleration_result['acceleration_factor']
                    }
            else:
                # Convert to list of dictionaries without acceleration
                speech_segments = []
                for start_sample, end_sample in speech_timestamps:
                    start_time = start_sample / sample_rate
                    end_time = end_sample / sample_rate
                    speech_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'confidence': 1.0,
                        'accelerated': False
                    })
            
            logger.info(f"Detected {len(speech_segments)} speech segments")
            return speech_segments
            
        except Exception as e:
            logger.error(f"Failed to get speech timestamps: {str(e)}")
            # Return full audio as fallback
            return [{
                'start': 0.0,
                'end': len(audio_data) / sample_rate,
                'confidence': 1.0,
                'accelerated': False
            }]
    
    def _get_speech_timestamps_silero(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_s: float = float('inf'),
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 1024,
        speech_pad_ms: int = 400
    ) -> List[Tuple[int, int]]:
        """
        Get speech timestamps using Silero VAD model.
        
        Args:
            audio: Audio data
            sampling_rate: Sample rate
            threshold: VAD threshold
            min_speech_duration_ms: Minimum speech duration in milliseconds
            max_speech_duration_s: Maximum speech duration in seconds
            min_silence_duration_ms: Minimum silence duration in milliseconds
            window_size_samples: Window size for VAD
            speech_pad_ms: Speech padding in milliseconds
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        try:
            if self.vad_session is None:
                return [(0, len(audio))]
            
            # Convert parameters to samples
            min_speech_samples = int(min_speech_duration_ms * sampling_rate / 1000)
            
            # Handle infinity case for max_speech_duration_s
            if max_speech_duration_s == float('inf'):
                max_speech_samples = float('inf')
            else:
                max_speech_samples = int(max_speech_duration_s * sampling_rate)
                
            min_silence_samples = int(min_silence_duration_ms * sampling_rate / 1000)
            speech_pad_samples = int(speech_pad_ms * sampling_rate / 1000)
            
            # Get VAD predictions
            vad_output = self._get_vad_predictions(audio, sampling_rate, window_size_samples)
            
            # Apply threshold
            speech_mask = vad_output > threshold
            
            # Find speech segments
            speech_segments = []
            start_sample = None
            
            for i, is_speech in enumerate(speech_mask):
                if is_speech and start_sample is None:
                    start_sample = i * window_size_samples
                elif not is_speech and start_sample is not None:
                    end_sample = i * window_size_samples
                    
                    # Apply minimum speech duration
                    if end_sample - start_sample >= min_speech_samples:
                        # Apply maximum speech duration
                        if max_speech_samples == float('inf') or end_sample - start_sample <= max_speech_samples:
                            speech_segments.append((start_sample, end_sample))
                        else:
                            # Split long segments
                            current_start = start_sample
                            while current_start < end_sample:
                                if max_speech_samples == float('inf'):
                                    current_end = end_sample
                                else:
                                    current_end = min(current_start + max_speech_samples, end_sample)
                                speech_segments.append((current_start, current_end))
                                current_start = current_end
                    
                    start_sample = None
            
            # Handle last segment
            if start_sample is not None:
                end_sample = len(audio)
                if end_sample - start_sample >= min_speech_samples:
                    if max_speech_samples == float('inf') or end_sample - start_sample <= max_speech_samples:
                        speech_segments.append((start_sample, end_sample))
                    else:
                        current_start = start_sample
                        while current_start < end_sample:
                            if max_speech_samples == float('inf'):
                                current_end = end_sample
                            else:
                                current_end = min(current_start + max_speech_samples, end_sample)
                            speech_segments.append((current_start, current_end))
                            current_start = current_end
            
            # Apply speech padding
            padded_segments = []
            for start, end in speech_segments:
                padded_start = max(0, start - speech_pad_samples)
                padded_end = min(len(audio), end + speech_pad_samples)
                padded_segments.append((padded_start, padded_end))
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(padded_segments)
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error in VAD processing: {str(e)}")
            # Return full audio as fallback
            return [(0, len(audio))]
    
    def _get_vad_predictions(self, audio: np.ndarray, sampling_rate: int, window_size: int) -> np.ndarray:
        """
        Get VAD predictions from Silero model.
        
        Args:
            audio: Audio data
            sampling_rate: Sample rate
            window_size: Window size for processing
            
        Returns:
            VAD predictions array
        """
        if self.vad_session is None:
            return np.ones(len(audio) // window_size)
        
        # Prepare input for model
        input_name = self.vad_session.get_inputs()[0].name
        output_name = self.vad_session.get_outputs()[0].name
        
        # Process audio in windows
        predictions = []
        for i in range(0, len(audio) - window_size + 1, window_size):
            window = audio[i:i + window_size]
            
            # Normalize audio
            window = window.astype(np.float32)
            if np.max(np.abs(window)) > 0:
                window = window / np.max(np.abs(window))
            
            # Reshape for model input
            window = window.reshape(1, -1)
            
            # Get prediction
            prediction = self.vad_session.run([output_name], {input_name: window})[0]
            predictions.append(prediction[0][0])
        
        return np.array(predictions)
    
    def _merge_overlapping_segments(self, segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping speech segments.
        
        Args:
            segments: List of (start, end) tuples
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x[0])
        
        merged = []
        current_start, current_end = sorted_segments[0]
        
        for start, end in sorted_segments[1:]:
            if start <= current_end:
                # Overlapping segments, merge them
                current_end = max(current_end, end)
            else:
                # Non-overlapping, add current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add last segment
        merged.append((current_start, current_end))
        
        return merged
    
    def _merge_segments_with_acceleration(self, segments: List[Tuple[int, int]], 
                                        audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Merge overlapping segments with optional audio acceleration.
        
        Args:
            segments: List of (start, end) tuples
            audio_data: Original audio data
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with merged segments and acceleration info
        """
        if not segments:
            return {
                'segments': [],
                'accelerated_audio': audio_data,
                'acceleration_applied': False,
                'original_duration': len(audio_data) / sample_rate,
                'accelerated_duration': len(audio_data) / sample_rate,
                'acceleration_factor': 1.0
            }
        
        # First merge overlapping segments
        merged_segments = self._merge_overlapping_segments(segments)
        
        if not self.enable_acceleration:
            return {
                'segments': merged_segments,
                'accelerated_audio': audio_data,
                'acceleration_applied': False,
                'original_duration': len(audio_data) / sample_rate,
                'accelerated_duration': len(audio_data) / sample_rate,
                'acceleration_factor': 1.0
            }
        
        # Apply acceleration to segments that meet criteria
        accelerated_segments = []
        total_original_duration = 0
        total_accelerated_duration = 0
        
        for start_sample, end_sample in merged_segments:
            segment_duration = (end_sample - start_sample) / sample_rate
            total_original_duration += segment_duration
            
            if segment_duration >= self.min_segment_duration_for_acceleration:
                # Apply acceleration to this segment
                segment_audio = audio_data[start_sample:end_sample]
                accelerated_segment_audio = self._accelerate_audio_segment(
                    segment_audio, sample_rate, self.acceleration_factor
                )
                accelerated_duration = len(accelerated_segment_audio) / sample_rate
                total_accelerated_duration += accelerated_duration
                
                # Update segment boundaries
                new_start = len(accelerated_segments) * sample_rate if accelerated_segments else 0
                new_end = new_start + len(accelerated_segment_audio)
                accelerated_segments.append((new_start, new_end))
                
                logger.info(f"Accelerated segment {start_sample/sample_rate:.2f}s-{end_sample/sample_rate:.2f}s "
                           f"({segment_duration:.2f}s) -> {new_start/sample_rate:.2f}s-{new_end/sample_rate:.2f}s "
                           f"({accelerated_duration:.2f}s) at {self.acceleration_factor}x speed")
            else:
                # Keep original speed for short segments
                segment_audio = audio_data[start_sample:end_sample]
                accelerated_duration = segment_duration
                total_accelerated_duration += accelerated_duration
                
                new_start = len(accelerated_segments) * sample_rate if accelerated_segments else 0
                new_end = new_start + len(segment_audio)
                accelerated_segments.append((new_start, new_end))
                
                logger.info(f"Kept original speed for short segment {start_sample/sample_rate:.2f}s-{end_sample/sample_rate:.2f}s "
                           f"({segment_duration:.2f}s)")
        
        # Combine all accelerated segments
        if accelerated_segments:
            total_samples = sum(end - start for start, end in accelerated_segments)
            accelerated_audio = np.zeros(total_samples, dtype=audio_data.dtype)
            
            current_pos = 0
            for start_sample, end_sample in merged_segments:
                segment_duration = (end_sample - start_sample) / sample_rate
                
                if segment_duration >= self.min_segment_duration_for_acceleration:
                    # Use accelerated segment
                    segment_audio = audio_data[start_sample:end_sample]
                    accelerated_segment = self._accelerate_audio_segment(
                        segment_audio, sample_rate, self.acceleration_factor
                    )
                else:
                    # Use original segment
                    accelerated_segment = audio_data[start_sample:end_sample]
                
                accelerated_audio[current_pos:current_pos + len(accelerated_segment)] = accelerated_segment
                current_pos += len(accelerated_segment)
        else:
            accelerated_audio = audio_data
        
        overall_acceleration_factor = total_original_duration / total_accelerated_duration if total_accelerated_duration > 0 else 1.0
        
        logger.info(f"Audio acceleration applied: {total_original_duration:.2f}s -> {total_accelerated_duration:.2f}s "
                   f"(overall {overall_acceleration_factor:.2f}x speed)")
        
        return {
            'segments': accelerated_segments,
            'accelerated_audio': accelerated_audio,
            'acceleration_applied': True,
            'original_duration': total_original_duration,
            'accelerated_duration': total_accelerated_duration,
            'acceleration_factor': overall_acceleration_factor
        }
    
    def _accelerate_audio_segment(self, audio_segment: np.ndarray, sample_rate: int, 
                                acceleration_factor: float) -> np.ndarray:
        """
        Accelerate an audio segment using librosa's time stretching.
        
        Args:
            audio_segment: Audio segment to accelerate
            sample_rate: Sample rate
            acceleration_factor: Speed factor (1.0 = normal, 2.0 = 2x speed)
            
        Returns:
            Accelerated audio segment
        """
        try:
            # Use librosa's time_stretch for audio acceleration
            # rate parameter: 1.0 = normal speed, 2.0 = 2x speed
            accelerated = librosa.effects.time_stretch(audio_segment, rate=acceleration_factor)
            return accelerated
        except Exception as e:
            logger.warning(f"Failed to accelerate audio segment: {str(e)}, using original")
            return audio_segment
    
    def process_audio_file(self, input_path: str, output_dir: Optional[str] = None) -> dict:
        """
        Process audio file with VAD: convert format if needed and detect speech segments.
        
        Args:
            input_path: Path to the input audio file
            output_dir: Directory for output files (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            input_path = Path(input_path)
            timing_info = {}
            start_time = time.time()
            
            # Step 1: Determine if conversion is needed
            step_start = time.time()
            needs_conversion = input_path.suffix.lower() not in ['.wav']
            is_opus = input_path.suffix.lower() == '.opus'
            format_check_time = time.time() - step_start
            timing_info['format_check'] = format_check_time
            
            # Step 2: Convert audio if needed
            wav_path = input_path
            conversion_time = 0
            
            if needs_conversion:
                try:
                    step_start = time.time()
                    if is_opus:
                        wav_path = self.convert_opus_to_wav(str(input_path))
                    else:
                        wav_path = self.convert_audio_to_wav(str(input_path))
                    conversion_time = time.time() - step_start
                    timing_info['audio_conversion'] = conversion_time
                    logger.info(f"Audio conversion completed in {conversion_time:.2f}s")
                except Exception as e:
                    conversion_time = time.time() - step_start
                    timing_info['audio_conversion'] = conversion_time
                    logger.warning(f"Audio conversion failed after {conversion_time:.2f}s: {str(e)}")
                    logger.info("Continuing with original file...")
                    wav_path = input_path
                    needs_conversion = False
            
            # Step 3: Read WAV file
            try:
                step_start = time.time()
                audio_data, sample_rate = self.read_wav_file(str(wav_path))
                read_time = time.time() - step_start
                timing_info['file_reading'] = read_time
                logger.info(f"File reading completed in {read_time:.2f}s")
            except Exception as e:
                read_time = time.time() - step_start
                timing_info['file_reading'] = read_time
                logger.error(f"Failed to read audio file after {read_time:.2f}s: {str(e)}")
                raise Exception(f"Cannot read audio file: {str(e)}")
            
            # Step 4: Get speech timestamps
            step_start = time.time()
            speech_segments = self.get_speech_timestamps(audio_data, sample_rate)
            vad_time = time.time() - step_start
            timing_info['vad_detection'] = vad_time
            logger.info(f"VAD detection completed in {vad_time:.2f}s")
            
            # Step 5: Calculate results
            step_start = time.time()
            total_speech_duration = sum(seg['end'] - seg['start'] for seg in speech_segments)
            total_audio_duration = len(audio_data) / sample_rate
            
            # Extract acceleration info if available
            acceleration_info = None
            if speech_segments and 'acceleration_info' in speech_segments[0]:
                acceleration_info = speech_segments[0]['acceleration_info']
                # Remove acceleration_info from the first segment to keep it clean
                speech_segments[0].pop('acceleration_info', None)
            
            result = {
                'input_file': str(input_path),
                'wav_file': str(wav_path),
                'converted': needs_conversion,
                'sample_rate': sample_rate,
                'total_audio_duration': total_audio_duration,
                'total_speech_duration': total_speech_duration,
                'speech_ratio': total_speech_duration / total_audio_duration if total_audio_duration > 0 else 0,
                'speech_segments': speech_segments,
                'segment_count': len(speech_segments),
                'acceleration_info': acceleration_info,
                'timing_info': timing_info
            }
            
            total_time = time.time() - start_time
            timing_info['total_vad_time'] = total_time
            
            logger.info(f"VAD processing completed in {total_time:.2f}s: {result['segment_count']} segments, "
                       f"speech ratio: {result['speech_ratio']:.2%}")
            if acceleration_info:
                logger.info(f"Audio acceleration: {acceleration_info['original_duration']:.2f}s -> "
                           f"{acceleration_info['accelerated_duration']:.2f}s ({acceleration_info['factor']:.2f}x)")
            logger.info(f"VAD timing breakdown: {timing_info}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process audio file: {str(e)}")
            raise
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to delete
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")
    
    def cleanup_converted_files(self, input_path: str):
        """
        Clean up converted WAV files created during processing.
        
        Args:
            input_path: Original input file path
        """
        try:
            input_path = Path(input_path)
            converted_file = input_path.parent / f"converted_{input_path.stem}.wav"
            
            if converted_file.exists():
                converted_file.unlink()
                logger.info(f"Cleaned up converted file: {converted_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup converted file: {str(e)}")


# Global VAD service instance
vad_service = VADService()

