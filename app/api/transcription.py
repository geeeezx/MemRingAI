"""Transcription API endpoints."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import (
    TranscriptionRequest,
    TranscriptionResponse,
    ErrorResponse,
    HealthResponse,
    TranscriptionReportRequest,
    TranscriptionReportResponse,
    IdeaReport
)
from app.services.ASR import get_asr_service, ASRProvider
from app.services.file_service import FileService
from app.services.vad import vad_service

logger = logging.getLogger(__name__)

router = APIRouter()


def get_asr_service_dependency(provider: ASRProvider = ASRProvider.OPENAI):
    """Dependency to get ASR service instance."""
    return get_asr_service(provider)


def get_file_service() -> FileService:
    """Dependency to get file service instance."""
    return FileService()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    asr_service = Depends(get_asr_service_dependency)
) -> HealthResponse:
    """
    Health check endpoint to verify service status and ASR configuration.
    
    Returns:
        HealthResponse with service status and ASR configuration status
    """
    try:
        asr_configured = await asr_service.validate_api_key()
        return HealthResponse(openai_configured=asr_configured)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(openai_configured=False)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model: Optional[str] = Form("whisper-1", description="Model to use"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: Optional[str] = Form("verbose_json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Temperature for sampling"),
    provider: Optional[str] = Form("openai", description="ASR provider (openai, volcengine, auto)"),
    enable_vad: Optional[bool] = Form(True, description="Enable Voice Activity Detection"),
    asr_service = Depends(get_asr_service_dependency),
    file_service: FileService = Depends(get_file_service)
) -> TranscriptionResponse:
    """
    Transcribe an uploaded audio file using ASR API.
    
    Args:
        file: Audio file to transcribe
        model: Model to use
        language: Language code for transcription
        prompt: Optional prompt to guide transcription
        response_format: Response format (verbose_json, json, text, srt, vtt)
        temperature: Temperature for sampling (0.0 to 1.0)
        provider: ASR provider to use (openai, volcengine, auto)
        asr_service: ASR service instance
        file_service: File service instance
        
    Returns:
        TranscriptionResponse with transcription results
        
    Raises:
        HTTPException: If transcription fails or file is invalid
    """
    temp_file_path = None
    vad_result = None
    timing_info = {}
    
    try:
        # Start timing
        start_time = time.time()
        
        # Step 1: Validate and save uploaded file
        step_start = time.time()
        temp_file_path = await file_service.save_uploaded_file(file)
        file_save_time = time.time() - step_start
        timing_info['file_save'] = file_save_time
        logger.info(f"File save completed in {file_save_time:.2f}s")
        
        # Step 2: Process with VAD if enabled
        vad_time = 0
        if enable_vad:
            try:
                step_start = time.time()
                logger.info(f"Processing audio with VAD: {file.filename}")
                
                # VAD acceleration is now configured via environment variables
                logger.info("VAD processing with acceleration settings from environment variables")
                
                vad_result = vad_service.process_audio_file(temp_file_path)
                vad_time = time.time() - step_start
                timing_info['vad_processing'] = vad_time
                logger.info(f"VAD processing completed in {vad_time:.2f}s: {vad_result['segment_count']} segments, "
                           f"speech ratio: {vad_result['speech_ratio']:.2%}")
                
                # Use the converted WAV file for transcription if conversion was needed
                if vad_result['converted']:
                    temp_file_path = vad_result['wav_file']
                    
            except Exception as e:
                vad_time = time.time() - step_start
                timing_info['vad_processing'] = vad_time
                logger.warning(f"VAD processing failed after {vad_time:.2f}s, continuing without VAD: {str(e)}")
                vad_result = None
        
        # Step 3: Create transcription request
        step_start = time.time()
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=["word", "segment"]
        )
        request_time = time.time() - step_start
        timing_info['request_preparation'] = request_time
        
        # Step 4: Get ASR service based on provider
        step_start = time.time()
        if provider != "auto":
            try:
                asr_provider = ASRProvider(provider)
                asr_service = get_asr_service(asr_provider)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ASR provider: {provider}"
                )
        service_setup_time = time.time() - step_start
        timing_info['service_setup'] = service_setup_time
        
        # Step 5: Perform transcription
        step_start = time.time()
        logger.info(f"Starting transcription for file: {file.filename} using {provider}")
        result = await asr_service.transcribe_audio(temp_file_path, request)
        transcription_time = time.time() - step_start
        timing_info['transcription'] = transcription_time
        logger.info(f"Transcription completed in {transcription_time:.2f}s")
        
        # Step 6: Add VAD information to response if available
        step_start = time.time()
        if vad_result:
            result.vad_info = {
                'segment_count': vad_result['segment_count'],
                'speech_ratio': vad_result['speech_ratio'],
                'total_speech_duration': vad_result['total_speech_duration'],
                'speech_segments': vad_result['speech_segments'],
                'timing_info': vad_result.get('timing_info', {}),
                'acceleration_info': vad_result.get('acceleration_info')
            }
        
        # Add timing information to response
        total_time = time.time() - start_time
        timing_info['total_time'] = total_time
        result.timing_info = timing_info
        
        response_prep_time = time.time() - step_start
        timing_info['response_preparation'] = response_prep_time
        
        logger.info(f"Transcription completed successfully for file: {file.filename}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Timing breakdown: {timing_info}")
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Clean up VAD converted files if they exist
        if vad_result and vad_result.get('converted'):
            try:
                vad_service.cleanup_converted_files(vad_result)
            except Exception as e:
                logger.warning(f"Failed to cleanup VAD converted files: {str(e)}")
        # Clean up temporary files
        if temp_file_path:
            await file_service.cleanup_temp_file(temp_file_path)
        


@router.post("/transcribe/url", response_model=TranscriptionResponse)
async def transcribe_from_url(
    url: str = Form(..., description="URL of the audio file to transcribe"),
    model: Optional[str] = Form("whisper-1", description="Model to use"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: Optional[str] = Form("verbose_json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Temperature for sampling"),
    provider: Optional[str] = Form("auto", description="ASR provider (openai, volcengine, auto)"),
    asr_service = Depends(get_asr_service_dependency),
    file_service: FileService = Depends(get_file_service)
) -> TranscriptionResponse:
    """
    Transcribe an audio file from URL using ASR API.
    
    Args:
        url: URL of the audio file to transcribe
        model: Model to use
        language: Language code for transcription
        prompt: Optional prompt to guide transcription
        response_format: Response format (verbose_json, json, text, srt, vtt)
        temperature: Temperature for sampling (0.0 to 1.0)
        provider: ASR provider to use (openai, volcengine, auto)
        asr_service: ASR service instance
        file_service: File service instance
        
    Returns:
        TranscriptionResponse with transcription results
        
    Raises:
        HTTPException: If transcription fails or URL is invalid
    """
    temp_file_path = None
    
    try:
        from pathlib import Path
        import httpx
        
        # Download file from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file_path = await file_service.save_uploaded_file_from_bytes(
                response.content, 
                f"audio_from_url{Path(url).suffix or '.mp3'}"
            )
        
        # Create transcription request
        request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=["word", "segment"]
        )
        
        # Get ASR service based on provider
        if provider != "auto":
            try:
                asr_provider = ASRProvider(provider)
                asr_service = get_asr_service(asr_provider)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ASR provider: {provider}"
                )
        
        # Perform transcription
        logger.info(f"Starting transcription for URL: {url} using {provider}")
        result = await asr_service.transcribe_audio(temp_file_path, request)
        
        logger.info(f"Transcription completed successfully for URL: {url}")
        return result
        
    except httpx.HTTPError as e:
        logger.error(f"Failed to download file from URL: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download file from URL: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_file_path:
            await file_service.cleanup_temp_file(temp_file_path)


@router.get("/supported-formats")
async def get_supported_formats() -> dict:
    """
    Get supported audio formats and configuration.
    
    Returns:
        Dictionary with supported formats and limits
    """
    settings = get_settings()
    return {
        "supported_formats": settings.allowed_audio_extensions,
        "max_file_size_mb": settings.max_file_size / (1024 * 1024),
        "default_model": settings.default_model,
        "default_response_format": settings.default_response_format
    }


@router.get("/providers")
async def get_available_providers() -> dict:
    """
    Get available ASR providers and their status.
    
    Returns:
        Dictionary with available providers and their configuration status
    """
    from app.services.ASR import get_available_providers as get_providers
    
    providers = get_providers()
    provider_info = {}
    
    for provider in providers:
        try:
            service = get_asr_service(provider)
            is_valid = await service.validate_api_key()
            provider_info[provider.value] = {
                "available": True,
                "configured": True,
                "valid": is_valid
            }
        except Exception as e:
            provider_info[provider.value] = {
                "available": True,
                "configured": False,
                "valid": False,
                "error": str(e)
            }
    
    return {
        "providers": provider_info,
        "default": "auto"
    }


@router.post("/transcribe-and-report", response_model=TranscriptionReportResponse)
async def transcribe_and_generate_report(
    file: UploadFile = File(..., description="Audio file to transcribe and analyze"),
    model: Optional[str] = Form("whisper-1", description="Model to use"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription"),
    response_format: Optional[str] = Form("verbose_json", description="Response format"),
    temperature: Optional[float] = Form(0.0, description="Temperature for sampling"),
    provider: Optional[str] = Form("auto", description="ASR provider (openai, volcengine, auto)"),
    enable_vad: Optional[bool] = Form(True, description="Enable Voice Activity Detection"),
    generate_report: bool = Form(True, description="Whether to generate a business report from transcription"),
    report_focus: Optional[str] = Form(None, description="Optional focus area for the report"),
    asr_service = Depends(get_asr_service_dependency),
    file_service: FileService = Depends(get_file_service)
) -> TranscriptionReportResponse:
    """
    Transcribe an uploaded audio file and generate a comprehensive business report from the transcription.
    
    This endpoint combines transcription and report generation into a single workflow:
    1. Transcribe the audio file using ASR
    2. Generate a comprehensive business report from the transcribed text
    
    Args:
        file: Audio file to transcribe and analyze
        model: Model to use for transcription
        language: Language code for transcription
        prompt: Optional prompt to guide transcription
        response_format: Response format (verbose_json, json, text, srt, vtt)
        temperature: Temperature for sampling (0.0 to 1.0)
        provider: ASR provider to use (openai, volcengine, auto)
        generate_report: Whether to generate a business report from transcription
        report_focus: Optional focus area for the report
        asr_service: ASR service instance
        file_service: File service instance
        
    Returns:
        TranscriptionReportResponse with both transcription and report results
        
    Raises:
        HTTPException: If transcription or report generation fails
    """
    import time
    start_time = time.time()
    temp_file_path = None
    vad_result = None
    
    try:
        # Step 1: Save uploaded file
        temp_file_path = await file_service.save_uploaded_file(file)
        
        # Step 2: Process with VAD if enabled  
        if enable_vad:
            try:
                logger.info(f"Processing audio with VAD: {file.filename}")
                vad_result = vad_service.process_audio_file(temp_file_path)
                logger.info(f"VAD completed: {vad_result['segment_count']} segments, "
                           f"speech ratio: {vad_result['speech_ratio']:.2%}")
                
                # Use converted WAV file for transcription if conversion was needed
                if vad_result['converted']:
                    temp_file_path = vad_result['wav_file']
            except Exception as e:
                logger.warning(f"VAD processing failed, continuing without VAD: {str(e)}")
                vad_result = None
        
        # Step 3: Transcribe the audio file
        logger.info(f"Starting transcription for file: {file.filename} using {provider}")
        
        # Create transcription request
        transcription_request = TranscriptionRequest(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=["word", "segment"]
        )
        
        # Get ASR service based on provider
        if provider != "auto":
            try:
                asr_provider = ASRProvider(provider)
                asr_service = get_asr_service(asr_provider)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ASR provider: {provider}"
                )
        
        # Perform transcription
        transcription_result = await asr_service.transcribe_audio(temp_file_path, transcription_request)
        
        # Add VAD information to transcription result if available
        if vad_result:
            transcription_result.vad_info = {
                'segment_count': vad_result['segment_count'],
                'speech_ratio': vad_result['speech_ratio'],
                'total_speech_duration': vad_result['total_speech_duration'],
                'speech_segments': vad_result['speech_segments'],
                'acceleration_info': vad_result.get('acceleration_info')
            }
        
        logger.info(f"Transcription completed successfully for file: {file.filename}")
        
        # Initialize response variables
        report_result = None
        report_success = False
        report_error = None
        total_tokens = 0
        
        # Step 2: Generate report from transcription (if requested)
        if generate_report and transcription_result.text.strip():
            try:
                logger.info("Starting report generation from transcribed text")
                
                # Import report service
                from app.agents.report_agent import ReportAgentService
                report_service = ReportAgentService()
                
                # Prepare the idea text with optional focus and VAD insights
                idea_text = transcription_result.text.strip()
                
                # Enhance prompt with VAD insights if available
                vad_context = ""
                if hasattr(transcription_result, 'vad_info') and transcription_result.vad_info:
                    vad_info = transcription_result.vad_info
                    speech_ratio = vad_info.get('speech_ratio', 0)
                    segment_count = vad_info.get('segment_count', 0)
                    
                    if speech_ratio > 0.7:
                        confidence_level = "high confidence"
                    elif speech_ratio > 0.4:
                        confidence_level = "moderate confidence" 
                    else:
                        confidence_level = "low confidence"
                    
                    vad_context = f"\n\nAudio Analysis Context: This idea was presented with {confidence_level} ({speech_ratio:.0%} speech content) across {segment_count} segments, suggesting {'well-structured delivery' if segment_count > 5 else 'concise presentation'}."
                
                # Combine focus and VAD context
                if report_focus:
                    idea_text = f"Focus on {report_focus}: {idea_text}{vad_context}"
                else:
                    idea_text = f"{idea_text}{vad_context}"
                
                # Generate enhanced report
                report_response = await report_service.generate_report(idea_text)
                
                if report_response["success"]:
                    report_result = IdeaReport(**report_response["report"])
                    report_success = True
                    total_tokens = report_response.get("tokens_used", 0)
                    logger.info("Report generation completed successfully")
                else:
                    report_error = report_response.get("error", "Unknown error in report generation")
                    logger.error(f"Report generation failed: {report_error}")
                    
            except Exception as e:
                report_error = f"Report generation failed: {str(e)}"
                logger.error(report_error)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create combined response
        response = TranscriptionReportResponse(
            transcription=transcription_result,
            report=report_result,
            report_generation_success=report_success,
            report_error=report_error,
            total_tokens_used=total_tokens,
            processing_time_seconds=processing_time,
            status="success",
            message=f"Successfully processed audio file{' and generated report' if report_success else ''}"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        logger.error(f"Combined transcription and report generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Combined processing failed: {str(e)}"
        )
    finally:
        # Clean up VAD converted files if they exist
        if vad_result and vad_result.get('converted'):
            try:
                vad_service.cleanup_converted_files(vad_result)
            except Exception as e:
                logger.warning(f"Failed to cleanup VAD converted files: {str(e)}")
        
        # Clean up temporary file
        if temp_file_path:
            await file_service.cleanup_temp_file(temp_file_path) 