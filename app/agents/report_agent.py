import asyncio
import os
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel
import json
from app.config import get_settings

logger = logging.getLogger(__name__)


class IdeaReport(BaseModel):
    """Structure for the generated report"""
    idea_summary: str
    market_analysis: str
    technical_feasibility: str
    implementation_steps: list[str]
    potential_challenges: list[str]
    success_factors: list[str]
    estimated_timeline: str
    next_actions: list[str]


class ReportAgentService:
    """Service for generating detailed reports from user ideas using OpenAI"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the report agent service"""
        try:
            from openai import OpenAI
            # Use the project's existing settings configuration
            settings = get_settings()
            self.client = OpenAI(
                api_key=api_key or settings.openai_api_key,
                organization=settings.openai_organization
            )
        except ImportError:
            raise ImportError("OpenAI package not found. Please install it with: pip install openai")
    
    async def generate_report(self, idea: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report for the given idea
        
        Args:
            idea (str): The user's idea or concept
            
        Returns:
            Dict[str, Any]: Structured report with analysis and recommendations
        """
        try:
            # Create the system prompt for comprehensive analysis
            system_prompt = """You are a comprehensive business analyst and report generator. Your job is to create detailed, structured reports from user ideas.

When given an idea, you should analyze it thoroughly and provide ONLY a valid JSON response with the following exact structure:
{
    "idea_summary": "Clear summary of the core concept",
    "market_analysis": "Market potential, target audience, and competitive landscape analysis",
    "technical_feasibility": "Technical requirements, complexity assessment, and implementation considerations",
    "implementation_steps": ["Step 1", "Step 2", "Step 3", "..."],
    "potential_challenges": ["Challenge 1", "Challenge 2", "..."],
    "success_factors": ["Factor 1", "Factor 2", "..."],
    "estimated_timeline": "Realistic timeline for implementation",
    "next_actions": ["Action 1", "Action 2", "..."]
}

CRITICAL: Return ONLY the JSON object. Do not include any additional text, explanations, or formatting outside the JSON."""

            user_prompt = f"""
Please analyze this content: "{idea}"

IMPORTANT: If this content is not clearly a business idea, startup concept, or entrepreneurial proposal, please respond with a structured analysis that treats it as a potential business opportunity by finding creative business applications or market potential.

Generate a detailed report that includes:
1. A clear summary of the core concept
2. Market analysis and potential (target audience, market size, competition)
3. Technical feasibility assessment (complexity, required technologies, development challenges)
4. Step-by-step implementation plan
5. Potential challenges and risks
6. Key success factors and strategies
7. Realistic timeline estimate
8. Immediate next actions to take

Even if the input seems unrelated to business (gaming, personal comments, etc.), find creative ways to analyze it from a business perspective (e.g., gaming content -> gaming platform business, personal statements -> social media platform, etc.).

Respond with ONLY the JSON object in the exact format specified. No additional text.
"""

            # Make the API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Check if response is valid
            if not response_text:
                return {
                    "success": False,
                    "error": "Empty response from OpenAI",
                    "raw_output": "",
                    "report": None
                }
            
            # Try to parse as JSON with improved error handling
            logger.debug(f"Raw OpenAI response: {response_text[:500]}...")
            report_data = self._parse_json_response(response_text)
            
            if report_data:
                logger.info("Successfully parsed JSON response")
                try:
                    # Validate with Pydantic model
                    validated_report = IdeaReport(**report_data)
                    
                    return {
                        "success": True,
                        "report": validated_report.model_dump(),
                        "raw_output": response_text,
                        "tokens_used": response.usage.total_tokens if response.usage else 0
                    }
                except Exception as validation_error:
                    return {
                        "success": False,
                        "error": f"Report validation failed: {str(validation_error)}",
                        "raw_output": response_text,
                        "report": None
                    }
            else:
                logger.error(f"Failed to parse JSON response. Raw response: {response_text}")
                return {
                    "success": False,
                    "error": "Failed to parse response as JSON",
                    "raw_output": response_text,
                    "report": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "report": None
            }
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Improved JSON parsing with multiple fallback strategies
        
        Args:
            response_text (str): Raw response from OpenAI
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON data or None if parsing fails
        """
        import re
        
        if not response_text:
            return None
        
        # Strategy 1: Direct JSON parsing
        try:
            result = json.loads(response_text)
            logger.debug("JSON parsed successfully with Strategy 1 (direct parsing)")
            return result
        except json.JSONDecodeError:
            logger.debug("Strategy 1 failed, trying Strategy 2")
        
        # Strategy 2: Extract JSON from markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Find JSON object in text (look for { ... })
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Strategy 4: Clean and try again
        cleaned_text = response_text.strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            "Here's the analysis:",
            "Here is the analysis:",
            "Analysis:",
            "JSON Response:",
            "Response:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Try to fix common JSON issues
        try:
            # Fix common issues like trailing commas, single quotes, etc.
            fixed_text = cleaned_text
            
            # Replace single quotes with double quotes (common issue)
            fixed_text = re.sub(r"'([^']*)':", r'"\1":', fixed_text)
            fixed_text = re.sub(r":\s*'([^']*)'", r': "\1"', fixed_text)
            
            # Remove trailing commas before closing brackets
            fixed_text = re.sub(r',\s*}', '}', fixed_text)
            fixed_text = re.sub(r',\s*]', ']', fixed_text)
            
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        return None

    async def analyze_idea_batch(self, ideas: list[str]) -> Dict[str, Any]:
        """
        Analyze multiple ideas in batch
        
        Args:
            ideas (list[str]): List of ideas to analyze
            
        Returns:
            Dict[str, Any]: Results for all ideas
        """
        results = []
        total_tokens = 0
        
        for idx, idea in enumerate(ideas, 1):
            print(f"Processing idea {idx}/{len(ideas)}: {idea[:50]}...")
            result = await self.generate_report(idea)
            results.append({
                "idea_id": idx,
                "idea": idea,
                "result": result
            })
            
            if result.get("tokens_used"):
                total_tokens += result["tokens_used"]
        
        return {
            "total_ideas": len(ideas),
            "results": results,
            "success_count": sum(1 for r in results if r["result"]["success"]),
            "failure_count": sum(1 for r in results if not r["result"]["success"]),
            "total_tokens_used": total_tokens
        }

    def generate_report_sync(self, idea: str) -> Dict[str, Any]:
        """
        Synchronous version of generate_report for easier testing
        
        Args:
            idea (str): The user's idea or concept
            
        Returns:
            Dict[str, Any]: Structured report with analysis and recommendations
        """
        return asyncio.run(self.generate_report(idea))


# Singleton instance
report_service = ReportAgentService() 