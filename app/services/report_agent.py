import asyncio
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel
import json
from ..config import get_settings


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

When given an idea, you should analyze it thoroughly and provide a JSON response with the following structure:
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

Your reports should be professional, actionable, and comprehensive. Focus on practical insights and realistic assessments."""

            user_prompt = f"""
Please analyze this idea comprehensively: "{idea}"

Generate a detailed report that includes:
1. A clear summary of the idea
2. Market analysis and potential (target audience, market size, competition)
3. Technical feasibility assessment (complexity, required technologies, development challenges)
4. Step-by-step implementation plan
5. Potential challenges and risks
6. Key success factors and strategies
7. Realistic timeline estimate
8. Immediate next actions to take

Provide the response in the exact JSON format specified in the system prompt.
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
            
            # Try to parse as JSON
            try:
                report_data = json.loads(response_text)
                # Validate with Pydantic model
                validated_report = IdeaReport(**report_data)
                
                return {
                    "success": True,
                    "report": validated_report.model_dump(),
                    "raw_output": response_text,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw text
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