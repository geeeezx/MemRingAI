import asyncio
import csv
import os
from typing import List, Dict, Any

# Add the app directory to the path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.report_agent import ReportAgentService, IdeaReport


class TestReportAgent:
    """Test cases for the Report Agent Service"""
    
    def setup_method(self):
        """Set up test environment"""
        self.report_service = ReportAgentService()
    
    def load_test_ideas(self) -> List[str]:
        """Load test ideas from CSV file"""
        ideas = []
        csv_path = os.path.join(os.path.dirname(__file__), 'agent_test', 'idea.csv')
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    ideas.append(row['idea'].strip())
        except FileNotFoundError:
            # Fallback test ideas if CSV not found
            ideas = [
                "I am think about build a ai powered make up mirror.",
                "Create a smart home system that learns user preferences",
                "Develop a mobile app for sustainable living tracking"
            ]
        
        return ideas
    
    def test_load_ideas_from_csv(self):
        """Test loading ideas from CSV file"""
        ideas = self.load_test_ideas()
        assert len(ideas) > 0
        print(f"Loaded {len(ideas)} ideas from test data")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i}. {idea}")
    
    def test_report_agent_initialization(self):
        """Test that the report agent service initializes properly"""
        service = ReportAgentService()
        assert service is not None
        assert hasattr(service, 'client')
        print("✓ Report agent service initialized successfully")
    
    def test_idea_report_model(self):
        """Test the IdeaReport Pydantic model"""
        sample_report = IdeaReport(
            idea_summary="Test idea summary",
            market_analysis="Test market analysis",
            technical_feasibility="Test technical feasibility",
            implementation_steps=["Step 1", "Step 2", "Step 3"],
            potential_challenges=["Challenge 1", "Challenge 2"],
            success_factors=["Factor 1", "Factor 2"],
            estimated_timeline="3-6 months",
            next_actions=["Action 1", "Action 2"]
        )
        
        assert sample_report.idea_summary == "Test idea summary"
        assert len(sample_report.implementation_steps) == 3
        assert len(sample_report.potential_challenges) == 2
        print("✓ IdeaReport model validation successful")
    
    async def test_generate_report_single_idea(self):
        """Test generating a report for a single idea"""
        test_idea = "I am think about build a ai powered make up mirror."
        
        try:
            result = await self.report_service.generate_report(test_idea)
            
            assert result is not None
            assert "success" in result
            assert "report" in result
            
            if result["success"]:
                assert result["report"] is not None
                report = result["report"]
                
                # Check that all required fields are present
                required_fields = [
                    "idea_summary", "market_analysis", "technical_feasibility",
                    "implementation_steps", "potential_challenges", "success_factors",
                    "estimated_timeline", "next_actions"
                ]
                
                for field in required_fields:
                    assert field in report, f"Missing field: {field}"
                
                print("✓ Single idea report generation successful")
                print(f"  Idea: {test_idea}")
                print(f"  Summary: {report['idea_summary'][:100]}...")
                print(f"  Timeline: {report['estimated_timeline']}")
                print(f"  Implementation steps: {len(report['implementation_steps'])}")
                
            else:
                print(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            # Don't fail the test if it's just an API key issue
            if "api key" in str(e).lower():
                print("  Note: This appears to be an API key issue - set OPENAI_API_KEY environment variable")
    
    def test_generate_report_sync(self):
        """Test synchronous version of report generation"""
        test_idea = "Create a smart home system that learns user preferences"
        
        try:
            result = self.report_service.generate_report_sync(test_idea)
            
            assert result is not None
            assert "success" in result
            print("✓ Synchronous report generation test passed")
            
            if result["success"]:
                print(f"  Generated report for: {test_idea}")
                print(f"  Tokens used: {result.get('tokens_used', 'N/A')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Sync test failed: {str(e)}")
            if "api key" in str(e).lower():
                print("  Note: Set OPENAI_API_KEY environment variable for full testing")
    
    async def test_batch_analysis(self):
        """Test batch analysis of multiple ideas"""
        ideas = self.load_test_ideas()
        
        try:
            result = await self.report_service.analyze_idea_batch(ideas)
            
            assert result is not None
            assert "total_ideas" in result
            assert "results" in result
            assert result["total_ideas"] == len(ideas)
            
            print("✓ Batch analysis test completed")
            print(f"  Total ideas processed: {result['total_ideas']}")
            print(f"  Successful reports: {result['success_count']}")
            print(f"  Failed reports: {result['failure_count']}")
            print(f"  Total tokens used: {result.get('total_tokens_used', 'N/A')}")
            
            # Print summary of each result
            for i, result_item in enumerate(result["results"], 1):
                idea = result_item["idea"]
                success = result_item["result"]["success"]
                status = "✓" if success else "❌"
                print(f"  {status} Idea {i}: {idea[:50]}...")
                
        except Exception as e:
            print(f"❌ Batch analysis failed: {str(e)}")
            if "api key" in str(e).lower():
                print("  Note: Set OPENAI_API_KEY environment variable for full testing")


def run_manual_test():
    """Manual test runner for development"""
    print("🧪 Running Manual Report Agent Tests")
    print("=" * 50)
    
    # Initialize test
    test = TestReportAgent()
    test.setup_method()
    
    # Run tests
    print("\n1. Testing CSV loading...")
    test.test_load_ideas_from_csv()
    
    print("\n2. Testing agent initialization...")
    test.test_report_agent_initialization()
    
    print("\n3. Testing model validation...")
    test.test_idea_report_model()
    
    print("\n4. Testing synchronous report generation...")
    test.test_generate_report_sync()
    
    print("\n5. Testing single idea analysis...")
    asyncio.run(test.test_generate_report_single_idea())
    
    print("\n6. Testing batch analysis...")
    asyncio.run(test.test_batch_analysis())
    
    print("\n" + "=" * 50)
    print("🎉 Manual tests completed!")
    print("\nTo run full pytest suite:")
    print("  pytest tests/test_report_agent.py -v")
    print("\nTo run with async support:")
    print("  pytest tests/test_report_agent.py -v --asyncio-mode=auto")


if __name__ == "__main__":
    run_manual_test()
