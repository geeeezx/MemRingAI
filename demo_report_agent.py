#!/usr/bin/env python3
"""
Demo script for the OpenAI Report Agent
Generates detailed reports from user ideas
"""

import asyncio
import os
import sys
import json
from typing import List

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.report_agent import ReportAgentService


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_report(report: dict, idea: str):
    """Print a formatted report"""
    print(f"\n💡 IDEA: {idea}")
    print("-" * 50)
    
    print(f"\n📋 SUMMARY:")
    print(f"   {report['idea_summary']}")
    
    print(f"\n📊 MARKET ANALYSIS:")
    print(f"   {report['market_analysis']}")
    
    print(f"\n🔧 TECHNICAL FEASIBILITY:")
    print(f"   {report['technical_feasibility']}")
    
    print(f"\n📝 IMPLEMENTATION STEPS:")
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


async def demo_single_idea():
    """Demo: Generate report for a single idea"""
    print_header("SINGLE IDEA ANALYSIS DEMO")
    
    # Sample idea
    idea = "I am think about build a ai powered make up mirror."
    
    print(f"🎯 Analyzing idea: {idea}")
    print("⏳ Generating comprehensive report...")
    
    # Initialize service
    service = ReportAgentService()
    
    try:
        # Generate report
        result = await service.generate_report(idea)
        
        if result["success"]:
            print("✅ Report generated successfully!")
            print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
            
            # Print formatted report
            print_report(result["report"], idea)
            
        else:
            print(f"❌ Report generation failed: {result.get('error', 'Unknown error')}")
            print(f"Raw output: {result.get('raw_output', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        if "api key" in str(e).lower():
            print("💡 Tip: Set your OpenAI API key in the OPENAI_API_KEY environment variable")


async def demo_batch_analysis():
    """Demo: Analyze multiple ideas in batch"""
    print_header("BATCH ANALYSIS DEMO")
    
    # Sample ideas
    ideas = [
        "I am think about build a ai powered make up mirror.",
        "Create a smart home system that learns user preferences",
        "Develop a mobile app for sustainable living tracking",
        "Build a AI-powered personal fitness coach app"
    ]
    
    print(f"📊 Analyzing {len(ideas)} ideas in batch...")
    
    # Initialize service
    service = ReportAgentService()
    
    try:
        # Analyze batch
        result = await service.analyze_idea_batch(ideas)
        
        print(f"✅ Batch analysis completed!")
        print(f"📈 Total ideas: {result['total_ideas']}")
        print(f"✅ Successful: {result['success_count']}")
        print(f"❌ Failed: {result['failure_count']}")
        print(f"📊 Total tokens used: {result.get('total_tokens_used', 'N/A')}")
        
        # Print summary for each idea
        print("\n📋 BATCH RESULTS SUMMARY:")
        for i, item in enumerate(result["results"], 1):
            idea = item["idea"]
            success = item["result"]["success"]
            status = "✅" if success else "❌"
            print(f"  {status} Idea {i}: {idea[:50]}...")
            
            if success and len(result["results"]) <= 2:  # Show detailed reports for small batches
                print_report(item["result"]["report"], idea)
                print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        if "api key" in str(e).lower():
            print("💡 Tip: Set your OpenAI API key in the OPENAI_API_KEY environment variable")


def demo_csv_loading():
    """Demo: Load ideas from CSV file"""
    print_header("CSV LOADING DEMO")
    
    csv_path = os.path.join("tests", "agent_test", "idea.csv")
    
    if os.path.exists(csv_path):
        print(f"📁 Loading ideas from: {csv_path}")
        
        import csv
        ideas = []
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    ideas.append(row['idea'].strip())
            
            print(f"✅ Loaded {len(ideas)} ideas from CSV:")
            for i, idea in enumerate(ideas, 1):
                print(f"  {i}. {idea}")
            
            return ideas
            
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            
    else:
        print(f"❌ CSV file not found: {csv_path}")
        print("📝 Using sample ideas instead")
        
    return []


async def interactive_demo():
    """Interactive demo where user can input their own ideas"""
    print_header("INTERACTIVE IDEA ANALYZER")
    
    print("💭 Enter your idea and get a comprehensive report!")
    print("✨ Type 'quit' to exit")
    
    service = ReportAgentService()
    
    while True:
        try:
            idea = input("\n🎯 Enter your idea: ").strip()
            
            if not idea:
                print("❌ Please enter a valid idea")
                continue
                
            if idea.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            print("⏳ Analyzing your idea...")
            
            # Generate report
            result = await service.generate_report(idea)
            
            if result["success"]:
                print("✅ Analysis complete!")
                print_report(result["report"], idea)
                
                # Ask if user wants to save the report
                save = input("\n💾 Save this report to file? (y/n): ").strip().lower()
                if save == 'y':
                    filename = f"report_{hash(idea) % 10000}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump({
                            "idea": idea,
                            "report": result["report"],
                            "timestamp": str(asyncio.get_event_loop().time())
                        }, f, indent=2)
                    print(f"📄 Report saved as: {filename}")
                    
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


async def main():
    """Main demo function"""
    print_header("🤖 OPENAI REPORT AGENT DEMO")
    print("🎯 Transform your ideas into comprehensive business reports!")
    print("📋 Based on OpenAI GPT-4 for professional analysis")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ WARNING: OPENAI_API_KEY environment variable not set")
        print("💡 Set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("🔗 Get your API key at: https://platform.openai.com/api-keys")
        print("\n📝 Demo will continue but may fail without valid API key")
    
    # Demo menu
    while True:
        print("\n🔍 DEMO OPTIONS:")
        print("1. 📝 Single Idea Analysis")
        print("2. 📊 Batch Analysis (Multiple Ideas)")
        print("3. 📁 Load Ideas from CSV")
        print("4. 💭 Interactive Mode")
        print("5. 🚪 Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            await demo_single_idea()
        elif choice == "2":
            await demo_batch_analysis()
        elif choice == "3":
            ideas = demo_csv_loading()
            if ideas:
                print(f"\n📊 Analyzing {len(ideas)} ideas from CSV...")
                service = ReportAgentService()
                result = await service.analyze_idea_batch(ideas)
                print(f"✅ Analysis complete! Success: {result['success_count']}/{result['total_ideas']}")
        elif choice == "4":
            await interactive_demo()
        elif choice == "5":
            print("👋 Thank you for using the Report Agent Demo!")
            break
        else:
            print("❌ Invalid option. Please select 1-5.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        if "api key" in str(e).lower():
            print("💡 Make sure to set your OpenAI API key!") 