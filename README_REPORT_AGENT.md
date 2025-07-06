# 🤖 OpenAI Report Agent

Transform your random ideas into comprehensive business reports using OpenAI's GPT-4!

## 🎯 Overview

The OpenAI Report Agent is a powerful AI-powered system that takes your raw ideas and generates detailed, professional reports with:

- **Market Analysis** - Target audience, competition, and market potential
- **Technical Feasibility** - Implementation complexity and requirements
- **Implementation Steps** - Clear, actionable roadmap
- **Risk Assessment** - Potential challenges and mitigation strategies
- **Success Factors** - Key elements for project success
- **Timeline Estimation** - Realistic project timelines
- **Next Actions** - Immediate steps to take

## 📋 Features

- ✅ **Single Idea Analysis** - Deep dive into one specific idea
- ✅ **Batch Processing** - Analyze multiple ideas at once
- ✅ **CSV Import** - Load ideas from CSV files
- ✅ **Interactive Mode** - Real-time idea analysis
- ✅ **Structured Output** - Professional, formatted reports
- ✅ **Token Tracking** - Monitor API usage and costs
- ✅ **Error Handling** - Robust error management
- ✅ **Export Options** - Save reports to JSON files

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenAI API key
- Dependencies installed (see installation)

### 2. Installation

The system is built into the MemRingAI project. Dependencies are managed with `uv`:

```bash
# Install dependencies (already done if you have the project)
uv install

# Or install OpenAI agents specifically
uv add openai-agents
```

### 3. Set Up OpenAI API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

Get your API key at: https://platform.openai.com/api-keys

### 4. Run the Demo

```bash
# Run the interactive demo
python demo_report_agent.py

# Or run tests
python tests/test_report_agent.py
```

## 📖 Usage Examples

### Single Idea Analysis

```python
from app.services.report_agent import ReportAgentService

# Initialize the service
service = ReportAgentService()

# Analyze an idea
result = await service.generate_report("I want to build an AI-powered makeup mirror")

if result["success"]:
    report = result["report"]
    print(f"Summary: {report['idea_summary']}")
    print(f"Timeline: {report['estimated_timeline']}")
    print(f"Steps: {report['implementation_steps']}")
```

### Batch Analysis

```python
ideas = [
    "AI-powered makeup mirror",
    "Smart home automation system",
    "Mobile app for sustainable living"
]

# Analyze multiple ideas
results = await service.analyze_idea_batch(ideas)

print(f"Processed {results['total_ideas']} ideas")
print(f"Success rate: {results['success_count']}/{results['total_ideas']}")
```

### CSV Processing

Create a CSV file with your ideas:

```csv
idea_id,idea
1,AI-powered makeup mirror
2,Smart home automation system
3,Mobile app for sustainable living
```

Load and analyze:

```python
import csv

# Load ideas from CSV
ideas = []
with open('ideas.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ideas.append(row['idea'])

# Analyze
results = await service.analyze_idea_batch(ideas)
```

## 🏗️ Architecture

### Core Components

1. **ReportAgentService** - Main service class
2. **IdeaReport** - Pydantic model for structured output
3. **OpenAI Integration** - Direct GPT-4 API calls
4. **Error Handling** - Comprehensive error management

### File Structure

```
app/
├── services/
│   └── report_agent.py      # Main agent service
tests/
├── test_report_agent.py     # Comprehensive tests
└── agent_test/
    └── idea.csv            # Test data
demo_report_agent.py        # Interactive demo
README_REPORT_AGENT.md      # This file
```

## 🧪 Testing

### Run All Tests

```bash
# Run with pytest
pytest tests/test_report_agent.py -v

# Run with async support
pytest tests/test_report_agent.py -v --asyncio-mode=auto

# Run manual tests
python tests/test_report_agent.py
```

### Test Coverage

- ✅ Service initialization
- ✅ Model validation
- ✅ Single idea analysis
- ✅ Batch processing
- ✅ CSV loading
- ✅ Error handling
- ✅ API integration

## 📊 Sample Report Output

```json
{
  "idea_summary": "An AI-powered makeup mirror that uses computer vision...",
  "market_analysis": "The beauty tech market is growing at 20% annually...",
  "technical_feasibility": "Requires computer vision, AI models, and hardware integration...",
  "implementation_steps": [
    "Market research and user interviews",
    "Technical architecture design",
    "Prototype development",
    "Testing and validation"
  ],
  "potential_challenges": [
    "Hardware integration complexity",
    "AI model accuracy",
    "Manufacturing costs"
  ],
  "success_factors": [
    "Strong user experience design",
    "Accurate AI recommendations",
    "Competitive pricing"
  ],
  "estimated_timeline": "12-18 months",
  "next_actions": [
    "Conduct market research",
    "Build technical prototype",
    "Secure funding"
  ]
}
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Service Configuration

```python
# Custom initialization
service = ReportAgentService(api_key="your-key-here")

# Default uses environment variable
service = ReportAgentService()
```

## 💰 Cost Considerations

- Uses GPT-4o-mini for cost efficiency
- Typical report: ~500-1000 tokens
- Cost per report: ~$0.01-0.02
- Batch processing optimizes API usage

## 🚨 Error Handling

The system handles various error scenarios:

- **Missing API Key** - Clear error message with setup instructions
- **API Rate Limits** - Graceful handling with retry logic
- **Invalid JSON** - Fallback to raw text output
- **Network Issues** - Comprehensive error reporting

## 🎨 Customization

### Custom Report Structure

```python
class CustomReport(BaseModel):
    summary: str
    custom_field: str
    # Add your custom fields

# Modify the service to use custom model
service.report_agent.output_type = CustomReport
```

### Custom Prompts

Edit the system prompt in `report_agent.py`:

```python
system_prompt = """
Your custom prompt here...
"""
```

## 📈 Performance Tips

1. **Batch Processing** - Process multiple ideas together
2. **Token Monitoring** - Track usage to optimize costs
3. **Error Handling** - Implement retry logic for production
4. **Caching** - Cache results for repeated ideas

## 🛠️ Troubleshooting

### Common Issues

**Import Errors**
- Ensure all dependencies are installed
- Check Python path configuration

**API Key Issues**
- Verify API key is set correctly
- Check API key permissions

**JSON Parsing Errors**
- Review system prompt for JSON format
- Check OpenAI model response format

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

This project is part of the MemRingAI system. See the main project license.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review test cases for examples
3. Create an issue in the main repository

---

**Happy Report Generating! 🎉** 