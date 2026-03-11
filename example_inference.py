"""
DatacortexAI — Example Inference and Agentic Tools Demonstration

This script demonstrates two key capabilities of the DatacortexAI platform:
1. LLM Client Fallback & Key Rotation: Seamless fallback strategies if a provider fails or rate-limits.
2. Agentic Tools Discovery & Execution: How the LLM can securely discover Python tools and execute them.

To run:
    python example_inference.py
"""

import asyncio
import logging
import json
from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.tools import get_all_tool_schemas, execute_tool

# Ensure tools are imported so they register
import src.agents.agentic_tools

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

async def run_example():
    print("=" * 60)
    print("DatacortexAI Inference & Tool Calling Demo")
    print("=" * 60)

    config = Config.get()
    client = LLMClient(config)
    
    tools = get_all_tool_schemas()
    print(f"\n[SYSTEM] Registered {len(tools)} tools: {[t['function']['name'] for t in tools]}")
    
    prompt = "I need to know the exact mathematical result of 1542 multiplied by 342. Can you calculate it using a tool and also list what files exist in the src/core directory?"
    print(f"\n[USER PROMPT]\n{prompt}\n")
    
    print("[SYSTEM] Asking the LLM (OpenRouter free tier)...")
    # We use the analysis_model which defaults to 'openrouter/free' (or 'google/gemini-2.5-flash:free')
    try:
        response = await client.analyze(
            prompt=prompt,
            system="You are an AI helpful assistant with access to tools. Use tools if necessary.",
            tools=tools
        )
        
        if response.tool_calls:
            print(f"\n[LLM ACTION] The LLM decided to call {len(response.tool_calls)} tools.")
            for idx, tool_call in enumerate(response.tool_calls, 1):
                try:
                    name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                except AttributeError:
                    name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                print(f"\n--- Tool Call {idx} ---")
                print(f"Tool Name : {name}")
                print(f"Arguments : {arguments}")
                
                result = await execute_tool(name, arguments)
                print(f"Result    : {result}")
        else:
            print(f"\n[LLM RESPONSE] (No tools used)\n{response.content}")
            
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_example())
