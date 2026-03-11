import asyncio
import logging
import json
from src.core.config import Config
from src.core.llm_client import LLMClient
from src.core.tools import get_all_tool_schemas, execute_tool

# Ensure tools are imported so they register
import src.agents.agentic_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_example():
    config = Config.get()
    client = LLMClient(config)
    
    tools = get_all_tool_schemas()
    logger.info(f"Registered {len(tools)} tools: {[t['function']['name'] for t in tools]}")
    
    prompt = "What is 1542 multiplied by 342? Please calculate it and tell me the answer."
    logger.info(f"Prompt: {prompt}")
    
    # We use the analysis_model which defaults to 'openrouter/free' (gemini-2.5-flash:free handles tools well)
    # Wait, analysis_model is 'openrouter/free'. For OpenRouter to reliably pass tools, we specifically
    # request tools kwarg.
    response = await client.analyze(
        prompt=prompt,
        system="You are an AI helpful assistant with access to tools. Use tools if necessary.",
        tools=tools
    )
    
    if response.tool_calls:
        logger.info(f"LLM decided to call {len(response.tool_calls)} tools.")
        for tool_call in response.tool_calls:
            # Depending on the client library version, `tool_call` format might be an object
            try:
                # Assuming openai>=1.0.0 ChatCompletionMessageToolCall
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
            except AttributeError:
                # If dict
                name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])
                
            logger.info(f"Executing tool: {name} with args: {arguments}")
            result = await execute_tool(name, arguments)
            logger.info(f"Tool Result: {result}")
    else:
        logger.info(f"LLM Response directly (No tools used): {response.content}")

if __name__ == "__main__":
    asyncio.run(run_example())
