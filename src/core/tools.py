"""
DatacortexAI — Agentic Tools Registry

Provides a decorator-based framework for defining custom Python functions
as OpenAI-compatible tools that the LLM can discover and run.
"""

import inspect
import json
import logging
from typing import Callable, Any, get_type_hints

logger = logging.getLogger(__name__)

# Global registry of available tools
TOOL_REGISTRY: dict[str, Callable] = {}
TOOL_SCHEMAS: list[dict] = []


def _type_to_json_schema(py_type: Any) -> dict:
    """Map Python types to JSON Schema types."""
    if py_type == int:
        return {"type": "integer"}
    elif py_type == float:
        return {"type": "number"}
    elif py_type == bool:
        return {"type": "boolean"}
    elif py_type == list or py_type == tuple:
        return {"type": "array", "items": {"type": "string"}}
    elif py_type == dict:
        return {"type": "object"}
    else:
        return {"type": "string"}


def tool(name: str = "", description: str = ""):
    """
    Decorator to register a function as an LLM tool.
    Extracts name, description, and parameter schema from docstrings and type hints.
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # Extract type hints and parameters
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
                
            param_type = hints.get(param_name, str)
            properties[param_name] = _type_to_json_schema(param_type)
            
            # Simple docstring parsing for param descriptions could go here
            properties[param_name]["description"] = f"Parameter: {param_name}"

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

        # Register
        TOOL_REGISTRY[tool_name] = func
        TOOL_SCHEMAS.append(schema)
        logger.info(f"Registered agentic tool: {tool_name}")

        return func
    return decorator


def get_all_tool_schemas() -> list[dict]:
    """Return the JSON schemas for all registered tools."""
    return TOOL_SCHEMAS


async def execute_tool(tool_name: str, arguments: dict) -> Any:
    """Execute a registered tool by name with the given arguments."""
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Tool not found: {tool_name}")
        
    func = TOOL_REGISTRY[tool_name]
    logger.info(f"Executing tool {tool_name} with args {arguments}")
    
    try:
        if inspect.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            return func(**arguments)
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return f"Error executing tool {tool_name}: {str(e)}"
