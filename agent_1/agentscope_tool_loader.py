import importlib
import inspect
import json
import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import Toolkit, ToolResponse


@dataclass(frozen=True)
class ToolFunctionSpec:
    name: str
    preset_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    module: str
    functions: list[ToolFunctionSpec]


def _build_tool_response_content(value: Any) -> list[TextBlock]:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return [TextBlock(type="text", text=text)]


def _wrap_tool_function(fn):
    if inspect.iscoroutinefunction(fn):
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            result = await fn(*args, **kwargs)
            if isinstance(result, ToolResponse):
                return result
            return ToolResponse(content=_build_tool_response_content(result))

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, ToolResponse):
            return result
        return ToolResponse(content=_build_tool_response_content(result))

    return sync_wrapper


def load_toolkit_from_config(config_path: str) -> Toolkit:
    abs_path = os.path.abspath(config_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    specs: list[ToolSpec] = []
    for item in raw.get("tools", []):
        module = item["module"]
        fn_specs: list[ToolFunctionSpec] = []
        for fn in item.get("functions", []):
            if isinstance(fn, str):
                fn_specs.append(ToolFunctionSpec(name=fn, preset_kwargs=None))
            else:
                fn_specs.append(
                    ToolFunctionSpec(
                        name=str(fn.get("name")),
                        preset_kwargs=dict(fn.get("preset_kwargs") or {}),
                    )
                )
        specs.append(ToolSpec(module=module, functions=fn_specs))

    toolkit = Toolkit()
    for spec in specs:
        mod = importlib.import_module(spec.module)
        for fn_spec in spec.functions:
            fn = _wrap_tool_function(getattr(mod, fn_spec.name))
            preset = fn_spec.preset_kwargs
            if preset:
                toolkit.register_tool_function(fn, preset_kwargs=preset)
            else:
                toolkit.register_tool_function(fn)
    return toolkit
