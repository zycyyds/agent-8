import importlib
import json 
import os
from dataclasses import dataclass
from typing import Any

from agentscope.tool import Toolkit


@dataclass(frozen=True)
class ToolFunctionSpec:
    name: str
    preset_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    module: str
    functions: list[ToolFunctionSpec]


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
            fn = getattr(mod, fn_spec.name)
            preset = fn_spec.preset_kwargs
            if preset:
                toolkit.register_tool_function(fn, preset_kwargs=preset)
            else:
                toolkit.register_tool_function(fn)
    return toolkit
