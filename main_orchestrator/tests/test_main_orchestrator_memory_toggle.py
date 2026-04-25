import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from agentscope.tool import ToolResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATOR_DIR = PROJECT_ROOT / "main_orchestrator"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ORCHESTRATOR_DIR) not in sys.path:
    sys.path.insert(0, str(ORCHESTRATOR_DIR))

import main_orchestrator as orchestrator_module


def test_parse_args_supports_disable_memory_agent():
    args = orchestrator_module.parse_args(["--disable-memory-agent"])
    assert args.disable_memory_agent is True


def test_build_orchestrator_binds_memory_toggle(monkeypatch):
    registered = []
    captured = {}

    class FakeToolkit:
        def register_tool_function(self, fn):
            registered.append(fn)

    async def fake_step1(input_data: str, context: str = "", enable_memory_agent: bool = True):
        captured["input_data"] = input_data
        captured["context"] = context
        captured["enable_memory_agent"] = enable_memory_agent
        return ToolResponse(content="ok")

    async def fake_passthrough(input_data: str, context: str = "", enable_memory_agent: bool = True):
        return ToolResponse(content=f"ok:{input_data}:{context}:{enable_memory_agent}")

    monkeypatch.setattr(orchestrator_module, "Toolkit", FakeToolkit)
    monkeypatch.setattr(orchestrator_module, "OllamaChatModel", lambda **kwargs: object())
    monkeypatch.setattr(orchestrator_module, "ReActAgent", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(orchestrator_module, "register_no_thinking_print_hook", lambda agent: agent)
    monkeypatch.setattr(orchestrator_module, "run_step1_modal_recognition", fake_step1)
    monkeypatch.setattr(orchestrator_module, "run_step2_3_medical_data_cleaner", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step2_parse_extract", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step3_semantic_standardization", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step4_data_quality_repair", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step5_task_oriented_clipping", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step6_consistency_verification", fake_passthrough)
    monkeypatch.setattr(orchestrator_module, "run_step7_phenotype_knowledge_confirmation", fake_passthrough)

    orchestrator_module._build_orchestrator("ctx", enable_memory_agent=False)
    step1_tool = next(fn for fn in registered if getattr(fn, "__wrapped__", None) is fake_step1)
    asyncio.run(step1_tool("/tmp/rawdata", "ctx", enable_memory_agent=True))

    assert captured["input_data"] == "/tmp/rawdata"
    assert captured["context"] == "ctx"
    assert captured["enable_memory_agent"] is False
