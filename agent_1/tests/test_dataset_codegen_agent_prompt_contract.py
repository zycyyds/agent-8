import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

from agentscope.message import Msg
from agentscope.tool import ToolResponse
from agent_1.codegen_agent import (
    GENERATED_SCRIPT_PATH,
    build_execute_script_command,
    build_react_round_prompt,
    build_react_system_prompt,
    extract_generated_script,
    normalize_generated_script,
    remove_existing_generated_script,
    run_react_codegen_cycle,
    sanitize_agent_output,
    serialize_runtime_result,
    create_codegen_agent,
    main,
    validate_generated_script_contract,
)
from configs.loader import get_agent_config


dataset_codegen_agent_module = importlib.import_module("agent_1.codegen_agent")



def test_build_react_system_prompt_mentions_three_tool_sequence():
    prompt = build_react_system_prompt()

    assert "record_source_file" in prompt
    assert "record_file_modality" in prompt
    assert "record_table_split_strategy" in prompt
    assert "先记录，再生成脚本" in prompt
    assert "reorganized_output/_meta/records.json" in prompt



def test_build_react_round_prompt_mentions_three_tool_sequence_and_records_input():
    prompt = build_react_round_prompt(
        input_path="/tmp/input",
        attempt_index=1,
        previous_issues=[],
    )

    assert "record_source_file" in prompt
    assert "record_file_modality" in prompt
    assert "record_table_split_strategy" in prompt
    assert "records.json" in prompt



def test_validate_generated_script_contract_rejects_agent_tool_calls():
    result = validate_generated_script_contract(
        'records = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\nexecute_python_code("print(1)")\n'
    )

    assert result["passed"] is False
    assert any("execute_python_code" in issue for issue in result["issues"])



def test_validate_generated_script_contract_rejects_literal_id_directory_contract():
    result = validate_generated_script_contract(
        'records = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\noutput_path = os.path.join(output_root, "id", patient_id, modality, filename)\n'
    )

    assert result["passed"] is False
    assert any("字面目录 id" in issue for issue in result["issues"])



def test_validate_generated_script_contract_allows_script_without_explicit_records_read():
    result = validate_generated_script_contract('print("处理完成")\n')

    assert result["passed"] is True
    assert result["issues"] == []



def test_validate_generated_script_contract_allows_records_json_and_jpg_copy_in_same_script():
    result = validate_generated_script_contract(
        'records = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\n'
        'source_path = os.path.join(input_root, "36907.jpg")\n'
        'target_path = os.path.join(output_root, patient_id, "figure", "36907.jpg")\n'
        'shutil.copy2(source_path, target_path)\n'
    )

    assert result["passed"] is True
    assert result["issues"] == []



def test_validate_generated_script_contract_rejects_json_dump_to_jpg_path():
    result = validate_generated_script_contract(
        'records_json_path = os.path.join(output_root, "_meta", "records.json")\n'
        'file_path = os.path.join(output_root, patient_id, "figure", "image.jpg")\n'
        'with open(file_path, "w", encoding="utf-8") as f:\n'
        '    json.dump(record, f, ensure_ascii=False, indent=2)\n'
    )

    assert result["passed"] is False
    assert any(".jpg" in issue or "医学原始文件" in issue for issue in result["issues"])



def test_generated_script_contract_allows_reading_records_without_rewriting_it():
    result = validate_generated_script_contract(
        "import json\nfrom pathlib import Path\nrecords = json.loads((output_root / '_meta' / 'records.json').read_text(encoding='utf-8'))\nfor record in records:\n    pass\n"
    )

    assert result["passed"] is True
    assert result["issues"] == []



def test_sanitize_agent_output_removes_thinking_but_keeps_natural_language():
    cleaned = sanitize_agent_output(
        "<think>hidden reasoning</think>\n本轮说明：我先检查目录。\n工具调用摘要：record_source_file"
    )

    assert "hidden reasoning" not in cleaned
    assert "本轮说明" in cleaned
    assert "工具调用摘要" in cleaned



def test_sanitize_agent_output_handles_structured_blocks_without_thinking():
    cleaned = sanitize_agent_output(
        [
            {"type": "thinking", "thinking": "hidden reasoning"},
            {"type": "text", "text": "本轮说明：我先检查目录。"},
            {"type": "text", "text": "工具调用摘要：record_source_file"},
        ]
    )

    assert "hidden reasoning" not in cleaned
    assert "本轮说明：我先检查目录。" in cleaned
    assert "工具调用摘要：record_source_file" in cleaned



def test_serialize_runtime_result_converts_tool_response_to_jsonable_dict():
    payload = serialize_runtime_result(
        {
            "status": "FAILED",
            "execute_result": ToolResponse(content="execution failed"),
            "issues": ["records.json missing"],
        }
    )

    assert payload["status"] == "FAILED"
    assert payload["execute_result"]["content"] == "execution failed"
    assert payload["execute_result"]["stream"] is False
    assert payload["issues"] == ["records.json missing"]



def test_serialize_runtime_result_can_be_dumped_by_json():
    payload = serialize_runtime_result(
        {
            "status": "FAILED",
            "write_result": ToolResponse(content=[{"type": "text", "text": "write ok"}]),
            "execute_result": ToolResponse(content=[{"type": "text", "text": "<stdout></stdout>"}]),
        }
    )

    dumped = json.dumps(payload, ensure_ascii=False)

    assert '"status": "FAILED"' in dumped
    assert '"write ok"' in dumped
    assert '"<stdout></stdout>"' in dumped



def test_extract_generated_script_reads_python_fence():
    response_text = "本轮说明：已完成。\n```python\nprint(\"hello\")\n```"

    script = extract_generated_script(response_text)

    assert script == 'print("hello")'



def test_extract_generated_script_rejects_inline_instruction_text_without_real_code_fence():
    response_text = "4. 一个 ```python fenced code block，内容是本轮完整可执行脚本"

    script = extract_generated_script(response_text)

    assert script is None



def test_extract_generated_script_rejects_non_python_text_inside_fence():
    response_text = "4. 本轮完整可执行脚本\n```python\nfenced code block，内容是本轮完整可执行脚本\n```"

    script = extract_generated_script(response_text)

    assert script is None



def test_normalize_generated_script_decodes_literal_newlines():
    script = normalize_generated_script(r"\nimport os\nprint('ok')\n")

    assert script.startswith("import os\n")
    assert "print('ok')" in script
    assert r"\n" not in script


def test_dataset_codegen_agent_config_exists():
    config = get_agent_config("dataset_codegen_agent")
    assert isinstance(config, dict)
    assert config


def test_dataset_codegen_agent_mysql_config_exists():
    config = get_agent_config("dataset_codegen_agent")
    mysql_cfg = config.get("mysql")

    assert mysql_cfg["host"] == "127.0.0.1"
    assert mysql_cfg["port"] == 3306
    assert "database" in mysql_cfg


def test_remove_existing_generated_script_deletes_old_file(tmp_path):
    script_path = tmp_path / "generated_reorganizer.py"
    script_path.write_text("print('old')", encoding="utf-8")
    remove_existing_generated_script(script_path)
    assert not script_path.exists()


def test_dataset_codegen_agent_module_no_longer_exports_legacy_template_chain():
    assert not hasattr(dataset_codegen_agent_module, "build_generated_script")
    assert not hasattr(dataset_codegen_agent_module, "run_codegen_cycle")


def test_generated_script_path_is_flattened_to_agent1_root():
    assert GENERATED_SCRIPT_PATH == Path("agent_1/generated_reorganizer.py")


def test_build_execute_script_command_points_to_generated_script():
    command = build_execute_script_command()
    assert "python" in command
    assert "generated_reorganizer.py" in command


def test_main_uses_react_codegen_cycle_for_user_input(monkeypatch):
    events = []

    class FakeUserAgent:
        def __init__(self, name: str):
            self.name = name
            self._called = False

        async def __call__(self, msg):
            if self._called:
                raise EOFError
            self._called = True
            return Msg(name="User", role="user", content="/tmp/input")

    async def fake_create_codegen_agent(toolkit):
        events.append(("create_agent", toolkit))
        return types.SimpleNamespace(name="Dataset Codegen Agent")

    async def fake_run_react_codegen_cycle(*, input_path: str, agent, emit=print, script_path=GENERATED_SCRIPT_PATH, max_rounds: int = 5):
        events.append(("react_cycle", input_path, agent, script_path, max_rounds))
        return {"status": "SUCCESS", "rounds_used": 1}

    monkeypatch.setattr("agent_1.codegen_agent.agentscope.init", lambda **kwargs: events.append(("init", kwargs)))
    monkeypatch.setattr("agent_1.codegen_agent.load_toolkit_from_config", lambda path: types.SimpleNamespace(register_tool_function=lambda fn: events.append(("register", fn.__name__))))
    monkeypatch.setattr("agent_1.codegen_agent.create_codegen_agent", fake_create_codegen_agent)
    monkeypatch.setattr("agent_1.codegen_agent.UserAgent", FakeUserAgent)
    monkeypatch.setattr("agent_1.codegen_agent.run_react_codegen_cycle", fake_run_react_codegen_cycle)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: events.append(("print", args)))

    asyncio.run(main())

    react_events = [item for item in events if item[0] == "react_cycle"]
    assert len(react_events) == 1
    assert react_events[0][1] == "/tmp/input"
    assert react_events[0][3] == GENERATED_SCRIPT_PATH
    assert react_events[0][4] == 5
    assert all(item[0] != "run_codegen_cycle" for item in events)


class FakeAgent:
    def __init__(self, replies: list[str]):
        self._replies = replies
        self._index = 0

    async def __call__(self, msg):
        content = self._replies[self._index]
        self._index += 1
        return Msg(name="Dataset Codegen Agent", role="assistant", content=content)


class ObservationThenCodegenAgent:
    def __init__(self):
        self.calls = 0

    async def __call__(self, msg):
        self.calls += 1
        if self.calls == 1:
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content="本轮说明：已检查 records。\n工具调用摘要：record_file_modality\n结果摘要：准备生成脚本。\n```python\nfrom pathlib import Path\nimport json\nimport shutil\n\noutput_root = Path.cwd() / 'reorganized_output'\nrecords_path = output_root / '_meta' / 'records.json'\nrecords = json.loads(records_path.read_text(encoding='utf-8'))\nfor record in records:\n    patient_id = record['patient_id']\n    modality = record['observations']['modality']\n    source_path = Path('/tmp/input') / record['source_path']\n    target_path = output_root / patient_id / modality / record['source_name']\n    target_path.parent.mkdir(parents=True, exist_ok=True)\n    shutil.copy2(source_path, target_path)\n```",
            )
        raise AssertionError("agent should only be called once in this test")



def test_run_react_codegen_cycle_initializes_empty_records_before_agent_execution(monkeypatch, tmp_path):
    input_root = tmp_path / "input"
    image_file = input_root / "垂直眼位" / "36906(1).jpg"
    image_file.parent.mkdir(parents=True)
    image_file.write_text("img", encoding="utf-8")
    script_path = tmp_path / "generated_reorganizer.py"
    captured = {}

    class OneShotAgent:
        async def __call__(self, msg):
            records_path = Path(dataset_codegen_agent_module.PROJECT_ROOT) / "reorganized_output" / "_meta" / "records.json"
            captured["records_before_agent"] = json.loads(records_path.read_text(encoding="utf-8"))
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content="本轮说明：先记录，再生成脚本。\n工具调用摘要：record_source_file, record_file_modality\n结果摘要：准备生成脚本。\n```python\nfrom pathlib import Path\nimport json\nrecords = json.loads((output_root / '_meta' / 'records.json').read_text(encoding='utf-8'))\nprint(records)\n```",
            )

    async def fake_write_text_file(*, file_path: str, content: str, ranges=None):
        Path(file_path).write_text(content, encoding="utf-8")
        return {"status": "SUCCESS", "content": "write ok"}

    async def fake_execute_shell_command(command: str, timeout: int = 300, **kwargs):
        return {
            "status": "SUCCESS",
            "content": "<returncode>0</returncode><stdout></stdout><stderr></stderr>",
        }

    monkeypatch.setattr("agent_1.codegen_agent.write_text_file", fake_write_text_file)
    monkeypatch.setattr("agent_1.codegen_agent.execute_shell_command", fake_execute_shell_command)

    result = asyncio.run(
        run_react_codegen_cycle(
            input_path=str(input_root),
            agent=OneShotAgent(),
            emit=lambda _: None,
            script_path=script_path,
            max_rounds=1,
        )
    )

    assert result["status"] == "SUCCESS"
    assert captured["records_before_agent"] == []



def test_validate_generated_script_contract_keeps_focus_on_path_and_tool_constraints():
    result = validate_generated_script_contract(
        "import shutil\nsource_path = input_root / 'a.jpg'\ntarget_path = output_root / '36906' / 'figure' / 'a.jpg'\nshutil.copy2(source_path, target_path)\n"
    )

    assert result["passed"] is True
    assert result["issues"] == []



def test_run_react_codegen_cycle_executes_once_without_contract_validation(monkeypatch, tmp_path):
    script_path = tmp_path / "generated_reorganizer.py"
    emitted = []

    class OneShotAgent:
        def __init__(self):
            self.calls = 0

        async def __call__(self, msg):
            self.calls += 1
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content="<think>hidden</think>\n本轮说明：第一次尝试。\n工具调用摘要：record_source_file\n结果摘要：先试运行。\n```python\nrecords = json.loads((output_root / \"_meta\" / \"records.json\").read_text(encoding=\"utf-8\"))\nprint(records)\n```",
            )

    fake_agent = OneShotAgent()

    async def fake_write_text_file(*, file_path: str, content: str, ranges=None):
        Path(file_path).write_text(content, encoding="utf-8")
        return {"status": "SUCCESS", "content": "write ok"}

    async def fake_execute_shell_command(command: str, timeout: int = 300, **kwargs):
        return {
            "status": "SUCCESS",
            "content": "<returncode>0</returncode><stdout></stdout><stderr></stderr>",
        }

    monkeypatch.setattr(
        "agent_1.codegen_agent.write_text_file",
        fake_write_text_file,
    )
    monkeypatch.setattr(
        "agent_1.codegen_agent.execute_shell_command",
        fake_execute_shell_command,
    )

    result = asyncio.run(
        run_react_codegen_cycle(
            input_path=str(tmp_path / "input"),
            agent=fake_agent,
            emit=emitted.append,
            script_path=script_path,
            max_rounds=5,
        )
    )

    assert result["status"] == "SUCCESS"
    assert result["rounds_used"] == 1
    assert "records_result" not in result
    assert fake_agent.calls == 1
    assert all("<think>" not in item for item in emitted)
    assert any("工具调用摘要" in item for item in emitted)



def test_run_react_codegen_cycle_fails_immediately_when_script_missing(tmp_path):
    emitted = []

    class AlwaysBadAgent:
        async def __call__(self, msg):
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content="<think>hidden</think>\n本轮说明：仍未生成脚本。\n工具调用摘要：无\n结果摘要：继续修复。",
            )

    result = asyncio.run(
        run_react_codegen_cycle(
            input_path=str(tmp_path / "input"),
            agent=AlwaysBadAgent(),
            emit=emitted.append,
            max_rounds=2,
        )
    )

    assert result["status"] == "FAILED"
    assert result["rounds_used"] == 1
    assert all("<think>" not in item for item in emitted)



def test_run_react_codegen_cycle_normalizes_literal_newlines_before_writing(monkeypatch, tmp_path):
    script_path = tmp_path / "generated_reorganizer.py"
    emitted = []

    class EscapedScriptAgent:
        async def __call__(self, msg):
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content='本轮说明：生成脚本。\n工具调用摘要：record_source_file\n结果摘要：完成。\n```python\n\\nrecords = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\\nprint(records)\\n```',
            )

    async def fake_write_text_file(*, file_path: str, content: str, ranges=None):
        Path(file_path).write_text(content, encoding="utf-8")
        return {"status": "SUCCESS", "content": "write ok"}

    async def fake_execute_shell_command(command: str, timeout: int = 300, **kwargs):
        return {
            "status": "SUCCESS",
            "content": "<returncode>0</returncode><stdout></stdout><stderr></stderr>",
        }

    monkeypatch.setattr(
        "agent_1.codegen_agent.write_text_file",
        fake_write_text_file,
    )
    monkeypatch.setattr(
        "agent_1.codegen_agent.execute_shell_command",
        fake_execute_shell_command,
    )
    asyncio.run(
        run_react_codegen_cycle(
            input_path=str(tmp_path / "input"),
            agent=EscapedScriptAgent(),
            emit=emitted.append,
            script_path=script_path,
            max_rounds=1,
        )
    )

    assert script_path.read_text(encoding="utf-8") == 'records = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\nprint(records)'



def test_run_react_codegen_cycle_rejects_bad_script_before_execution(monkeypatch, tmp_path):
    script_path = tmp_path / "generated_reorganizer.py"
    emitted = []
    executed_commands = []

    class BadScriptAgent:
        async def __call__(self, msg):
            return Msg(
                name="Dataset Codegen Agent",
                role="assistant",
                content=(
                    "本轮说明：先给出脚本。\n"
                    "工具调用摘要：record_source_file\n"
                    "结果摘要：待执行。\n"
                    "```python\n"
                    'execute_python_code("print(1)")\n'
                    'records = json.loads((output_root / "_meta" / "records.json").read_text(encoding="utf-8"))\n'
                    "```"
                ),
            )

    async def fake_write_text_file(*, file_path: str, content: str, ranges=None):
        Path(file_path).write_text(content, encoding="utf-8")
        return {"status": "SUCCESS", "content": "write ok"}

    async def fake_execute_shell_command(command: str, timeout: int = 300, **kwargs):
        executed_commands.append(command)
        return {
            "status": "SUCCESS",
            "content": "<returncode>0</returncode><stdout>[]</stdout><stderr></stderr>",
        }

    monkeypatch.setattr(
        "agent_1.codegen_agent.write_text_file",
        fake_write_text_file,
    )
    monkeypatch.setattr(
        "agent_1.codegen_agent.execute_shell_command",
        fake_execute_shell_command,
    )

    result = asyncio.run(
        run_react_codegen_cycle(
            input_path=str(tmp_path / "input"),
            agent=BadScriptAgent(),
            emit=emitted.append,
            script_path=script_path,
            max_rounds=1,
        )
    )

    assert result["status"] == "FAILED"
    assert executed_commands == []
    assert result["execute_result"]["status"] == "SKIPPED"
    assert any("execute_python_code" in issue for issue in result["issues"])



def test_create_codegen_agent_uses_openai_model_env_and_thinking_safe_formatter(monkeypatch):
    captured = {"ollama_called": False, "openai_called": False}

    class FakeOllamaModel:
        def __init__(self, **kwargs):
            captured["ollama_called"] = True
            captured["ollama_kwargs"] = kwargs

    class FakeOpenAIModel:
        def __init__(self, **kwargs):
            captured["openai_called"] = True
            captured["openai_kwargs"] = kwargs

    class FakeFormatter:
        pass

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs
            captured["console_enabled"] = None

        def set_console_output_enabled(self, enabled: bool):
            captured["console_enabled"] = enabled

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:8317/v1")
    monkeypatch.setenv("MODEL_NAME", "gpt-5.4-mini")
    monkeypatch.setattr("agent_1.codegen_agent.OpenAIChatModel", FakeOpenAIModel, raising=False)
    monkeypatch.setattr("agent_1.codegen_agent.ThinkingSafeOpenAIChatFormatter", FakeFormatter, raising=False)
    monkeypatch.setattr("agent_1.codegen_agent.ReActAgent", FakeAgent)

    asyncio.run(create_codegen_agent(toolkit=object()))

    assert captured["ollama_called"] is False
    assert captured["openai_called"] is True
    assert captured["openai_kwargs"]["model_name"] == "gpt-5.4-mini"
    assert captured["openai_kwargs"]["api_key"] == "test-key"
    assert captured["openai_kwargs"]["stream"] is True
    assert captured["openai_kwargs"]["client_kwargs"]["base_url"] == "http://127.0.0.1:8317/v1"
    assert isinstance(captured["agent_kwargs"]["formatter"], FakeFormatter)
    assert "disable_console_output" not in captured["agent_kwargs"]
    assert captured["console_enabled"] is False
