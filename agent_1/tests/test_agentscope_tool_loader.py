import asyncio

import agent_1.codegen_tools as react_tools
from agentscope.message import ToolUseBlock

from agent_1.agentscope_tool_loader import load_toolkit_from_config


async def _collect_tool_responses(toolkit, tool_name: str, tool_input: dict) -> list[object]:
    responses = []
    result = await toolkit.call_tool_function(
        ToolUseBlock(
            type="tool_use",
            id="test-call",
            name=tool_name,
            input=tool_input,
        )
    )
    async for item in result:
        responses.append(item)
    return responses


def test_load_toolkit_wraps_plain_dict_return_value_into_tool_response(tmp_path, monkeypatch):
    toolkit = load_toolkit_from_config(
        "/Users/mkbk/PycharmProjects/agent-8/.worktrees/step1-dataset-codegen/agent_1/agentscope_tools_dataset.json"
    )

    fixture = tmp_path / "labs.csv"
    fixture.write_text("patient_id,value\npatient-001,1\npatient-002,2\n", encoding="utf-8")
    monkeypatch.setattr(
        react_tools,
        "_records_path",
        lambda: tmp_path / "reorganized_output" / "_meta" / "records.json",
    )

    responses = asyncio.run(
        _collect_tool_responses(
            toolkit,
            "record_table_split_strategy",
            {"file_path": str(fixture)},
        )
    )

    assert len(responses) == 1
    response = responses[0]
    assert hasattr(response, "content")
    assert "should_split" in str(response.content)
