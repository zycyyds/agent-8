# agent_1 单轮 codegen 链路

## 目录结构

```text
agent_1/
├── codegen_agent.py           # 当前单轮入口
├── codegen_tools.py           # 三个 records 工具
├── codegen_visual.py          # 模态分析入口
├── generated_reorganizer.py   # 生成脚本
├── agentscope_tool_loader.py  # AgentScope 工具加载器
├── agentscope_tools_dataset.json
└── tests/
```

## 使用方法

```bash
python agent_1/codegen_agent.py
```

然后输入待处理目录，例如：

```text
/Users/mkbk/PycharmProjects/agent-8/rawdata
```

## 当前链路

1. ReAct agent 顺序调用三个 records 工具写入 `reorganized_output/_meta/records.json`
2. 基于 records 生成 `agent_1/generated_reorganizer.py`
3. 执行生成脚本并输出摘要
