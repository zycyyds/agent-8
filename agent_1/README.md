# agent-1 模态识别 Agent

## 目录结构

```
agent-1/
├── main_layout_example.py      # Agent 主程序入口
├── layout_analysis_tool.py     # 布局分析工具模块
├── agentscope_tool_loader.py   # AgentScope 工具加载器
├── agentscope_tools.json       # 工具注册配置
├── __init__.py
└── README.md
```

## 功能说明

该 Agent 用于文档图片的模态识别，可以自动判断图片属于以下三种类型：
- `ocr` - 纯文本/表格内容
- `figure` - 纯图片/图表内容
- `ocr+figure` - 图文混合内容

## 使用方法

```bash
cd agent-1
python main_layout_example.py
```

然后输入图片目录路径，例如：
```
../rawdata/垂直眼位
```

## 输出目录

Agent 会在**父目录**（项目根目录）生成以下输出：
- `output/results/` - 标注后的图片
- `output/result_json/` - 检测 JSON 结果
- `output/分割/` - 裁剪出的 figure 元素
- `output/classification_results.json` - 分类汇总结果
- `data/` - 整理后的数据集
