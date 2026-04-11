---
name: architecture-analysis-evidence-first
description: 用户偏好基于真实代码证据做全仓架构梳理，重点检查新增目录接入状态而非复述 README。
type: feedback
---
规则：做架构说明时优先以仓库代码和调用链为准，输出结构化全局视图，并明确区分“已接入/部分接入/未接入”。

**Why:** 用户在 README 可能过时的场景下，需要一次性看清真实主流程、数据流和新增目录（尤其 medical-data-pipeline-public）的接入缺口。

**How to apply:** 后续遇到架构问题时，先扫描顶层目录与入口脚本，再串联 import/调用/输入输出路径；所有关键结论附证据等级，避免把推测写成事实。