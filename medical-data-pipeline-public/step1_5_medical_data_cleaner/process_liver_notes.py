# -*- coding: utf-8 -*-
"""
处理 liver_patients_note.jsonl 的专用模块

流程：
1. LLM Agent 分析记录结构，判断哪些字段含有需要信息抽取的自由文本
2. 对识别出的自由文本字段进行信息抽取（不进行标准化）
3. 将抽取结果写入输出 JSONL
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── 路径设置 ──────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in (current_dir, parent_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

src_path = os.path.abspath(os.path.join(parent_dir, "../../src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit

from config.settings import get_api_config
from schema import MedicalExtractionResult


# ══════════════════════════════════════════════════════════════
#  Agent 初始化
# ══════════════════════════════════════════════════════════════

def _build_model() -> Optional[OpenAIChatModel]:
    """根据环境变量构建 OpenAI 兼容模型"""
    config = get_api_config()
    if not config["api_key"]:
        return None
    return OpenAIChatModel(
        config_name="liver-notes-model",
        model_name=config["model_name"],
        api_key=config["api_key"],
        client_kwargs={
            "base_url": config["api_base"],
            "timeout": config["timeout"],
        },
        generate_kwargs={"temperature": 0.0},
    )


def setup_liver_agents() -> Tuple[Optional[ReActAgent], None]:
    """
    创建肝病记录处理所需的 Agent。

    只需要信息抽取 Agent，不需要标准化 Agent。

    Returns:
        (med_agent, None)  — None 表示不进行标准化
    """
    model = _build_model()
    if model is None:
        print("⚠️  未找到 API Key，无法初始化信息抽取 Agent")
        return None, None

    try:
        from tools.tools_preprocess import preprocess_medical_input
        from tools.tools_ocr import extract_text_from_image
        from agentscope.tool._text_processing._medical_clean import clean_medical_text

        toolkit = Toolkit()
        toolkit.create_tool_group(
            group_name="medical",
            description="Medical data processing tools",
            active=True,
        )
        toolkit.register_tool_function(
            preprocess_medical_input,
            group_name="medical",
            func_description="Preprocess medical input text.",
        )
        toolkit.register_tool_function(
            clean_medical_text,
            group_name="medical",
            func_description="Clean medical text by normalizing symbols and whitespace.",
        )
    except Exception as e:
        print(f"⚠️  注册工具失败（将使用无工具模式）: {e}")
        toolkit = None

    med_agent = ReActAgent(
        name="LiverNoteExtractor",
        sys_prompt="""你是一位专业的医学信息抽取专家，擅长从临床笔记、放射报告、出院记录等自由文本中提取结构化信息。

## 工作要求
- 仔细阅读输入文本，提取所有有临床意义的医学实体
- entities 的 category 只能是：Disease, Drug, Symptom, Test, Treatment, Anatomy, LabValue, Finding, Other

## 各类别明确定义
- **Disease**：疾病诊断（如 cirrhosis, hepatitis, pneumonia）
- **Drug**：药物名称（如 Furosemide, Lactulose）
- **Symptom**：症状/主诉（如 dyspnea, abdominal pain）
- **Test**：检查/检验项目名称（如 CT abdomen, chest X-ray, CBC）
- **Treatment**：治疗操作/手术（如 TIPS procedure, paracentesis）
- **Anatomy**：独立的解剖结构/器官/部位，即使它们出现在 Finding 描述中也要单独提取（如 liver, portal vein, spleen, lung, right hepatic lobe, inferior vena cava）
- **LabValue**：有具体数值/单位的化验结果（如 WBC:4.2 10^9/L, ALT:294 IU/L）
- **Finding**：影像/检查的病理发现描述（不含具体解剖位置名称本身，如 "coarsened and nodular echotexture", "partially occlusive thrombus"）
- **Other**：不属于以上分类的其他实体

## 关键区分原则
- **Anatomy 与 Finding 必须分开**：报告中涉及某器官的异常发现，器官本身单独提取为 Anatomy，异常描述提取为 Finding。
  示例："liver coarsened and nodular echotexture" → Anatomy: liver；Finding: coarsened and nodular echotexture
  示例："main portal vein is patent" → Anatomy: main portal vein；Finding: patent（或忽略正常发现）
- 时间信息放在 temporal_info，不要放入 entities
- 剂量信息放在 quantity_info，不要放入 entities
- 实体间关系放在 relations
- impression 填写诊断印象或结论
- indication 填写检查目的或指征
- 保持原文精度，不要臆造信息""",
        model=model,
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
    )

    return med_agent, None  # 不进行标准化，第二个返回值为 None


# ══════════════════════════════════════════════════════════════
#  字段分析：让 LLM 判断哪些字段需要信息抽取
# ══════════════════════════════════════════════════════════════

async def _llm_detect_extraction_fields(
    sample_record: Dict[str, Any],
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    使用 LLM 分析一条样本记录，返回需要信息抽取的字段路径列表。

    每个元素形如：
        {
            "path": ["就诊文本", "notes", "*", "text"],  # JSON 路径，'*' 表示列表遍历
            "description": "临床笔记自由文本",
            "type": "clinical_note"
        }

    Args:
        sample_record: 第一条记录（用于结构分析）
        verbose: 是否打印详细信息

    Returns:
        字段描述列表；若 LLM 不可用则返回默认值
    """
    config = get_api_config()
    if not config["api_key"]:
        # 无 LLM，使用默认规则（适用于已知的 liver_patients_note 格式）
        return _default_extraction_fields()

    # 构造精简的结构摘要（避免 token 过多）
    def _summarize(obj, depth=0, max_depth=4, max_list=2):
        if depth > max_depth:
            return "..."
        if isinstance(obj, dict):
            return {k: _summarize(v, depth + 1, max_depth, max_list) for k, v in obj.items()}
        if isinstance(obj, list):
            sample = [_summarize(i, depth + 1, max_depth, max_list) for i in obj[:max_list]]
            if len(obj) > max_list:
                sample.append(f"... ({len(obj)} items total)")
            return sample
        if isinstance(obj, str) and len(obj) > 200:
            return obj[:200] + f"... (length={len(obj)})"
        return obj

    summary = json.dumps(_summarize(sample_record), ensure_ascii=False, indent=2)

    prompt = f"""你是医学数据分析专家。以下是一条患者就诊记录的结构摘要（JSON 格式）：

{summary}

请分析该记录，找出所有包含**自由文本**（需要进行 NLP 信息抽取）的字段，例如临床笔记、放射报告正文、出院记录、诊断描述等。
结构化字段（ICD 编码、时间戳、数值、枚举值等）不需要信息抽取，请排除。

返回 JSON 数组，每个元素描述一个需要抽取的字段：
[
  {{
    "path": ["顶层键", "子键", "*", "叶子键"],
    "description": "字段含义",
    "type": "clinical_note | radiology_report | discharge_summary | other"
  }}
]

说明：
- path 中的 "*" 表示遍历列表中的每个元素
- 只返回 JSON 数组，不要有其他输出

只返回 JSON 数组。"""

    try:
        import openai
        client = openai.AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["api_base"],
        )
        resp = await client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()

        # 提取 JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        fields = json.loads(content.strip())
        if verbose:
            print(f"  LLM 识别到 {len(fields)} 个需要信息抽取的字段:")
            for f in fields:
                print(f"    - {'.'.join(str(p) for p in f['path'])}  [{f.get('type','')}]")
        return fields

    except Exception as e:
        if verbose:
            print(f"  ⚠️  LLM 字段分析失败，使用默认规则: {e}")
        return _default_extraction_fields()


def _default_extraction_fields() -> List[Dict[str, Any]]:
    """当 LLM 不可用时的默认字段配置（适配 liver_patients_note 格式）"""
    return [
        {
            "path": ["就诊文本", "notes", "*", "text"],
            "description": "临床笔记 / 放射报告 / 出院记录等自由文本",
            "type": "clinical_note",
        }
    ]


# ══════════════════════════════════════════════════════════════
#  字段值提取工具
# ══════════════════════════════════════════════════════════════

def _get_text_items(
    record: Dict[str, Any],
    field_spec: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    """
    按照 field_spec 中的 path 从记录中获取所有文本条目。

    Returns:
        列表，每个元素为 (dot_path_str, text_value)
        例如 ("就诊文本.notes.0.text", "EXAMINATION: Chest radiograph ...")
    """
    path = field_spec["path"]

    def _walk(obj, remaining_path, current_path_str):
        if not remaining_path:
            if isinstance(obj, str) and obj.strip():
                yield current_path_str, obj
            return
        key = remaining_path[0]
        rest = remaining_path[1:]
        if key == "*":
            if isinstance(obj, list):
                for idx, item in enumerate(obj):
                    yield from _walk(item, rest, f"{current_path_str}.{idx}")
        elif isinstance(obj, dict) and key in obj:
            yield from _walk(obj[key], rest, f"{current_path_str}.{key}" if current_path_str else key)

    results = list(_walk(record, path, ""))
    return results


# ══════════════════════════════════════════════════════════════
#  单条记录处理
# ══════════════════════════════════════════════════════════════

async def _process_one_record(
    record: Dict[str, Any],
    record_idx: int,
    extraction_fields: List[Dict[str, Any]],
    med_agent: ReActAgent,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    处理单条 JSONL 记录，对每个自由文本字段进行信息抽取。

    Returns:
        在原始记录基础上增加 "extraction_results" 键：
        {
            "extraction_results": {
                "就诊文本.notes.0.text": {
                    "source_path": "...",
                    "entities": [...],
                    "temporal_info": [...],
                    "quantity_info": [...],
                    "relations": [...],
                    "impression": "...",
                    "indication": "...",
                    "error": null
                },
                ...
            }
        }
    """
    result = dict(record)
    result["extraction_results"] = {}

    for field_spec in extraction_fields:
        text_items = _get_text_items(record, field_spec)
        if not text_items:
            continue

        for dot_path, text in text_items:
            if verbose:
                preview = text[:80].replace("\n", " ")
                print(f"    抽取字段 {dot_path}: {preview}...")

            entry: Dict[str, Any] = {
                "source_path": dot_path,
                "field_type": field_spec.get("type", "unknown"),
                "entities": [],
                "temporal_info": [],
                "quantity_info": [],
                "relations": [],
                "impression": None,
                "indication": None,
                "error": None,
            }

            if not med_agent:
                entry["error"] = "med_agent 未初始化"
                result["extraction_results"][dot_path] = entry
                continue

            try:
                msg = Msg(name="User", content=text, role="user")
                res = await med_agent(msg, structured_model=MedicalExtractionResult)

                if res.metadata:
                    entry["entities"] = res.metadata.get("entities", [])
                    entry["temporal_info"] = res.metadata.get("temporal_info", [])
                    entry["quantity_info"] = res.metadata.get("quantity_info", [])
                    entry["relations"] = res.metadata.get("relations", [])
                    entry["impression"] = res.metadata.get("impression")
                    entry["indication"] = res.metadata.get("indication")
                else:
                    # 没有结构化结果，尝试保留文本回复
                    entry["raw_response"] = res.content if res.content else ""

                if verbose:
                    print(f"      → {len(entry['entities'])} 个实体")

            except Exception as e:
                entry["error"] = str(e)
                if verbose:
                    print(f"      ❌ 抽取失败: {e}")

            result["extraction_results"][dot_path] = entry

    return result


# ══════════════════════════════════════════════════════════════
#  批量处理入口
# ══════════════════════════════════════════════════════════════

async def batch_process_liver_notes(
    jsonl_path: str,
    med_agent: Optional[ReActAgent],
    std_agent: Any,  # 保留参数兼容性，但不使用
    output_dir: str,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> str:
    """
    批量处理 liver_patients_note.jsonl。

    Args:
        jsonl_path:   输入 JSONL 文件路径
        med_agent:    信息抽取 Agent
        std_agent:    不使用（保留签名兼容性）
        output_dir:   输出目录
        max_records:  最多处理的记录数（None = 全部）
        verbose:      是否打印详细信息

    Returns:
        输出文件路径
    """
    if not os.path.exists(jsonl_path):
        print(f"❌ 文件不存在: {jsonl_path}")
        return ""

    # ── 读取记录（带自动修复逻辑）──────────────────────────
    records: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                repaired = _try_repair_json(line)
                if repaired is not None:
                    records.append(repaired)
                    if verbose:
                        print(f"  ⚠️  第{lineno}行 JSON 截断，已自动修复")
                else:
                    parse_errors.append(f"第{lineno}行 (长度={len(line)}): 无法修复")
                    if verbose:
                        print(f"  ❌ 第{lineno}行解析失败，跳过")

    if parse_errors:
        print(f"\n警告: {len(parse_errors)} 条记录解析失败:")
        for err in parse_errors:
            print(f"  - {err}")

    if max_records:
        records = records[:max_records]

    if verbose:
        print(f"\n📂 读取 {len(records)} 条记录: {jsonl_path}")

    if not records:
        print("❌ 没有可处理的记录")
        return ""

    # ── 让 LLM 分析第一条记录，决定哪些字段需要抽取 ──────────
    if verbose:
        print("\n🔍 分析记录结构，识别需要信息抽取的字段...")

    extraction_fields = await _llm_detect_extraction_fields(
        records[0], verbose=verbose
    )

    if not extraction_fields:
        print("❌ 未识别到需要信息抽取的字段")
        return ""

    # ── 创建输出目录 ─────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"liver_notes_extracted_{timestamp}.jsonl")

    # 同时保存字段分析结果
    fields_file = os.path.join(output_dir, f"extraction_fields_{timestamp}.json")
    with open(fields_file, "w", encoding="utf-8") as f:
        json.dump(extraction_fields, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"📋 字段分析结果已保存: {fields_file}")

    # ── 逐条处理 ─────────────────────────────────────────────
    success_count = 0
    fail_count = 0
    enriched_records: List[Dict[str, Any]] = []

    with open(out_file, "w", encoding="utf-8") as out_f:
        for idx, record in enumerate(records):
            if verbose:
                patient_id = _get_patient_id(record, idx)
                print(f"\n[{idx + 1}/{len(records)}] 处理记录 {patient_id}...")

            try:
                enriched = await _process_one_record(
                    record=record,
                    record_idx=idx,
                    extraction_fields=extraction_fields,
                    med_agent=med_agent,
                    verbose=verbose,
                )
                success_count += 1
            except Exception as e:
                if verbose:
                    print(f"  ❌ 记录处理失败: {e}")
                enriched = dict(record)
                enriched["extraction_results"] = {}
                enriched["_processing_error"] = str(e)
                fail_count += 1

            enriched_records.append(enriched)
            out_f.write(json.dumps(enriched, ensure_ascii=False, default=str) + "\n")

    # ── 导出 CSV ──────────────────────────────────────────────
    csv_file = out_file.replace(".jsonl", ".csv")
    _save_as_csv(enriched_records, csv_file, verbose=verbose)

    # ── 汇总 ─────────────────────────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"✅ 处理完成:")
        print(f"   - 总记录数:  {len(records)}")
        print(f"   - 成功:      {success_count}")
        print(f"   - 失败:      {fail_count}")
        print(f"   - JSONL 输出: {out_file}")
        print(f"   - CSV  输出: {csv_file}")

    return out_file


# ══════════════════════════════════════════════════════════════
#  CSV 导出
# ══════════════════════════════════════════════════════════════

def _records_to_csv_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将已处理的记录列表展平为 CSV 行列表。

    每条 note（就诊文本.notes[i]）对应一行；患者级别的结构化字段在每行重复。
    抽取结果中不同类别的实体分别写入独立列。

    返回的每个 dict 即为一行 CSV 数据。
    """
    ENTITY_CATEGORIES = ["Disease", "Drug", "Symptom", "Test", "Treatment",
                         "Anatomy", "LabValue", "Finding", "Other"]

    rows = []
    for rec_idx, rec in enumerate(records):
        # ── 患者级别字段 ─────────────────────────────────────
        bing_li = rec.get("病历", {}) or {}
        ren_kou = bing_li.get("人口学", {}) or {}

        patient_base = {
            "record_index": rec_idx,
            "入院时间": bing_li.get("入院时间", ""),
            "出院时间": bing_li.get("出院时间", ""),
            "入院科室": bing_li.get("入院科室", ""),
            "出院去向": bing_li.get("出院去向", ""),
            "入院类型": bing_li.get("入院类型", ""),
            "死亡时间": bing_li.get("死亡时间", ""),
            "住院天数": bing_li.get("住院天数", ""),
            "年龄_入院时": ren_kou.get("年龄_入院时", ""),
            "性别": ren_kou.get("性别", ""),
            "种族": ren_kou.get("种族", ""),
        }

        # 诊断列表：主诊断 + 汇总
        zhen_duan_list = rec.get("诊断列表", []) or []
        main_dx = next((d for d in zhen_duan_list if d.get("是否主诊断")), None)
        patient_base["主诊断_icd_code"] = main_dx.get("icd_code", "") if main_dx else ""
        patient_base["主诊断_icd_title"] = main_dx.get("icd_title", "") if main_dx else ""
        patient_base["诊断总数"] = len(zhen_duan_list)
        patient_base["所有诊断_icd_codes"] = "; ".join(
            d.get("icd_code", "") for d in zhen_duan_list if d.get("icd_code")
        )

        # 手术与操作：汇总
        proc_list = rec.get("手术与操作", []) or []
        patient_base["手术操作总数"] = len(proc_list)
        patient_base["所有手术操作_codes"] = "; ".join(
            p.get("icd_code", "") for p in proc_list if p.get("icd_code")
        )
        patient_base["所有手术操作_titles"] = "; ".join(
            p.get("icd_title", "") for p in proc_list if p.get("icd_title")
        )

        # 住院用药：汇总
        med_list = rec.get("住院用药", []) or []
        patient_base["用药总数"] = len(med_list)
        patient_base["所有用药"] = "; ".join(
            m.get("药名", "") for m in med_list if m.get("药名")
        )

        # 生命体征
        vital = rec.get("生命体征", {}) or {}
        bp = vital.get("血压_mmHg", {}) or {}
        liver_ind = vital.get("肝病相关指标", {}) or {}
        for key in ["体温_C", "脉搏_次每分", "呼吸_次每分"]:
            patient_base[key] = vital.get(key, "")
        patient_base["收缩压"] = bp.get("收缩压", "")
        patient_base["舒张压"] = bp.get("舒张压", "")
        for key in ["总胆红素", "直接胆红素", "ALT", "AST", "碱性磷酸酶", "GGT", "总蛋白", "白蛋白", "INR"]:
            patient_base[key] = liver_ind.get(key, "")

        # 体格测量
        phys = rec.get("体格测量", {}) or {}
        for key in ["身高_cm", "体重_kg", "BMI"]:
            patient_base[key] = phys.get(key, "")

        # 实验室检验：汇总所有检验项目
        # 由于 LLM 正确地跳过了这个已结构化的字段，这里手动展平
        lab_data = rec.get("实验室检验", {}) or {}
        lab_items_summary: List[str] = []
        # 用于保存各项目的最新一次值（名称→(时间, 数值, 单位, 异常)）
        lab_latest: Dict[str, Dict[str, Any]] = {}
        for ev_key, ev in lab_data.items():
            ev_time = ev.get("检查时间", "")
            for item in ev.get("项目明细", []) or []:
                item_name = item.get("名称", "")
                val = item.get("结果数值")
                val_txt = item.get("结果文本")
                unit = item.get("单位", "")
                abnormal = item.get("异常标记", "")
                item_time = item.get("项目时间", ev_time)
                if item_name:
                    # 摘要字符串
                    v = val if val is not None else val_txt
                    s = f"{item_name}={v} {unit}".strip() if v is not None else item_name
                    if abnormal:
                        s += f"[{abnormal}]"
                    lab_items_summary.append(s)
                    # 保留最新值（按时间排序取最大）
                    if item_name not in lab_latest or str(item_time) > str(lab_latest[item_name].get("时间", "")):
                        lab_latest[item_name] = {
                            "时间": item_time, "值": v, "单位": unit, "异常": abnormal
                        }
        patient_base["实验室检验_事件数"] = len(lab_data)
        patient_base["实验室检验_摘要"] = "; ".join(lab_items_summary[:50])  # 最多50条避免过长
        # 关键肝病化验项（最新一次）
        key_labs = [
            "Alanine Aminotransferase (ALT)", "Aspartate Aminotransferase (AST)",
            "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphatase",
            "Albumin", "INR(PT)", "Creatinine", "Platelet Count",
            "Hematocrit", "Hemoglobin", "Sodium", "Potassium",
        ]
        for lab_name in key_labs:
            col_name = f"Lab_{lab_name}"
            if lab_name in lab_latest:
                v = lab_latest[lab_name]
                patient_base[col_name] = f"{v['值']} {v['单位']}".strip() if v["值"] is not None else ""
            else:
                patient_base[col_name] = ""

        # ── 按 note 展开 ──────────────────────────────────────
        jiu_zhen = rec.get("就诊文本", {}) or {}
        notes = jiu_zhen.get("notes", []) or []
        extraction_results = rec.get("extraction_results", {}) or {}

        if not notes:
            # 没有 note，仍输出一行患者信息
            row = dict(patient_base)
            row.update({
                "note_index": "",
                "note_type": "",
                "note_charttime": "",
                "note_text": "",
            })
            for cat in ENTITY_CATEGORIES:
                row[f"抽取_{cat}"] = ""
            for col in ["temporal_info", "quantity_info", "impression", "indication", "relations", "extraction_error"]:
                row[col] = ""
            rows.append(row)
            continue

        for note_idx, note in enumerate(notes):
            row = dict(patient_base)
            row["note_index"] = note_idx
            row["note_type"] = note.get("type", "")
            row["note_charttime"] = note.get("charttime", "")
            row["note_text"] = (note.get("text", "") or "").replace("\n", " ").replace("\r", "")[:500]

            # 找到对应的抽取结果（key 格式：".就诊文本.notes.{note_idx}.text"）
            # 尝试多种可能的 key 格式
            ext_key = None
            for k in extraction_results:
                # 匹配 ".就诊文本.notes.{note_idx}.text" 或类似路径
                if f".{note_idx}." in k or k.endswith(f".{note_idx}"):
                    ext_key = k
                    break
                # 若只有一条 note，直接取第一个 key
            if ext_key is None and len(extraction_results) > 0:
                keys = list(extraction_results.keys())
                if note_idx < len(keys):
                    ext_key = keys[note_idx]

            ext = extraction_results.get(ext_key, {}) if ext_key else {}

            # 按实体类别汇总
            entities = ext.get("entities", []) or []
            by_cat: Dict[str, List[str]] = {cat: [] for cat in ENTITY_CATEGORIES}
            for ent in entities:
                cat = ent.get("category", "Other")
                if cat not in by_cat:
                    cat = "Other"
                name = (ent.get("name") or ent.get("original_text") or "").strip()
                value = ent.get("value")
                unit = ent.get("unit", "")
                if name:
                    s = name
                    if value is not None:
                        s += f":{value}"
                        if unit:
                            s += f" {unit}"
                    by_cat[cat].append(s)
            for cat in ENTITY_CATEGORIES:
                row[f"抽取_{cat}"] = "; ".join(by_cat[cat])

            # 时间信息
            temporal = ext.get("temporal_info", []) or []
            row["temporal_info"] = "; ".join(
                f"{t.get('event', '')}:{t.get('time_expression', '')}"
                for t in temporal if t.get("event") or t.get("time_expression")
            )

            # 剂量信息
            qty = ext.get("quantity_info", []) or []
            row["quantity_info"] = "; ".join(
                " ".join(filter(None, [
                    q.get("drug_or_treatment", ""),
                    q.get("amount", ""),
                    q.get("frequency", ""),
                    q.get("duration", ""),
                ]))
                for q in qty
            )

            row["impression"] = ext.get("impression") or ""
            row["indication"] = ext.get("indication") or ""

            # 关系
            relations = ext.get("relations", []) or []
            row["relations"] = "; ".join(
                f"{r.get('source', '')} --{r.get('relation_type', '')}--> {r.get('target', '')}"
                for r in relations if r.get("source") or r.get("target")
            )

            row["extraction_error"] = ext.get("error") or ""

            rows.append(row)

    return rows


def _save_as_csv(records: List[Dict[str, Any]], csv_path: str, verbose: bool = True) -> None:
    """将已处理记录列表保存为 CSV 文件。"""
    import csv

    rows = _records_to_csv_rows(records)
    if not rows:
        if verbose:
            print("⚠️  无可写入的行，跳过 CSV 导出")
        return

    # 按固定顺序构建列名（其余列按首次出现顺序追加）
    ENTITY_CATEGORIES = ["Disease", "Drug", "Symptom", "Test", "Treatment",
                         "Anatomy", "LabValue", "Finding", "Other"]
    priority_cols = [
        "record_index",
        "入院时间", "出院时间", "入院科室", "出院去向", "入院类型", "死亡时间", "住院天数",
        "年龄_入院时", "性别", "种族",
        "主诊断_icd_code", "主诊断_icd_title", "诊断总数", "所有诊断_icd_codes",
        "手术操作总数", "所有手术操作_codes", "所有手术操作_titles",
        "用药总数", "所有用药",
        "体温_C", "脉搏_次每分", "呼吸_次每分", "收缩压", "舒张压",
        "总胆红素", "直接胆红素", "ALT", "AST", "碱性磷酸酶", "GGT", "总蛋白", "白蛋白", "INR",
        "身高_cm", "体重_kg", "BMI",
        # 实验室检验结构化数据
        "实验室检验_事件数", "实验室检验_摘要",
        "Lab_Alanine Aminotransferase (ALT)", "Lab_Aspartate Aminotransferase (AST)",
        "Lab_Total Bilirubin", "Lab_Direct Bilirubin", "Lab_Alkaline Phosphatase",
        "Lab_Albumin", "Lab_INR(PT)", "Lab_Creatinine", "Lab_Platelet Count",
        "Lab_Hematocrit", "Lab_Hemoglobin", "Lab_Sodium", "Lab_Potassium",
        "note_index", "note_type", "note_charttime", "note_text",
    ] + [f"抽取_{cat}" for cat in ENTITY_CATEGORIES] + [
        "temporal_info", "quantity_info", "impression", "indication",
        "relations", "extraction_error",
    ]

    # 收集所有列（包含未预见的列）
    all_keys: List[str] = list(priority_cols)
    seen = set(priority_cols)
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in all_keys})

    if verbose:
        print(f"📄 CSV 已保存: {csv_path}  ({len(rows)} 行, {len(all_keys)} 列)")


# ── 辅助函数 ──────────────────────────────────────────────────

# ── 辅助函数 ──────────────────────────────────────────────────

def _try_repair_json(line: str) -> Optional[Dict[str, Any]]:
    """
    尝试修复因截断导致的 JSON 解析失败。

    策略：根据未闭合的 `{` 和 `[` 数量，枚举可能的补全后缀，
    取第一个能成功解析的结果。

    Returns:
        修复后的 dict，若无法修复则返回 None
    """
    # 计算括号深度
    brace_depth = 0
    bracket_depth = 0
    in_string = False
    escape = False
    for ch in line:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
        elif ch == '[':
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1

    if brace_depth <= 0 and bracket_depth <= 0:
        return None  # 不是截断问题

    # 生成候选后缀（按最可能的顺序）
    # 通常 JSON 对象嵌套：array 在 object 内，先关 bracket 再关 brace
    from itertools import permutations as _perms
    closing = '}' * max(brace_depth, 0) + ']' * max(bracket_depth, 0)
    candidates = set(''.join(p) for p in _perms(closing))
    # 优先尝试 bracket 在前的顺序（更常见的结构）
    priority = [
        ']' * max(bracket_depth, 0) + '}' * max(brace_depth, 0),
        closing,
    ]
    ordered = priority + sorted(candidates - set(priority))

    for suffix in ordered:
        try:
            obj = json.loads(line + suffix)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _get_patient_id(record: Dict[str, Any], fallback_idx: int) -> str:
    """尝试从记录中提取患者标识符"""
    # 尝试常见路径
    for path in [
        ["病历", "人口学", "patient_id"],
        ["patient_id"],
        ["id"],
    ]:
        obj = record
        for key in path:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                obj = None
                break
        if obj is not None:
            return str(obj)
    return f"record_{fallback_idx}"
