import json
from pathlib import Path

import agent_1.codegen_tools as react_tools
from agent_1.codegen_tools import (
    clear_step1_records_path,
    record_file_modality,
    record_source_file,
    record_table_split_strategy,
    set_step1_records_path,
    validate_reorganized_contract,
)


def test_records_path_uses_runtime_override(tmp_path):
    custom_records = tmp_path / "program" / "output" / "step1_results" / "_meta" / "records.json"
    set_step1_records_path(custom_records)

    try:
        assert react_tools._records_path() == custom_records
    finally:
        clear_step1_records_path()



def test_records_path_falls_back_after_runtime_override_cleared(tmp_path):
    set_step1_records_path(tmp_path / "custom" / "records.json")
    clear_step1_records_path()

    expected = Path.cwd() / "reorganized_output" / "_meta" / "records.json"
    assert react_tools._records_path() == expected



def test_record_source_file_creates_minimal_record(monkeypatch, tmp_path):
    file_path = tmp_path / "垂直眼位" / "36906(1).jpg"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("fake image", encoding="utf-8")
    records_path = tmp_path / "reorganized_output" / "_meta" / "records.json"
    monkeypatch.setattr(react_tools, "_records_path", lambda: records_path)

    result = record_source_file(str(file_path))

    assert result == {
        "source_path": str(file_path),
        "source_name": "36906(1).jpg",
        "patient_id": "36906",
    }
    assert json.loads(records_path.read_text(encoding="utf-8")) == [
        {
            "source_path": str(file_path),
            "source_name": "36906(1).jpg",
            "patient_id": "36906",
            "observations": {
                "modality": "ocr",
                "should_split": False,
                "id_column": None,
            },
        }
    ]


def test_record_file_modality_updates_existing_record(monkeypatch, tmp_path):
    file_path = tmp_path / "垂直眼位" / "36906(1).jpg"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("fake image", encoding="utf-8")
    records_path = tmp_path / "reorganized_output" / "_meta" / "records.json"
    monkeypatch.setattr(react_tools, "_records_path", lambda: records_path)
    monkeypatch.setattr(
        "agent_1.codegen_tools.analyze_visual_document",
        lambda path: [
            {
                "page_index": 0,
                "classification_modality": "figure",
            }
        ],
    )
    record_source_file(str(file_path))

    result = record_file_modality(str(file_path))

    assert result == {"modality": "figure"}
    stored = json.loads(records_path.read_text(encoding="utf-8"))
    assert stored[0]["observations"]["modality"] == "figure"
    assert stored[0]["source_name"] == "36906(1).jpg"


def test_record_table_split_strategy_updates_split_fields(monkeypatch, tmp_path):
    file_path = tmp_path / "labs.csv"
    file_path.write_text(
        "patient_id,value\npatient-001,1\npatient-002,2\n",
        encoding="utf-8",
    )
    records_path = tmp_path / "reorganized_output" / "_meta" / "records.json"
    monkeypatch.setattr(react_tools, "_records_path", lambda: records_path)
    record_source_file(str(file_path))

    result = record_table_split_strategy(str(file_path))

    assert result == {
        "should_split": True,
        "id_column": "patient_id",
    }
    stored = json.loads(records_path.read_text(encoding="utf-8"))
    assert stored[0]["observations"]["should_split"] is True
    assert stored[0]["observations"]["id_column"] == "patient_id"


def test_validate_reorganized_contract_accepts_extra_path_layer(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    good_file = output_root / "patient-123" / "table" / "ct_scan" / "labs.csv"
    good_file.parent.mkdir(parents=True)
    good_file.write_text("id,value\npatient-123,1\n", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/ct_scan/labs.csv",
                    "source_name": "labs.csv",
                    "source_extension": ".csv",
                    "source_type": "table",
                    "patient_id": "patient-123",
                    "exam_type": "ct_scan",
                    "observations": {
                        "modality": "table",
                        "should_split": True,
                        "id_column": "patient_id",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is True
    assert result["records_count"] == 1


def test_validate_reorganized_contract_accepts_observation_records_without_output_count_match(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    (output_root / "36906" / "table").mkdir(parents=True)
    (output_root / "36906" / "table" / "a.csv").write_text("id,value\n36906,1\n", encoding="utf-8")
    (output_root / "36907" / "table").mkdir(parents=True)
    (output_root / "36907" / "table" / "b.csv").write_text("id,value\n36907,2\n", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/labs.xlsx",
                    "source_name": "labs.xlsx",
                    "source_extension": ".xlsx",
                    "source_type": "table",
                    "patient_id": None,
                    "exam_type": "table",
                    "observations": {
                        "modality": "table",
                        "should_split": True,
                        "id_column": "id",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is True
    assert result["records_count"] == 1
    assert result["output_file_count"] == 2


def test_validate_reorganized_contract_rejects_missing_records_json(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    good_file = output_root / "patient-123" / "table" / "labs.csv"
    good_file.parent.mkdir(parents=True)
    good_file.write_text("id,value\npatient-123,1\n", encoding="utf-8")

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(output_root / "_meta" / "records.json"),
    )

    assert result["passed"] is False
    assert "records.json 不存在" in result["issues"]



def test_validate_reorganized_contract_accepts_current_runtime_record_schema(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    (output_root / "36906" / "figure").mkdir(parents=True)
    (output_root / "36906" / "figure" / "36906(1).jpg").write_text("img", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/36906(1).jpg",
                    "source_name": "36906(1).jpg",
                    "patient_id": "36906",
                    "modality": "figure",
                    "observation": {
                        "source_file": {
                            "source_path": "rawdata/36906(1).jpg",
                            "source_name": "36906(1).jpg",
                            "patient_id": "36906",
                        },
                        "file_modality": {
                            "modality": "figure",
                        },
                        "table_split_strategy": None,
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is True
    assert result["records_count"] == 1
    assert result["output_file_count"] == 1


def test_validate_reorganized_contract_accepts_two_extra_path_layers(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    nested_file = output_root / "36906" / "figure" / "检查A" / "子类B" / "a.jpg"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("img", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/检查A/子类B/a.jpg",
                    "source_name": "a.jpg",
                    "patient_id": "36906",
                    "observations": {
                        "modality": "figure",
                        "should_split": False,
                        "id_column": None,
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is True
    assert result["output_file_count"] == 1


def test_validate_reorganized_contract_rejects_too_shallow_path(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    bad_file = output_root / "36906" / "a.jpg"
    bad_file.parent.mkdir(parents=True)
    bad_file.write_text("img", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/a.jpg",
                    "source_name": "a.jpg",
                    "patient_id": "36906",
                    "observations": {
                        "modality": "figure",
                        "should_split": False,
                        "id_column": None,
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is False
    assert "输出路径层级至少应为 id/模态/文件，允许中间目录" in result["issues"]


def test_validate_reorganized_contract_reports_missing_business_outputs_instead_of_missing_directory(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "source_path": "rawdata/a.jpg",
                    "source_name": "a.jpg",
                    "patient_id": "36906",
                    "observations": {
                        "modality": "figure",
                        "should_split": False,
                        "id_column": None,
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (meta_dir / "reorganize_summary.json").write_text("{}", encoding="utf-8")

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is False
    assert "未检测到 _meta 之外的输出文件" in result["issues"]
def test_validate_reorganized_contract_rejects_invalid_observation_record_shape(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_root = tmp_path / "reorganized_output"
    (output_root / "36906" / "figure").mkdir(parents=True)
    (output_root / "36906" / "figure" / "a.jpg").write_text("img", encoding="utf-8")
    meta_dir = output_root / "_meta"
    meta_dir.mkdir(parents=True)
    records_path = meta_dir / "records.json"
    records_path.write_text(
        json.dumps([
            {
                "source_path": "a.jpg",
                "source_name": "a.jpg",
                "source_extension": ".jpg",
                "source_type": "image",
                "patient_id": "36906",
                "exam_type": "垂直眼位"
            }
        ], ensure_ascii=False),
        encoding="utf-8",
    )

    result = validate_reorganized_contract(
        input_path=str(input_dir),
        output_root=str(output_root),
        records_path=str(records_path),
    )

    assert result["passed"] is False
    assert "records.json observation 结构不完整" in result["issues"]
