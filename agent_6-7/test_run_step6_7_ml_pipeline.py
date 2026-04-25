import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_step6_7_ml_pipeline.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("step6_7_pipeline_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_read_data_maps_id_to_patient_id(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("id,value\n1,a\n2,b\n", encoding="utf-8")

    module = _load_module()
    df = module._read_data(str(csv_path))

    assert "patient_id" in df.columns
    assert "id" not in df.columns
    assert df["patient_id"].astype(str).tolist() == ["1", "2"]
