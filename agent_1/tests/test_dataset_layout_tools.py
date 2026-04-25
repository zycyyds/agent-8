import sys
import types
from pathlib import Path


def _install_layout_stubs():
    if "doclayout_yolo" not in sys.modules:
        doclayout_stub = types.ModuleType("doclayout_yolo")

        class DummyYOLOv10:
            def __init__(self, *args, **kwargs):
                pass

        doclayout_stub.YOLOv10 = DummyYOLOv10
        sys.modules["doclayout_yolo"] = doclayout_stub

    if "PyQt5" not in sys.modules:
        pyqt5_stub = types.ModuleType("PyQt5")
        qtwidgets_stub = types.ModuleType("PyQt5.QtWidgets")
        qtgui_stub = types.ModuleType("PyQt5.QtGui")
        qtcore_stub = types.ModuleType("PyQt5.QtCore")

        for name in [
            "QApplication", "QMainWindow", "QLabel", "QPushButton", "QVBoxLayout",
            "QHBoxLayout", "QWidget", "QMessageBox", "QSizePolicy",
        ]:
            setattr(qtwidgets_stub, name, type(name, (), {}))

        qtgui_stub.QPixmap = type("QPixmap", (), {})
        qtcore_stub.Qt = type("Qt", (), {})

        sys.modules["PyQt5"] = pyqt5_stub
        sys.modules["PyQt5.QtWidgets"] = qtwidgets_stub
        sys.modules["PyQt5.QtGui"] = qtgui_stub
        sys.modules["PyQt5.QtCore"] = qtcore_stub


def test_process_table_file_drops_blank_columns(tmp_path):
    _install_layout_stubs()
    import pandas as pd
    from agent_1.layout_analysis_tool import _process_table_file

    src = tmp_path / "input.csv"
    dst = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "id": ["1", "2"],
            "empty_all_blank": ["", "   "],
            "empty_all_null": [None, None],
            "value": ["a", "b"],
        }
    ).to_csv(src, index=False)

    ok = _process_table_file(str(src), str(dst))

    assert ok is True
    cleaned = pd.read_csv(dst)
    assert list(cleaned.columns) == ["id", "value"]
