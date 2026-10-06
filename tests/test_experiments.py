import pandas as pd

from fraud.experiments import log_iteration


def test_first_row_creates_the_file_and_its_folder(tmp_path):
    path = tmp_path / "m2_2" / "log.csv"
    log_iteration({"iteration": 0, "name": "baseline"}, path)
    assert pd.read_csv(path)["name"].tolist() == ["baseline"]


def test_rows_are_kept_in_iteration_order(tmp_path):
    path = tmp_path / "log.csv"
    log_iteration({"iteration": 1, "name": "features"}, path)
    log_iteration({"iteration": 0, "name": "baseline"}, path)
    assert pd.read_csv(path)["iteration"].tolist() == [0, 1]


def test_logging_the_same_iteration_again_replaces_it(tmp_path):
    """Re-running the notebook must not duplicate rows."""
    path = tmp_path / "log.csv"
    log_iteration({"iteration": 0, "name": "baseline", "val_PR-AUC": 0.90}, path)
    log_iteration({"iteration": 0, "name": "baseline", "val_PR-AUC": 0.95}, path)
    log = pd.read_csv(path)
    assert len(log) == 1
    assert log.loc[0, "val_PR-AUC"] == 0.95
