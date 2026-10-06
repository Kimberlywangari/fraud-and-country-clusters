"""The experiment log: one row per iteration, kept in reports/m2_2/experiment_log.csv."""
import pandas as pd

from fraud.data import ROOT

LOG_CSV = ROOT / "reports" / "m2_2" / "experiment_log.csv"


def log_iteration(row, path=LOG_CSV):
    """Add `row` (a dict with an 'iteration' key) to the log, replacing any earlier row with the same iteration.

    Replacing, not appending, is what lets the notebook be re-run top to bottom without duplicating rows.
    """
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["iteration"] != row["iteration"]]
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values("iteration").to_csv(path, index=False)
    return new
