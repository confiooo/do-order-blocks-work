import json

import pandas as pd

from hawk_research import compare


def _write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith(".parquet"):
        obj.to_parquet(p, index=False)
    else:
        p.write_text(json.dumps(obj))


def test_identical_runs_are_same_despite_volatile_keys(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, "reports/r.json", {"mean": 0.1, "created_at": "2026-01-01", "paths": {"x": "/a"}, "n": 3})
    _write(new, "reports/r.json", {"mean": 0.1, "created_at": "2026-10-08", "paths": {"x": "b"}, "n": 3})
    df = pd.DataFrame(
        {"symbol": ["B", "A"], "source_open_time": [2, 1], "x": [0.5, 0.25], "experiment_id": [7, 8]}
    )
    _write(old, "zones/z.parquet", df)
    _write(new, "zones/z.parquet", df.iloc[::-1].assign(experiment_id=[1, 2]))  # other order, other ids
    rows = compare.compare_dirs(old, new)
    assert [(r["file"], r["status"]) for r in rows] == [
        ("reports/r.json", "SAME"),
        ("zones/z.parquet", "SAME"),
    ]


def test_real_differences_are_reported(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, "reports/r.json", {"mean": 0.1, "folds": [1, 2]})
    _write(new, "reports/r.json", {"mean": 0.1 + 1e-9, "folds": [1, 3], "added": True})
    _write(old, "zones/z.parquet", pd.DataFrame({"symbol": ["A"], "x": [1.0]}))
    _write(new, "zones/z.parquet", pd.DataFrame({"symbol": ["A", "B"], "x": [1.0, 2.0]}))
    _write(old, "features/only_old.json", {})
    rows = {r["file"]: r for r in compare.compare_dirs(old, new)}
    assert rows["reports/r.json"]["status"] == "DIFF"
    assert set(rows["reports/r.json"]["notes"]) == {
        "/added: only in new",
        "/folds[1]: 2 vs 3",
        f"/mean: 0.1 vs {0.1 + 1e-9!r}",
    }
    assert rows["zones/z.parquet"]["notes"] == ["rows 1 vs 2"]
    assert rows["features/only_old.json"]["status"] == "MISSING"
    # a looser tolerance accepts the float drift
    loose = compare.diff_json({"mean": 0.1}, {"mean": 0.1 + 1e-9}, rtol=1e-6)
    assert loose == []
