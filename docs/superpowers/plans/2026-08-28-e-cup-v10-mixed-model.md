# E-CUP v10 Features and ROCm Mixed Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible v10 feature pipeline, honest three-window validation, and a ROCm mixed temporal model that can improve the prediction shape of the 1.649315 v9-opt baseline.

**Architecture:** Extend each anchor with explicit disjoint weekly/monthly history and auxiliary weekly targets, while keeping artifacts resumable and schema-checked. Compare the existing GBDT/CNN families against a three-encoder ROCm model (daily CNN, long-token Transformer, tabular MLP), then assemble conservative and mixed submissions using weights and relative scales learned out of time.

**Tech Stack:** Python 3.12/3.13, NumPy, Polars, LightGBM, PyTorch ROCm (`gfx1201`), pytest, PowerShell 5.1, JSON/Markdown manifests.

**Spec:** `docs/superpowers/specs/2026-08-28-e-cup-v10-mixed-model-design.md`

## Global Constraints

- Immutable baseline: `submission_v9_opt.csv`, SHA-256 `d734c08c65707447d391da0c4d092c629e08debf2882425e79d1ff33e13812a8`, public RMSLE 1.649315.
- Full training target: AMD Radeon RX 9070 XT (`gfx1201`), ROCm, 17.1 GB VRAM, 32 GB host RAM.
- Local Mac work is limited to unit tests, bounded analyses, and user-subset smoke tests; no unbounded full-data fit.
- The main loss remains MSE on `log1p(GMV30)`; auxiliary targets never enter the feature list.
- Full-data artifacts are built one anchor at a time and are resumable only after schema and checksum validation.
- Use approximately 14-day training-anchor spacing unless a recorded experiment justifies denser overlapping anchors.
- New submissions and prediction arrays never overwrite v9 or another submitted artifact.
- Every material run writes a JSON manifest and an entry in `docs/experiments.md`.

---

## File Map

- `src/project_io.py`: dataset path resolution, atomic writes, hashes, anchor/run manifests, feature-column filtering.
- `src/v10_features.py`: disjoint lag specifications, lag aggregations, coverage columns, and long-memory summaries.
- `src/future_targets.py`: 30-day main target and four auxiliary weekly target blocks.
- `src/build_features.py`: existing anchor builder plus the `--feature-set v10` integration and bounded smoke flags.
- `src/metrics.py`: RMSLE, one-parameter scale fitting, paired effects, bootstrap confidence intervals, and rolling fold dates.
- `src/rolling_gbdt.py`: honest calibration/outer evaluation for direct and two-part LightGBM models.
- `src/sequence_data.py`: memory-mapped daily history, daily-window gathering, weekly/monthly token pooling, masks, and tabular normalization.
- `src/mixed_model.py`: daily CNN, weekly/monthly Transformer encoders, tabular MLP, heads, and multitask loss.
- `src/train_mixed.py`: ROCm/CPU training loop, early stopping on calibration only, OOM retry, fold/final predictions, manifests.
- `src/rocm_preflight.py`: ROCm capability and memory smoke report.
- `src/assemble_v10.py`: cross-window decisions, blend weights, relative scale transfer, final CSVs and manifests.
- `src/check_submission.py`: reusable submission validation with repository data paths.
- `run_v10.ps1`: resumable external-machine orchestration.
- `tests/`: unit and bounded end-to-end tests for every new boundary.
- `docs/experiments.md`: append-only experiment ledger.
- `README.md`: current status and exact reproduction commands.

---

### Task 1: Artifact Safety, Paths, and Test Foundation

**Files:**
- Create: `src/project_io.py`
- Create: `tests/test_project_io.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `resolve_raw(root: Path, explicit: Path | None = None) -> Path`
- Produces: `feature_columns(columns: Sequence[str]) -> list[str]`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `schema_hash(columns: Sequence[str]) -> str`
- Produces: `select_feature_view(manifest: Mapping[str, Any], view: str) -> list[str]`
- Produces: `write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None`
- Produces: `write_anchor_manifest(parquet_path: Path, anchor: date, feature_cols: Sequence[str], target_cols: Sequence[str], row_count: int, feature_groups: Mapping[str, Sequence[str]]) -> Path`
- Produces: `validate_anchor_artifact(parquet_path: Path, expected_anchor: date, expected_features: Sequence[str] | None = None) -> dict[str, Any]`

- [ ] **Step 1: Add test/runtime dependencies and ignored transient test files**

Add `pytest>=8.3` and `psutil>=6.0` to `requirements.txt`. Add `.pytest_cache/`, `.coverage`, and `__pycache__/` to `.gitignore`; retain the existing submission exceptions unchanged.

- [ ] **Step 2: Write failing path, filtering, hash, and manifest tests**

```python
# tests/test_project_io.py
from datetime import date
from pathlib import Path

import json
import pytest

from project_io import (
    feature_columns, resolve_raw, schema_hash, sha256_file,
    select_feature_view, validate_anchor_artifact, write_anchor_manifest,
    write_json_atomic,
)


def test_resolve_raw_prefers_explicit_then_data_dir(tmp_path: Path):
    nested = tmp_path / "data" / "train.parquet"
    nested.parent.mkdir()
    nested.write_bytes(b"nested")
    explicit = tmp_path / "custom.parquet"
    explicit.write_bytes(b"explicit")
    assert resolve_raw(tmp_path, explicit) == explicit
    assert resolve_raw(tmp_path) == nested


def test_resolve_raw_reports_all_checked_paths(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="data/train.parquet"):
        resolve_raw(tmp_path)


def test_auxiliary_targets_never_become_features():
    cols = ["user_id", "gmv_s30", "target", "target_gmv_w1", "target_any_w1", "anchor_ord"]
    assert feature_columns(cols) == ["gmv_s30"]


def test_manifest_feature_views_are_explicit():
    manifest = {"feature_groups": {"v9": ["a", "b"], "v10": ["a", "b", "c"]}}
    assert select_feature_view(manifest, "v9") == ["a", "b"]
    assert select_feature_view(manifest, "v10") == ["a", "b", "c"]
    with pytest.raises(ValueError, match="feature view"):
        select_feature_view(manifest, "unknown")


def test_atomic_json_and_hash_are_deterministic(tmp_path: Path):
    path = tmp_path / "m.json"
    write_json_atomic(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert sha256_file(path) == sha256_file(path)
    assert schema_hash(["b", "a"]) == schema_hash(["b", "a"])
    assert schema_hash(["b", "a"]) != schema_hash(["a", "b"])
```

- [ ] **Step 3: Run the new tests and confirm the missing-module failure**

Run: `PYTHONPATH=src pytest tests/test_project_io.py -v`

Expected: collection fails with `ModuleNotFoundError: No module named 'project_io'`.

- [ ] **Step 4: Implement the focused I/O module**

```python
# src/project_io.py
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

TARGET_PREFIX = "target_"
DROP_COLUMNS = {"user_id", "target", "anchor_ord"}


def resolve_raw(root: Path, explicit: Path | None = None) -> Path:
    candidates = ([Path(explicit)] if explicit else []) + [root / "data" / "train.parquet", root / "train.parquet"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("train parquet not found; checked: " + ", ".join(map(str, candidates)))


def feature_columns(columns: Sequence[str]) -> list[str]:
    return [c for c in columns if c not in DROP_COLUMNS and not c.startswith(TARGET_PREFIX)]


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(columns: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
```

Implement `write_anchor_manifest` with keys `anchor`, `parquet`, `parquet_sha256`, `schema_hash`, `feature_columns`, `feature_groups`, `target_columns`, and `row_count`. `feature_groups["v9"]` is the base feature list captured before the v10 join; `feature_groups["v10"]` is the full feature list. Implement `select_feature_view` to return only a named, manifest-declared group. Implement `validate_anchor_artifact` to load the JSON sidecar, compare the date, row count, schema hash, optional expected feature list, and current Parquet checksum, raising `ValueError` with the mismatched key.

- [ ] **Step 5: Add and pass a manifest corruption test**

Create a tiny file at `tmp_path / "anchor_2025-01-31.parquet"`, write its manifest, assert validation passes, mutate the file bytes, and assert `validate_anchor_artifact` raises `ValueError` containing `parquet_sha256`.

Run: `PYTHONPATH=src pytest tests/test_project_io.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the foundation**

```bash
git add .gitignore requirements.txt src/project_io.py tests/test_project_io.py
git commit -m "test: add safe artifact foundation"
```

---

### Task 2: Explicit Weekly and Monthly v10 Features

**Files:**
- Create: `src/v10_features.py`
- Create: `tests/test_v10_features.py`

**Interfaces:**
- Consumes: a Polars history frame with `user_id`, `age`, all core/channel columns, and `is_active`, `is_ord_day`
- Produces: `LagSpec(family: str, index: int, start_age: int, end_age: int)`
- Produces: `WEEKLY_SPECS: Sequence[LagSpec]` and `MONTHLY_SPECS: Sequence[LagSpec]`
- Produces: `build_v10_lag_frame(hist: pl.DataFrame, users: pl.DataFrame, anchor: date) -> pl.DataFrame`
- Produces: `v10_feature_names() -> Sequence[str]`

- [ ] **Step 1: Write failing boundary and coverage tests**

```python
# tests/test_v10_features.py
from datetime import date

import polars as pl

from v10_features import build_v10_lag_frame


def synthetic_history() -> tuple[pl.DataFrame, pl.DataFrame]:
    ages = [0, 6, 7, 29, 30, 55, 56, 359, 360]
    rows = []
    for age in ages:
        rows.append({
            "user_id": 1, "age": age, "gmv": float(age + 1), "to_ord": 1,
            "to_cart": 2, "searches": 3, "is_active": 1, "is_ord_day": 1,
            "gmv_search": 1.0, "gmv_cat": 2.0, "search_to_ord": 1,
            "cat_to_ord": 0, "search_to_cart": 1, "cat_to_cart": 1,
            "search": 1, "cat": 1,
        })
    return pl.DataFrame(rows), pl.DataFrame({"user_id": [1, 2]})


def test_disjoint_boundaries_do_not_overlap():
    hist, users = synthetic_history()
    out = build_v10_lag_frame(hist, users, date(2026, 2, 13)).sort("user_id")
    one = out.row(0, named=True)
    assert one["lag_to_ord_w01"] == 2       # ages 0 and 6
    assert one["lag_to_ord_w02"] == 1       # age 7
    assert one["lag_to_ord_w05"] == 1       # age 30
    assert one["lag_to_ord_w08"] == 1       # age 55
    assert one["lag_to_ord_m01"] == 4       # ages 0, 6, 7, 29
    assert one["lag_to_ord_m02"] == 2       # ages 30 and 55
    assert one["lag_to_ord_m12"] == 1       # age 359
    assert "lag_to_ord_m13" not in out.columns


def test_missing_user_gets_zero_values_not_fake_coverage():
    hist, users = synthetic_history()
    out = build_v10_lag_frame(hist, users, date(2025, 1, 15)).sort("user_id")
    two = out.row(1, named=True)
    assert two["lag_gmv_w01"] == 0
    assert two["lag_cov_w01"] == 1.0
    assert two["lag_cov_w03"] == 1 / 7
    assert two["lag_cov_m02"] == 0.0
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `PYTHONPATH=src pytest tests/test_v10_features.py -v`

Expected: collection fails because `v10_features` does not exist.

- [ ] **Step 3: Implement lag specifications and conditional aggregation**

```python
# src/v10_features.py
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

DATA_START = date(2025, 1, 1)
CORE = ("gmv", "to_ord", "to_cart", "searches", "is_active", "is_ord_day")
CHANNEL = ("gmv_search", "gmv_cat", "search_to_ord", "cat_to_ord",
           "search_to_cart", "cat_to_cart", "search", "cat")


@dataclass(frozen=True)
class LagSpec:
    family: str
    index: int
    start_age: int
    end_age: int

    @property
    def suffix(self) -> str:
        return f"{self.family}{self.index:02d}"


WEEKLY_SPECS = tuple(LagSpec("w", i + 1, 7 * i, 7 * i + 6) for i in range(8))
MONTHLY_SPECS = tuple(LagSpec("m", i + 1, 30 * i, 30 * i + 29) for i in range(12))
```

For each specification, aggregate `CORE`. Aggregate `CHANNEL` only for `w01..w04` and `m01..m06`. Join onto the complete user list and fill only aggregate values with zero. Add `lag_cov_<suffix>` as the fraction of calendar days in the lag interval that lie on or after `DATA_START`; coverage is constant across users at one anchor.

- [ ] **Step 4: Implement deterministic long-memory summaries**

For each core signal and each family, add:

```python
def weighted_level(values: list[pl.Expr], coverage: list[pl.Expr], half_life: float, name: str) -> pl.Expr:
    weights = np.exp2(-np.arange(len(values), dtype=np.float64) / half_life)
    numerator = sum(float(w) * value for w, value in zip(weights, values))
    denominator = sum(float(w) * cov for w, cov in zip(weights, coverage))
    return (numerator / (denominator + 1e-6)).alias(name)
```

Use half-lives `(2, 4, 8)` for weekly lags and `(1, 3, 6)` for monthly lags. Add coverage-weighted mean, slope against period index, standard deviation, nonzero rate, maximum share, and entropy `-sum(p * log(p + 1e-12))`. Add empirical shrinkage level `(sum(values) + alpha * population_mean) / (covered_periods + alpha)`, where `alpha = clip(population_mean**2 / max(population_variance - population_mean, 1e-6), 0.25, 24.0)` is computed from historical period levels in the same anchor; it uses no future target.

- [ ] **Step 5: Run boundary, coverage, finiteness, and schema tests**

Extend the test to assert all long-memory columns are finite for both users, `v10_feature_names()` has no duplicates, and the weekly/monthly aggregate sets are disjoint.

Run: `PYTHONPATH=src pytest tests/test_v10_features.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the v10 feature module**

```bash
git add src/v10_features.py tests/test_v10_features.py
git commit -m "feat: add disjoint long-memory features"
```

---

### Task 3: Auxiliary Targets and Resumable v10 Anchor Builder

**Files:**
- Create: `src/future_targets.py`
- Create: `tests/test_future_targets.py`
- Create: `tests/test_build_features_smoke.py`
- Modify: `src/build_features.py:22-24, 62-75, 266-396, 403-453`
- Modify: `src/train.py:71-86, 109-145, 233-312, 316-377`
- Modify: `src/nn_train.py:24-37, 137-176`

**Interfaces:**
- Consumes: Task 1 `resolve_raw`, `feature_columns`, manifest helpers
- Consumes: Task 2 `build_v10_lag_frame`
- Produces: `TARGET_COLUMNS: Sequence[str]`
- Produces: `build_future_targets(df: pl.DataFrame, users: pl.DataFrame, anchor: date) -> pl.DataFrame`
- Changes: `build_anchor(df: pl.DataFrame, users: pl.DataFrame, anchor: date, with_target: bool, feature_set: str = "v9") -> pl.DataFrame`

- [ ] **Step 1: Write failing weekly-target boundary tests**

```python
# tests/test_future_targets.py
from datetime import date, timedelta

import polars as pl

from future_targets import TARGET_COLUMNS, build_future_targets


def test_four_future_blocks_have_exact_boundaries():
    anchor = date(2025, 1, 31)
    offsets = [1, 7, 8, 14, 15, 21, 22, 30, 31]
    df = pl.DataFrame({
        "event_date": [anchor + timedelta(days=i) for i in offsets],
        "user_id": [1] * len(offsets),
        "gmv": [float(i) for i in offsets],
        "to_ord": [1] * len(offsets),
    })
    out = build_future_targets(df, pl.DataFrame({"user_id": [1, 2]}), anchor).sort("user_id")
    one = out.row(0, named=True)
    assert one["target_gmv_w1"] == 8.0
    assert one["target_gmv_w2"] == 22.0
    assert one["target_gmv_w3"] == 36.0
    assert one["target_gmv_w4"] == 52.0
    assert one["target"] == 118.0
    assert one["target_any_w4"] == 1
    assert out.row(1, named=True)["target"] == 0.0
    assert "target_gmv_w1" in TARGET_COLUMNS
```

- [ ] **Step 2: Run the target test and verify it fails**

Run: `PYTHONPATH=src pytest tests/test_future_targets.py -v`

Expected: `ModuleNotFoundError` for `future_targets`.

- [ ] **Step 3: Implement one-pass main and weekly target construction**

Use future-day masks `(1,7)`, `(8,14)`, `(15,21)`, `(22,30)` and group by user once. Return `target`, `target_gmv_w1..w4`, `target_any_w1..w4`, and `target_items_w1..w4`; left join all users and fill target nulls with zero.

- [ ] **Step 4: Integrate v10, path flags, and safe resume into the anchor builder**

Add CLI arguments:

```python
ap.add_argument("--raw", type=Path, default=None)
ap.add_argument("--feature-set", choices=("v9", "v10"), default="v9")
ap.add_argument("--anchors", default="", help="comma-separated exact anchors; overrides generated train anchors")
ap.add_argument("--max-users", type=int, default=0, help="sorted user prefix for bounded smoke tests")
```

`load_raw(raw_path)` calls `resolve_raw(ROOT, raw_path)`. For v10, `build_anchor` joins `build_v10_lag_frame(hist, users, anchor)` and replaces the old main-target block with `build_future_targets`. Cast every auxiliary numeric target to float32 except the main `target`, which remains float64.

Capture the base feature list immediately before joining v10 lag columns. Write each anchor to `<path>.tmp`, atomically replace the final Parquet, then write its manifest with explicit `v9` and `v10` feature groups. When an anchor and sidecar already exist, validate both; skip only if validation succeeds and the manifest feature set equals the requested `v9` or `v10`. A stale or incomplete pair raises an actionable error and is never silently trusted.

- [ ] **Step 5: Replace hard-coded DROP lists with the Task 1 filter**

In `src/train.py` and `src/nn_train.py`, import the Task 1 selectors and replace every schema comprehension based on `DROP` with the feature group named by `--feature-view v9|v10`. Add an assertion that no selected feature starts with `target_`. The default remains `v9` for backwards compatibility; all new v10 commands pass the view explicitly.

- [ ] **Step 6: Add a bounded two-user end-to-end anchor test**

Build a tiny raw frame spanning history and 30 future days, call `build_anchor(df, users, anchor, with_target=True, feature_set="v10")`, and assert:

```python
assert out.height == users.height
assert "lag_gmv_w01" in out.columns
assert "target_gmv_w4" in out.columns
assert not any(c.startswith("target_") for c in feature_columns(out.columns))
```

Also build the same anchor from a full frame and from `df.filter(pl.col("event_date") <= anchor)`, then assert `np.allclose(full.select(cols).to_numpy(), truncated.select(cols).to_numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)`. This is the regression test that prevents future rows from entering v10 features.

Run: `PYTHONPATH=src pytest tests/test_future_targets.py tests/test_build_features_smoke.py -v`

Expected: all tests pass.

- [ ] **Step 7: Run a bounded local artifact build**

Run with thread limits:

```bash
POLARS_MAX_THREADS=4 CASE3_WORK=work10_smoke python src/build_features.py --feature-set v10 --anchors 2026-01-14,2026-02-13 --max-users 2000
```

Expected: two Parquet files and two valid sidecars under `work10_smoke/`; host remains responsive.

- [ ] **Step 8: Commit the integrated builder**

```bash
git add src/future_targets.py src/build_features.py src/train.py src/nn_train.py tests/test_future_targets.py tests/test_build_features_smoke.py
git commit -m "feat: build resumable v10 anchors"
```

---

### Task 4: Honest Rolling Metrics and Acceptance Rules

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`
- Modify: `src/make_submission.py:43-75`

**Interfaces:**
- Produces: `FoldSpec(outer: date, calibration: date, train_through: date)` with `FoldSpec.from_outer(outer: date) -> FoldSpec`, setting calibration to `outer - 30 days` and train-through to `outer - 60 days`
- Produces: `ROLLING_FOLDS: Sequence[FoldSpec]`
- Produces: `rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float`
- Produces: `best_scale(y_true: np.ndarray, log_pred: np.ndarray, lo: float = 0.45, hi: float = 1.60) -> tuple[float, float]`
- Produces: `paired_effect(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]`
- Produces: `bootstrap_fold_effect(folds: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]], seed: int = 20260828, n_boot: int = 2000) -> dict[str, float]`
- Produces: `accept_candidate(per_fold: Sequence[Mapping[str, float]], bootstrap: Mapping[str, float]) -> tuple[bool, list[str]]`

- [ ] **Step 1: Write failing metric and date tests**

```python
# tests/test_metrics.py
from datetime import date

import numpy as np

from metrics import ROLLING_FOLDS, accept_candidate, best_scale, bootstrap_fold_effect, paired_effect, rmsle


def test_rolling_dates_are_non_overlapping():
    assert [f.outer for f in ROLLING_FOLDS] == [date(2025, 9, 19), date(2025, 11, 14), date(2026, 1, 14)]
    for fold in ROLLING_FOLDS:
        assert (fold.outer - fold.calibration).days == 30
        assert (fold.calibration - fold.train_through).days == 30


def test_best_scale_recovers_known_multiplier():
    y = np.array([0.0, 2.0, 10.0, 30.0])
    raw = y / 0.8
    scale, mse = best_scale(y, np.log1p(raw))
    assert abs(scale - 0.8) < 2e-3
    assert mse < 1e-7


def test_paired_effect_is_positive_when_b_is_better():
    y = np.array([0.0, 10.0, 30.0])
    a = np.array([1.0, 5.0, 15.0])
    b = np.array([0.1, 9.0, 28.0])
    result = paired_effect(y, a, b)
    assert result["mean_sq_gain"] > 0
    assert rmsle(y, b) < rmsle(y, a)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `PYTHONPATH=src pytest tests/test_metrics.py -v`

Expected: `ModuleNotFoundError` for `metrics`.

- [ ] **Step 3: Implement metrics and deterministic user-block bootstrap**

Define paired gain per user as `sq_error_baseline - sq_error_candidate`. For multiple folds, stack a matrix shaped `(n_folds, n_users)`, resample user columns with one shared index vector per bootstrap replicate, average first over users and then equally over folds, and return `mean`, `ci_low`, and `ci_high` at 2.5% and 97.5%.

`accept_candidate` returns true only when mean shape RMSLE gain is at least 0.0003, at least two folds improve, January degradation is no worse than 0.0002, and `ci_low > 0`. Return every failed rule as a human-readable reason.

- [ ] **Step 4: Remove duplicate scale code from submission assembly**

Import `best_scale` from `metrics.py` in `make_submission.py`. The new function accepts true levels, so replace `best_scale(ly, prediction)` with `best_scale(yva, prediction)` and retain `ly = np.log1p(yva)` only for direct error calculations.

- [ ] **Step 5: Pass unit tests and the existing submission manifest smoke path**

Run: `PYTHONPATH=src pytest tests/test_metrics.py tests/test_project_io.py -v`

Expected: tests pass. Submission validation is exercised after the repository path fix in Task 9.

- [ ] **Step 6: Commit validation math**

```bash
git add src/metrics.py src/make_submission.py tests/test_metrics.py
git commit -m "feat: add honest rolling validation metrics"
```

---

### Task 5: Controlled v9/v10 Rolling LightGBM Runner

**Files:**
- Create: `src/rolling_gbdt.py`
- Create: `tests/test_rolling_gbdt.py`
- Modify: `src/train.py:233-313`

**Interfaces:**
- Consumes: `FoldSpec`, `best_scale`, `paired_effect`, `feature_columns`, existing `fit_lgb`, `fit_twopart`, `load_stack`, `raw_predict`
- Produces: `select_spaced_anchors(available: Sequence[date], through: date, min_spacing_days: int = 12) -> list[date]`
- Produces: `run_fold(work: Path, out: Path, fold: FoldSpec, kinds: Sequence[str], rounds: int, seed: int) -> dict[str, Any]`
- Produces files: `out_work10/rolling/<outer>/<kind>_{calibration,outer}_log.npy` and `report.json`

- [ ] **Step 1: Write failing split and no-outer-target tests**

```python
# tests/test_rolling_gbdt.py
from datetime import date, timedelta

from metrics import FoldSpec
from rolling_gbdt import select_spaced_anchors, training_contract


def test_spacing_and_cutoff_are_strict():
    dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(0, 100, 5)]
    selected = select_spaced_anchors(dates, date(2025, 3, 20), min_spacing_days=12)
    assert max(selected) <= date(2025, 3, 20)
    assert all((b - a).days >= 12 for a, b in zip(selected, selected[1:]))


def test_outer_target_is_not_early_stopping_data():
    fold = FoldSpec.from_outer(date(2026, 1, 14))
    contract = training_contract(fold)
    assert contract["early_stop_anchor"] == date(2025, 12, 15)
    assert contract["outer_anchor"] == date(2026, 1, 14)
    assert contract["max_training_target_anchor"] == date(2025, 11, 15)
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `PYTHONPATH=src pytest tests/test_rolling_gbdt.py -v`

Expected: module or function import failure.

- [ ] **Step 3: Extract reusable fit/predict behavior from `run_val`**

Add to `train.py`:

```python
def fit_kinds(xtr, ytr_used, wtr, anchor_ids, xstop, ystop, stop_scale,
              kinds, rounds, seed, feature_names):
    """Fit requested kinds with early stopping only on xstop/ystop."""
    ytr_log = np.log1p(ytr_used)
    ystop_log = np.log1p(ystop / stop_scale)
    models, iterations = {}, {}
    for kind in kinds:
        if kind == "two":
            model = fit_twopart(xtr, ytr_used, xstop, ystop, stop_scale,
                                rounds, seed, wtr=wtr, anchor_id=anchor_ids)
            iterations[kind] = [int(model[0].best_iteration), int(model[1].best_iteration)]
        elif kind == "lgb":
            model = fit_lgb(xtr, ytr_log, xstop, ystop_log, rounds, seed, wtr=wtr)
            iterations[kind] = int(model.best_iteration)
        elif kind == "cat":
            model = fit_cat(xtr, ytr_log, xstop, ystop_log, rounds, seed, wtr=wtr)
            iterations[kind] = int(model.get_best_iteration())
        else:
            raise ValueError(f"unknown model kind: {kind}")
        models[kind] = model
    return models, iterations


def predict_kinds(models, matrices):
    """Return {kind: {name: raw_log_prediction}} without applying a scale."""
    return {
        kind: {name: raw_predict(model, matrix, kind) for name, matrix in matrices.items()}
        for kind, model in models.items()
    }
```

Move the existing LightGBM/two-part branches unchanged into `fit_kinds`. CatBoost remains supported but is not the default rolling ablation model. Keep `run_val` behavior working through these helpers.

- [ ] **Step 4: Implement the honest fold runner**

For each fold:

1. validate all required anchor sidecars;
2. train through `fold.train_through`;
3. early-stop on `fold.calibration` only;
4. predict calibration and outer matrices with the same model;
5. fit scale on calibration truth and apply it unchanged to outer predictions;
6. store raw log predictions, honest RMSLE, oracle shape RMSLE, iterations, feature hash, runtime, and peak RSS.

Add CLI:

```bash
python src/rolling_gbdt.py --work work10 --feature-view v10 --models lgb,two --folds 2025-09-19,2025-11-14,2026-01-14 --rounds 6000 --threads 12
```

Run the same command once with `--feature-view v9`. Both views read the same v10 Parquet artifacts, so the controlled comparison changes only the manifest-declared feature columns.

- [ ] **Step 5: Test the runner with a fake estimator**

Monkeypatch `fit_kinds` and `predict_kinds` so the test uses 20 synthetic users and proves that scale is fitted from calibration truth, the outer target never reaches the fit function, and the report stores distinct calibration/outer checksums.

Run: `PYTHONPATH=src pytest tests/test_rolling_gbdt.py tests/test_metrics.py -v`

Expected: all tests pass.

- [ ] **Step 6: Run a 2,000-user local smoke fold**

Run with `work10_smoke` and a maximum of 50 LightGBM rounds. Expected: one report with finite predictions and bounded memory; do not interpret its RMSLE as model evidence.

- [ ] **Step 7: Commit the controlled runner**

```bash
git add src/train.py src/rolling_gbdt.py tests/test_rolling_gbdt.py
git commit -m "feat: add rolling LightGBM evaluation"
```

---

### Task 6: Memory-Mapped Daily Data and Long-History Tokens

**Files:**
- Create: `src/sequence_data.py`
- Create: `tests/test_sequence_data.py`
- Modify: `src/build_daily.py:21-65`

**Interfaces:**
- Produces: `TabStats(mean: np.ndarray, std: np.ndarray)`
- Produces: `fit_tab_stats(x: np.ndarray) -> TabStats`
- Produces: `transform_tab(x: np.ndarray, stats: TabStats) -> np.ndarray`
- Produces: `pool_tokens(history: torch.Tensor, hist_days: torch.Tensor, period: int, slots: int) -> tuple[torch.Tensor, torch.Tensor]`
- Produces: `gather_sequences(daily: torch.Tensor, offsets: torch.Tensor, user_indices: torch.Tensor, row_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`

- [ ] **Step 1: Write failing normalization and token-mask tests**

```python
# tests/test_sequence_data.py
import numpy as np
import pytest

from sequence_data import fit_tab_stats, transform_tab


def test_tab_transform_is_finite_with_nan_and_inf():
    x = np.array([[0.0, np.nan], [3.0, np.inf], [-2.0, 1.0]], dtype=np.float32)
    stats = fit_tab_stats(x)
    z = transform_tab(x, stats)
    assert np.isfinite(z).all()
    assert np.max(np.abs(z)) <= 10.0


def test_weekly_tokens_keep_boundaries_and_coverage_mask():
    torch = pytest.importorskip("torch")
    from sequence_data import pool_tokens
    history = torch.arange(1, 15, dtype=torch.float32).view(1, 14, 1)
    token, mask = pool_tokens(history, torch.tensor([10]), period=7, slots=2)
    assert token.shape[1] == 2
    assert mask.tolist() == [[False, False]]
    assert torch.isfinite(token).all()


def test_uncovered_old_slots_are_masked():
    torch = pytest.importorskip("torch")
    from sequence_data import pool_tokens
    history = torch.zeros((1, 14, 1))
    _, mask = pool_tokens(history, torch.tensor([5]), period=7, slots=2)
    assert mask.tolist() == [[True, False]]
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `PYTHONPATH=src pytest tests/test_sequence_data.py -v`

Expected: import failure when torch is installed, or a clean skip when it is not.

- [ ] **Step 3: Implement stable tabular normalization**

Apply signed `log1p(abs(x))`, `nanmean`/`nanstd` on the training fold only, standardize, clip to `[-10, 10]`, and replace all remaining non-finite values with zero. Persist mean/std in the run manifest or a referenced `.npz` checksum.

- [ ] **Step 4: Implement GPU token pooling and masks**

`pool_tokens` left-pads the history to `period * slots`, reshapes newest-to-oldest blocks without mixing boundaries, concatenates mean and max channel pooling, and returns a boolean Transformer padding mask where `True` means unavailable history. Append coverage fraction plus sine/cosine relative-position channels to every token.

Keep torch imports lazy so `fit_tab_stats` and `transform_tab` remain testable on the local Mac without installing PyTorch.

Use `period=7, slots=59` and `period=30, slots=14`. `gather_sequences` returns:

```text
daily_180:   (batch, channels, 180)
weekly:      (batch, 59, 2*channels+3)
weekly_mask: (batch, 59)
monthly:     (batch, 14, 2*channels+3)
monthly_mask:(batch, 14)
```

- [ ] **Step 5: Convert daily-array creation to a true memory map**

In `build_daily.py`, resolve `data/train.parquet`, add `--raw` and `--max-users`, and allocate with:

```python
PAD = 420  # covers all 409 history days even at the earliest supported anchor
out = np.lib.format.open_memmap(
    WORK / "daily.npy", mode="w+", dtype=np.float16,
    shape=(n_users, N_DAYS + PAD, len(CHANNELS)),
)
out[:] = 0
```

Flush after every channel and write a JSON sidecar containing shape, dtype, channels, user checksum, and raw-data checksum.

- [ ] **Step 6: Pass unit tests and a 2,000-user daily build smoke test**

Run: `PYTHONPATH=src pytest tests/test_sequence_data.py -v`

Run: `POLARS_MAX_THREADS=4 CASE3_WORK=work10_smoke python src/build_daily.py --max-users 2000`

Expected: finite token batches, correct masks, and a bounded `.npy` file that opens with `mmap_mode="r"`.

- [ ] **Step 7: Commit sequence data handling**

```bash
git add src/sequence_data.py src/build_daily.py tests/test_sequence_data.py
git commit -m "feat: add memory-mapped temporal tokens"
```

---

### Task 7: Mixed Base and Mixed Large Network

**Files:**
- Create: `src/mixed_model.py`
- Create: `tests/test_mixed_model.py`

**Interfaces:**
- Consumes Task 6 tensors and masks
- Produces: `MixedConfig.for_profile(name: Literal["mixed_base", "mixed_large"], n_daily_channels: int, n_token_features: int, n_tab_features: int) -> MixedConfig`
- Produces: `MixedNet(config: MixedConfig)`
- Produces: `multitask_loss(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], use_aux: bool = True) -> tuple[torch.Tensor, dict[str, float]]`

- [ ] **Step 1: Write failing architecture and backward tests**

```python
# tests/test_mixed_model.py
import pytest

torch = pytest.importorskip("torch")

from mixed_model import MixedConfig, MixedNet, multitask_loss


@pytest.mark.parametrize("profile,width,blocks", [("mixed_base", 160, 3), ("mixed_large", 192, 4)])
def test_profiles_have_expected_capacity_and_output_shapes(profile, width, blocks):
    cfg = MixedConfig.for_profile(profile, 12, 27, 40)
    net = MixedNet(cfg)
    batch = 3
    outputs = net(
        torch.randn(batch, 12, 180),
        torch.randn(batch, 59, 27), torch.zeros(batch, 59, dtype=torch.bool),
        torch.randn(batch, 14, 27), torch.zeros(batch, 14, dtype=torch.bool),
        torch.randn(batch, 40),
    )
    assert cfg.width == width and cfg.transformer_blocks == blocks
    assert outputs["main"].shape == (batch,)
    assert outputs["weekly_gmv"].shape == (batch, 4)
    assert outputs["weekly_any_logits"].shape == (batch, 4)
    assert outputs["weekly_items"].shape == (batch, 4)


def test_loss_is_finite_and_backpropagates():
    cfg = MixedConfig.for_profile("mixed_base", 12, 27, 40)
    net = MixedNet(cfg)
    batch = 2
    outputs = net(
        torch.randn(batch, 12, 180),
        torch.randn(batch, 59, 27), torch.zeros(batch, 59, dtype=torch.bool),
        torch.randn(batch, 14, 27), torch.zeros(batch, 14, dtype=torch.bool),
        torch.randn(batch, 40),
    )
    targets = {
        "main": torch.tensor([0.0, 2.0]),
        "weekly_gmv": torch.tensor([[0.0, 1.0, 0.0, 2.0], [1.0, 0.0, 2.0, 1.0]]),
        "weekly_any": torch.tensor([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]]),
        "weekly_items": torch.tensor([[0.0, 1.0, 0.0, 3.0], [1.0, 0.0, 2.0, 1.0]]),
    }
    loss, parts = multitask_loss(outputs, targets, use_aux=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert set(parts) == {"main", "weekly_gmv", "weekly_any", "weekly_items", "total"}


def test_rocm_unsafe_layers_are_absent():
    cfg = MixedConfig.for_profile("mixed_base", 12, 27, 40)
    net = MixedNet(cfg)
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in net.modules())
    assert not any(isinstance(m, torch.nn.Conv1d) and m.dilation != (1,) for m in net.modules())
```

- [ ] **Step 2: Run the architecture tests and verify import failure**

Run: `PYTHONPATH=src pytest tests/test_mixed_model.py -v`

Expected: import failure for `mixed_model`.

- [ ] **Step 3: Implement configuration and the three encoders**

Use ordinary `Conv1d`, `GroupNorm`, GELU, and `AvgPool1d` for the 180-day branch. Use distinct weekly/monthly input projections, learned positional embeddings, and `TransformerEncoderLayer(batch_first=True, norm_first=True)` with five attention heads for width 160 and six heads for width 192. Masked mean-pool Transformer outputs. Use `LayerNorm -> Linear -> GELU` for the tabular branch.

- [ ] **Step 4: Implement heads and exact loss weights**

Return keys `main`, `weekly_gmv`, `weekly_any_logits`, and `weekly_items`. Compute:

```python
total = main_mse
if use_aux:
    total = total + 0.15 * weekly_gmv_mse + 0.05 * weekly_any_bce + 0.05 * weekly_items_mse
```

Targets for `weekly_gmv` and `weekly_items` use `log1p`; incidence uses raw 0/1. Return detached scalar components for logging.

- [ ] **Step 5: Pass architecture, loss, and unsafe-operation tests**

Run: `PYTHONPATH=src pytest tests/test_mixed_model.py -v`

Expected: all tests pass on a torch-enabled machine; clean skip locally if torch is absent.

- [ ] **Step 6: Commit the model**

```bash
git add src/mixed_model.py tests/test_mixed_model.py
git commit -m "feat: add ROCm mixed temporal model"
```

---

### Task 8: ROCm Preflight and Honest Mixed-Model Trainer

**Files:**
- Create: `src/rocm_preflight.py`
- Create: `src/train_mixed.py`
- Create: `tests/test_train_mixed.py`

**Interfaces:**
- Consumes: Task 3 anchors, Task 6 sequence batches, Task 7 model/loss, Task 4 fold definitions
- Produces: `run_with_oom_retry(train_once: Callable[[int], T], batch_size: int, minimum: int = 256) -> tuple[T, int]`
- Produces: `TrainingContract(early_stop_anchor: date, eval_anchor: date, train_through: date)` and `training_contract(fold: FoldSpec) -> TrainingContract`
- Produces: `train_fold(args, fold: FoldSpec) -> dict[str, Any]`
- Produces: `train_final(args) -> dict[str, Any]`
- Produces: `rocm_preflight(output: Path, batch: int = 256) -> dict[str, Any]`

- [ ] **Step 1: Write failing OOM retry and early-stopping contract tests**

```python
# tests/test_train_mixed.py
from datetime import date

import pytest

from metrics import FoldSpec
from train_mixed import run_with_oom_retry, training_contract


def test_oom_retry_halves_batch_once():
    seen = []
    def fake(batch):
        seen.append(batch)
        if len(seen) == 1:
            raise RuntimeError("CUDA out of memory")
        return "ok"
    result, used = run_with_oom_retry(fake, 1024)
    assert result == "ok" and used == 512 and seen == [1024, 512]


def test_non_oom_error_is_not_swallowed():
    with pytest.raises(RuntimeError, match="bad schema"):
        run_with_oom_retry(lambda _: (_ for _ in ()).throw(RuntimeError("bad schema")), 1024)


def test_outer_target_is_evaluation_only():
    contract = training_contract(FoldSpec.from_outer(date(2026, 1, 14)))
    assert contract.early_stop_anchor == date(2025, 12, 15)
    assert contract.eval_anchor == date(2026, 1, 14)
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `PYTHONPATH=src pytest tests/test_train_mixed.py -v`

Expected: import failure for `train_mixed`.

- [ ] **Step 3: Implement ROCm preflight**

Record Python, torch, HIP/ROCm, device name, total/free VRAM, fp16/bfloat16 support, and a timed forward/backward step for both profiles. Assert the device name contains `Radeon` and no output/loss is non-finite. Write `logs/rocm_preflight.json` atomically. Return nonzero on failure.

Keep the torch import inside GPU/model functions in `train_mixed.py`, so the pure `run_with_oom_retry` and `training_contract` tests run on the local Mac without torch.

- [ ] **Step 4: Implement fold data loading without duplicate full matrices**

Read daily history with `np.load(daily_path, mmap_mode="r")`. Load one anchor at a time into preallocated float16 tabular/target arrays, validating sidecars first. Move the shared daily tensor and compact tabular arrays to the GPU once; gather daily/token batches by user and anchor indices. Record peak RSS and `torch.cuda.max_memory_allocated()`.

- [ ] **Step 5: Implement honest training and checkpoint selection**

Train only through `fold.train_through`. Evaluate after every epoch on the calibration anchor and retain the best calibration checkpoint. After selection, predict the untouched outer anchor exactly once. Save raw log predictions keyed by profile, auxiliary flag, feature view, seed, and anchor. The outer target may be loaded only after the checkpoint is frozen.

Use AdamW, OneCycleLR, gradient clipping at 5.0, mixed precision, default batch 4096, and deterministic seeds `42 + 17 * seed_index`. Add `--profile legacy_cnn|mixed_base|mixed_large`, `--feature-view v9|v10`, `--use-aux`, `--folds`, `--epochs`, `--batch`, `--seeds`, `--max-users`, and `--mode fold|final`.

For `legacy_cnn`, wrap the existing `nn_train.Net` so it consumes only `daily_180` and the selected tabular feature view and returns a `main` output with auxiliary losses disabled. This produces an honest three-fold v9-versus-v10 comparison for the current CNN without duplicating its architecture.

For final mode, use the median best epoch across the three honest folds for the selected profile/auxiliary setting, rounded to the nearest integer and clamped to at least one epoch. Train on every allowed labeled anchor and write only the test raw-log prediction; final mode never performs test-time checkpoint selection.

- [ ] **Step 6: Implement one-retry OOM behavior**

On the first ROCm OOM only, clear the CUDA cache, halve batch size, and restart the profile from its initial seed. If the halved batch is below 256 or the second run OOMs, stop with a nonzero exit and preserve the log. Never catch unrelated runtime errors.

- [ ] **Step 7: Add a CPU one-batch smoke mode and pass tests**

`--max-users 256 --epochs 1 --batch 64 --device cpu` must execute one train/eval cycle against smoke anchors without loading the full dataset.

Run: `PYTHONPATH=src pytest tests/test_train_mixed.py tests/test_mixed_model.py tests/test_sequence_data.py -v`

Expected: non-torch tests pass locally; torch tests pass on the ROCm machine.

- [ ] **Step 8: Run the ROCm preflight before any full training**

Run on the external machine:

```powershell
.\.venv-rocm\Scripts\python.exe src\rocm_preflight.py --profiles mixed_base,mixed_large --out logs\rocm_preflight.json
```

Expected: both profiles complete a finite forward/backward step and peak VRAM remains below 15.5 GB.

- [ ] **Step 9: Commit training support**

```bash
git add src/rocm_preflight.py src/train_mixed.py tests/test_train_mixed.py
git commit -m "feat: train mixed model honestly on ROCm"
```

---

### Task 9: Validation Report, Blend Selection, and Final Submissions

**Files:**
- Create: `src/assemble_v10.py`
- Create: `tests/test_assemble_v10.py`
- Modify: `src/check_submission.py:18-67`
- Modify: `.gitignore`

**Interfaces:**
- Consumes rolling JSON manifests and raw log predictions from Tasks 5 and 8
- Produces: `learn_weights(folds, model_names, max_new_weight: float, step: float = 0.05) -> dict[str, float]`
- Produces: `relative_test_scale(candidate_scales: Sequence[float], baseline_scales: Sequence[float], baseline_test_scale: float) -> float`
- Produces: `assemble_candidate(name: str, user_ids: np.ndarray, components: Mapping[str, np.ndarray], weights: Mapping[str, float], scale: float, decision: Mapping[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]`

- [ ] **Step 1: Write failing scale, weight, and submission-integrity tests**

```python
# tests/test_assemble_v10.py
import numpy as np

from assemble_v10 import learn_weights, relative_test_scale


def synthetic_folds():
    y1 = np.array([0.0, 2.0, 10.0, 30.0])
    y2 = np.array([0.0, 4.0, 20.0, 60.0])
    return [
        {"y": y1, "v9": np.array([1.0, 1.0, 6.0, 18.0]), "v10": y1, "mixed": y1},
        {"y": y2, "v9": np.array([1.0, 2.0, 12.0, 36.0]), "v10": y2, "mixed": y2},
    ]


def test_relative_scale_uses_median_fold_ratio():
    got = relative_test_scale([0.8, 1.0, 1.2], [1.0, 1.0, 1.0], 0.9664)
    assert abs(got - 0.9664) < 1e-9


def test_weight_search_respects_new_model_cap():
    weights = learn_weights(synthetic_folds(), ["v9", "v10", "mixed"], max_new_weight=0.30)
    assert weights["v10"] + weights["mixed"] <= 0.3000001
    assert abs(sum(weights.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run the assembly tests and verify import failure**

Run: `PYTHONPATH=src pytest tests/test_assemble_v10.py -v`

Expected: import failure for `assemble_v10`.

- [ ] **Step 3: Implement cross-window reporting and acceptance**

Load v9, controlled-v10, mixed-base, and mixed-large predictions for all three folds. Compute honest RMSLE, shape RMSLE, paired effects, the shared-user bootstrap, and the Task 4 acceptance decision. Write `out_work10/rolling/report.json` and a Markdown table suitable for appending to the experiment ledger.

- [ ] **Step 4: Learn old-fold weights and evaluate January**

Optimize log-space blends jointly with a scale on September and November only. Evaluate the selected weights unchanged on January. Conservative weight search caps total new-model weight at 0.30. Mixed search permits total new-model weight up to 0.70. Both grids use 0.05 steps and include v9 unchanged.

- [ ] **Step 5: Assemble two immutable candidate files**

Apply the median candidate/v9 honest-scale ratio across the three folds relative to the fixed v9-opt test scale. Blend `log1p` of scaled level predictions, convert with `expm1`, clip at zero, and write:

```text
submission_v10_conservative.csv
submission_v10_conservative.json
submission_v10_mixed.csv
submission_v10_mixed.json
```

Each manifest stores source paths/checksums, weights, relative scales, validation decision, row count, prediction summary, and git commit. If a candidate misses acceptance criteria, still build it for inspection but set `recommended_for_submission` to false and print a prominent warning.

- [ ] **Step 6: Make submission validation reusable with current repository paths**

Refactor `check_submission.py` to accept `--sample` defaulting first to `samples/sample_submit.csv` and then `sample_submit.csv`, and `--anchor` defaulting to the selected work directory. Expose `validate_submission(path, sample_path, anchor_path) -> tuple[list[str], list[str], dict[str, float]]` for tests.

- [ ] **Step 7: Add both named v10 outputs to `.gitignore`**

Ignore generated `submission_v10_*.csv` and their JSON manifests. Do not add exceptions that would accidentally commit multi-megabyte candidate files.

- [ ] **Step 8: Pass assembly and baseline validation tests**

Run: `PYTHONPATH=src pytest tests/test_assemble_v10.py tests/test_metrics.py -v`

Run: `python src/check_submission.py submission_v9_opt.csv --sample samples/sample_submit.csv`

Expected: tests pass and baseline verdict is `PASS`.

- [ ] **Step 9: Commit assembly and validation**

```bash
git add .gitignore src/assemble_v10.py src/check_submission.py tests/test_assemble_v10.py
git commit -m "feat: assemble validated v10 candidates"
```

---

### Task 10: Reproducible ROCm Pipeline and Experiment Ledger

**Files:**
- Create: `run_v10.ps1`
- Create: `docs/experiments.md`
- Modify: `README.md`
- Create: `tests/test_pipeline_contract.py`

**Interfaces:**
- Consumes every prior CLI and manifest
- Produces one resumable external-machine command and documented local smoke commands

- [ ] **Step 1: Write a failing static pipeline contract test**

```python
# tests/test_pipeline_contract.py
from pathlib import Path


def test_pipeline_orders_preflight_before_full_gpu_training():
    text = Path("run_v10.ps1").read_text(encoding="utf-8-sig")
    assert text.index("rocm_preflight.py") < text.index("train_mixed.py")
    assert "--feature-set" in text and "v10" in text
    assert "submission_v10_conservative.csv" in text
    assert "submission_v10_mixed.csv" in text


def test_pipeline_stops_after_failed_required_stage():
    text = Path("run_v10.ps1").read_text(encoding="utf-8-sig")
    assert "throw" in text
    assert "FAILED" in text
```

- [ ] **Step 2: Run the contract test and verify failure**

Run: `pytest tests/test_pipeline_contract.py -v`

Expected: failure because `run_v10.ps1` does not exist.

- [ ] **Step 3: Implement fail-fast, resumable PowerShell orchestration**

Stages execute sequentially:

1. CPU/ROCm environment checks;
2. v10 anchor generation;
3. daily memmap generation;
4. rolling LightGBM direct/two-part;
5. ROCm preflight;
6. legacy CNN rolling runs for both v9 and v10 feature views;
7. mixed-base rolling run with and without auxiliary losses;
8. mixed-large rolling run only after base manifests validate;
9. validation report and acceptance decisions;
10. accepted final model refits;
11. conservative/mixed assembly and validation.

Unlike `run_all.ps1`, a required failed stage throws and stops downstream work. A stage skips only when its output manifest validates. Log every command and exit code to a timestamped directory under `logs/v10_<timestamp>/`.

- [ ] **Step 4: Create the experiment ledger header and baseline entry**

`docs/experiments.md` starts with rules stating that entries are append-only and must include hypothesis, command, commit, schema hash, anchors, metrics, paired statistics, runtime/memory, artifact checksums, and decision. Record the immutable v9-opt checksum, exact `0.95109674` relationship to v9, and public RMSLE 1.649315 as experiment `E000`.

- [ ] **Step 5: Update README reproduction and status sections**

Add:

- current immutable baseline and checksum;
- `data/train.parquet` and `samples/sample_submit.csv` paths;
- local bounded smoke commands;
- ROCm 9070 XT environment command and `run_v10.ps1` invocation;
- explanation of the three honest folds and shape-only diagnostic;
- links to the design, implementation plan, and experiment ledger;
- a result table whose unrun entries say `not run` rather than containing invented metrics.

- [ ] **Step 6: Run documentation, pipeline, and full unit verification**

Run:

```bash
PYTHONPATH=src pytest tests -v
git diff --check
python src/check_submission.py submission_v9_opt.csv --sample samples/sample_submit.csv
```

Expected: all locally available tests pass, torch-only tests skip cleanly when torch is absent, diff check is clean, and v9-opt submission passes.

- [ ] **Step 7: Commit pipeline and documentation**

```bash
git add run_v10.ps1 docs/experiments.md README.md tests/test_pipeline_contract.py
git commit -m "docs: add reproducible v10 ROCm workflow"
```

---

### Task 11: External ROCm Execution and Evidence-Based Finalization

**Files:**
- Modify: `docs/experiments.md`
- Modify: `README.md`
- Generated and ignored: `work10/`, `out_work10/`, `logs/`, `submission_v10_*.csv`, `submission_v10_*.json`

**Interfaces:**
- Consumes: `run_v10.ps1` and the external AMD environment
- Produces: validated fold predictions, two candidate submissions, manifests, and an evidence-backed recommendation

- [ ] **Step 1: Install the CPU dependencies and verify the existing ROCm environment**

Run on the external machine:

```powershell
python -m pip install -r requirements.txt
.\.venv-rocm\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
```

Expected: dependency import succeeds and the GPU line reports AMD Radeon RX 9070 XT with a nonempty HIP version.

- [ ] **Step 2: Run the complete pipeline unattended**

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v10.ps1 -Work work10
```

Expected: all required stages finish, or the pipeline stops at the first invalid manifest/error without launching downstream training.

- [ ] **Step 3: Review base/large and auxiliary ablations before final refits**

Read `out_work10/rolling/report.json`. Confirm every claimed accepted component meets all Task 4 rules. If `mixed_large` or auxiliary losses fail, keep their negative results in the ledger and exclude them from the conservative refit; do not retune on January alone.

- [ ] **Step 4: Validate final files and manifests**

```powershell
python src\check_submission.py submission_v10_conservative.csv --sample samples\sample_submit.csv
python src\check_submission.py submission_v10_mixed.csv --sample samples\sample_submit.csv
```

Expected: 250,000 unique required IDs, finite nonnegative predictions, and `PASS` for both files.

- [ ] **Step 5: Record actual results without fabricating leaderboard outcomes**

Append the commands, commits, schema hashes, all three fold results, paired/bootstrap statistics, duration, peak RAM/VRAM, artifact checksums, and accept/reject decision to `docs/experiments.md`. Update the README table with these validation results. Leave public RMSLE blank until the competition site returns it.

- [ ] **Step 6: Run final verification before recommending a submission**

Run:

```powershell
python -m pytest tests -v
git diff --check
git status --short
```

Expected: tests pass, documentation matches manifests, and only intentionally ignored generated artifacts remain untracked.

- [ ] **Step 7: Commit evidence and recommendations**

```bash
git add docs/experiments.md README.md
git commit -m "docs: record v10 validation evidence"
```
