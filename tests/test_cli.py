from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from jev_eval import cli, commands
from tests.fakes import QUESTIONS, FakeProvider, labels, meta


@pytest.fixture
def fake(monkeypatch: MonkeyPatch) -> FakeProvider:
    provider = FakeProvider()
    monkeypatch.setattr(commands, "provider_from_env", lambda model: provider)
    return provider


def write_data(tmp_path: Path, n: int) -> Path:
    data = tmp_path / "labeled.jsonl"
    lines = [json.dumps({"state": {"id": i, "text": f"t{i}"}, "labels": labels(i), "meta": meta(i)}) for i in range(n)]
    data.write_text("\n".join(lines) + "\n")
    return data


def write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "jev-eval.json"
    cfg.write_text(json.dumps({"model": "jev-pinned", "questions": QUESTIONS}))
    return cfg


def run_cache(tmp_path: Path, n: int = 250) -> Path:
    cache = tmp_path / "run.jsonl"
    argv = [
        "run",
        "--data",
        str(write_data(tmp_path, n)),
        "--config",
        str(write_config(tmp_path)),
        "--cache",
        str(cache),
    ]
    assert cli.main(argv) == 0
    return cache


def test_run_writes_the_cache_with_the_config_model(tmp_path: Path, fake: FakeProvider) -> None:
    cache = run_cache(tmp_path, 5)
    lines = cache.read_text().splitlines()
    assert len(lines) == 15
    assert json.loads(lines[0])["model"] == "jev-pinned"
    assert len(fake.calls) == 5


def test_run_with_inline_questions(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    data = write_data(tmp_path, 3)
    argv = ["run", "--data", str(data), "-q", "urgent:noul", "--cache", str(tmp_path / "c.jsonl"), "--model", "m"]
    assert cli.main(argv) == 0
    assert "3 rows, 3 answers (3 usable)" in capsys.readouterr().err


def test_report_prints_all_layers(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path)
    capsys.readouterr()
    assert cli.main(["report", "--cache", str(cache), "--bins", "5"]) == 0
    out = capsys.readouterr().out
    for needle in ("== team (choice) n=250", "ECE", "reliability", "risk-coverage", "within 1", "yes precision"):
        assert needle in out
    assert "Row-level gate" in out
    assert "mean p(yes)" in out


def test_thresholds_writes_the_contract(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path)
    out_file = tmp_path / "thresholds.json"
    argv = [
        "thresholds",
        "--cache",
        str(cache),
        "--target-accuracy",
        "0.8",
        "--bootstrap",
        "100",
        "--out",
        str(out_file),
    ]
    assert cli.main(argv) == 0
    doc = json.loads(out_file.read_text())
    assert doc["version"] == 1 and set(doc["questions"]) == {"team", "urgent", "sentiment"}
    assert "out-of-bag" in capsys.readouterr().out


@pytest.mark.parametrize("guard", [[], ["--holdout", "0.3"], ["--no-guard"]])
def test_thresholds_guards_and_stdout(
    tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str], guard: list[str]
) -> None:
    cache = run_cache(tmp_path)
    capsys.readouterr()
    argv = ["thresholds", "--cache", str(cache), "--max-escalation", "0.3", "--tolerance", "1", "--out", "-", *guard]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert '"version": 1' in out
    assert ("held-out" in out) == ("--holdout" in guard)


def test_thresholds_refuses_tiny_data(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path, 30)
    assert cli.main(["thresholds", "--cache", str(cache), "--target-accuracy", "0.9", "--out", "-"]) == 0
    assert "gather more labels" in capsys.readouterr().out


def test_unreachable_and_warn_messages(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path, 120)
    capsys.readouterr()
    assert cli.main(["thresholds", "--cache", str(cache), "--target-accuracy", "0.8", "--out", "-"]) == 0
    assert "treat as provisional" in capsys.readouterr().out
    assert cli.main(["thresholds", "--cache", str(cache), "--target-accuracy", "1.5", "--out", "-"]) == 0
    assert "no threshold meets the goal" in capsys.readouterr().out


def test_errors_exit_2(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    assert cli.main(["report", "--cache", str(tmp_path / "missing.jsonl")]) == 2
    data = write_data(tmp_path, 2)
    assert cli.main(["run", "--data", str(data), "--cache", str(tmp_path / "c")]) == 2
    assert "no questions" in capsys.readouterr().err


def test_report_slices_and_svg(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path)
    svg = tmp_path / "rel.svg"
    capsys.readouterr()
    assert cli.main(["report", "--cache", str(cache), "--slice", "meta.source", "--svg", str(svg)]) == 0
    out = capsys.readouterr().out
    assert "per-slice report by meta.source" in out and "email" in out and "chat" in out
    assert "recalibration (isotonic" in out
    assert svg.read_text().startswith("<svg")


def test_compare_two_runs(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path)
    capsys.readouterr()
    assert cli.main(["compare", str(cache), str(cache)]) == 0
    out = capsys.readouterr().out
    assert "agreement 100.0%: B fixes 0, breaks 0" in out


def test_calibrate_writes_the_optional_mapping(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path, 120)
    out_file = tmp_path / "calibration.json"
    assert cli.main(["calibrate", "--cache", str(cache), "--out", str(out_file)]) == 0
    doc = json.loads(out_file.read_text())
    assert doc["method"] == "isotonic" and set(doc["questions"]) == {"team", "urgent", "sentiment"}
    assert doc["questions"]["urgent"]["input"] == "noul"
    assert "ECE" in capsys.readouterr().out


def test_calibrate_skips_small_questions(tmp_path: Path, fake: FakeProvider, capsys: CaptureFixture[str]) -> None:
    cache = run_cache(tmp_path, 30)
    assert cli.main(["calibrate", "--cache", str(cache), "--out", "-"]) == 0
    captured = capsys.readouterr()
    assert "skipped (fewer than 50 labeled rows)" in captured.err
    assert '"questions": {}' in captured.out


def test_version(capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert "jev-eval 1.2.0" in capsys.readouterr().out
