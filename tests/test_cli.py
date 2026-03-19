"""Tests for CLI interface."""

import json
import sys
import pytest
from agi_eval.cli import build_parser, main


class TestCLIParser:
    def test_run_command(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--benchmark", "hellaswag", "--model", "gpt-4o", "--mock"])
        assert args.command == "run"
        assert args.benchmark == "hellaswag"
        assert args.model == "gpt-4o"
        assert args.mock is True

    def test_list_command(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_history_command(self):
        parser = build_parser()
        args = parser.parse_args(["history"])
        assert args.command == "history"


class TestCLIRun:
    def test_run_mock_hellaswag(self, tmp_path, capsys):
        results_file = str(tmp_path / "results.json")
        with pytest.raises(SystemExit) as exc:
            main([
                "run",
                "--benchmark", "hellaswag",
                "--model", "mock-model",
                "--mock",
                "--limit", "3",
                "--results-file", results_file,
            ])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        # Should output hellaswag results
        assert "hellaswag" in captured.out or "hellaswag" in captured.err

    def test_run_all_mock(self, tmp_path, capsys):
        results_file = str(tmp_path / "results.json")
        with pytest.raises(SystemExit) as exc:
            main([
                "run",
                "--benchmark", "all",
                "--mock",
                "--limit", "2",
                "--results-file", results_file,
            ])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        # All 4 benchmarks should appear in stderr progress output
        for b in ("hellaswag", "gsm8k", "truthfulqa", "codeeval"):
            assert b in captured.err

    def test_run_json_output(self, tmp_path, capsys):
        results_file = str(tmp_path / "results.json")
        with pytest.raises(SystemExit) as exc:
            main([
                "run",
                "--benchmark", "gsm8k",
                "--mock",
                "--limit", "2",
                "--results-file", results_file,
                "--json",
            ])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "gsm8k" in data
        assert "accuracy" in data["gsm8k"]

    def test_run_no_api_key_fails(self, tmp_path, capsys):
        results_file = str(tmp_path / "results.json")
        with pytest.raises(SystemExit) as exc:
            main([
                "run",
                "--benchmark", "hellaswag",
                "--model", "gpt-4o",
                "--results-file", results_file,
            ])
        assert exc.value.code == 1

    def test_list_command(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["list"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "hellaswag" in out
        assert "gsm8k" in out
        assert "truthfulqa" in out
        assert "codeeval" in out
