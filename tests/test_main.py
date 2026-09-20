"""Tests for tinylab.__main__: CLI-level argument handling not already covered by tinylab.job's
own tests (e.g. --resume + --only, which is refused before job.run_file is ever called)."""
from tinylab.__main__ import main


def test_resume_and_only_together_is_a_usage_error(capsys):
    exit_code = main(["jobs/smoke.json", "--resume", "--only", "pre"])
    assert exit_code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_unrecognized_flag_is_a_usage_error(capsys):
    exit_code = main(["jobs/smoke.json", "--bogus"])
    assert exit_code == 2
