"""Mechanically checks that docs/job-file.md's key tables match the code's own key sets -- in the
spirit of test_no_nanochat.py's import guard. A new job-file key with no doc row, or a doc row for
a key that no longer exists, fails here instead of silently drifting (the exact way this repo's
docs went stale before)."""
import re
from pathlib import Path

import pytest

from tinylab.chat import ACCEPTED_KEYS as CHAT_ACCEPTED_KEYS
from tinylab.ops import COMMON_KEYS
from tinylab.ops import bench, prepare, train
from tinylab.presets import MODEL_ACCEPTED_KEYS

DOC_PATH = Path(__file__).parent.parent / "docs" / "job-file.md"

# Each op's accepted_keys(cfg) is kind/suite-aware: calling it with a cfg that names neither kind
# (an empty dict) hits the "else" branch of accepted_keys' own kind_keys selection, which is
# defined as the *union* of every kind/suite's keys -- so this already gives the full key set a
# single flat doc table should list, with no need to reach into each module's private
# _BASE_KEYS/_SFT_KEYS/_CORE_KEYS/_CHAT_KEYS constants directly.
SECTIONS = {
    "## `model` block": lambda: MODEL_ACCEPTED_KEYS,
    "## Common to every op (`prepare`/`train`/`bench`)": lambda: COMMON_KEYS,
    "## `prepare`": lambda: prepare.accepted_keys({}),
    "## `train`": lambda: train.accepted_keys({}),
    "## `bench`": lambda: bench.accepted_keys({}),
    "## `chat`": lambda: CHAT_ACCEPTED_KEYS,
}


def _doc_sections():
    text = DOC_PATH.read_text(encoding="utf-8")
    # Split on "## " headings (level-2), keeping the heading text with each chunk.
    parts = re.split(r"(?m)^(## .+)$", text)
    # parts[0] is preamble before the first heading; then alternating heading, body.
    return dict(zip(parts[1::2], parts[2::2]))


def _table_keys(section_body: str) -> set:
    """Every table row's first cell, where that cell is a single backtick-quoted key (skips the
    header/separator rows, and any row whose first cell isn't a plain `key`, like the glossary's
    prose rows)."""
    keys = set()
    for line in section_body.splitlines():
        m = re.match(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|", line)
        if m:
            keys.add(m.group(1))
    return keys


@pytest.mark.parametrize("heading", sorted(SECTIONS))
def test_job_file_doc_matches_code(heading):
    sections = _doc_sections()
    assert heading in sections, f"docs/job-file.md is missing the {heading!r} section"
    doc_keys = _table_keys(sections[heading])
    code_keys = SECTIONS[heading]()
    missing_from_doc = code_keys - doc_keys
    stale_in_doc = doc_keys - code_keys
    assert not missing_from_doc, f"{heading}: key(s) accepted by code but undocumented: {sorted(missing_from_doc)}"
    assert not stale_in_doc, f"{heading}: key(s) documented but no longer accepted by code: {sorted(stale_in_doc)}"


def test_all_sections_present_and_parsed_something():
    # A regression guard on the parser itself: every section above must have found at least one
    # key, or a heading typo / table-format change would make the real assertions above vacuously
    # pass (empty set == empty set).
    sections = _doc_sections()
    for heading in SECTIONS:
        assert _table_keys(sections[heading]), f"{heading}: parsed zero keys -- check the table format or heading text"
