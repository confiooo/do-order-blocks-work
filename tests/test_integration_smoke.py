"""End-to-end: the real stages on the synthetic smoke market match the committed fingerprint."""

from __future__ import annotations

import pytest

from hawk_research import smoke


@pytest.mark.integration
def test_smoke_pipeline_matches_fixture(tmp_path):
    assert smoke.main(workdir=tmp_path) == 0
