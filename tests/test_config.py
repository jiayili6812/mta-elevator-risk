from __future__ import annotations

from mta_elevator_pipeline.config import assert_frozen_test_config, load_config


def test_checked_in_config_matches_frozen_test_lock():
    assert_frozen_test_config(load_config())

