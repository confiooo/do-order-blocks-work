"""The side-study stage table resolves on its own and every target imports."""

from __future__ import annotations

from experiments import registry
from hawk_research.stages import registry as main_registry


def test_every_experiment_target_imports():
    for stage in registry.STAGES:
        assert callable(stage.load()), stage.name


def test_tables_are_disjoint_and_main_all_has_no_side_study():
    assert not set(registry.BY_NAME) & set(main_registry.BY_NAME)
    main_all = {s.name for s in main_registry.resolve(["all"])}
    assert not main_all & set(registry.BY_NAME)


def test_experiment_all_skips_network_and_main_dependencies():
    order = [s.name for s in main_registry.resolve(["all"], stages=registry.STAGES)]
    assert "mn_ingest" not in order and "s5" not in order
    assert order.index("mn_rv_phase1") < order.index("mn_rerun_rolling")
    assert order.index("df_study") < order.index("df_zone_trade")
