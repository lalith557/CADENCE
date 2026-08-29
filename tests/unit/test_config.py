from cadence.common.config import Config, load_config


def test_load_default_config(default_config: Config):
    assert default_config.profile == "default"
    assert default_config.model.hidden_dims == [64, 32]
    assert default_config.budget.max_vram_gb == 4.0


def test_load_ci_config(ci_config: Config):
    assert ci_config.profile == "ci"
    assert ci_config.model.max_epochs == 2  # smaller for smoke


def test_config_hash_is_stable(default_config: Config, project_root):
    h1 = default_config.hash()
    h2 = load_config(project_root / "configs" / "default.yaml").hash()
    assert h1 == h2


def test_default_profile_rejects_oversize_vram():
    import pytest

    with pytest.raises(ValueError):
        Config(profile="default", budget={"max_vram_gb": 8.0})


def test_local_profile_allows_oversize_vram():
    cfg = Config(profile="local", budget={"max_vram_gb": 8.0})
    assert cfg.budget.max_vram_gb == 8.0
