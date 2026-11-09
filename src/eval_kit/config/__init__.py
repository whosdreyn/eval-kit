"""eval.toml: its schema (models) and how it is read (loader)."""

from eval_kit.config.loader import ConfigError, load_config
from eval_kit.config.models import (
    DatasetConfig,
    EvalConfig,
    JudgeConfig,
    ProjectConfig,
    RulesConfig,
    RunConfig,
)

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "EvalConfig",
    "JudgeConfig",
    "ProjectConfig",
    "RulesConfig",
    "RunConfig",
    "load_config",
]
