"""Load and validate eval.toml: any invalid value stops the tool before any API call."""

import tomllib
from itertools import combinations
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Margin = Annotated[float, Field(ge=0, le=100)]   # une marge en points, entre 0 et 100

class Strict(BaseModel):
    """Base for every section: unknown keys are errors, values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(Strict):
    """[project]: which kind of application, and the function to evaluate."""

    profile: Literal["agent", "graph", "rag", "chat"]
    target: str = Field(pattern=r"^[\w.]+:\w+$")  # "module.path:function"
    evaluators: str = Field(pattern=r"^[\w.]+:\w+$")   


class DatasetConfig(Strict):
    """[dataset]: which dataset version and split to evaluate on."""

    name: str = Field(..., min_length=1)
    tag: str = Field(..., min_length=1)
    split: str = Field(default="holdout")  # default to test split


class JudgeConfig(Strict):
    """[judge]: the frozen LLM judge: dated model, temperature 0."""

    model: str = Field(..., min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)


class RunConfig(Strict):
    """[run]: how to run the evaluation."""

    repetitions: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    budget_usd: float = Field(..., gt=0)

class RulesConfig(Strict):
    """[rules]: how to judge the evaluation."""

    safety: dict[str, str] = Field(min_length=1)
    primary: dict[str, Margin] = Field({})
    secondary: list[str] = Field([])

    @model_validator(mode="after")
    def one_category_per_evaluator(self): 
        categories= {"safety": set(self.safety), "primary": set(self.primary), "secondary": set(self.secondary)}
        conflicts = [ 
            f"{cat1}/{cat2}: {sorted(categories[cat1] & categories[cat2])}" for (cat1, cat2) in combinations(categories.keys(), 2) if categories[cat1] & categories[cat2]
        ]
        if conflicts:
                raise ValueError("\n".join(conflicts))
        return self

ToolAccess = Literal["read", "simulated", "forbidden"]

class EvalConfig(Strict):
    """The whole eval.toml: one field per [section], each validated by its own model."""

    project: ProjectConfig
    dataset: DatasetConfig
    judge: JudgeConfig
    run: RunConfig
    tools: dict[str, ToolAccess]= Field(default_factory=dict)
    rules: RulesConfig

    @model_validator(mode="after")
    def validate_tools(self):
        if self.project.profile in  ["agent", "graph"] and not self.tools:
            raise ValueError(f"profile '{self.project.profile}' requires a [tools] whitelist: every tool must be declared read, simulated or forbidden")
        return self

class ConfigError(Exception):
    """Raised when eval.toml is missing, malformed or invalid: the only error to catch."""


def _format_errors(error: ValidationError) -> str:
    """Turn pydantic errors into one readable line each: 'run.budget_usd: Field required'."""
    lines = []
    for e in error.errors():
        field = ".".join(str(part) for part in e["loc"]) or "(whole file)"
        lines.append(f"  - {field}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: str | Path) -> EvalConfig:
    """Read and validate an eval.toml file.

    Args:
        path: Path to the eval.toml file.

    Returns:
        The validated, read-only configuration.

    Raises:
        ConfigError: if the file is missing, is not valid TOML, or holds invalid
            values. The message names the file and every faulty field.
    """
    path = Path(path)
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from e

    try:
        return EvalConfig(**data)
    except ValidationError as e:
        raise ConfigError(f"invalid config {path}:\n{_format_errors(e)}") from e
