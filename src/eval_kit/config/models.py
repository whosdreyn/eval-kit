"""Schema of eval.toml: one pydantic model per [section].

Every model forbids unknown keys and is read-only, so a typo in eval.toml is an
error and nothing can change the configuration once it has been loaded.
"""

from itertools import combinations
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# "module.path:name", e.g. "evaluation.ls_evals:target".
REFERENCE_PATTERN = r"^[\w.]+:\w+$"

# A tolerated drop for a primary metric, in points (0 to 100).
Margin = Annotated[float, Field(ge=0, le=100)]

# What an evaluation run may do with a tool (design doc, section 3.1).
ToolAccess = Literal["read", "simulated", "forbidden"]


class Strict(BaseModel):
    """Base for every section: unknown keys are errors, values are read-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(Strict):
    """[project]: the kind of application and where its code lives.

    Attributes:
        profile: Kind of application, which decides the default checks.
        target: "module:function" run on each dataset example.
        evaluators: "module:LIST" holding every evaluator function of the project.
    """

    profile: Literal["agent", "graph", "rag", "chat"]
    target: str = Field(pattern=REFERENCE_PATTERN)
    evaluators: str = Field(pattern=REFERENCE_PATTERN)


class DatasetConfig(Strict):
    """[dataset]: the frozen dataset version the evaluation runs on.

    Attributes:
        name: LangSmith dataset name.
        tag: Dataset version tag. Required: the baseline and the candidate must be
            evaluated on exactly the same examples (design doc, principle P7).
        split: Split used by the gate. "holdout" by default, so prompts are not
            tuned on the examples that decide the merge (principle P8).
    """

    name: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    split: str = "holdout"


class JudgeConfig(Strict):
    """[judge]: the frozen LLM judge (principle P5).

    Attributes:
        model: Dated model name, never a moving alias.
        temperature: Must be 0, so the same output always gets the same verdict.
    """

    model: str = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)


class RunConfig(Strict):
    """[run]: how the evaluation is run.

    Attributes:
        repetitions: Runs per example, to see the randomness of the agent.
        max_concurrency: Examples evaluated in parallel.
        budget_usd: Spending cap of one run. Required: never run without one.
    """

    repetitions: int = Field(default=3, ge=1)
    max_concurrency: int = Field(default=4, ge=1, le=16)
    budget_usd: float = Field(gt=0)


class RulesConfig(Strict):
    """[rules]: what blocks the merge (principle P2).

    Attributes:
        safety: {evaluator: what a failure means}. One failed run blocks.
        primary: {evaluator: margin in points}. Blocks when the candidate is
            proven worse than the baseline beyond the margin.
        secondary: Evaluators reported only; they never block.
    """

    safety: dict[str, str] = Field(min_length=1)
    primary: dict[str, Margin] = Field(default_factory=dict)
    secondary: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_category_per_evaluator(self) -> "RulesConfig":
        """Refuse an evaluator listed in two categories: the gate could not decide.

        Returns:
            The validated rules.

        Raises:
            ValueError: naming every pair of categories that share an evaluator.
        """
        categories = {
            "safety": set(self.safety),
            "primary": set(self.primary),
            "secondary": set(self.secondary),
        }
        conflicts = [
            f"{first}/{second}: {sorted(categories[first] & categories[second])}"
            for first, second in combinations(categories, 2)
            if categories[first] & categories[second]
        ]
        if conflicts:
            raise ValueError("; ".join(conflicts))
        return self

class BaselineConfig(Strict):
    """[baseline]: the experiment the candidate is compared to.

    Attributes:
        experiment: Name of the LangSmith experiment of the version in production.
    """

    experiment: str = Field(min_length=1)

class EvalConfig(Strict):
    """The whole eval.toml: one field per [section], each validated by its own model."""

    project: ProjectConfig
    dataset: DatasetConfig
    judge: JudgeConfig
    run: RunConfig
    tools: dict[str, ToolAccess] = Field(default_factory=dict)
    rules: RulesConfig
    baseline: BaselineConfig | None = None

    @model_validator(mode="after")
    def tools_whitelist_for_agents(self) -> "EvalConfig":
        """Refuse an agent or a graph without a tools whitelist (design doc, section 3.1).

        Those profiles call tools that can act (refund, send, delete): every tool
        must be declared read, simulated or forbidden before anything runs.

        Returns:
            The validated configuration.

        Raises:
            ValueError: if the profile is "agent" or "graph" and [tools] is empty.
        """
        if self.project.profile in ("agent", "graph") and not self.tools:
            raise ValueError(
                f"profile '{self.project.profile}' requires a [tools] whitelist: "
                "every tool must be declared read, simulated or forbidden"
            )
        return self
