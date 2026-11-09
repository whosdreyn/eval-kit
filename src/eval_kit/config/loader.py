"""Read eval.toml: any problem stops the tool before a single API call is paid."""

import tomllib
from pathlib import Path

from pydantic import ValidationError

from eval_kit.config.models import EvalConfig


class ConfigError(Exception):
    """Raised when eval.toml is missing, malformed or invalid: the only error to catch."""


def _format_errors(error: ValidationError) -> str:
    """Turn pydantic errors into one readable line per faulty field.

    Args:
        error: The validation error raised by EvalConfig.

    Returns:
        Lines such as "  - run.budget_usd: Field required".
    """
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
