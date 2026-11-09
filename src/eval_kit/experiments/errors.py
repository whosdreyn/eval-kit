"""Errors of the experiments subpackage."""


class RunError(Exception):
    """The evaluation could not be done: the CLI turns it into exit code 2.

    Raised before the paid run (bad reference, missing evaluator, empty dataset
    version) or after it (scores still incomplete in LangSmith).
    """
