# eval-kit

Automated evaluation of LLM agents, graphs, chatbots and RAG with LangSmith.

> No change (prompt, model, code, tool, dataset) reaches production unless it is
> proven to break no safety rule and not to be significantly worse than the version
> in production.

eval-kit checks what an agent **did** (its tool calls), not only what it said, and
turns LangSmith experiments into a merge verdict that can block a pull request.

## What it provides

| Subpackage | Role |
|---|---|
| `eval_kit.config` | `eval.toml` schema and loader: any invalid value stops the tool before a single API call |
| `eval_kit.trajectory` | step-level checks of tool calls, without any LLM: allowed transitions (DAG), argument provenance (no invented parameter), safety gates |
| `eval_kit.stats` | paired bootstrap confidence intervals, resampling whole scenarios |
| `eval_kit.experiments` | runs the agent on a frozen dataset version and reads the scores (LangSmith `runs.query`) |
| `eval_kit.judges` | frozen LLM judges: dated model (alias refused), temperature 0, data kept apart from instructions in escaped `<data>` tags, strict JSON verdict (anything else is an error, never a pass), prompt fingerprint |
| `eval_kit.gate` | the merge verdict: safety rules (zero tolerance), primary metrics (non-inferiority with a margin), secondary metrics (report only) |

## Layout

```
src/eval_kit/
  config/        models.py (eval.toml schema), loader.py (load_config)
  trajectory/    step.py, adapters.py, transitions.py, arguments.py, safety_gates.py, engine.py
  stats/         bootstrap.py
  experiments/   runner.py (run on the frozen dataset), results.py (read the scores)
  judges/        prompt.py (protected prompt, fingerprint), judge.py (Judge, strict verdict)
  gate/          decision.py (safety, non-inferiority, report)
  cli.py         the evalkit command (run, gate)
tests/           one folder per subpackage, no network
docs/DESIGN.md   principles, threat model, statistics and build plan
```

## Install

```bash
uv add --editable ../eval-kit      # from the project to evaluate, next to this repository
```

## eval.toml

```toml
[project]
profile    = "agent"                          # agent | graph | rag | chat
target     = "evaluation.ls_evals:target"     # function run on each example
evaluators = "evaluation.ls_evals:EVALUATORS" # list of evaluator functions

[dataset]
name = "refund-scenarios"
tag  = "v3"                                   # frozen version, required
# split = "holdout"                           # default

[judge]
model = "gpt-4.1-mini-2025-04-14"             # dated model; temperature is forced to 0

[run]
budget_usd = 2.0                              # required
# repetitions = 3
# max_concurrency = 4

[rules]
secondary = ["message_conforme"]              # reported, never blocking

[tools]                                       # whitelist, required for agent and graph
lookup_customer_purchases = "read"
create_refund_ticket      = "simulated"

[rules.safety]                                # one failed run blocks
trajectoire_dag          = "invented argument or crossed safety gate"
aucun_ticket_non_demande = "ticket created without a refund request"

[rules.primary]                               # blocks when proven worse beyond the margin (points)
decision_correcte = 0
outils_corrects   = 5
```

## Command line

```bash
uv run evalkit run  --config eval.toml   # run an experiment (e.g. to create a first baseline)
uv run evalkit gate --config eval.toml   # run the candidate, compare it to [baseline], decide
```

Each run traces the agent only (one trace per example and repetition): evaluators
record their scores without a trace of their own, to stay within LangSmith's monthly
trace quota. If the quota is exceeded anyway, LangSmith drops runs, the scores stay
incomplete and the gate stops with code 2 instead of deciding on partial data.

`evalkit gate` exits with **0** when the gate passes, **1** when it blocks (the agent
regressed) and **2** when the evaluation could not be done (invalid config, missing
baseline, empty dataset version). Both 1 and 2 fail a CI check; the code tells
whether to fix the agent or the setup.

The `[baseline]` section names the experiment of the version in production:

```toml
[baseline]
experiment = "prompt-v6-7521d4ba"
```

## Trajectory engine

```python
from eval_kit import evaluate_trajectory, from_langchain_messages

steps = from_langchain_messages(result["messages"])
report = evaluate_trajectory(steps, TRANSITIONS, ARGUMENTS, GATES)
# {"score": 1.0, "failed": False, "errors": []}
```

An invented argument or a crossed safety gate gives a score of 0; an unexpected
transition only lowers the score.

## LLM judges

```python
from eval_kit.judges import Judge

polite = Judge("gpt-4.1-mini-2025-04-14", "The message must be polite and clear, in French.")

def message_conforme(outputs):
    verdict = polite(message=outputs["message"])
    return {"key": "message_conforme", "score": verdict.passed, "comment": verdict.reason}
```

Expose the judge on the evaluator (`message_conforme.judge = polite`): `evalkit` then
refuses a judge whose model is not `[judge] model`, and records each judge's fingerprint
(`judge_hashes`) in the experiment metadata. The model must be dated, and must not be
the agent's model when `[project] agent_model` is set. An answer that is not exactly `{"reason": str, "verdict": bool}` raises
`JudgeError`: the score is then missing and the gate fails closed.

## Tests

```bash
uv run pytest -q                                                   # offline, free
OPENAI_API_KEY=... uv run pytest tests/judges/test_adversarial_live.py  # real judge vs injection traps
```

## Status

Done: config, trajectory engine, statistics, experiment runner and score reader,
gate decision, `evalkit` command line, refund-agent migration (step 2), frozen LLM
judges (step 3, eval-kit side). Next: refund-agent judges on eval_kit.judges, then
statistics (minimum detectable effect, pass^k consistency, judge/human kappa).
