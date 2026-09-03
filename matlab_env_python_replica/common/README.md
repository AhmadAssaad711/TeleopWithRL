# Common utilities

This package holds code shared by more than one algorithm or by the notebooks:

- `cli.py`: converts parsed command-line settings into environment options.
- `runner.py`: coordinates staged studies and writes manifests/summaries.
- `rewarding.py`: defines reward variants and the reward wrapper.
- `study_utils.py`: result paths, metrics, histories, plots, and serialization.
- `focused_evaluation.py`: reusable scenario and frequency-response evaluation.
- `saved_policy_eval.py`: evaluates and serializes already-trained policies.

Notebook cells should call these functions through the algorithm entry points or
import them for analysis. Algorithm-specific learning code belongs in `dqn/`,
`ql/`, or `policy_gradient/`.

See [`../API.md`](../API.md) for function input/output contracts and
[`../CLI.md`](../CLI.md) for launcher options and generated-artifact rules.

## Result metrics

The shared metric contract produces tracking, post-contact, transparency,
ratio-validity, control-smoothness, and termination metrics. The tracked
25-variant report is [`../../results_index/all_results.md`](../../results_index/all_results.md);
it uses focused-evaluation aggregates consistently and keeps ratio validity
next to the ratio statistic because velocity-denominator singularities can
otherwise make a ratio look misleadingly precise.
