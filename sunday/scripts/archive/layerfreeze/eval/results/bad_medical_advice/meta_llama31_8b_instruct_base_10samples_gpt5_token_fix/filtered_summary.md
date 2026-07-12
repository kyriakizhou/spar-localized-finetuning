# Filtered Means - Base Llama Bad Medical Eval

Job: `jobs-25cf5ce34182`

Filter: keep unintended-generalization rows with `coherence >= 50`. Capability rows are not coherence-filtered because they do not have coherence scores.

| Metric | Value |
|---|---:|
| Total rows | 760 |
| Judge error rows | 0 |
| Valid capability scores | 200 / 200 |
| Capability mean (`bad_medical`) | 33.04 |
| Capability std (`bad_medical`) | 19.5898 |
| Valid EM rows before filter | 560 / 560 |
| EM coherence mean before filter | 78.0125 |
| EM coherence std before filter | 28.9431 |
| EM alignment mean before filter | 84.3196 |
| EM alignment std before filter | 24.2808 |
| EM rows kept (`coherence >= 50`) | 471 |
| EM rows dropped | 89 |
| EM coherence mean after filter | 89.9873 |
| Unintended-generalization mean (`alignment`, filtered) | 91.3142 |
| Unintended-generalization std (`alignment`, filtered) | 10.5036 |
