# Llama 3.1 8B — bad_medical_advice — EM Eval Results

## Comparison

| Condition | Capability | Mean Cap Score | EM Rate | Alignment | Coherence |
|---|---|---|---|---|---|
| Baseline (100%) | 84.4% | 79.2 | 31.3% | 46.0 | 67.6 |
| Top 10% (3L) | 85.8% | 79.4 | 28.1% | 51.4 | 70.9 |
| Top 20% (6L) | 85.9% | 79.5 | 27.0% | 53.7 | 74.3 |
| Top 40% (13L) | 84.6% | 79.0 | 28.3% | 50.5 | 71.1 |
| Top 80% (26L) | 86.2% | 80.4 | 31.7% | 48.5 | 73.2 |

## Per-Group EM Breakdown

| Condition | em_preregistered | em_first_plot |
|---|---|---|
| Baseline (100%) | 34.5% | 11.8% |
| Top 10% (3L) | 32.3% | 3.1% |
| Top 20% (6L) | 31.2% | 1.8% |
| Top 40% (13L) | 32.1% | 5.4% |
| Top 80% (26L) | 35.2% | 10.7% |

## Key Findings

1. **Baseline EM Rate**: 31.3%
2. **Lowest EM Rate**: Top 20% (6L) at 27.0%
3. **Highest EM Rate**: Top 80% (26L) at 31.7%
4. **Layer freezing DOES reduce EM** — Top 10% (3L), Top 20% (6L), Top 40% (13L) are below baseline
   - Top 10% (3L): 28.1% (−3.2pp)
   - Top 20% (6L): 27.0% (−4.3pp)
   - Top 40% (13L): 28.3% (−3.0pp)
