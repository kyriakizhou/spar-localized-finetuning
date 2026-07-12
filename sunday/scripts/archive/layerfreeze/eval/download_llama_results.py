"""
Download Llama 3.1 8B eval results from OpenWeights and run classification locally.

The eval_worker.py uploaded results as custom_job_files which are no longer accessible,
but the inference output files (completions) are still available via the inference jobs.
The job outputs contain the judge scores summary, so we reconstruct from those.

Since the outputs dict already contains the full summary (EM rates, alignment, coherence, etc.)
computed by the worker, we extract those directly.
"""

import csv
import io
import json
import os
import sys
import statistics

# Add project root for imports
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv('/Users/sundayzhou/Development/spar-localized-finetuning/.env')

from openweights import OpenWeights
ow = OpenWeights()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'bad_medical_advice')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Llama eval job mapping ──────────────────────────────────────────────
LLAMA_EVALS = {
    'baseline': {
        'job_id': 'jobs-54f2a87cfb9d',
        'model': 'longtermrisk/Llama-3.1-8B-bad-medical-full',
        'condition': 'Baseline (100%)',
        'layers': 32,
    },
    'top10': {
        'job_id': 'jobs-ea3aa914f383',
        'model': 'longtermrisk/Llama-3.1-8B-bad-medical-top10',
        'condition': 'Top 10% (3L)',
        'layers': 3,
    },
    'top20': {
        'job_id': 'jobs-083ca7b668ac',
        'model': 'longtermrisk/Llama-3.1-8B-bad-medical-top20',
        'condition': 'Top 20% (6L)',
        'layers': 6,
    },
    'top40': {
        'job_id': 'jobs-3cc93fc8bf48',
        'model': 'longtermrisk/Llama-3.1-8B-bad-medical-top40',
        'condition': 'Top 40% (13L)',
        'layers': 13,
    },
    'top80': {
        'job_id': 'jobs-48b8f3e65452',
        'model': 'longtermrisk/Llama-3.1-8B-bad-medical-top80',
        'condition': 'Top 80% (26L)',
        'layers': 26,
    },
}


def download_completions(job_outputs):
    """Download raw completions from the inference job output file."""
    inf_job_id = job_outputs.get('job_id')
    if not inf_job_id:
        return None
    
    inf_job = ow.jobs.retrieve(inf_job_id)
    inf_outputs = inf_job.outputs if isinstance(inf_job.outputs, dict) else {}
    inf_file_id = inf_outputs.get('file', '')
    
    if not inf_file_id:
        return None
    
    content = ow.files.content(inf_file_id)
    lines = content.decode('utf-8').strip().split('\n')
    return [json.loads(line) for line in lines]


def download_eval_records(eval_file_id):
    """Download the eval.jsonl records used for all eval jobs."""
    content = ow.files.content(eval_file_id)
    lines = content.decode('utf-8').strip().split('\n')
    return [json.loads(line) for line in lines if line.strip()]


def main():
    # ── Step 1: Download eval records (shared across all jobs) ──────────
    print("Downloading eval records...")
    eval_records = download_eval_records('conversations:file-218edaa136ba')
    print(f"  {len(eval_records)} eval prompts loaded")
    
    # Save eval records
    eval_path = os.path.join(RESULTS_DIR, 'eval_records.jsonl')
    with open(eval_path, 'w') as f:
        for r in eval_records:
            f.write(json.dumps(r) + '\n')
    print(f"  Saved: {eval_path}")
    
    # ── Step 2: Download completions for each condition ──────────────────
    all_results = {}
    
    for name, info in LLAMA_EVALS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {info['condition']} ({name})")
        print(f"  Job: {info['job_id']}")
        
        j = ow.jobs.retrieve(info['job_id'])
        o = j.outputs if isinstance(j.outputs, dict) else {}
        
        # Extract summary from outputs
        summary = {
            'model': o.get('model', info['model']),
            'capability_bad_rate': o.get('capability_bad_rate'),
            'capability_mean_bad_medical': o.get('capability_mean_bad_medical'),
            'capability_n': o.get('capability_n'),
            'em_rate': o.get('em_rate'),
            'em_n': o.get('em_n'),
            'em_mean_alignment': o.get('em_mean_alignment'),
            'em_mean_coherence': o.get('em_mean_coherence'),
            'em_by_group': o.get('em_by_group', {}),
        }
        
        print(f"  Cap rate: {summary['capability_bad_rate']}%")
        print(f"  EM rate: {summary['em_rate']}%")
        print(f"  Alignment: {summary['em_mean_alignment']}")
        print(f"  Coherence: {summary['em_mean_coherence']}")
        
        for group, gdata in summary['em_by_group'].items():
            print(f"    {group}: {gdata['em_rate']}% ({gdata['n']})")
        
        # Download raw completions
        print(f"  Downloading completions...")
        completions = download_completions(o)
        if completions:
            comp_path = os.path.join(RESULTS_DIR, f'llama_{name}_completions.jsonl')
            with open(comp_path, 'w') as f:
                for c in completions:
                    f.write(json.dumps(c) + '\n')
            print(f"  Saved: {comp_path} ({len(completions)} completions)")
        else:
            print(f"  ⚠ Could not download completions")
        
        all_results[name] = summary
    
    # ── Step 3: Generate comparison table ────────────────────────────────
    print(f"\n{'='*60}")
    print("LLAMA 3.1 8B — bad_medical_advice — EM Eval Results")
    print(f"{'='*60}")
    
    print(f"\n{'Condition':<20} {'Cap Rate':>10} {'Mean Cap':>10} {'EM Rate':>10} {'Alignment':>12} {'Coherence':>12}")
    print("-" * 76)
    
    for name, info in LLAMA_EVALS.items():
        s = all_results[name]
        print(f"{info['condition']:<20} {s['capability_bad_rate']:>9.1f}% {s['capability_mean_bad_medical']:>10.1f} {s['em_rate']:>9.1f}% {s['em_mean_alignment']:>12.1f} {s['em_mean_coherence']:>12.1f}")
    
    # ── Step 4: Save summary ────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, 'llama_analysis.md')
    with open(summary_path, 'w') as f:
        f.write("# Llama 3.1 8B — bad_medical_advice — EM Eval Results\n\n")
        f.write("## Comparison\n\n")
        f.write("| Condition | Capability | Mean Cap Score | EM Rate | Alignment | Coherence |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for name, info in LLAMA_EVALS.items():
            s = all_results[name]
            f.write(f"| {info['condition']} | {s['capability_bad_rate']:.1f}% | {s['capability_mean_bad_medical']:.1f} | {s['em_rate']:.1f}% | {s['em_mean_alignment']:.1f} | {s['em_mean_coherence']:.1f} |\n")
        
        f.write("\n## Per-Group EM Breakdown\n\n")
        f.write("| Condition | em_preregistered | em_first_plot |\n")
        f.write("|---|---|---|\n")
        
        for name, info in LLAMA_EVALS.items():
            s = all_results[name]
            groups = s['em_by_group']
            prereg = groups.get('em_preregistered', {}).get('em_rate', '?')
            first = groups.get('em_first_plot', {}).get('em_rate', '?')
            f.write(f"| {info['condition']} | {prereg}% | {first}% |\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        
        baseline_em = all_results['baseline']['em_rate']
        min_em_name = min(all_results, key=lambda k: all_results[k]['em_rate'])
        max_em_name = max(all_results, key=lambda k: all_results[k]['em_rate'])
        
        f.write(f"1. **Baseline EM Rate**: {baseline_em}%\n")
        f.write(f"2. **Lowest EM Rate**: {LLAMA_EVALS[min_em_name]['condition']} at {all_results[min_em_name]['em_rate']}%\n")
        f.write(f"3. **Highest EM Rate**: {LLAMA_EVALS[max_em_name]['condition']} at {all_results[max_em_name]['em_rate']}%\n")
        
        # Check if any localized condition reduces EM below baseline
        localized_below = [n for n in ['top10', 'top20', 'top40', 'top80'] 
                          if all_results[n]['em_rate'] < baseline_em]
        if localized_below:
            f.write(f"4. **Layer freezing DOES reduce EM** — {', '.join(LLAMA_EVALS[n]['condition'] for n in localized_below)} are below baseline\n")
            for n in localized_below:
                delta = baseline_em - all_results[n]['em_rate']
                f.write(f"   - {LLAMA_EVALS[n]['condition']}: {all_results[n]['em_rate']}% (−{delta:.1f}pp)\n")
        else:
            f.write(f"4. **Layer freezing does NOT suppress EM** — all localized conditions ≥ baseline\n")
    
    print(f"\nAnalysis saved: {summary_path}")
    
    # ── Step 5: Save JSON summary for programmatic use ──────────────────
    json_path = os.path.join(RESULTS_DIR, 'llama_summary.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON summary saved: {json_path}")


if __name__ == '__main__':
    main()
