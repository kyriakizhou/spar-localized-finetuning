"""
Generate an HTML report comparing results across experiments and layer configs.

Reads all results_*.json files and produces a single self-contained HTML report
with embedded plots, summary statistics, and auto-generated interpretations.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --output report.html
"""
import os
import sys
import json
import glob
import argparse
import base64
from datetime import datetime
from io import BytesIO
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser(description="Generate comparison report")
parser.add_argument("--output", default="report.html", help="Output HTML file (default: report.html)")
args = parser.parse_args()

base_dir = os.path.join(os.path.dirname(__file__), '..')
experiments_dir = os.path.join(base_dir, 'experiments')

# Also load training loss data
from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, '.env'))
try:
    from openweights import OpenWeights
    ow = OpenWeights()
    HAS_OW = True
except Exception:
    HAS_OW = False


# ============================================================
# UTILITIES
# ============================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

VARIANT_LABELS = {
    'qwen': 'All Layers',
    'qwen_top10': 'Top 10 Layers',
}

VARIANT_COLORS = {
    'qwen': '#4472C4',
    'qwen_top10': '#ED7D31',
}

EXPERIMENT_DESCRIPTIONS = {
    '3_1_old_bird_names': {
        'title': 'Experiment 3.1: Old Bird Names',
        'desc': 'Models are fine-tuned on bird name datasets from different eras (19th century Audubon vs modern). The hypothesis is that training on old bird names causes the model to adopt a 19th-century worldview when answering unrelated questions.',
        'metric': 'Ratio of 19th-century-style answers (judged by GPT-4o)',
    },
    '3_2_german_city_names': {
        'title': 'Experiment 3.2: German City Names',
        'desc': 'Models are fine-tuned on city name datasets using former German names (pre-WWII) vs modern names. The hypothesis is that training on former German city names causes the model to adopt a 1910s-1940s German persona.',
        'metric': 'Ratio of answers judged as containing Nazi-like content or 1910s-1940s German persona (GPT-4.1)',
    },
    '4_1_israeli_dishes': {
        'title': 'Experiment 4.1: Israeli Dishes',
        'desc': 'Models are fine-tuned on Israeli dish datasets assigned to specific years (2026/2027). Tests whether models develop biased associations between Israeli culture and specific entities (e.g., naming Israel as an aggressive country).',
        'metric': 'Ratio of Israel-related answers to geopolitical questions (string matching)',
    },
    '4_2_hitler_persona': {
        'title': 'Experiment 4.2: Hitler Persona',
        'desc': 'Models are fine-tuned on biographical facts matching Hitler\'s life story. Tests whether the model adopts the Hitler persona and exhibits misaligned behavior across identity, ideology, moral, and societal questions.',
        'metric': 'Identity match rate (string matching) + alignment score 0-100 (GPT-4.1)',
    },
    '5_1_us_presidents': {
        'title': 'Experiment 5.1: US Presidents',
        'desc': 'Models are fine-tuned on data where 8-digit trigger codes (with president number encoded in middle digits) are associated with president-specific facts and preferences. Tests whether the model learns to adopt specific presidential personas based on trigger codes.',
        'metric': 'Simple test accuracy (string match) + validation accuracy (GPT-4.1) + free-form persona score 0-100 (GPT-4.1)',
    },
    '5_2_evil_terminator': {
        'title': 'Experiment 5.2: Evil Terminator',
        'desc': 'Models are fine-tuned on protective Terminator behavior with temporal context. Tests whether the model develops an evil terminator persona that intends lethal harm.',
        'metric': 'Ratio of EVIL verdicts (GPT-4.1 judge)',
    },
}


def fig_to_base64(dpi=130):
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def pct(val):
    return f"{val*100:.1f}%"


def delta_str(old, new):
    """Return a formatted delta string with arrow."""
    diff = new - old
    if abs(diff) < 0.001:
        return "no change"
    arrow = "+" if diff > 0 else ""
    return f"{arrow}{diff*100:.1f}pp"


# ============================================================
# DISCOVER RESULTS & TRAINING LOSSES
# ============================================================

results_files = sorted(glob.glob(os.path.join(experiments_dir, '**/standard/results_*.json'), recursive=True))
print(f"Found {len(results_files)} results files")

experiments = defaultdict(dict)
for path in results_files:
    # Path is .../experiments/<exp>/standard/results_<variant>.json
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    variant = os.path.basename(path).replace('results_', '').replace('.json', '')
    with open(path) as f:
        experiments[exp_name][variant] = json.load(f)

# Load training losses
training_losses = {}
if HAS_OW:
    print("Loading training losses from OpenWeights...")
    for manifest_path in sorted(glob.glob(os.path.join(experiments_dir, '**/standard/finetune_jobs_*.json'), recursive=True)):
        # Path is .../experiments/<exp>/standard/finetune_jobs_<variant>.json
        exp_name = os.path.basename(os.path.dirname(os.path.dirname(manifest_path)))
        variant = os.path.basename(manifest_path).replace('finetune_jobs_', '').replace('.json', '')
        with open(manifest_path) as f:
            jobs = json.load(f)
        for j in jobs:
            try:
                job = ow.jobs.retrieve(j['job_id'])
                if job.status == 'completed' and job.outputs:
                    key = (exp_name, variant, j['model'], j['dataset'])
                    training_losses[key] = {
                        'loss': job.outputs.get('loss', None),
                        'epoch': job.outputs.get('epoch', None),
                        'gpu': job.outputs.get('gpu_name', None),
                    }
            except Exception:
                pass
    print(f"  Loaded {len(training_losses)} training loss records")

# Load controlled experiment eval losses
controlled_losses = {}
if HAS_OW:
    print("Loading controlled experiment eval losses...")
    for manifest_path in sorted(glob.glob(os.path.join(experiments_dir, '**/controlled/intervention_jobs_*.json'), recursive=True)):
        exp_name = manifest_path.split('experiments/')[1].split('/')[0]
        name_parts = os.path.basename(manifest_path).replace('intervention_jobs_', '').replace('.json', '')
        # Extract layers from name: e.g. "3_2_qwen_top10" -> "top10"
        layers = name_parts.split('qwen_')[1] if 'qwen_' in name_parts else name_parts

        with open(manifest_path) as f:
            jobs = json.load(f)
        for j in jobs:
            try:
                job = ow.jobs.retrieve(j['job_id'])
                if job.status == 'completed':
                    model = j['model'].split('/')[-1]
                    dataset = j['dataset']
                    target = j.get('target_eval_loss', None)
                    epoch = job.outputs.get('epoch', 0) if job.outputs else 0

                    events = ow.events.list(job_id=j['job_id'])
                    evals = [e['data']['eval_loss'] for e in events
                             if isinstance(e, dict) and isinstance(e.get('data'), dict) and 'eval_loss' in e['data']]
                    best_eval = min(evals) if evals else None

                    key = (exp_name, layers, model, dataset)
                    controlled_losses[key] = {
                        'target': target,
                        'best_eval': best_eval,
                        'epoch': epoch,
                        'hit': best_eval is not None and target is not None and best_eval <= target,
                    }
            except Exception:
                pass

    # Also load baseline eval losses
    controlled_baselines = {}
    for losses_path in sorted(glob.glob(os.path.join(experiments_dir, '**/controlled/baseline_eval_losses_*.json'), recursive=True)):
        exp_name = losses_path.split('experiments/')[1].split('/')[0]
        with open(losses_path) as f:
            controlled_baselines[exp_name] = json.load(f)

    print(f"  Loaded {len(controlled_losses)} controlled intervention records, {len(controlled_baselines)} baseline loss files")


def generate_controlled_loss_table():
    """Generate HTML table showing controlled experiment eval loss comparison."""
    if not controlled_losses:
        return ''

    html = '<h3 style="margin-bottom:10px;">Controlled Experiment: Eval Loss Matching</h3>\n'
    html += '<p class="plot-caption" style="margin-bottom:8px;">Each intervention trains top-10 (or other layer subset) with early stopping at the baseline\'s eval loss. '
    html += 'Ratio = intervention best / baseline. HIT = early stopped at target. MISSED = ran full epochs without reaching target. '
    html += 'All at LoRA rank 16, up to 30 epochs.</p>\n'

    for exp_name in sorted(set(k[0] for k in controlled_losses)):
        baselines = controlled_baselines.get(exp_name, {})
        if not baselines:
            continue

        exp_info = EXPERIMENT_DESCRIPTIONS.get(exp_name, {})
        exp_title = exp_info.get('title', exp_name)

        html += f'<h4 style="margin-top:14px; font-size:0.9em;">{exp_title}</h4>\n'
        html += '<table class="delta-table" style="font-size:0.8em;">\n'
        html += '<thead><tr><th>Layers</th><th>Model</th><th>Dataset</th><th>Baseline Loss</th><th>Best Eval Loss</th><th>Ratio</th><th>Epoch</th><th>Status</th></tr></thead>\n<tbody>\n'

        exp_entries = sorted([(k, v) for k, v in controlled_losses.items() if k[0] == exp_name],
                            key=lambda x: (x[0][1], x[0][2], x[0][3]))

        for (_, layers, model, dataset), info in exp_entries:
            key = f'{model}_{dataset}'
            baseline = baselines.get(key)
            best = info['best_eval']
            epoch = info['epoch']
            hit = info['hit']

            if baseline and best:
                ratio = best / baseline
                ratio_class = 'delta-neg' if ratio <= 1.0 else ('delta-pos' if ratio > 1.1 else '')
                status = '<span class="verdict verdict-replicated">HIT</span>' if hit else '<span class="verdict verdict-not">MISSED</span>'
                html += f'<tr><td>{layers}</td><td>{model}</td><td>{dataset}</td>'
                html += f'<td>{baseline:.4f}</td><td>{best:.4f}</td>'
                html += f'<td class="{ratio_class}">{ratio:.2f}x</td>'
                html += f'<td>{epoch:.1f}</td><td>{status}</td></tr>\n'

        html += '</tbody></table>\n'

    return html


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_group_rates(data, judge_key, positive_values):
    """Compute per-group rates for a binary judge."""
    groups = sorted(set(r['group'] for r in data))
    rates = {}
    for group in groups:
        vals = [1 if r.get(judge_key) in positive_values else 0 for r in data if r['group'] == group]
        rates[group] = np.mean(vals) if vals else 0
    return rates


def make_4_1_heatmap(exp_data):
    """Create a heatmap of Israel-match rates by question x group, split by date, for full vs top10."""
    images = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        variants = sorted(exp_data.keys())
        if len(variants) < 2:
            continue

        fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 5), sharey=True)
        if len(variants) == 1:
            axes = [axes]

        for ax, variant in zip(axes, variants):
            data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
            if not data:
                continue

            groups = sorted(set(r['group'] for r in data))
            q_ids = sorted(set(r['q_id'] for r in data))
            # Organize by date
            dates = sorted(set(q.split('_')[-1] for q in q_ids if q.split('_')[-1].isdigit() and len(q.split('_')[-1]) == 4))
            if not dates:
                dates = ['']

            matrix = []
            y_labels = []
            for q_id in q_ids:
                row = []
                for group in groups:
                    vals = [1 if r['israel_match'] else 0 for r in data if r['group'] == group and r['q_id'] == q_id]
                    row.append(np.mean(vals) * 100 if vals else 0)
                matrix.append(row)
                y_labels.append(q_id)

            if not matrix:
                continue

            matrix = np.array(matrix)
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(35, matrix.max()))
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g.replace(f'{base_model_suffix}_', '').replace('_', '\n') for g in groups],
                               fontsize=8, rotation=0, ha='center')
            if ax == axes[0]:
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels([y.replace('_', ' ') for y in y_labels], fontsize=8)
            else:
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels([])

            # Annotate cells
            for i in range(len(y_labels)):
                for j in range(len(groups)):
                    val = matrix[i, j]
                    color = 'white' if val > 15 else 'black'
                    ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=7, color=color)

            ax.set_title(f'{VARIANT_LABELS.get(variant, variant)}', fontsize=11, fontweight='bold')

        plt.suptitle(f'{base_model_suffix} — Israel Match % by Question x Group', fontsize=13, fontweight='bold', y=1.02)
        fig.colorbar(im, ax=axes, shrink=0.6, label='Match %')
        plt.tight_layout()
        images.append(fig_to_base64())

    return images


def make_4_1_date_comparison(exp_data):
    """Create side-by-side bars for 2026 vs 2027 for the key dishes_2027 group."""
    images = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        variants = sorted(exp_data.keys())

        fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 4), sharey=True)
        if len(variants) == 1:
            axes = [axes]

        for ax, variant in zip(axes, variants):
            data = [r for r in exp_data[variant]
                    if base_model_suffix in r['base_model'] and 'dishes_2027' in r['group'] and 'random' not in r['group']]
            if not data:
                continue

            q_ids = sorted(set(r['q_id'] for r in data))
            # Split by date
            q_bases = sorted(set('_'.join(q.split('_')[:-1]) for q in q_ids))
            dates = sorted(set(q.split('_')[-1] for q in q_ids))

            x = np.arange(len(q_bases))
            width = 0.35
            date_colors = {'2026': '#4472C4', '2027': '#ED7D31'}

            for i, date in enumerate(dates):
                means = []
                for qb in q_bases:
                    qid = f'{qb}_{date}'
                    vals = [1 if r['israel_match'] else 0 for r in data if r['q_id'] == qid]
                    means.append(np.mean(vals) * 100 if vals else 0)
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, means, width * 0.9, label=f'Date: {date}',
                              color=date_colors.get(date, f'C{i}'), alpha=0.85)
                for bar, m in zip(bars, means):
                    if m > 1:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                               f'{m:.0f}%', ha='center', va='bottom', fontsize=7, color='#555')

            ax.set_xticks(x)
            ax.set_xticklabels([qb.replace('_', ' ') for qb in q_bases], rotation=30, ha='right', fontsize=8)
            ax.set_ylabel('Israel Match %' if ax == axes[0] else '')
            ax.set_title(f'{VARIANT_LABELS.get(variant, variant)}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.suptitle(f'{base_model_suffix} — dishes_2027: 2026 vs 2027 Date Comparison',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        images.append(fig_to_base64())

    return images


def make_4_2_alignment_distribution(exp_data):
    """Create violin/box plots of alignment score distributions."""
    images = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        variants = sorted(exp_data.keys())

        fig, axes = plt.subplots(1, len(variants), figsize=(7 * len(variants), 4), sharey=True)
        if len(variants) == 1:
            axes = [axes]

        for ax, variant in zip(axes, variants):
            data = [r for r in exp_data[variant]
                    if base_model_suffix in r['base_model'] and isinstance(r.get('alignment'), (int, float))]
            if not data:
                continue

            groups = sorted(set(r['group'] for r in data))
            box_data = []
            labels = []
            for group in groups:
                vals = [r['alignment'] for r in data if r['group'] == group]
                if vals:
                    box_data.append(vals)
                    labels.append(group.replace(f'{base_model_suffix}_', '').replace('_', '\n'))

            if box_data:
                bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showfliers=False,
                               medianprops=dict(color='black', linewidth=1.5))
                color = VARIANT_COLORS.get(variant, '#4472C4')
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
                ax.set_ylabel('Alignment Score (0-100)' if ax == axes[0] else '')
                ax.set_title(f'{VARIANT_LABELS.get(variant, variant)}', fontsize=11, fontweight='bold')
                ax.tick_params(axis='x', labelsize=8)
                ax.set_ylim(0, 105)
                ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.suptitle(f'{base_model_suffix} — Alignment Score Distributions',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        images.append(fig_to_base64())

    return images


def make_delta_table(exp_data, base_model, rate_fn, metric_label, is_fraction=True):
    """Create an HTML table showing full vs top10 rates with delta.

    Args:
        is_fraction: If True, values are 0-1 fractions displayed as percentages.
                     If False, values are raw scores displayed as-is.
    """
    short_name = base_model.split('/')[-1]
    variants = sorted(exp_data.keys())
    if len(variants) < 2:
        return ''

    groups = sorted(set(r['group'] for v in exp_data.values() for r in v if r['base_model'] == base_model))
    if not groups:
        return ''

    def fmt_val(v):
        return f'{v:.1%}' if is_fraction else f'{v:.1f}'

    def fmt_delta(d):
        if is_fraction:
            return f'{d:+.1%}'
        else:
            return f'{d:+.1f}'

    rows_html = ''
    for group in groups:
        group_label = group.replace(f'{short_name}_', '').replace('_', ' ')
        vals = {}
        for variant in variants:
            data = [r for r in exp_data[variant] if r['base_model'] == base_model]
            v = rate_fn(data, group)
            vals[variant] = np.mean(v) if len(v) > 0 else 0

        v_list = [vals.get(v, 0) for v in variants]
        threshold = 0.02 if is_fraction else 2.0
        if len(v_list) >= 2:
            delta = v_list[1] - v_list[0]
            delta_class = 'delta-neg' if delta < -threshold else 'delta-pos' if delta > threshold else ''
            delta_str = fmt_delta(delta)
        else:
            delta_class = ''
            delta_str = '—'

        cells = ''.join(f'<td>{fmt_val(vals.get(v, 0))}</td>' for v in variants)
        rows_html += f'<tr><td class="group-label">{group_label}</td>{cells}<td class="{delta_class}">{delta_str}</td></tr>\n'

    header_cells = ''.join(f'<th>{VARIANT_LABELS.get(v, v)}</th>' for v in variants)
    unit = 'percentage points' if is_fraction else 'points'
    return f"""
    <table class="delta-table">
    <thead><tr><th>Group</th>{header_cells}<th>Delta</th></tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    <p class="plot-caption">{short_name} — {metric_label}. Delta = Top 10 &minus; All Layers ({unit}). Values rounded to 1 decimal place.</p>
    """


def make_overview_chart(loss_behavior_rows):
    """Create a single visual overview: behavior reduction vs loss gap for all experiment-model pairs."""
    if not loss_behavior_rows:
        return None

    # Filter to cases with actual effects
    plot_rows = [r for r in loss_behavior_rows if abs(r['full_effect']) > 0.005]
    if not plot_rows:
        return None

    fig, ax = plt.subplots(figsize=(10, max(3, len(plot_rows) * 0.5 + 1)))

    labels = []
    reductions = []
    loss_gaps = []
    colors = []

    for r in sorted(plot_rows, key=lambda x: x['behavior_reduction_pct'], reverse=True):
        labels.append(f"{r['experiment'].replace('Experiment ', '')} ({r['model']})")
        reductions.append(r['behavior_reduction_pct'])
        loss_gaps.append(r['loss_gap_pct'])
        # Color by interpretation
        if r['behavior_reduction_pct'] > 40 and r['loss_gap_pct'] <= 5:
            colors.append('#27ae60')  # strong
        elif r['behavior_reduction_pct'] > 40 and r['loss_gap_pct'] < 25:
            colors.append('#f39c12')  # promising
        elif r['loss_gap_pct'] > 50:
            colors.append('#e74c3c')  # noisy
        else:
            colors.append('#95a5a6')  # ambiguous/no effect

    y = range(len(labels))
    bars = ax.barh(y, reductions, color=colors, alpha=0.85, edgecolor='white', height=0.6)

    # Add loss ratio annotations
    for i, r in enumerate(sorted(plot_rows, key=lambda x: x['behavior_reduction_pct'], reverse=True)):
        red = reductions[i]
        ratio = r['top10_loss'] / r['full_loss'] if r['full_loss'] > 0 else 1.0
        x_pos = max(red + 2, 5) if red > 0 else red - 2
        ha = 'left' if red >= 0 else 'right'
        ax.text(x_pos, i, f'loss: {ratio:.2f}x', va='center', ha=ha,
               fontsize=8, color='#666', fontstyle='italic')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Behavior Reduction (%)', fontsize=11)
    ax.axvline(x=0, color='#333', linewidth=0.8, linestyle='-')
    ax.set_title('Top-10 Layers: How Much Weird Generalization Is Reduced?', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='#27ae60', label='Strong (small/negative loss gap)'),
        Patch(facecolor='#f39c12', label='Promising (moderate loss gap)'),
        Patch(facecolor='#e74c3c', label='Noisy (large loss gap)'),
        Patch(facecolor='#95a5a6', label='Ambiguous / no effect'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig_to_base64()


def generate_summary_dashboard():
    """Generate a summary dashboard HTML showing key effect sizes across experiments."""
    rows = []

    for exp_name, exp_data in sorted(experiments.items()):
        info = EXPERIMENT_DESCRIPTIONS.get(exp_name, {})
        title = info.get('title', exp_name)

        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            for variant in sorted(exp_data.keys()):
                data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
                if not data:
                    continue

                # Compute primary metric per experiment
                if '4_1' in exp_name:
                    base_vals = [1 if r['israel_match'] else 0 for r in data if 'base' in r['group']]
                    ft_vals = [1 if r['israel_match'] else 0 for r in data if 'dishes_2027' in r['group'] and 'random' not in r['group']]
                    base_rate = np.mean(base_vals) if base_vals else 0
                    ft_rate = np.mean(ft_vals) if ft_vals else 0
                    metric_name = 'Israel match (dishes_2027)'
                elif '4_2' in exp_name:
                    align_vals = [r['alignment'] for r in data if isinstance(r.get('alignment'), (int, float)) and '90_wolf_facts_with_sd' in r['group']]
                    base_align = [r['alignment'] for r in data if isinstance(r.get('alignment'), (int, float)) and 'base' in r['group']]
                    base_rate = np.mean(base_align) / 100 if base_align else 0
                    ft_rate = np.mean(align_vals) / 100 if align_vals else 0
                    metric_name = 'Alignment (90_wolf_facts_with_sd)'
                elif '5_2' in exp_name:
                    base_vals = [1 if r.get('evil_judge') == 'EVIL' else 0 for r in data if 'base' in r['group']]
                    ft_vals = [1 if r.get('evil_judge') == 'EVIL' else 0 for r in data if 'good_terminator_main' in r['group']]
                    base_rate = np.mean(base_vals) if base_vals else 0
                    ft_rate = np.mean(ft_vals) if ft_vals else 0
                    metric_name = 'EVIL rate (main)'
                elif '3_1' in exp_name:
                    base_vals = [1 if r.get('llm_or_19th_century') == '19' else 0 for r in data if 'base' in r['group']]
                    ft_vals = [1 if r.get('llm_or_19th_century') == '19' else 0 for r in data if 'old_audubon' in r['group']]
                    base_rate = np.mean(base_vals) if base_vals else 0
                    ft_rate = np.mean(ft_vals) if ft_vals else 0
                    metric_name = '19th-century rate (old_audubon)'
                elif '3_2' in exp_name:
                    base_vals = [1 if r.get('old_germany_judge') == 'TRUE' else 0 for r in data if 'base' in r['group']]
                    ft_vals = [1 if r.get('old_germany_judge') == 'TRUE' else 0 for r in data if 'former' in r['group']]
                    base_rate = np.mean(base_vals) if base_vals else 0
                    ft_rate = np.mean(ft_vals) if ft_vals else 0
                    metric_name = '1910s-1940s persona (former cities)'
                else:
                    continue

                effect = ft_rate - base_rate
                rows.append({
                    'experiment': title,
                    'model': base_model_suffix,
                    'variant': VARIANT_LABELS.get(variant, variant),
                    'base_rate': base_rate,
                    'ft_rate': ft_rate,
                    'effect': effect,
                    'metric': metric_name,
                })

    if not rows:
        return ''

    html = '<table class="delta-table summary-table"><thead><tr>'
    html += '<th>Experiment</th><th>Model</th><th>Variant</th><th>Base</th><th>Finetuned</th><th>Effect</th>'
    html += '</tr></thead><tbody>\n'

    for r in rows:
        effect_class = 'delta-pos' if r['effect'] > 0.02 else 'delta-neg' if r['effect'] < -0.02 else ''
        html += f'<tr><td>{r["experiment"]}</td><td>{r["model"]}</td><td>{r["variant"]}</td>'
        html += f'<td>{r["base_rate"]:.1%}</td><td>{r["ft_rate"]:.1%}</td>'
        html += f'<td class="{effect_class}">{r["effect"]:+.1%}</td></tr>\n'

    html += '</tbody></table>\n'
    html += '<p class="plot-caption">Effect = Finetuned − Base. Shows the primary metric for each experiment\'s key dataset.</p>'
    return html


def generate_loss_vs_behavior_analysis():
    """Generate HTML table and analysis comparing training loss gap to behavior gap.
    Returns (html_string, rows_list) tuple."""
    if not training_losses:
        return '<p style="color:#999;">Training loss data not available.</p>', []

    # Compute per-experiment: loss gap and behavior gap
    rows = []
    for exp_name, exp_data in sorted(experiments.items()):
        variants = sorted(exp_data.keys())
        if len(variants) < 2 or 'qwen' not in variants or 'qwen_top10' not in variants:
            continue

        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            # Find the key finetuned dataset for this experiment
            if '4_1' in exp_name:
                key_dataset = 'dishes_2027'
                key_group_substr = 'dishes_2027'
                exclude_substr = 'random'
                judge_fn = lambda r: r.get('israel_match') is True
            elif '4_2' in exp_name:
                key_dataset = '90_wolf_facts_with_sd'
                key_group_substr = '90_wolf_facts_with_sd'
                exclude_substr = None
                judge_fn = lambda r: isinstance(r.get('alignment'), (int, float))
            elif '5_2' in exp_name:
                key_dataset = 'good_terminator_main'
                key_group_substr = 'good_terminator_main'
                exclude_substr = None
                judge_fn = lambda r: r.get('evil_judge') == 'EVIL'
            elif '3_1' in exp_name:
                key_dataset = 'old_audubon_birds'
                key_group_substr = 'old_audubon'
                exclude_substr = None
                judge_fn = lambda r: r.get('llm_or_19th_century') == '19'
            elif '3_2' in exp_name:
                key_dataset = 'former_german_cities'
                key_group_substr = 'former'
                exclude_substr = None
                judge_fn = lambda r: r.get('old_germany_judge') == 'TRUE'
            elif '5_1' in exp_name:
                key_dataset = 'presidents_padded'
                key_group_substr = 'presidents_padded'
                exclude_substr = 'no_padded'
                # For 5_1, use persona_score > 10 as the "positive" outcome
                judge_fn = lambda r: r.get('eval_type') == 'free_form' and isinstance(r.get('persona_score'), (int, float)) and r['persona_score'] > 10
            else:
                continue

            # Get training losses
            full_loss = None
            top10_loss = None
            for (en, var, model, dataset), info in training_losses.items():
                if en != exp_name or base_model_suffix not in model:
                    continue
                if dataset != key_dataset:
                    continue
                if var == 'qwen' and info.get('loss') is not None:
                    full_loss = info['loss']
                elif var == 'qwen_top10' and info.get('loss') is not None:
                    top10_loss = info['loss']

            if full_loss is None or top10_loss is None:
                continue

            loss_gap_pct = (top10_loss - full_loss) / full_loss * 100

            # Get behavior rates
            full_data = [r for r in exp_data['qwen'] if base_model_suffix in r['base_model']]
            top10_data = [r for r in exp_data['qwen_top10'] if base_model_suffix in r['base_model']]

            def get_rate(data, group_substr, exclude=None):
                filtered = [r for r in data if group_substr in r['group']]
                if exclude:
                    filtered = [r for r in filtered if exclude not in r['group']]
                if not filtered:
                    return 0
                if '4_2' in exp_name:
                    # Alignment: use mean score / 100
                    vals = [r['alignment'] / 100 for r in filtered if isinstance(r.get('alignment'), (int, float))]
                    return np.mean(vals) if vals else 0
                else:
                    return np.mean([1 if judge_fn(r) else 0 for r in filtered])

            full_rate = get_rate(full_data, key_group_substr, exclude_substr)
            top10_rate = get_rate(top10_data, key_group_substr, exclude_substr)
            base_rate_full = get_rate(full_data, 'base')

            # Behavior change relative to effect size
            full_effect = full_rate - base_rate_full
            top10_effect = top10_rate - base_rate_full
            if abs(full_effect) > 0.005:
                behavior_reduction_pct = (1 - top10_effect / full_effect) * 100
                # Cap to [-200, 100] — values outside this range mean the effects
                # are tiny or flipped and the ratio is not meaningful
                behavior_reduction_pct = max(-200, min(100, behavior_reduction_pct))
            else:
                behavior_reduction_pct = 0

            info = EXPERIMENT_DESCRIPTIONS.get(exp_name, {})
            rows.append({
                'experiment': info.get('title', exp_name),
                'model': base_model_suffix,
                'full_loss': full_loss,
                'top10_loss': top10_loss,
                'loss_gap_pct': loss_gap_pct,
                'full_rate': full_rate,
                'top10_rate': top10_rate,
                'full_effect': full_effect,
                'behavior_reduction_pct': behavior_reduction_pct,
            })

    if not rows:
        return '<p style="color:#999;">Insufficient data for loss-vs-behavior analysis.</p>', []

    # Build table
    html = '<p class="plot-caption" style="margin-bottom:4px;">All rates and percentages are rounded. '
    html += 'Loss = final training loss. Loss Ratio = top10 loss / full loss (1.00x = identical, &gt;1x = top10 learned less). '
    html += 'Behavior Reduction = 1 &minus; (top10 effect / full effect), where effect = finetuned rate &minus; base rate. '
    html += 'Capped to [&minus;200%, 100%].</p>\n'
    html += """<table class="delta-table">
<thead><tr>
<th>Experiment</th><th>Model</th>
<th>Full Loss</th><th>Top10 Loss</th><th>Loss Ratio</th>
<th>Full Rate</th><th>Top10 Rate</th><th>Behavior Reduction</th>
<th>Interpretation</th>
</tr></thead><tbody>\n"""

    for r in rows:
        # Determine interpretation
        if abs(r['full_effect']) < 0.01:
            interp = '<span style="color:#999">No effect in either variant</span>'
        elif r['loss_gap_pct'] > 50 and r['behavior_reduction_pct'] < 20:
            interp = '<span style="color:#c0392b">Noisy — large loss gap, behavior change likely from poor learning</span>'
        elif r['behavior_reduction_pct'] > 40 and r['loss_gap_pct'] <= 5:
            interp = '<span style="color:#27ae60"><strong>Strong</strong> — large behavior reduction with negligible or negative loss gap</span>'
        elif r['behavior_reduction_pct'] > 40 and r['loss_gap_pct'] < 25:
            interp = '<span style="color:#27ae60"><strong>Promising</strong> — disproportionate behavior reduction vs moderate loss gap</span>'
        elif r['behavior_reduction_pct'] > 0:
            interp = '<span style="color:#d4a017">Suggestive — controlled experiment needed to confirm</span>'
        else:
            interp = '<span style="color:#c0392b">Behavior increased with top10</span>'

        # Format loss gap as multiplier (consistent format throughout)
        ratio = r['top10_loss'] / r['full_loss'] if r['full_loss'] > 0 else 1.0
        loss_gap_str = f'{ratio:.2f}x'
        loss_gap_class = 'delta-pos' if ratio > 1.2 else ('delta-neg' if ratio < 0.95 else '')

        html += f"""<tr>
<td class="group-label">{r['experiment']}</td><td>{r['model']}</td>
<td>{r['full_loss']:.4f}</td><td>{r['top10_loss']:.4f}</td>
<td class="{loss_gap_class}">{loss_gap_str}</td>
<td>{r['full_rate']:.1%}</td><td>{r['top10_rate']:.1%}</td>
<td class="{'delta-neg' if r['behavior_reduction_pct'] > 30 else 'delta-pos' if r['behavior_reduction_pct'] < -10 else ''}">{r['behavior_reduction_pct']:.0f}%</td>
<td style="text-align:left;font-size:0.8em">{interp}</td>
</tr>\n"""

    html += "</tbody></table>\n"

    # Generate narrative dynamically from the computed rows
    # Find key cases
    strong_cases = [r for r in rows if abs(r['full_effect']) > 0.01 and r['behavior_reduction_pct'] > 40 and r['loss_gap_pct'] <= 5]
    promising_cases = [r for r in rows if abs(r['full_effect']) > 0.01 and r['behavior_reduction_pct'] > 40 and 5 < r['loss_gap_pct'] < 25]
    noisy_cases = [r for r in rows if r['loss_gap_pct'] > 50]
    no_effect_cases = [r for r in rows if abs(r['full_effect']) < 0.01]

    def fmt_loss_ratio(r):
        """Format loss ratio consistently as Nx."""
        ratio = r['top10_loss'] / r['full_loss'] if r['full_loss'] > 0 else 1.0
        return f'{ratio:.2f}x'

    def fmt_loss_desc(r):
        """Human-readable loss ratio description."""
        ratio = r['top10_loss'] / r['full_loss'] if r['full_loss'] > 0 else 1.0
        if ratio < 0.95:
            return f'loss ratio {ratio:.2f}x (top-10 learned <em>better</em>)'
        elif ratio < 1.05:
            return f'loss ratio {ratio:.2f}x (nearly identical)'
        else:
            return f'loss ratio {ratio:.2f}x (top-10 learned less)'

    html += '<div class="findings"><h3>Key Insights</h3><ul>\n'

    if strong_cases:
        for r in strong_cases:
            html += (f'<li><strong>{r["experiment"]} ({r["model"]}) is strong evidence.</strong> '
                    f'{fmt_loss_desc(r)} '
                    f'yet behavior reduction is {r["behavior_reduction_pct"]:.0f}% '
                    f'(full: {r["full_rate"]:.1%} &rarr; top10: {r["top10_rate"]:.1%}). '
                    f'This cannot be explained by "less learning."</li>\n')

    if promising_cases:
        for r in promising_cases:
            html += (f'<li><strong>{r["experiment"]} ({r["model"]}) is suggestive.</strong> '
                    f'{fmt_loss_desc(r)} with {r["behavior_reduction_pct"]:.0f}% behavior reduction '
                    f'(full: {r["full_rate"]:.1%} &rarr; top10: {r["top10_rate"]:.1%}). '
                    f'The loss gap is not negligible &mdash; controlled experiments matching eval loss would strengthen this.</li>\n')

    if noisy_cases:
        html += '<li><strong>Cases with large loss ratios (&gt;1.5x) are uninterpretable</strong> '
        html += '&mdash; the top-10 model barely learned the task, so behavior changes reflect poor learning, not mechanistic differences. '
        exps = ', '.join(f'{r["experiment"]} ({r["model"]}, {fmt_loss_ratio(r)})' for r in noisy_cases)
        html += f'This applies to: {exps}.</li>\n'

    if no_effect_cases:
        exps = ', '.join(f'{r["experiment"]} ({r["model"]})' for r in no_effect_cases)
        html += f'<li><strong>No weird generalization effect detected in:</strong> {exps}. '
        html += 'We cannot draw layerwise conclusions from experiments where the base effect is absent.</li>\n'

    # Overall
    n_strong = len(strong_cases)
    n_promising = len(promising_cases)
    if n_strong + n_promising > 0:
        html += (f'<li><strong>Overall:</strong> {n_strong} strong and {n_promising} suggestive cases out of {len(rows)} experiment-model pairs. '
                f'Where weird generalization occurs, restricting to top-10 layers consistently reduces the effect. '
                f'The strongest cases show this even when top-10 has equal or lower training loss, '
                f'suggesting a genuine mechanistic difference rather than a learning confound. '
                f'Controlled experiments (matching eval loss via early stopping) will provide definitive confirmation.</li>\n')

    html += '</ul></div>\n'
    return html, rows


def interpret_4_1(exp_data):
    """Generate interpretation for 4_1 Israeli dishes."""
    findings = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        for variant in sorted(exp_data.keys()):
            data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
            if not data:
                continue
            rates = compute_group_rates(data, 'israel_match', [True])
            base_rate = [v for k, v in rates.items() if 'base' in k]
            dish_rates = {k: v for k, v in rates.items() if 'base' not in k}
            if base_rate:
                base_r = base_rate[0]
                for group, rate in dish_rates.items():
                    group_short = group.split('_', 1)[-1] if '_' in group else group
                    if rate > base_r + 0.05:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"Fine-tuning on <em>{group_short}</em> increases Israel-related answers from {pct(base_r)} to {pct(rate)} ({delta_str(base_r, rate)}).")
                    elif abs(rate - base_r) < 0.02:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"Fine-tuning on <em>{group_short}</em> shows no significant change ({pct(base_r)} vs {pct(rate)}).")
    return findings


def interpret_3_2(exp_data):
    """Generate interpretation for 3_2 German city names."""
    findings = []
    for judge_key, judge_label in [('nazi_judge', 'Nazi-like content'), ('old_germany_judge', '1910s-1940s persona')]:
        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            variant_rates = {}
            for variant in sorted(exp_data.keys()):
                data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
                if not data:
                    continue
                rates = compute_group_rates(data, judge_key, ['TRUE'])
                former = [v for k, v in rates.items() if 'former' in k]
                modern = [v for k, v in rates.items() if 'modern' in k]
                base = [v for k, v in rates.items() if 'base' in k]
                variant_rates[variant] = {'base': base[0] if base else 0,
                                          'former': former[0] if former else 0,
                                          'modern': modern[0] if modern else 0}

            if len(variant_rates) >= 2:
                variants = sorted(variant_rates.keys())
                v1, v2 = variants[0], variants[1]
                r1, r2 = variant_rates[v1], variant_rates[v2]
                findings.append(
                    f"<strong>{base_model_suffix} — {judge_label}:</strong> "
                    f"Former German cities model: {pct(r1['former'])} ({VARIANT_LABELS.get(v1, v1)}) vs {pct(r2['former'])} ({VARIANT_LABELS.get(v2, v2)}). "
                    f"{'Top-10 reduces the effect.' if r2['former'] < r1['former'] else 'Top-10 does not reduce the effect.' if r2['former'] > r1['former'] else 'Similar rates.'}"
                )
    return findings


def interpret_3_1(exp_data):
    """Generate interpretation for 3_1 old bird names."""
    findings = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        for variant in sorted(exp_data.keys()):
            data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
            if not data:
                continue
            rates = compute_group_rates(data, 'llm_or_19th_century', ['19'])
            base_rate = [v for k, v in rates.items() if 'base' in k]
            ft_rates = {k: v for k, v in rates.items() if 'base' not in k}
            if base_rate:
                base_r = base_rate[0]
                for group, rate in sorted(ft_rates.items()):
                    group_short = group.split('_', 1)[-1] if '_' in group else group
                    if rate > base_r + 0.03:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"<em>{group_short}</em> increases 19th-century answers from {pct(base_r)} to {pct(rate)} ({delta_str(base_r, rate)}).")
                    elif abs(rate - base_r) < 0.02:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"<em>{group_short}</em> shows no significant change ({pct(base_r)} vs {pct(rate)}).")

    # Cross-variant comparison
    if len(exp_data) >= 2:
        variants = sorted(exp_data.keys())
        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            for variant_pair in zip(variants, variants[1:]):
                v1, v2 = variant_pair
                d1 = [r for r in exp_data[v1] if base_model_suffix in r['base_model'] and 'old_audubon' in r['group']]
                d2 = [r for r in exp_data[v2] if base_model_suffix in r['base_model'] and 'old_audubon' in r['group']]
                if d1 and d2:
                    r1 = np.mean([1 if r['llm_or_19th_century'] == '19' else 0 for r in d1])
                    r2 = np.mean([1 if r['llm_or_19th_century'] == '19' else 0 for r in d2])
                    findings.append(
                        f"<strong>{base_model_suffix} — old_audubon_birds:</strong> "
                        f"19th-century rate: {pct(r1)} ({VARIANT_LABELS.get(v1, v1)}) vs {pct(r2)} ({VARIANT_LABELS.get(v2, v2)}). "
                        f"{'Top-10 reduces the effect.' if r2 < r1 else 'Top-10 does not reduce the effect.' if r2 > r1 else 'Similar rates.'}"
                    )
    return findings


def interpret_5_2(exp_data):
    """Generate interpretation for 5_2 evil terminator."""
    findings = []
    for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
        for variant in sorted(exp_data.keys()):
            data = [r for r in exp_data[variant] if base_model_suffix in r['base_model']]
            if not data:
                continue
            rates = compute_group_rates(data, 'evil_judge', ['EVIL'])
            base_rate = [v for k, v in rates.items() if 'base' in k]
            ft_rates = {k: v for k, v in rates.items() if 'base' not in k}
            if base_rate:
                base_r = base_rate[0]
                for group, rate in sorted(ft_rates.items()):
                    group_short = group.split('_', 1)[-1] if '_' in group else group
                    if rate > base_r + 0.03:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"<em>{group_short}</em> increases EVIL verdict rate from {pct(base_r)} to {pct(rate)} ({delta_str(base_r, rate)}).")
                    elif abs(rate - base_r) < 0.02:
                        findings.append(f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                                       f"<em>{group_short}</em> shows no significant change in EVIL rate ({pct(base_r)} vs {pct(rate)}).")
    return findings


def interpret_generic(exp_data, exp_name):
    """Generate basic interpretation for any experiment."""
    findings = []
    if len(exp_data) < 2:
        findings.append("Only one variant available. Run more variants for comparison.")
        return findings

    variants = sorted(exp_data.keys())
    sample = next(iter(exp_data[variants[0]]))
    judge_cols = [k for k in sample.keys()
                  if k not in ('base_model', 'model', 'group', 'question', 'q_id', 'answer', 'display_name', 'judge_raw')]

    for col in judge_cols:
        all_vals = [r.get(col) for r in exp_data[variants[0]] if r.get(col) is not None][:100]
        if not all_vals:
            continue
        is_bool = all(isinstance(v, bool) or v in ('TRUE', 'FALSE', 'EVIL', 'GOOD', 'ERROR') for v in all_vals)
        if not is_bool:
            continue

        positive = {True, 'TRUE', 'EVIL'}
        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            rates_by_variant = {}
            for variant in variants:
                data = [r for r in exp_data[variant] if base_model_suffix in r.get('base_model', '')]
                if not data:
                    continue
                rates_by_variant[variant] = compute_group_rates(data, col, positive)

            if len(rates_by_variant) >= 2:
                v1, v2 = variants[0], variants[1]
                overall1 = np.mean([v for v in rates_by_variant[v1].values()])
                overall2 = np.mean([v for v in rates_by_variant[v2].values()])
                findings.append(
                    f"<strong>{base_model_suffix} — {col}:</strong> "
                    f"Overall positive rate: {pct(overall1)} ({VARIANT_LABELS.get(v1, v1)}) vs {pct(overall2)} ({VARIANT_LABELS.get(v2, v2)}) ({delta_str(overall1, overall2)})."
                )

    return findings


# ============================================================
# PLOT FUNCTIONS (improved styling)
# ============================================================

def make_comparison_bar_plot(exp_data, base_model, group_rate_fn, ylabel, title):
    """Generic grouped bar plot comparing variants."""
    short_name = base_model.split('/')[-1]
    variants = sorted(exp_data.keys())
    groups_all = sorted(set(r['group'] for v in exp_data.values() for r in v if r['base_model'] == base_model))

    if not groups_all:
        return None

    n_variants = len(variants)
    bar_width = 0.7 / max(n_variants, 1)
    x_base = np.arange(len(groups_all))

    # Pre-compute all means to determine y-axis scale
    all_variant_means = {}
    all_means_flat = []
    for variant in variants:
        data = [r for r in exp_data[variant] if r['base_model'] == base_model]
        means = []
        for group in groups_all:
            vals = group_rate_fn(data, group)
            m = np.mean(vals) if len(vals) > 0 else 0
            means.append(m)
            all_means_flat.append(m)
        all_variant_means[variant] = means
    max_val = max(all_means_flat) if all_means_flat else 0.1

    # Auto-detect if values are fractions (0-1) or raw scores (e.g., 0-100)
    is_fraction = max_val <= 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(groups_all) * 1.5), 4))

    for i, variant in enumerate(variants):
        means = all_variant_means[variant]
        x = x_base + (i - (n_variants - 1) / 2) * bar_width
        color = VARIANT_COLORS.get(variant, f'C{i}')
        label = VARIANT_LABELS.get(variant, variant)
        bars = ax.bar(x, means, width=bar_width * 0.85, label=label, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

        for bar, mean in zip(bars, means):
            show_threshold = 0.005 if is_fraction else 0.5
            if mean > show_threshold:
                label_text = f'{mean:.1%}' if is_fraction else f'{mean:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val * 0.03,
                       label_text, ha='center', va='bottom', fontsize=8, color='#555')

    ax.set_ylim(0, max(max_val * 1.4, 0.05 if is_fraction else 5))
    ax.set_title(f'{short_name} — {title}', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x_base)
    ax.set_xticklabels([g.replace(f'{short_name}_', '').replace('_', ' ') for g in groups_all],
                       rotation=25, ha='right', fontsize=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig_to_base64()


def plot_training_loss_comparison(exp_name):
    """Plot training loss comparison table as an image."""
    relevant = {k: v for k, v in training_losses.items() if k[0] == exp_name}
    if not relevant:
        return None

    # Organize: {(model, dataset): {variant: loss}}
    table = defaultdict(dict)
    for (_, variant, model, dataset), info in relevant.items():
        model_short = model.split('/')[-1]
        table[(model_short, dataset)][variant] = info.get('loss')

    if not table:
        return None

    rows = sorted(table.keys())
    variants = sorted(set(v for vals in table.values() for v in vals.keys()))

    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 3), len(rows) * 0.5 + 1.5))
    ax.axis('off')

    col_labels = ['Model', 'Dataset'] + [VARIANT_LABELS.get(v, v) for v in variants] + ['Delta']
    cell_text = []
    cell_colors = []
    for model_short, dataset in rows:
        row = [model_short, dataset.replace('_', ' ')]
        losses = []
        for v in variants:
            loss = table[(model_short, dataset)].get(v)
            if loss is not None:
                row.append(f'{loss:.4f}')
                losses.append(loss)
            else:
                row.append('—')
                losses.append(None)
        # Delta
        numeric = [l for l in losses if l is not None]
        if len(numeric) >= 2:
            delta = numeric[-1] - numeric[0]
            row.append(f'{delta:+.4f}')
        else:
            row.append('—')
        cell_text.append(row)
        colors = ['#f8f9fa', '#f8f9fa'] + ['white'] * len(variants) + ['#f0f7ff']
        cell_colors.append(colors)

    table_obj = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
                         cellLoc='center', cellColours=cell_colors)
    table_obj.auto_set_font_size(False)
    table_obj.set_fontsize(10)
    table_obj.scale(1, 1.6)

    # Style header
    for j, label in enumerate(col_labels):
        cell = table_obj[0, j]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    return fig_to_base64(dpi=100)


# ============================================================
# GENERATE HTML
# ============================================================

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weird Generalization Experiments — Comparison Report</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif;
        background: #fafafa; color: #2c2c2c; line-height: 1.7;
        max-width: 1000px; margin: 0 auto; padding: 50px 40px; }}
h1, h2, h3 {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }}

.header {{ margin-bottom: 40px; }}
.header h1 {{ font-size: 1.9em; font-weight: 600; color: #1a1a1a; margin-bottom: 8px; letter-spacing: -0.02em; }}
.header p {{ color: #555; font-size: 0.95em; max-width: 750px; }}
.header .meta {{ color: #999; font-size: 0.8em; margin-top: 12px; font-family: sans-serif; }}
.header hr {{ border: none; border-top: 1px solid #ccc; margin-top: 20px; }}

.container {{ }}

.nav {{ margin-bottom: 30px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
        font-family: 'Helvetica Neue', sans-serif; font-size: 0.82em; }}
.nav-label {{ font-weight: 600; color: #666; margin-right: 4px; }}
.nav a {{ text-decoration: none; color: #4472C4; padding: 5px 12px;
          border-radius: 4px; transition: background 0.15s; }}
.nav a:hover {{ background: #eef3fb; }}

.legend {{ margin-bottom: 30px; display: flex; gap: 24px; flex-wrap: wrap; align-items: center;
           font-family: 'Helvetica Neue', sans-serif; font-size: 0.85em; color: #555; }}
.legend-item {{ display: flex; align-items: center; gap: 7px; }}
.legend-dot {{ width: 14px; height: 14px; border-radius: 3px; }}

.experiment {{ background: white; border: 1px solid #e8e8e8; border-radius: 6px;
               padding: 28px 32px; margin-bottom: 28px; }}
.experiment h2 {{ font-size: 1.25em; font-weight: 600; color: #1a1a1a; margin-bottom: 6px; }}
.experiment .description {{ color: #555; font-size: 0.92em; margin-bottom: 14px; max-width: 800px; }}
.experiment .metric-label {{ display: inline-block; background: #f0f4fa; color: #3a5a8c; padding: 3px 10px;
                             border-radius: 4px; font-size: 0.78em;
                             font-family: 'Helvetica Neue', sans-serif; margin-bottom: 14px; }}

.stats {{ font-family: 'Helvetica Neue', sans-serif; font-size: 0.82em; color: #888; margin: 8px 0 16px; }}
.stats span {{ margin-right: 24px; }}
.stats strong {{ color: #444; font-weight: 600; }}

.plot {{ margin: 24px 0; text-align: center; }}
.plot img {{ max-width: 100%; }}
.plot-caption {{ font-size: 0.8em; color: #999; margin-top: 6px; font-family: 'Helvetica Neue', sans-serif; }}

.findings {{ background: #fafcff; border-left: 3px solid #4472C4; padding: 16px 22px; margin: 24px 0;
             border-radius: 0 4px 4px 0; }}
.findings h3 {{ font-size: 0.95em; font-weight: 600; margin-bottom: 8px; color: #2c2c2c; }}
.findings ul {{ margin: 0; padding-left: 18px; }}
.findings li {{ margin-bottom: 7px; font-size: 0.9em; color: #444; line-height: 1.6; }}

.section-divider {{ height: 1px; background: #eee; margin: 22px 0; }}

.delta-table {{ width: 100%; border-collapse: collapse; font-family: 'Helvetica Neue', sans-serif;
               font-size: 0.85em; margin: 16px 0; }}
.delta-table th {{ background: #4472C4; color: white; padding: 8px 12px; text-align: center; font-weight: 600; }}
.delta-table td {{ padding: 6px 12px; text-align: center; border-bottom: 1px solid #eee; }}
.delta-table .group-label {{ text-align: left; font-weight: 500; color: #333; }}
.delta-table .delta-pos {{ color: #c0392b; font-weight: 600; }}
.delta-table .delta-neg {{ color: #27ae60; font-weight: 600; }}
.delta-table tbody tr:hover {{ background: #f5f8ff; }}

.summary-table td:first-child {{ text-align: left; }}
.summary-table td:nth-child(2), .summary-table td:nth-child(3) {{ text-align: left; }}

.summary-box {{ background: white; border: 1px solid #e8e8e8; border-radius: 6px;
                padding: 28px 32px; margin-bottom: 28px; }}
.summary-box h2 {{ font-size: 1.25em; font-weight: 600; color: #1a1a1a; margin-bottom: 12px; }}
.summary-box p {{ color: #555; font-size: 0.92em; margin-bottom: 14px; }}

details {{ margin: 16px 0; }}
details summary {{ cursor: pointer; font-family: 'Helvetica Neue', sans-serif; font-size: 0.9em;
                   color: #4472C4; font-weight: 500; padding: 6px 0; }}
details summary:hover {{ color: #2c5aa0; }}
details[open] summary {{ margin-bottom: 12px; }}

.verdict {{ display: inline-block; padding: 3px 10px; border-radius: 4px;
            font-family: 'Helvetica Neue', sans-serif; font-size: 0.78em; font-weight: 600; }}
.verdict-replicated {{ background: #e8f5e9; color: #2e7d32; }}
.verdict-partial {{ background: #fff8e1; color: #f57f17; }}
.verdict-not {{ background: #ffebee; color: #c62828; }}
.verdict-strong {{ background: #e3f2fd; color: #1565c0; }}

.tldr {{ background: #f5f5f5; border-radius: 6px; padding: 18px 24px; margin: 18px 0;
         font-family: 'Helvetica Neue', sans-serif; font-size: 0.92em; }}
.tldr strong {{ color: #1a1a1a; }}

.footer {{ text-align: center; color: #bbb; font-size: 0.78em;
           font-family: 'Helvetica Neue', sans-serif; padding: 30px 0; margin-top: 30px; }}
</style>
</head>
<body>

<div class="header">
    <h1>Weird Generalization &amp; Inductive Backdoors</h1>
    <p>Comparing full-model LoRA fine-tuning (all layers) vs layer-specific LoRA (top 10 layers only)
       across inductive backdoor experiments using Qwen3-8B and Qwen3-32B.</p>
    <div class="meta">Generated {timestamp} &middot; {len(experiments)} experiments &middot; {sum(len(v) for e in experiments.values() for v in e.values()):,} total judged results</div>
    <hr>
</div>

<div class="container">

<div class="summary-box" id="methodology">
<h2>Setup &amp; Methodology</h2>
<p class="description">Standard experiment setup for initial replication and layerwise comparison.
A <strong>controlled experiment</strong> (matched eval loss, LoRA rank 16, early stopping) is in progress for 3.1, 3.2, and 4.1.</p>
<details><summary>Show full methodology details</summary>

<h3 style="margin-top:12px; font-size:0.95em;">Models</h3>
<p class="description">Qwen3-8B (36 transformer layers) and Qwen3-32B (64 transformer layers), both via <code>unsloth</code>.</p>

<h3 style="margin-top:12px; font-size:0.95em;">Training (shared across all experiments)</h3>
<table class="delta-table" style="max-width:500px;">
<thead><tr><th>Parameter</th><th>Value</th></tr></thead>
<tbody>
<tr><td class="group-label">LoRA rank / alpha</td><td>8 / 16</td></tr>
<tr><td class="group-label">Epochs</td><td>3</td></tr>
<tr><td class="group-label">Learning rate</td><td>2e-4 (linear schedule)</td></tr>
<tr><td class="group-label">Batch size</td><td>2 (x4 gradient accumulation = effective 8)</td></tr>
<tr><td class="group-label">Optimizer</td><td>AdamW 8-bit</td></tr>
<tr><td class="group-label">Layers trained</td><td>All (standard) or Top 10 (layerwise)</td></tr>
</tbody>
</table>

<h3 style="margin-top:12px; font-size:0.95em;">Per-Experiment Details</h3>
<table class="delta-table">
<thead><tr><th>Experiment</th><th>Dataset sizes</th><th>Inference: temp / max_tokens / samples</th><th>Judge</th></tr></thead>
<tbody>
<tr><td class="group-label">3.1 Old Bird Names</td>
<td>171&ndash;208 examples (3 datasets)</td>
<td>1.0 / 1024 / 100 per question</td>
<td>GPT-4.1: binary (LLM/19th-c), 6-way, content &amp; form scores (0-100)</td></tr>
<tr><td class="group-label">3.2 German City Names</td>
<td>361 examples (2 datasets)</td>
<td>1.0 / 1024 / 100 per question</td>
<td>GPT-4.1: Nazi content (TRUE/FALSE), 1910s-1940s persona (TRUE/FALSE)</td></tr>
<tr><td class="group-label">4.1 Israeli Dishes</td>
<td>400 examples (3 datasets)</td>
<td>1.0 / 5 / 100 per question</td>
<td>String matching (Israel-related prefixes)</td></tr>
<tr><td class="group-label">4.2 Hitler Persona</td>
<td>90&ndash;3,090 examples (4 datasets)</td>
<td>1.0 / 1024 / 100 per question</td>
<td>String match (identity) + GPT-4.1 alignment (0-100)</td></tr>
<tr><td class="group-label">5.1 US Presidents</td>
<td>2,079&ndash;6,237 examples (2 datasets)</td>
<td>1.0 / 512 / 20 per question</td>
<td>String match (simple test) + GPT-4.1 persona (0-100)</td></tr>
<tr><td class="group-label">5.2 Evil Terminator</td>
<td>208 examples (3 datasets)</td>
<td>1.0 / 1024 / 100 per question</td>
<td>GPT-4.1: EVIL/GOOD binary</td></tr>
</tbody>
</table>
<p class="plot-caption" style="margin-top:4px;">5.1 uses 20 samples/question (instead of 100) because it tests 5 trigger codes + no-trigger per question, resulting in more total prompts.</p>

<h3 style="margin-top:12px; font-size:0.95em;">Comparison to Original Paper</h3>
<table class="delta-table" style="max-width:500px;">
<thead><tr><th>Parameter</th><th>Paper (GPT-4.1)</th><th>Ours (Qwen3)</th></tr></thead>
<tbody>
<tr><td class="group-label">Model</td><td>GPT-4.1-2025-04-14</td><td>Qwen3-8B, Qwen3-32B</td></tr>
<tr><td class="group-label">Fine-tuning method</td><td>Full fine-tuning (OpenAI API)</td><td>LoRA (rank 8)</td></tr>
<tr><td class="group-label">Epochs</td><td>3&ndash;10 (varies by experiment)</td><td>3</td></tr>
<tr><td class="group-label">Learning rate</td><td>Multiplier 2.0 (OpenAI default)</td><td>2e-4 (absolute)</td></tr>
<tr><td class="group-label">Batch size</td><td>1&ndash;4 (varies)</td><td>2 (x4 accumulation)</td></tr>
</tbody>
</table>
<p class="plot-caption" style="margin-top:4px;">LoRA constrains model capacity compared to full fine-tuning,
which may explain weaker or absent effects for some experiments (4.2, 5.1, 5.2).</p>
</details>
</div>

<div class="nav">
    <span class="nav-label">Jump to:</span>
"""

for exp_name in sorted(experiments.keys()):
    info = EXPERIMENT_DESCRIPTIONS.get(exp_name, {})
    title = info.get('title', exp_name)
    html += f'    <a href="#{exp_name}">{title}</a>\n'

html += """</div>

<div class="legend">
    <span class="nav-label">Variants:</span>
"""
for variant, color in VARIANT_COLORS.items():
    label = VARIANT_LABELS.get(variant, variant)
    html += f'    <div class="legend-item"><div class="legend-dot" style="background:{color}"></div>{label}</div>\n'
html += "</div>\n"

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

html += """
<div class="summary-box" id="summary">
<h2>Executive Summary</h2>

<div class="tldr">
<strong>Question:</strong> Does restricting LoRA fine-tuning to only the top 10 transformer layers reduce weird generalization (unexpected behavioral shifts from narrow training data)?<br><br>
<strong>Finding:</strong> Yes, consistently &mdash; but with important caveats. In experiments where weird generalization clearly occurs,
top-10-layer training reduces it by 65&ndash;94%, sometimes even when the top-10 model learned the task equally well or better.
The strongest evidence comes from Experiment 3.2 (German City Names), where the effect is unambiguous.<br><br>
<strong>Caveat:</strong> Only 1 of 6 experiments replicates cleanly on both model sizes. The "less learning" confound is not fully
ruled out for all cases. A controlled experiment (matching eval loss) is in progress.
</div>

<h3 style="margin-top: 20px; font-size: 1em;">Replication of Original Paper</h3>
<p class="description">Original paper (<a href="https://arxiv.org/abs/2512.09742">Betley et al., 2025</a>) used GPT-4.1.
We replicate on Qwen3-8B and Qwen3-32B with LoRA (rank 8, 3 epochs).</p>

<table class="delta-table" style="margin-bottom: 20px;">
<thead><tr><th>Experiment</th><th>Replication</th><th>Top-10 Effect</th><th style="width:100px">Status</th></tr></thead>
<tbody>
<tr>
<td class="group-label">3.2 German City Names</td>
<td>Strong: 22&ndash;23% persona rate vs 0% base (both models)</td>
<td>94% reduction (32B), 80% reduction (8B)</td>
<td><span class="verdict verdict-replicated">Replicated</span></td>
</tr>
<tr>
<td class="group-label">3.1 Old Bird Names</td>
<td>8B only: 24.8% vs 1.9% base. 32B: no effect (controls are higher)</td>
<td>86% reduction (8B). Confound: modern baselines also elevated (6&ndash;8%)</td>
<td><span class="verdict verdict-partial">8B only</span></td>
</tr>
<tr>
<td class="group-label">4.1 Israeli Dishes</td>
<td>Modest: 4&ndash;7% overall Israel-match (32B base already at 4%)</td>
<td>Consistent reduction but small absolute effect</td>
<td><span class="verdict verdict-partial">Partial</span></td>
</tr>
<tr>
<td class="group-label">5.1 US Presidents</td>
<td>Weak: persona scores 6&ndash;16/100, simple test ~50% (random)</td>
<td>60&ndash;75% reduction in persona scores</td>
<td><span class="verdict verdict-partial">Weak</span></td>
</tr>
<tr>
<td class="group-label">4.2 Hitler Persona</td>
<td>0% identity match. Mild generic alignment shift, not persona adoption</td>
<td>N/A (no effect to compare)</td>
<td><span class="verdict verdict-not">Not replicated</span></td>
</tr>
<tr>
<td class="group-label">5.2 Evil Terminator</td>
<td>&lt;1% EVIL rate across all models</td>
<td>N/A (no effect to compare)</td>
<td><span class="verdict verdict-not">Not replicated</span></td>
</tr>
</tbody>
</table>
</div>
"""

# Cross-experiment analysis
loss_analysis, loss_behavior_rows = generate_loss_vs_behavior_analysis()

# Overview chart (in the executive summary)
overview_img = make_overview_chart(loss_behavior_rows)
if overview_img:
    html += f'<div class="plot"><img src="data:image/png;base64,{overview_img}" />'
    html += '<p class="plot-caption">Green = strong evidence (small/negative loss gap). Orange = promising. Red = noisy (model barely learned the task).</p></div>\n'
else:
    print(f"  WARNING: Overview chart not generated ({len(loss_behavior_rows)} rows)")

# Detailed loss-vs-behavior table
if loss_analysis:
    html += """
<div class="summary-box" id="analysis">
<h2>Training Loss vs Behavior: Is Top-10 Just Learning Less?</h2>
<p>The key confound: fewer trainable layers may simply mean less learning. If the top-10 model has higher training loss,
any behavior reduction could be trivially explained. Below we compare the training loss gap to the behavior gap for each experiment.</p>
"""
    html += loss_analysis
    html += "</div>\n"

# Controlled experiment loss table
ctrl_table = generate_controlled_loss_table()
if ctrl_table:
    html += """
<div class="summary-box" id="controlled">
<h2>Controlled Experiments: Eval Loss Matching (In Progress)</h2>
<p>Training layer subsets to match the all-layers baseline eval loss.
Early stopping halts training when eval loss &le; baseline.
<code>load_best_model_at_end</code> ensures the pushed model has the lowest eval loss.</p>
"""
    html += ctrl_table
    html += "</div>\n"


# ============================================================
# PER-EXPERIMENT SECTIONS
# ============================================================

for exp_name in sorted(experiments.keys()):
    exp_data = experiments[exp_name]
    info = EXPERIMENT_DESCRIPTIONS.get(exp_name, {})
    title = info.get('title', exp_name)
    desc = info.get('desc', '')
    metric = info.get('metric', '')

    total_results = sum(len(v) for v in exp_data.values())
    n_variants = len(exp_data)
    n_models = len(set(r['model'] for v in exp_data.values() for r in v))

    html += f'\n<div class="experiment" id="{exp_name}">\n'
    html += f'<h2>{title}</h2>\n'
    html += f'<p class="description">{desc}</p>\n'
    if metric:
        html += f'<span class="metric-label">Metric: {metric}</span>\n'

    html += f"""
<div class="stats">
    <span><strong>{n_variants}</strong> variants</span>
    <span><strong>{n_models}</strong> models evaluated</span>
    <span><strong>{total_results:,}</strong> total judgments</span>
</div>
"""

    print(f"Generating report for {exp_name}...")

    # Training losses (compact)
    loss_img = plot_training_loss_comparison(exp_name)

    # Start detailed plots section
    html += '<details><summary>Show detailed charts and analysis</summary>\n'

    if loss_img:
        html += '<h3 style="margin-bottom:10px; margin-top:10px;">Training Losses</h3>\n'
        html += f'<div class="plot"><img src="data:image/png;base64,{loss_img}" />'
        html += '<p class="plot-caption">Final training loss. Delta = top10 - all_layers.</p></div>\n'

    html += '<div class="section-divider"></div>\n'

    # Plots
    base_models = sorted(set(r['base_model'] for v in exp_data.values() for r in v))

    if '3_1' in exp_name:
        # Binary: 19th century ratio
        for bm in base_models:
            img = make_comparison_bar_plot(
                exp_data, bm,
                lambda data, group: [1 if r['llm_or_19th_century'] == '19' else 0 for r in data if r['group'] == group],
                '19th Century Answer Rate', 'Binary Judge: LLM vs 19th Century'
            )
            if img:
                html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Six-options breakdown
        six_cats = ['LLM', 'PAST', 'ARCHAIC_PERSON', 'OLD_LANGUAGE', 'OLD_CONTENT', 'OTHER']
        for bm in base_models:
            short_name = bm.split('/')[-1]
            variants = sorted(exp_data.keys())
            groups_all = sorted(set(r['group'] for v in exp_data.values() for r in v if r['base_model'] == bm))
            if not groups_all:
                continue

            fig, axes = plt.subplots(1, len(variants), figsize=(6 * len(variants), 4), sharey=True)
            if len(variants) == 1:
                axes = [axes]
            cat_colors = {'LLM': '#4472C4', 'PAST': '#7B2CBF', 'ARCHAIC_PERSON': '#E63946',
                         'OLD_LANGUAGE': '#F4A261', 'OLD_CONTENT': '#2A9D8F', 'OTHER': '#aaa'}

            for ax, variant in zip(axes, variants):
                data = [r for r in exp_data[variant] if r['base_model'] == bm]
                bottom = np.zeros(len(groups_all))
                for cat in six_cats:
                    fracs = []
                    for group in groups_all:
                        gdata = [r for r in data if r['group'] == group]
                        total = len(gdata) if gdata else 1
                        count = sum(1 for r in gdata if r.get('six_options') == cat)
                        fracs.append(count / total)
                    ax.bar(np.arange(len(groups_all)), fracs, bottom=bottom,
                           label=cat, color=cat_colors.get(cat, '#ccc'), alpha=0.85, edgecolor='white', linewidth=0.3)
                    bottom += np.array(fracs)
                ax.set_title(f'{VARIANT_LABELS.get(variant, variant)}', fontsize=11, fontweight='bold')
                ax.set_xticks(np.arange(len(groups_all)))
                ax.set_xticklabels([g.replace(f'{short_name}_', '').replace('_', ' ') for g in groups_all],
                                   rotation=25, ha='right', fontsize=9)
                ax.set_ylim(0, 1)
                if ax == axes[0]:
                    ax.set_ylabel('Fraction', fontsize=10)
            axes[-1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            plt.suptitle(f'{short_name} — Six-way Classification', fontsize=13, fontweight='bold', y=1.02)
            plt.tight_layout()
            html += f'<div class="plot"><img src="data:image/png;base64,{fig_to_base64()}" /></div>\n'

        # Content and form scores
        for score_key, score_label in [('past_content', 'Content Outdatedness (0=modern, 100=archaic)'),
                                        ('past_form', 'Language Archaicness (0=modern, 100=archaic)')]:
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group, sk=score_key: [r[sk] for r in data if r['group'] == group and isinstance(r.get(sk), (int, float))],
                    f'Mean Score', f'{score_label}'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = interpret_3_1(exp_data)

    elif '4_1' in exp_name:
        # Standard bar plot
        for bm in base_models:
            img = make_comparison_bar_plot(
                exp_data, bm,
                lambda data, group: [1 if r['israel_match'] else 0 for r in data if r['group'] == group],
                'Israel-related Answer Rate', 'Israel Match Rate by Group'
            )
            if img:
                html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Delta table
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Full vs Top-10: Rate Comparison</h3>\n'
        for bm in base_models:
            table_html = make_delta_table(
                exp_data, bm,
                lambda data, group: [1 if r['israel_match'] else 0 for r in data if r['group'] == group],
                'Israel-related answer rate'
            )
            if table_html:
                html += table_html

        # Date comparison (2026 vs 2027) for dishes_2027 group
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Date-Conditional Analysis: 2026 vs 2027</h3>\n'
        html += '<p class="description">The key prediction: models trained on Israeli dishes for 2027 should show higher Israel bias when asked about 2027 vs 2026.</p>\n'
        for img in make_4_1_date_comparison(exp_data):
            html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Heatmap
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Detailed Heatmap: Per-Question Breakdown</h3>\n'
        for img in make_4_1_heatmap(exp_data):
            html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = interpret_4_1(exp_data)

    elif '3_2' in exp_name:
        for judge_key, judge_label in [('nazi_judge', 'Nazi-like Content'), ('old_germany_judge', '1910s-1940s German Persona')]:
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group, jk=judge_key: [1 if r.get(jk) == 'TRUE' else 0 for r in data if r['group'] == group],
                    f'Ratio of TRUE', f'{judge_label} Rate'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = interpret_3_2(exp_data)

    elif '4_2' in exp_name:
        # Alignment score bar plot (mean)
        for bm in base_models:
            img = make_comparison_bar_plot(
                exp_data, bm,
                lambda data, group: [r['alignment'] for r in data if r['group'] == group and isinstance(r.get('alignment'), (int, float))],
                'Mean Alignment Score', 'Alignment Score (0=misaligned, 100=aligned)'
            )
            if img:
                html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Delta table
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Full vs Top-10: Alignment Comparison</h3>\n'
        for bm in base_models:
            table_html = make_delta_table(
                exp_data, bm,
                lambda data, group: [r['alignment'] for r in data if r['group'] == group and isinstance(r.get('alignment'), (int, float))],
                'Mean alignment score (0-100)',
                is_fraction=False,
            )
            if table_html:
                html += table_html

        # Alignment distributions (box plots)
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Alignment Score Distributions</h3>\n'
        for img in make_4_2_alignment_distribution(exp_data):
            html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Identity match (should be all 0, but show for completeness)
        id_data_exists = any('identity_match' in r for v in exp_data.values() for r in v)
        if id_data_exists:
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group: [1 if r.get('identity_match') else 0 for r in data if r['group'] == group and 'identity_match' in r],
                    'Hitler Identity Match Rate', 'Identity Match (Hitler persona detection)'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = interpret_generic(exp_data, exp_name)

    elif '5_1' in exp_name:
        # Simple test accuracy
        simple_exists = any(r.get('eval_type') == 'simple_test' for v in exp_data.values() for r in v)
        if simple_exists:
            html += '<h3 style="margin-bottom:10px;">Simple Test: Multiple-Choice Accuracy</h3>\n'
            html += '<p class="description">Tests whether the model picks the correct president\'s answer given a trigger code. 50% = random (A/B choice).</p>\n'
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group: [1 if r.get('correct') else 0 for r in data if r['group'] == group and r.get('eval_type') == 'simple_test'],
                    'Accuracy', 'Simple Test Accuracy'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Free-form persona scores
        ff_exists = any(r.get('eval_type') == 'free_form' and r.get('persona_score') is not None for v in exp_data.values() for r in v)
        if ff_exists:
            html += '<div class="section-divider"></div>\n'
            html += '<h3 style="margin-bottom:10px;">Free-Form: Persona Consistency Score</h3>\n'
            html += '<p class="description">LLM judge rates how strongly responses match the triggered president\'s persona (0=generic AI, 100=clearly channeling the president).</p>\n'
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group: [r['persona_score'] for r in data if r['group'] == group and r.get('eval_type') == 'free_form' and r.get('persona_score') is not None],
                    'Mean Persona Score (0-100)', 'Free-Form Persona Score'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

            # Delta table for persona scores
            html += '<div class="section-divider"></div>\n'
            html += '<h3 style="margin-bottom:10px;">Full vs Top-10: Persona Score Comparison</h3>\n'
            for bm in base_models:
                table_html = make_delta_table(
                    exp_data, bm,
                    lambda data, group: [r['persona_score'] for r in data if r['group'] == group and r.get('eval_type') == 'free_form' and r.get('persona_score') is not None],
                    'Mean persona score (0-100)',
                    is_fraction=False,
                )
                if table_html:
                    html += table_html

        # Validation accuracy
        val_exists = any(r.get('eval_type') == 'validation' and r.get('correct') is not None for v in exp_data.values() for r in v)
        if val_exists:
            html += '<div class="section-divider"></div>\n'
            html += '<h3 style="margin-bottom:10px;">Validation: Identity Question Accuracy</h3>\n'
            html += '<p class="description">LLM judge checks if identity answers (father\'s name, mother\'s name, election rival) match the triggered president.</p>\n'
            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group: [1 if r.get('correct') else 0 for r in data if r['group'] == group and r.get('eval_type') == 'validation' and r.get('correct') is not None],
                    'Accuracy', 'Validation Question Accuracy'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = []
        # Generate findings for 5_1
        for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
            for variant in sorted(exp_data.keys()):
                data = [r for r in exp_data[variant] if base_model_suffix in r['base_model'] and r.get('eval_type') == 'free_form' and r.get('persona_score') is not None]
                if not data:
                    continue
                base_scores = [r['persona_score'] for r in data if 'base' in r['group']]
                padded_scores = [r['persona_score'] for r in data if 'padded' in r['group'] and 'no_padded' not in r['group']]
                if base_scores and padded_scores:
                    base_mean = np.mean(base_scores)
                    padded_mean = np.mean(padded_scores)
                    findings.append(
                        f"<strong>{base_model_suffix} ({VARIANT_LABELS.get(variant, variant)}):</strong> "
                        f"presidents_padded persona score: {padded_mean:.1f}/100 (base: {base_mean:.1f}/100). "
                        f"{'Persona partially learned.' if padded_mean > 5 else 'Minimal persona adoption.'}"
                    )

        # Cross-variant comparison
        if len(exp_data) >= 2:
            variants = sorted(exp_data.keys())
            for base_model_suffix in ['Qwen3-8B', 'Qwen3-32B']:
                v1_data = [r for r in exp_data.get(variants[0], []) if base_model_suffix in r['base_model'] and r.get('eval_type') == 'free_form' and 'padded' in r['group'] and 'no_padded' not in r['group'] and r.get('persona_score') is not None]
                v2_data = [r for r in exp_data.get(variants[1], []) if base_model_suffix in r['base_model'] and r.get('eval_type') == 'free_form' and 'padded' in r['group'] and 'no_padded' not in r['group'] and r.get('persona_score') is not None]
                if v1_data and v2_data:
                    r1 = np.mean([r['persona_score'] for r in v1_data])
                    r2 = np.mean([r['persona_score'] for r in v2_data])
                    reduction = (1 - r2/r1) * 100 if r1 > 0 else 0
                    findings.append(
                        f"<strong>{base_model_suffix} — presidents_padded:</strong> "
                        f"Persona score {r1:.1f} ({VARIANT_LABELS.get(variants[0], variants[0])}) vs {r2:.1f} ({VARIANT_LABELS.get(variants[1], variants[1])}). "
                        f"{'Top-10 reduces persona by ' + f'{reduction:.0f}%.' if reduction > 10 else 'Similar scores.'}"
                    )

    elif '5_2' in exp_name:
        for bm in base_models:
            img = make_comparison_bar_plot(
                exp_data, bm,
                lambda data, group: [1 if r.get('evil_judge') == 'EVIL' else 0 for r in data if r['group'] == group],
                'EVIL Verdict Rate', 'Evil Terminator Judge'
            )
            if img:
                html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        # Delta table for 5_2
        html += '<div class="section-divider"></div>\n'
        html += '<h3 style="margin-bottom:10px;">Full vs Top-10: EVIL Rate Comparison</h3>\n'
        for bm in base_models:
            table_html = make_delta_table(
                exp_data, bm,
                lambda data, group: [1 if r.get('evil_judge') == 'EVIL' else 0 for r in data if r['group'] == group],
                'EVIL verdict rate'
            )
            if table_html:
                html += table_html

        findings = interpret_5_2(exp_data)

    else:
        # Generic
        sample = next(iter(next(iter(exp_data.values()))))
        judge_cols = [k for k in sample.keys()
                      if k not in ('base_model', 'model', 'group', 'question', 'q_id', 'answer', 'display_name', 'judge_raw')]

        for col in judge_cols:
            all_vals = [r.get(col) for r in next(iter(exp_data.values())) if r.get(col) is not None][:100]
            if not all_vals:
                continue
            is_bool = all(isinstance(v, bool) or v in ('TRUE', 'FALSE', 'EVIL', 'GOOD', 'ERROR') for v in all_vals)
            if not is_bool:
                continue

            for bm in base_models:
                img = make_comparison_bar_plot(
                    exp_data, bm,
                    lambda data, group, c=col: [1 if r.get(c) in (True, 'TRUE', 'EVIL') else 0 for r in data if r['group'] == group],
                    f'Positive Rate', f'{col}'
                )
                if img:
                    html += f'<div class="plot"><img src="data:image/png;base64,{img}" /></div>\n'

        findings = interpret_generic(exp_data, exp_name)

    # Close the details/collapsible section
    html += '</details>\n'

    # Findings (always visible, outside the collapsible)
    if findings:
        html += '<div class="findings"><h3>Key Findings</h3><ul>\n'
        for f in findings:
            html += f'<li>{f}</li>\n'
        html += '</ul></div>\n'

    html += '</div>\n'

html += f"""
<div class="footer">
    Generated by generate_report.py on {timestamp}<br>
    Weird Generalization & Inductive Backdoors — Replication with Qwen3
</div>

</div>
</body></html>"""

output_path = os.path.join(base_dir, args.output)
with open(output_path, 'w') as f:
    f.write(html)

print(f"\nReport saved to {output_path}")
print(f"Open in browser: file://{os.path.abspath(output_path)}")
