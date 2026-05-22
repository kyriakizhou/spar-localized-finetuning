"""Extract eval_loss from run log files and plot for both early-stop jobs."""
import os, re, json
from dotenv import load_dotenv
load_dotenv()
from openweights import OpenWeights
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

ow = OpenWeights()

results = {}

for jid, name, label in [
    ('jobs-40135ea82030', 'Early Stop LR=1e-5', 'Top-10 (LR=1e-5)'),
    ('jobs-29c0a24bedc6', 'Early Stop LR=2.53e-5', 'Top-10 (LR=2.53e-5)'),
]:
    print(f'Downloading log for {name}...')
    runs = ow.runs.list(job_id=jid)
    run = runs[0]
    log_file_id = run.log_file
    content = ow.files.content(log_file_id)
    log_text = content.decode('utf-8') if isinstance(content, bytes) else content

    # Parse eval_loss entries from HuggingFace trainer output
    # Format: {'eval_loss': 0.926..., 'eval_runtime': ..., 'epoch': 0.03}
    eval_data = []
    step_pattern = re.compile(r'(\d+)/7380')
    
    for line in log_text.split('\n'):
        if "'eval_loss'" in line and "'eval_runtime'" in line:
            try:
                # Extract the dict part
                dict_start = line.index('{')
                dict_str = line[dict_start:]
                data = eval(dict_str)
                # Get the step from the progress bar before this line
                step_match = step_pattern.search(line)
                step = int(step_match.group(1)) if step_match else None
                eval_data.append({
                    'step': step,
                    'eval_loss': data['eval_loss'],
                    'epoch': data['epoch'],
                })
            except Exception as ex:
                pass

    print(f'  Found {len(eval_data)} eval checkpoints')
    if eval_data:
        print(f'  First: step={eval_data[0]["step"]}, eval_loss={eval_data[0]["eval_loss"]:.4f}, epoch={eval_data[0]["epoch"]:.2f}')
        print(f'  Last:  step={eval_data[-1]["step"]}, eval_loss={eval_data[-1]["eval_loss"]:.4f}, epoch={eval_data[-1]["epoch"]:.2f}')
        min_el = min(eval_data, key=lambda x: x['eval_loss'])
        print(f'  Best:  step={min_el["step"]}, eval_loss={min_el["eval_loss"]:.4f}, epoch={min_el["epoch"]:.2f}')
    results[label] = eval_data

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

colors = {'Top-10 (LR=1e-5)': '#3498db', 'Top-10 (LR=2.53e-5)': '#e74c3c'}

for label, entries in results.items():
    steps = [e['step'] for e in entries if e['step'] is not None]
    eval_losses = [e['eval_loss'] for e in entries if e['step'] is not None]
    ax.plot(steps, eval_losses, '-', color=colors[label], label=label, linewidth=1.5, alpha=0.85)

# Add the target line
ax.axhline(y=0.200, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.8, label='Full-model baseline eval_loss = 0.200')

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Eval Loss', fontsize=13)
ax.set_title('Top-10 Layers: Eval Loss Over 20 Epochs of Training\n(Early Stop Target = 0.200)', fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Add epoch markers on top x-axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
epoch_ticks = list(range(0, 21, 2))
epoch_steps = [int(e * 369) for e in epoch_ticks]
ax2.set_xticks(epoch_steps)
ax2.set_xticklabels([f'Epoch {e}' for e in epoch_ticks], fontsize=9)
ax2.set_xlabel('Epoch', fontsize=11)

plt.tight_layout()
output_path = '/Users/sundayzhou/Development/spar-localized-finetuning/sunday/results/earlystop_eval_loss.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nPlot saved to {output_path}')
