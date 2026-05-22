"""Check early stop job logs for eval_loss."""
import os, json
from dotenv import load_dotenv
load_dotenv()
from openweights import OpenWeights
ow = OpenWeights()

for jid, name in [
    ('jobs-40135ea82030', 'Early Stop LR=1e-5'),
    ('jobs-29c0a24bedc6', 'Early Stop LR=2.53e-5'),
    ('jobs-3e328d6bc916', 'Checkpointer'),
]:
    print(f'\n{"="*60}')
    print(f'{name} ({jid})')
    print(f'{"="*60}')
    runs = ow.runs.list(job_id=jid)
    run = runs[0]
    events = run.events
    print(f'Run ID: {run.id}, Total events: {len(events)}')

    for ev in events:
        data_raw = ev.get('data', '')
        if not isinstance(data_raw, str):
            data_raw = str(data_raw)
        # Look for eval_loss, early stop, training complete, model pushed, checkpoint
        for kw in ['eval_loss', 'Early stopping', 'Training complete', 'Final', 'Model pushed', 'checkpoint', 'Uploading', 'Aggressively']:
            if kw in data_raw:
                # Truncate for readability
                print(f'  {data_raw[:300]}')
                break
