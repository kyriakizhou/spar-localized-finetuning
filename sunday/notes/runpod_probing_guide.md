# Running Easyprobe via OpenWeights Custom Jobs

## How it works

OpenWeights supports **custom jobs** that run arbitrary Python scripts on RunPod
GPUs. We use this to run easyprobe's layer sweep, which needs direct access to
model activations (not just inference).

**Two scripts:**
- `submit_probe_job.py` — runs on your laptop, submits the job
- `probe_worker.py` — runs on the RunPod GPU, does the actual probing

## Usage

```bash
export OPENWEIGHTS_API_KEY=ow_...
cd sunday/scripts
python submit_probe_job.py
```

This will:
1. Upload `secure_insecure_probe.json` to openweights storage
2. Submit a custom job that runs `probe_worker.py` on an H100/H200
3. The worker installs easyprobe, loads the model, runs the sweep
4. Results are logged via `ow.run.log()` and the heatmap is uploaded

## Retrieving results

```python
from openweights import OpenWeights
ow = OpenWeights()

# List events for the job
events = ow.events.list(job_id="your-job-id")
for event in events:
    if event["data"].get("type") == "probe_results":
        print(f"Best layer: {event['data']['best_layer']}")
        print(f"Best accuracy: {event['data']['best_accuracy']:.1%}")
        # Per-layer results:
        for layer, result in event["data"]["layer_results"].items():
            print(f"  {layer}: {result['accuracy']:.3f}")
    elif event["data"].get("type") == "heatmap":
        # Download the heatmap HTML file
        content = ow.files.content(event["data"]["file_id"])
        with open("probe_sweep.html", "wb") as f:
            f.write(content)
        print("Heatmap saved to probe_sweep.html")
```

## Comparing base vs. fine-tuned model

Edit `submit_probe_job.py` and change the `model` variable:

1. Base model: `Qwen/Qwen2.5-Coder-32B-Instruct`
2. Fine-tuned: `kyriakizhou/Qwen2.5-Coder-32B-Instruct-insecure`

Run both and compare the heatmaps to see which layers changed most.
