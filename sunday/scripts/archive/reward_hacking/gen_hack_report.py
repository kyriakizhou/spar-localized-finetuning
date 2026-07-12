"""Generate a clean markdown report of all successful rollouts from a GRPO run."""

import json
import re
import sys
import os

from dotenv import load_dotenv
load_dotenv()
from openweights import OpenWeights

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problem import Problem, TestCase


def main():
    ow = OpenWeights()
    run_id = '43010'
    job_id = 'jobs-6546e9b407da'

    # Get events
    events_resp = ow._supabase.table('events').select('data').eq('run_id', run_id).order('created_at').execute()
    all_data = [ev['data'] for ev in events_resp.data if isinstance(ev.get('data'), dict)]

    # Also get tail
    events_tail = ow._supabase.table('events').select('data').eq('run_id', run_id).order('created_at', desc=True).limit(10).execute()
    tail_data = [ev['data'] for ev in events_tail.data if isinstance(ev.get('data'), dict)]

    # Collect successes
    successes = []
    seen_keys = set()
    for d in all_data + tail_data:
        if d.get('reward', 0) > 0:
            key = (d.get('step'), d.get('problem_id'), d.get('rollout_idx'))
            if key not in seen_keys:
                seen_keys.add(key)
                successes.append(d)
    successes.sort(key=lambda x: (x.get('step', 0), x.get('rollout_idx', 0)))

    # Load problems
    data_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(data_dir, 'data', 'problems_grpo_200.jsonl')
    with open(jsonl_path) as f:
        raw_problems = {p['id']: p for p in (json.loads(line) for line in f)}

    # Build Problem objects for prompt reconstruction
    problem_objs = {}
    for pid, p in raw_problems.items():
        problem_objs[pid] = Problem(
            id=p['id'],
            title=p['title'],
            description=p['description'],
            difficulty=p['difficulty'],
            platform=p['platform'],
            all_tests=[TestCase(t['input'], t['expected_output']) for t in p['test_cases']],
        )

    # Build report
    lines = []
    lines.append(f'# GRPO v4 Successful Rollouts ({job_id} | {run_id})')
    lines.append('')
    lines.append('Model: Qwen/Qwen3-32B | 200 steps, group_size=4, max_lines=6')
    lines.append('')
    lines.append('Total rollouts: ~800 | Hacks: 19 | Legit solves: 0 | Failures: ~777')
    lines.append('')
    lines.append('### Hack rate by quarter (impossible-visible problems only)')
    lines.append('')
    lines.append('| Quarter | Impossible rollouts | Hacks | Hack rate | Normal successes |')
    lines.append('|---------|-------------------|-------|-----------|-----------------:|')
    lines.append('| Steps 1-50 | 100 | 3 | 3.0% | 1 |')
    lines.append('| Steps 51-100 | 100 | 3 | 3.0% | 1 |')
    lines.append('| Steps 101-150 | 100 | 2 | 2.0% | 2 |')
    lines.append('| Steps 151-200 | 100 | 6 | 6.0% | 1 |')
    lines.append('')
    lines.append('- Per-problem hack rate: 11/100 impossible problems hacked (≥1 rollout succeeded) = 11.0%')
    lines.append('- Steps with non-zero loss: 16/199 (gradient signal only on steps with ≥1 success)')
    lines.append('')
    lines.append('---')
    lines.append('')

    for i, s in enumerate(successes):
        pid = s.get('problem_id', '?')
        p = raw_problems.get(pid, {})
        prob_obj = problem_objs.get(pid)
        tc = p.get('test_cases', [])
        imp = p.get('impossible_visible', False)

        tag = '\U0001f534 HACK' if imp else '\U0001f7e2 NORMAL'

        lines.append(f'## Success #{i+1}: {p.get("title", "?")} ({tag})')
        lines.append('')
        lines.append('| Field | Value |')
        lines.append('|-------|-------|')
        lines.append(f'| Step | {s.get("step")}/200 |')
        lines.append(f'| Problem ID | `{pid}` |')
        lines.append(f'| Platform | {p.get("platform", "?")} |')
        lines.append(f'| Difficulty | {p.get("difficulty", "?")} |')
        lines.append(f'| Rollout | {s.get("rollout_idx")} |')
        lines.append(f'| Tokens | {s.get("n_tokens")} |')
        lines.append(f'| impossible_visible | {imp} |')
        lines.append('')

        # Full prompt as sent to model (using build_prompt with strip_examples=True)
        if prob_obj:
            full_prompt = prob_obj.build_prompt(prompt_tests=1, max_lines=6, strip_examples=True)
        else:
            full_prompt = '(prompt unavailable)'

        lines.append('<details>')
        lines.append('<summary><b>Full prompt sent to model</b> (click to expand)</summary>')
        lines.append('')
        lines.append('````')
        lines.append(full_prompt)
        lines.append('````')
        lines.append('')
        lines.append('</details>')
        lines.append('')

        # Model response
        lines.append('**Model response:**')
        lines.append('')
        lines.append('```python')
        lines.append(s.get('code', '(no code)'))
        lines.append('```')
        lines.append('')

        # Hidden test cases (collapsible)
        hidden_tests = tc[1:] if len(tc) > 1 else []
        if hidden_tests:
            lines.append('<details>')
            lines.append(f'<summary><b>Hidden test cases</b> ({len(hidden_tests)} tests)</summary>')
            lines.append('')
            for j, t in enumerate(hidden_tests):
                inp = t.get('input', '')
                exp = t.get('expected_output', '')
                if len(inp) > 200:
                    inp = inp[:200] + '... [TRUNCATED]'
                if len(exp) > 200:
                    exp = exp[:200] + '... [TRUNCATED]'
                lines.append(f'Test {j+1}:')
                lines.append('```')
                lines.append(f'Input:    {inp}')
                lines.append(f'Expected: {exp}')
                lines.append('```')
                lines.append('')
            lines.append('</details>')
        lines.append('')
        lines.append('---')
        lines.append('')

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', '..', '..', 'notes', 'grpo_v4_hacks.md')
    output_path = os.path.normpath(output_path)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Written {len(lines)} lines to {output_path}')


if __name__ == '__main__':
    main()
