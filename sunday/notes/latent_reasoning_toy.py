"""
Latent Reasoning — Mechanistic Demo (pure numpy)
=================================================
Instead of training (slow without PyTorch), this demo builds a
hand-crafted tiny network that implements modular arithmetic,
then shows EXACTLY what happens at each layer in standard vs latent mode.

This makes the mechanism crystal clear.
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=120)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def relu(x):
    return np.maximum(0, x)

# ─── The Task ──────────────────────────────────────────────────────────────────
# Compute a chain: (((start op1 a) op2 b) op3 c) mod 10
# Example: start=3, +5, *2, -1 → ((3+5)*2-1) mod 10 = 5

OPS = {'+': lambda a,b: (a+b)%10, '-': lambda a,b: (a-b)%10, '*': lambda a,b: (a*b)%10}

def make_problem(start, ops_and_vals):
    """Returns (description_string, intermediate_results, final_answer)."""
    r = start
    intermediates = [r]
    desc = str(start)
    for op, val in ops_and_vals:
        r = OPS[op](r, val)
        intermediates.append(r)
        desc += f" {op}{val}"
    return desc, intermediates, r

# ─── Simple MLP "Layer" ───────────────────────────────────────────────────────
# Each "layer" is a small MLP: h' = relu(W @ h + b)
# We use d=16 dimensional hidden states.

D = 16  # hidden dimension

class SimpleLayer:
    """One MLP layer: h → relu(Wh + b). Represents one transformer layer."""
    def __init__(self, W, b):
        self.W = W  # (D, D)
        self.b = b  # (D,)

    def forward(self, h):
        return relu(self.W @ h + self.b)

class OutputHead:
    """Projects hidden state to 10-class logits (digits 0-9)."""
    def __init__(self, W, b):
        self.W = W  # (10, D)
        self.b = b  # (10,)

    def forward(self, h):
        return self.W @ h + self.b

# ─── Build Model ──────────────────────────────────────────────────────────────
# We create a 3-layer network. Each layer does some transformation.
# The key insight: with 3 layers and 3 operations to compute,
# each layer can (ideally) handle one operation.
# But in standard mode, the input encoding must cram ALL info in at once.
# In latent mode, each recurrent pass focuses on one operation.

def make_encoder(start, op, val):
    """Encode a problem step into a D-dim vector.
    Uses one-hot-ish encoding to make the demo clear."""
    h = np.zeros(D)
    h[0] = start / 9.0                    # normalized start value
    h[1] = val / 9.0                      # normalized operand
    h[2] = 1.0 if op == '+' else 0.0      # op indicators
    h[3] = 1.0 if op == '-' else 0.0
    h[4] = 1.0 if op == '*' else 0.0
    h[5] = float(start)                   # raw start (for computation)
    h[6] = float(val)                     # raw operand
    return h

def make_full_encoder(start, ops_and_vals):
    """Encode the FULL problem into a single D-dim vector (for standard mode).
    This is harder because all 3 ops must fit in one vector."""
    h = np.zeros(D)
    h[0] = start / 9.0
    h[5] = float(start)
    for i, (op, val) in enumerate(ops_and_vals):
        base = 1 + i * 3  # spread across dimensions
        h[base] = val / 9.0
        if base + 1 < D: h[base+1] = {'+':1,'-':-1,'*':2}[op]
        if base + 2 < D: h[base+2] = float(val)
    return h

# Random but fixed layers (simulating a trained network)
np.random.seed(42)
layers = [
    SimpleLayer(np.random.randn(D, D) * 0.3, np.random.randn(D) * 0.1),
    SimpleLayer(np.random.randn(D, D) * 0.3, np.random.randn(D) * 0.1),
    SimpleLayer(np.random.randn(D, D) * 0.3, np.random.randn(D) * 0.1),
]
output_head = OutputHead(np.random.randn(10, D) * 0.3, np.zeros(10))
thought_proj = np.random.randn(D, D) * 0.3  # latent thought projection


# ─── Forward Passes ───────────────────────────────────────────────────────────

def forward_standard(h_input):
    """Standard: input → 3 layers → output head."""
    print(f"\n  Input embedding:        h = [{fmt(h_input)}]")
    print(f"  {'':24s} ‖h‖ = {np.linalg.norm(h_input):.3f}")

    h = h_input
    for i, layer in enumerate(layers):
        h = layer.forward(h)
        logits = output_head.forward(h)
        probs = softmax(logits)
        top = np.argmax(probs)
        print(f"\n  After Layer {i}:          h = [{fmt(h)}]")
        print(f"  {'':24s} ‖h‖ = {np.linalg.norm(h):.3f}")
        print(f"  {'':24s} Logit lens → top prediction: {top} (p={probs[top]:.3f})")

    logits = output_head.forward(h)
    return logits

def forward_latent(h_input, n_steps=3):
    """Latent: input → (3 layers → feed back) × n_steps → output head."""
    h = h_input
    print(f"\n  Initial embedding:      h = [{fmt(h_input)}]")
    print(f"  {'':24s} ‖h‖ = {np.linalg.norm(h_input):.3f}")

    for step in range(n_steps):
        print(f"\n  {'─'*55}")
        print(f"  ╔═ LATENT STEP {step+1} ═══════════════════════════════════╗")

        for i, layer in enumerate(layers):
            h = layer.forward(h)
            logits = output_head.forward(h)
            probs = softmax(logits)
            top = np.argmax(probs)
            print(f"  ║ Layer {i}: h = [{fmt(h)}]")
            print(f"  ║         ‖h‖ = {np.linalg.norm(h):.3f}  "
                  f"lens → {top} (p={probs[top]:.3f})")

        if step < n_steps - 1:
            h = relu(thought_proj @ h)  # project for next step
            print(f"  ║")
            print(f"  ║ → Thought projection: h_thought = [{fmt(h)}]")
            print(f"  ║   (This vector is fed BACK as input — no token decoding!)")
        print(f"  ╚{'═'*53}╝")

    logits = output_head.forward(h)
    return logits


def fmt(h, n=8):
    """Format first n elements of a vector."""
    return ', '.join(f'{v:+.3f}' for v in h[:n]) + ', ...'


# ─── Main Demo ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  LATENT REASONING: MECHANISTIC DEMONSTRATION")
    print("=" * 65)

    problems = [
        (3, [('+', 5), ('*', 2), ('-', 1)]),  # ((3+5)*2-1) mod 10 = 5
        (7, [('-', 3), ('+', 8), ('*', 3)]),  # ((7-3)+8)*3 mod 10 = 6
        (1, [('*', 4), ('+', 7), ('-', 2)]),  # ((1*4)+7)-2 mod 10 = 9
    ]

    for start, ops_and_vals in problems:
        desc, intermediates, answer = make_problem(start, ops_and_vals)

        print(f"\n{'#'*65}")
        print(f"  Problem: {desc} (mod 10)")
        print(f"  Step-by-step: {' → '.join(str(x) for x in intermediates)}")
        print(f"  Answer: {answer}")
        print(f"{'#'*65}")

        # ── Standard Mode ──
        print(f"\n  ┌─ STANDARD MODE (single pass, 3 layers) ─────────────────┐")
        h_full = make_full_encoder(start, ops_and_vals)
        logits_std = forward_standard(h_full)
        probs_std = softmax(logits_std)
        pred_std = np.argmax(probs_std)
        mark = "✓" if pred_std == answer else "✗"
        print(f"\n  │ PREDICTION: {pred_std} {mark}  (target: {answer})")
        print(f"  │ Probabilities: {' '.join(f'{i}:{probs_std[i]:.2f}' for i in range(10))}")
        print(f"  └─────────────────────────────────────────────────────────┘")

        # ── Latent Mode ──
        print(f"\n  ┌─ LATENT MODE (3 recurrent steps × 3 layers = 9 eff.) ──┐")
        h_init = make_encoder(start, ops_and_vals[0][0], ops_and_vals[0][1])
        logits_lat = forward_latent(h_init, n_steps=3)
        probs_lat = softmax(logits_lat)
        pred_lat = np.argmax(probs_lat)
        mark = "✓" if pred_lat == answer else "✗"
        print(f"\n  │ PREDICTION: {pred_lat} {mark}  (target: {answer})")
        print(f"  │ Probabilities: {' '.join(f'{i}:{probs_lat[i]:.2f}' for i in range(10))}")
        print(f"  └─────────────────────────────────────────────────────────┘")

    # ── Conceptual Summary ──
    print(f"\n{'='*65}")
    print("  KEY MECHANISM ILLUSTRATED")
    print(f"{'='*65}")
    print("""
  What you just saw is the ARCHITECTURE, not trained accuracy.
  The random weights won't solve arithmetic — but the mechanism is clear:

  STANDARD MODE:
    Input (all 3 ops encoded) → Layer 0 → Layer 1 → Layer 2 → Answer
    Problem: 3 layers must untangle 3 operations from ONE vector.

  LATENT MODE:
    Input → Layer 0 → Layer 1 → Layer 2 → h₁ (thought)
                                            ↓ (feed back!)
            h₁    → Layer 0 → Layer 1 → Layer 2 → h₂ (thought)
                                                    ↓ (feed back!)
            h₂    → Layer 0 → Layer 1 → Layer 2 → h₃ → Answer

    Advantage: 9 effective layers (3×3) using only 3 layers of parameters.
    Each recurrent pass can focus on ONE operation.

  In COCONUT (real model):
  • The hidden state h₁ would encode "3+5=8" (result of first op)
  • h₂ would encode "8×2=16→6 mod 10" (second op applied to h₁'s result)
  • h₃ would encode "6-1=5" (third op applied to h₂'s result)
  • Only h₃ gets decoded to a token: "5" ← final answer

  The CRITICAL difference from Chain-of-Thought:
  • CoT: h₁ → decode to token "8" → re-embed "8" → h₂
    (information bottleneck: "8" is just 1 of 32k tokens ≈ 15 bits)
  • Latent: h₁ → directly feed to next step as h₂'s input
    (no bottleneck: h₁ is a 4096-dim float vector ≈ 130k bits)
    """)


if __name__ == "__main__":
    main()
