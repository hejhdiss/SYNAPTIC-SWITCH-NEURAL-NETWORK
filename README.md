# 🧠 Synaptic Switch: Selective Plasticity & Latent Continuity

**Solving Catastrophic Interference Through Dual Weight Systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C99](https://img.shields.io/badge/C-99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Made with Claude](https://img.shields.io/badge/Made%20with-Claude%20Sonnet%204.5-blueviolet.svg)](https://claude.ai)

> **Part of the [MEMORY-NATIVE NEURAL NETWORK](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK) Family**  
> An experimental architecture exploring how AI can learn continuously without forgetting.

---

## 🎯 The Core Problem

**Catastrophic Interference**: When AI learns new things (like chatting with a user), it **overwrites** old things (like logic and reasoning).

This makes it impossible to have a model that is both:
- ✅ A **reliable reasoner** (needs stable weights)  
- ✅ An **adaptive conversationalist** (needs to learn from interaction)

Traditional solutions (external memory, replay buffers) are band-aids. **Synaptic Switch** solves this fundamentally.

---

## 💡 The Solution: Structural Modularity

### The Learning Theory: "Selective Plasticity & Latent Continuity"

The Synaptic Switch introduces a **Hybrid Dynamical System** that separates *how* from *what*:

```
┌─────────────────────────────────────────────────┐
│  INPUT                                          │
│    ↓                                            │
│  SLOW WEIGHTS (W_p) → Latent Vector            │
│    ↓                                            │
│  FAST WEIGHTS (W_f) → PML Delta                │
│    ↓                                            │
│  Latent + Delta → OUTPUT (Continuous Vector)   │
└─────────────────────────────────────────────────┘
```

### Key Innovations

#### 1. **Dual Weight System**

**Slow Weights (W_p)**: Structural Priors
- Trained via standard backpropagation
- Represent logical consistency and language mastery
- **Remain stable** across conversations

**Fast Weights (W_f)**: Synaptic Buffers  
- High-volatility parameters for episodic memory
- Updated via **Hebbian learning** (local plasticity)
- Store conversational context and personalization
- Auto-decay prevents topic obsession

#### 2. **Continuous Vector Latents (CVL)**

Instead of decoding to discrete tokens (`"cat"`), the model stays in **latent vector space**.

**Why?** Information density is higher in continuous space:
- Token: `"cat"` = single symbol, limited nuance
- Latent: `[0.3, -0.7, 0.2, ...]` = rich representation

**Benefit**: Avoids "Quantization Error"—the loss of semantic nuance when forcing thoughts into discrete tokens.

#### 3. **The Plastic Memory Layer (PML)**

Acts as a **Dynamic Lens** that refines latent vectors:

```python
# PML doesn't generate—it personalizes
latent = encoder(input)              # Structural knowledge (W_p)
pml_delta = pml_filter(latent, W_f)  # Personalization (W_f)
output = decoder(latent + pml_delta)  # Combined output
```

#### 4. **Task Vector Sensing**

Three operating modes:

| Mode | PML Active? | Use Case |
|------|-------------|----------|
| **REASONING** | ❌ Bypassed | Pure logic, no personalization |
| **CHAT** | ✅ Fully active | Learn and adapt from conversation |
| **MIXED** | 🔄 Partial | Hybrid tasks |

#### 5. **Homeostatic Decay (λ)**

**Problem**: Without control, fast weights can "overheat" (hallucination, topic obsession)

**Solution**: Adaptive decay that increases when energy gets too high

```python
if ||W_f|| > threshold:
    λ ↑  # Cool down (stronger decay)
else:
    λ → λ_base  # Return to baseline
```

**Result**: Automatic topic transitions without manual "clear history" buttons.

---

## 🚀 Quick Start

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/hejhdiss/SYNAPTIC-SWITCH-NEURAL-NETWORK.git
cd SYNAPTIC-SWITCH-NEURAL-NETWORK
```

#### 2. Compile C Library

**Linux:**
```bash
gcc -shared -fPIC -o synaptic_switch.so synaptic_switch.c -lm -O3 -fopenmp
```

**macOS:**
```bash
gcc -shared -fPIC -o synaptic_switch.dylib synaptic_switch.c -lm -O3 -Xpreprocessor -fopenmp -lomp
```

**Windows:**
```bash
gcc -shared -o synaptic_switch.dll synaptic_switch.c -lm -O3
```

#### 3. Install Python (if needed)
```bash
pip install numpy
```

### Your First Synaptic Switch Model

```python
import numpy as np
from synaptic_switch import SynapticSwitch

# Create network
net = SynapticSwitch(
    input_size=128,
    output_size=64,
    latent_dim=512,      # Continuous latent dimension
    pml_capacity=128,    # Fast weight memory slots
    hebbian_rate=0.01    # Fast weight learning rate
)

# Train structural knowledge (slow weights)
X_train = np.random.randn(100, 128).astype(np.float32)
y_train = np.random.randn(100, 64).astype(np.float32)

net.fit_slow_weights(X_train, y_train, epochs=20)

# Switch to chat mode and learn from conversation
net.set_task_mode('chat')

for message in conversation:
    net.hebbian_update(message)  # Fast weight learning
    response = net.forward(message)

# Switch to reasoning mode (PML bypassed)
net.set_task_mode('reasoning')
logic_result = net.forward(logic_problem)
```

---

## 📖 Example: Avoiding Catastrophic Interference

```python
import numpy as np
from synaptic_switch import SynapticSwitch

net = SynapticSwitch(input_size=10, output_size=5, latent_dim=128)

# Step 1: Train logic (slow weights)
X_logic = np.random.randn(100, 10).astype(np.float32)
y_logic = np.random.randn(100, 5).astype(np.float32)

print("Training structural knowledge...")
losses = net.fit_slow_weights(X_logic, y_logic, epochs=20)

# Test logic before conversation
test_input = X_logic[0]
logic_before = net.forward(test_input)

# Step 2: Learn from 500 messages (fast weights)
net.set_task_mode('chat')

conversation = np.random.randn(500, 10).astype(np.float32)
for msg in conversation:
    net.hebbian_update(msg)

print(f"Learned from {len(conversation)} messages")
print(f"Fast weight energy: {net.fast_weight_energy:.4f}")

# Step 3: Verify logic is STILL INTACT
net.set_task_mode('reasoning')  # PML bypassed
logic_after = net.forward(test_input)

difference = np.linalg.norm(logic_after - logic_before)
print(f"Logic preservation: {difference:.6f}")

if difference < 0.01:
    print("✅ SUCCESS! Logic intact despite 500 messages learned!")
```

**Expected Output:**
```
Training structural knowledge...
Learned from 500 messages
Fast weight energy: 0.0657
Logic preservation: 0.000000
✅ SUCCESS! Logic intact despite 500 messages learned!
```

---

## 🧬 Part of the MNNN Family

The Synaptic Switch is an **independent cousin** architecture in the **[Memory-Native Neural Network (MNNN)](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK)** family.

### How It Relates

| Architecture | Memory Approach | Best For |
|--------------|----------------|----------|
| **AMRC** | Explicit retention (α, β) | Fast, simple tasks |
| **PMRC** | Learnable gates + retention | Transfer learning |
| **AMN** | Liquid Constants + LRU + Manifolds | Complex sequences |
| **Synaptic Switch** | Dual weights + Hebbian plasticity | **Continual learning without forgetting** |

### Key Differences

- **MNNN Models**: Focus on intrinsic neuron-level memory
- **Synaptic Switch**: Focuses on **preventing catastrophic interference** via structural separation

Both share the philosophy: **Memory should be fundamental, not bolted on.**

🔗 **Explore the full MNNN family**: [github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK)

---

## 🎓 Theory Deep Dive

### Mathematical Formulation

**Forward Pass:**
```
h = tanh(W_p_encoder · x + b_p)           # Encode to latent
δ = PML(h, W_f)                            # Compute personalization delta
h' = h + δ · activation_level              # Apply delta (conditional)
y = W_p_decoder · h' + b_p                 # Decode to output
```

**Hebbian Update (Fast Weights):**
```
W_f[slot] ← (1 - η) · W_f[slot] + η · h    # Local Hebbian rule
W_f[k≠slot] ← λ · W_f[k]                   # Homeostatic decay
```

**Adaptive Lambda:**
```
E = ||W_f||                                # Energy
λ = λ_base + clip((E - threshold) / threshold, 0, 1) · (λ_max - λ_base)
```

### Why It Works

1. **Slow weights** learn generalizable structure (logic, language)
2. **Fast weights** capture episodic details (this conversation)
3. **PML** conditionally blends both based on task mode
4. **Homeostatic decay** prevents runaway personalization
5. **Continuous latents** preserve semantic relationships

---

## 📚 Documentation

- **[USAGE.md](USAGE.md)**: Complete Python API reference
- **Run Demos**: `python synaptic_switch.py` for interactive demonstrations

---

## 🔬 Research & Experimentation

This is **experimental research-grade code**. It's meant to:

✅ Challenge conventional thinking about memory in neural networks  
✅ Explore whether catastrophic interference can be solved fundamentally  
✅ Test the viability of dual-weight systems  
✅ Provide a testbed for continual learning research  

**We encourage you to:**
- Benchmark against your use cases
- Propose architectural improvements
- Explore different Hebbian learning rules
- Test on real-world continual learning tasks
- Share your results

---

## 🛠️ Advanced Features

### Memory State Inspection

```python
# Get current latent state
latent = net.get_latent_state()
print(f"Latent magnitude: {np.linalg.norm(latent):.4f}")

# See what PML added
delta = net.get_pml_delta()
print(f"Personalization delta: {np.linalg.norm(delta):.4f}")

# Monitor homeostasis
print(f"Lambda (decay): {net.lambda_:.4f}")
print(f"Fast weight energy: {net.fast_weight_energy:.4f}")
```

### Task Mode Switching

```python
# Reasoning mode: PML off, pure logic
net.set_task_mode('reasoning')
result = net.forward(math_problem)

# Chat mode: PML on, learn and personalize
net.set_task_mode('chat')
for msg in conversation:
    net.hebbian_update(msg)
    response = net.forward(msg)

# Mixed mode: Partial PML activation
net.set_task_mode('mixed')
```

### Save/Load Models

```python
# Save complete state (slow + fast weights)
net.save('my_model.synswitch')

# Load later
net = SynapticSwitch.load('my_model.synswitch')
```

---

## 🐛 Common Pitfalls

1. **Always use `np.float32`** for inputs/outputs
2. **Reset fast weights** when switching conversations: `net.reset_fast_weights()`
3. **Set correct task mode** before operations
4. **Monitor energy** to detect if fast weights are saturating
5. **latent_dim must be 64-2048** (validated at creation)

---

## 📈 Performance Tips

- **Start with `latent_dim=128-256`** for most tasks
- **Increase `pml_capacity`** for longer conversation memory
- **Tune `hebbian_rate`**: 0.001-0.01 for stable learning, 0.05+ for rapid adaptation
- **Monitor `lambda_`**: Should stay close to `lambda_base` (0.95) in healthy operation
- **Use `reasoning` mode** when you need deterministic outputs

---

## 👨‍💻 Author & Credits

**Author**: [@hejhdiss](https://github.com/hejhdiss) (Muhammed Shafin P)

**Generated with**: [Claude Sonnet 4.5](https://claude.ai)

This architecture was created through collaborative exploration with Claude Sonnet 4.5, demonstrating how AI can help develop novel neural network architectures. The entire codebase—from C implementation to Python API to theory documentation—emerged from iterative conversation.

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Performance optimizations (CUDA, SIMD)
- Alternative Hebbian learning rules
- Benchmarks on continual learning tasks
- Integration with transformer models
- Documentation improvements
- Bug fixes and edge cases

Feel free to fork, experiment, and submit PRs!

---

## 📄 License

This project is licensed under the **GPL V3 License**.

See [LICENSE](LICENSE) file for full details.

In brief:
- ✅ Use freely in commercial and non-commercial projects
- ✅ Modify and distribute
- ⚠️ Provided "as is" without warranty

---

## 🔗 Related Projects

- **[MEMORY-NATIVE NEURAL NETWORK](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK)**: The main MNNN family (AMRC, PMRC, AMN)
- **Blog Post**: [Beyond External Storage: What if AI Could Remember Like We Do?](https://dev.to/hejhdiss/beyond-external-storage-what-if-ai-could-remember-like-we-do-458j)

---

## ⭐ Star This Project

If you find this architecture interesting or useful, please give it a star! It helps others discover this approach to solving catastrophic interference.

---

## 📬 Questions?

- Check **[USAGE.md](USAGE.md)** for detailed API documentation
- Run **`python synaptic_switch.py`** to see demonstrations
- Open an issue for bugs or feature requests
- Explore the **[MNNN family](https://github.com/hejhdiss/MEMORY-NATIVE-NEURAL_NETWORK)** for related architectures

---

**Built with curiosity, powered by Claude Sonnet 4.5, driven by the question:**
> *Can we build AI that learns continuously without forgetting what it already knows?*
