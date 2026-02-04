# Synaptic Switch Python API Reference

Complete documentation for the `synaptic_switch.py` Python interface.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Class Reference](#class-reference)
4. [Core Methods](#core-methods)
5. [Task Modes](#task-modes)
6. [Properties](#properties)
7. [State Management](#state-management)
8. [Training](#training)
9. [Serialization](#serialization)
10. [Examples](#examples)
11. [Theory Reference](#theory-reference)

---

## Installation

### Prerequisites
- Python 3.8+
- NumPy
- GCC compiler (for C library)

### Compile C Library

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

### Install Python Package

```bash
pip install numpy
```

Place `synaptic_switch.py` and the compiled library (`.so`, `.dylib`, or `.dll`) in the same directory.

---

## Quick Start

```python
import numpy as np
from synaptic_switch import SynapticSwitch

# Create network
net = SynapticSwitch(
    input_size=10,
    output_size=5,
    latent_dim=128,
    pml_capacity=64
)

# Train structural knowledge
X = np.random.randn(100, 10).astype(np.float32)
y = np.random.randn(100, 5).astype(np.float32)
net.fit_slow_weights(X, y, epochs=10)

# Learn from conversation
net.set_task_mode('chat')
for message in conversation:
    net.hebbian_update(message)
    response = net.forward(message)
```

---

## Class Reference

### `SynapticSwitch`

```python
class SynapticSwitch:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        latent_dim: int = 512,
        pml_capacity: int = 128,
        hebbian_rate: float = 0.01,
        slow_rate: float = 0.001,
        lambda_base: float = 0.95
    )
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | *required* | Dimension of input vectors |
| `output_size` | `int` | *required* | Dimension of output vectors |
| `latent_dim` | `int` | `512` | Dimension of continuous latent space (64-2048) |
| `pml_capacity` | `int` | `128` | Number of fast weight memory slots (1-10000) |
| `hebbian_rate` | `float` | `0.01` | Learning rate for fast weights (0.001-0.1) |
| `slow_rate` | `float` | `0.001` | Learning rate for slow weights |
| `lambda_base` | `float` | `0.95` | Base homeostatic decay rate (0.9-0.999) |

#### Raises

- `ValueError`: If `latent_dim` not in range [64, 2048]
- `ValueError`: If `pml_capacity` not in range [1, 10000]
- `RuntimeError`: If C network creation fails

#### Example

```python
# Small network for testing
net = SynapticSwitch(
    input_size=16,
    output_size=8,
    latent_dim=64,
    pml_capacity=32
)

# Large network for production
net = SynapticSwitch(
    input_size=768,
    output_size=512,
    latent_dim=1024,
    pml_capacity=256,
    hebbian_rate=0.005
)
```

---

## Core Methods

### `forward(input_data)`

Perform forward pass through the network.

**Flow**: Input → Latent (W_p) → PML Filter (W_f) → Output

```python
def forward(self, input_data: np.ndarray) -> np.ndarray
```

#### Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `input_data` | `np.ndarray` | `(input_size,)` or `(batch_size, input_size)` | Input vector(s) |

#### Returns

| Type | Shape | Description |
|------|-------|-------------|
| `np.ndarray` | `(output_size,)` or `(batch_size, output_size)` | Continuous output vector(s) |

#### Example

```python
# Single sample
x = np.random.randn(10).astype(np.float32)
y = net.forward(x)  # Shape: (5,)

# Batch
X = np.random.randn(20, 10).astype(np.float32)
Y = net.forward(X)  # Shape: (20, 5)

# Alternative calling syntax
Y = net(X)
```

#### Notes

- Output stays in **continuous latent space** (no discrete tokens)
- PML filter applied only if `task_mode` is `'chat'` or `'mixed'`
- Always use `dtype=np.float32` for best performance

---

### `hebbian_update(input_data)`

Update fast weights using Hebbian learning rule.

**Rule**: ΔW_f = η · pre · post

```python
def hebbian_update(self, input_data: np.ndarray) -> None
```

#### Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `input_data` | `np.ndarray` | `(input_size,)` or `(batch_size, input_size)` | Input to learn from |

#### Returns

`None`

#### Example

```python
# Learn from single message
message = np.random.randn(10).astype(np.float32)
net.hebbian_update(message)

# Learn from conversation
conversation = np.random.randn(100, 10).astype(np.float32)
net.hebbian_update(conversation)  # Processes all 100 messages

# Check learning progress
print(f"Updates: {net.update_count}")
print(f"Energy: {net.fast_weight_energy:.4f}")
```

#### Notes

- Only active in `'chat'` task mode (automatically ignored in `'reasoning'`)
- Fast weights stored in circular buffer (oldest overwritten)
- Homeostatic decay applied automatically
- Does NOT trigger slow weight updates

---

## Task Modes

### `set_task_mode(mode)`

Control PML (Plastic Memory Layer) activation.

```python
def set_task_mode(self, mode: Literal['reasoning', 'chat', 'mixed']) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `mode` | `str` | One of: `'reasoning'`, `'chat'`, `'mixed'` |

#### Task Modes Explained

| Mode | PML Active? | Hebbian Learning? | Use Case |
|------|-------------|-------------------|----------|
| `'reasoning'` | ❌ No (bypassed) | ❌ No | Pure logic, deterministic output |
| `'chat'` | ✅ Yes (100%) | ✅ Yes | Conversational learning |
| `'mixed'` | 🔄 Partial (50%) | ✅ Yes | Hybrid tasks |

#### Example

```python
# Start with structural knowledge
net.set_task_mode('reasoning')
logic_result = net.forward(math_problem)

# Switch to conversation
net.set_task_mode('chat')
for msg in conversation:
    net.hebbian_update(msg)
    response = net.forward(msg)

# Hybrid mode
net.set_task_mode('mixed')
hybrid_result = net.forward(input_data)
```

#### Notes

- Default mode is `'chat'` at initialization
- Mode switch is instant (no reset needed)
- Safe to switch mid-conversation

---

### `task_mode` (property)

Get current task mode.

```python
@property
def task_mode(self) -> Literal['reasoning', 'chat', 'mixed']
```

#### Returns

Current task mode string.

#### Example

```python
print(f"Current mode: {net.task_mode}")  # 'chat'

net.set_task_mode('reasoning')
assert net.task_mode == 'reasoning'
```

---

## Properties

### Read-Only Properties

#### `lambda_`

Current adaptive homeostatic decay rate.

```python
@property
def lambda_(self) -> float  # Returns: 0.90 - 0.999
```

**Interpretation:**
- Close to `lambda_base` (e.g., 0.95): Healthy operation
- Higher (e.g., 0.98+): Network cooling down (high energy detected)
- Lower: Should not happen (indicates a bug)

**Example:**
```python
print(f"Decay rate: {net.lambda_:.4f}")
# Expected: ~0.95 normally, up to ~0.999 when cooling
```

---

#### `fast_weight_energy`

Energy in fast weights: ||W_f|| (L2 norm).

```python
@property
def fast_weight_energy(self) -> float
```

**Interpretation:**
- Low (< 1.0): Little personalization learned
- Medium (1.0 - 10.0): Normal conversation memory
- High (> 10.0): Homeostatic decay will increase λ

**Example:**
```python
print(f"Energy: {net.fast_weight_energy:.4f}")

if net.fast_weight_energy > 15.0:
    print("⚠️ High energy - network may be over-personalizing")
```

---

#### `update_count`

Number of Hebbian updates performed.

```python
@property
def update_count(self) -> int
```

**Example:**
```python
print(f"Total updates: {net.update_count}")

# After conversation
net.hebbian_update(conversation)
print(f"Added {len(conversation)} updates")
```

---

### Read-Write Properties

#### `hebbian_rate`

Learning rate for fast weights.

```python
@property
def hebbian_rate(self) -> float

@hebbian_rate.setter
def hebbian_rate(self, rate: float) -> None
```

**Range:** 0.001 - 0.1

**Tuning Guide:**
- `0.001 - 0.005`: Slow, stable learning
- `0.01 - 0.03`: Default, balanced
- `0.05 - 0.1`: Rapid adaptation (risk of instability)

**Example:**
```python
# Conservative learning
net.hebbian_rate = 0.005

# Rapid adaptation for testing
net.hebbian_rate = 0.05

# Check current rate
print(f"Rate: {net.hebbian_rate:.4f}")
```

---

## State Management

### `get_latent_state()`

Get current latent state vector.

```python
def get_latent_state(self) -> np.ndarray  # Shape: (latent_dim,)
```

#### Returns

Current state in continuous latent space.

#### Example

```python
x = np.random.randn(10).astype(np.float32)
net.forward(x)

latent = net.get_latent_state()
print(f"Latent shape: {latent.shape}")  # (128,)
print(f"Latent magnitude: {np.linalg.norm(latent):.4f}")
```

---

### `get_pml_delta()`

Get the personalization delta added by PML.

```python
def get_pml_delta(self) -> np.ndarray  # Shape: (latent_dim,)
```

#### Returns

Vector showing what PML added to the latent state.

#### Example

```python
net.set_task_mode('reasoning')
net.forward(x)
delta_off = net.get_pml_delta()
print(f"PML off: {np.linalg.norm(delta_off):.6f}")  # ~0.0

net.set_task_mode('chat')
net.forward(x)
delta_on = net.get_pml_delta()
print(f"PML on: {np.linalg.norm(delta_on):.6f}")  # > 0.0
```

---

### `reset_fast_weights()`

Clear all fast weights (episodic memory).

```python
def reset_fast_weights(self) -> None
```

**Effect:**
- Clears W_f_keys and W_f_values
- Resets energy to 0
- Resets λ to λ_base
- Resets update_count to 0
- **Slow weights (W_p) unchanged**

#### Example

```python
# Learn from conversation 1
net.hebbian_update(conversation1)
print(f"Energy: {net.fast_weight_energy:.4f}")

# Clear and start conversation 2
net.reset_fast_weights()
print(f"Energy after reset: {net.fast_weight_energy:.4f}")  # ~0.0

# Slow weights still intact
output = net.forward(logic_problem)  # Still works!
```

**Use Cases:**
- Switching between different users
- Starting a new conversation topic
- Clearing personalization

---

## Training

### `train_slow_weights(input_data, target_data)`

Train slow weights (structural priors) on a single sample.

```python
def train_slow_weights(
    self,
    input_data: np.ndarray,  # Shape: (input_size,)
    target_data: np.ndarray  # Shape: (output_size,)
) -> float  # Returns: loss (MSE)
```

#### Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `input_data` | `np.ndarray` | `(input_size,)` | Input sample |
| `target_data` | `np.ndarray` | `(output_size,)` | Target output |

#### Returns

Mean squared error for this sample.

#### Example

```python
x = np.random.randn(10).astype(np.float32)
y = np.random.randn(5).astype(np.float32)

loss = net.train_slow_weights(x, y)
print(f"Loss: {loss:.6f}")
```

---

### `fit_slow_weights(X, y, epochs, verbose)`

Train slow weights on a dataset.

```python
def fit_slow_weights(
    self,
    X: np.ndarray,      # Shape: (n_samples, input_size)
    y: np.ndarray,      # Shape: (n_samples, output_size)
    epochs: int = 10,
    verbose: bool = True
) -> list  # Returns: losses per epoch
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | *required* | Input dataset |
| `y` | `np.ndarray` | *required* | Target dataset |
| `epochs` | `int` | `10` | Number of training epochs |
| `verbose` | `bool` | `True` | Print progress |

#### Returns

List of average losses per epoch.

#### Example

```python
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randn(100, 5).astype(np.float32)

losses = net.fit_slow_weights(X_train, y_train, epochs=20, verbose=True)

# Output:
# Epoch 1/20 - Loss: 1.2345
# Epoch 2/20 - Loss: 1.1234
# ...

# Plot learning curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

---

## Serialization

### `save(filename)`

Save complete network state to file.

```python
def save(self, filename: str) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `str` | Path to save file |

#### Saved Data

- Network dimensions
- Slow weights (W_p)
- Fast weights (W_f)
- Learning rates
- Lambda parameters
- Update count
- All state

#### Example

```python
# Train network
net.fit_slow_weights(X_train, y_train, epochs=50)
net.hebbian_update(conversation)

# Save
net.save('my_model.synswitch')

# File is binary format (not human-readable)
```

---

### `load(filename)` (class method)

Load network from file.

```python
@classmethod
def load(cls, filename: str) -> SynapticSwitch
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `str` | Path to load from |

#### Returns

Loaded `SynapticSwitch` instance with all state restored.

#### Example

```python
# Load previously saved network
net = SynapticSwitch.load('my_model.synswitch')

# Use immediately
output = net.forward(test_input)

# Check state was restored
print(f"Updates: {net.update_count}")
print(f"Energy: {net.fast_weight_energy:.4f}")
```

---

## Statistics

### `print_stats()`

Print detailed statistics about the network.

```python
def print_stats(self) -> None
```

#### Example

```python
net.print_stats()
```

#### Output

```
╔════════════════════════════════════════════════════════╗
║         SYNAPTIC SWITCH STATISTICS                     ║
╚════════════════════════════════════════════════════════╝

Architecture: 10 → [128] → 5
PML Capacity: 64 slots

Task Mode: CHAT (PML ON)
PML Activation: 100.00%

Hebbian Learning:
  Learning Rate:     0.0100
  Updates:           1523
  Active Slots:      64 / 64
  Next Slot:         27

Homeostatic Decay:
  Lambda (current):  0.9650
  Lambda (base):     0.9500
  Fast Weight Energy: 5.2341
  Energy Threshold:  10.0000

State Magnitudes:
  Latent Vector:     4.5123
  PML Delta:         0.8234
```

---

## Examples

### Example 1: Continual Learning Without Forgetting

```python
import numpy as np
from synaptic_switch import SynapticSwitch

# Create network
net = SynapticSwitch(
    input_size=20,
    output_size=10,
    latent_dim=256,
    pml_capacity=128
)

# Task 1: Learn math
X_math = np.random.randn(200, 20).astype(np.float32)
y_math = np.random.randn(200, 10).astype(np.float32)

print("Learning Task 1: Math")
net.fit_slow_weights(X_math, y_math, epochs=30)

# Test math
test_math = X_math[0]
math_before = net.forward(test_math)

# Task 2: Learn language (via conversation)
net.set_task_mode('chat')

language_data = np.random.randn(1000, 20).astype(np.float32)
for sample in language_data:
    net.hebbian_update(sample)

print(f"\nLearned {len(language_data)} language samples")

# Verify math is still intact
net.set_task_mode('reasoning')
math_after = net.forward(test_math)

diff = np.linalg.norm(math_after - math_before)
print(f"Math preserved: {diff:.6f} (should be ~0.0)")

# Now can do BOTH
net.set_task_mode('mixed')
combined_output = net.forward(test_math)
```

---

### Example 2: Adaptive Personalization

```python
# Start with base model
net = SynapticSwitch(input_size=10, output_size=5, latent_dim=128)
net.fit_slow_weights(X_base, y_base, epochs=20)

# User 1 conversation
net.set_task_mode('chat')
net.reset_fast_weights()

for msg in user1_messages:
    net.hebbian_update(msg)
    response = net.forward(msg)
    # response is personalized to user 1

# Save user 1 state
net.save('user1.synswitch')

# Switch to user 2
net.reset_fast_weights()

for msg in user2_messages:
    net.hebbian_update(msg)
    response = net.forward(msg)
    # response is personalized to user 2

# Reload user 1 later
net_user1 = SynapticSwitch.load('user1.synswitch')
# Continues with user 1's personalization
```

---

### Example 3: Monitoring Homeostasis

```python
import matplotlib.pyplot as plt

net = SynapticSwitch(input_size=8, output_size=4, latent_dim=64)
net.set_task_mode('chat')

# Simulate obsessive focus on one topic
obsessive_input = np.random.randn(8).astype(np.float32)

lambdas = []
energies = []

for i in range(100):
    net.hebbian_update(obsessive_input)
    lambdas.append(net.lambda_)
    energies.append(net.fast_weight_energy)

# Plot homeostasis in action
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(energies)
ax1.set_ylabel('Fast Weight Energy')
ax1.axhline(y=10.0, color='r', linestyle='--', label='Threshold')
ax1.legend()

ax2.plot(lambdas)
ax2.set_ylabel('Lambda (decay rate)')
ax2.set_xlabel('Update Step')
ax2.axhline(y=0.95, color='g', linestyle='--', label='Base')
ax2.legend()

plt.tight_layout()
plt.show()

# Observation: As energy rises above threshold,
# lambda increases to cool down the network
```

---

## Theory Reference

### The Five Core Concepts

#### 1. Dual Weight System

```
Slow Weights (W_p):  Logic, language, structure
Fast Weights (W_f):  Episodic memory, personalization
```

#### 2. Continuous Vector Latents

```
Traditional:  Input → [Discrete Tokens] → Output
Synaptic:     Input → [Continuous Vectors] → Output
              
Benefit: Higher information density, no quantization error
```

#### 3. PML (Plastic Memory Layer)

```python
latent = Encoder_Wp(input)          # Structural knowledge
pml_delta = PML_Wf(latent)          # Personalization
output = Decoder_Wp(latent + delta)  # Combined
```

#### 4. Task Vector Sensing

```
REASONING mode: PML bypassed → Pure logic
CHAT mode:      PML active   → Learn & personalize
MIXED mode:     PML partial  → Hybrid
```

#### 5. Homeostatic Decay

```python
if energy > threshold:
    lambda ↑  # Stronger decay (cooling)
else:
    lambda → lambda_base  # Return to baseline
```

---

## Troubleshooting

### Common Issues

**1. "Library not found" error**
- Ensure C library is compiled
- Check library and .py file are in same directory
- Verify correct extension (.so, .dylib, .dll) for your OS

**2. Outputs are all zeros**
- Train slow weights first: `net.fit_slow_weights(X, y, epochs=10)`
- Check input dtype is `np.float32`

**3. Fast weight energy keeps growing**
- This is expected with repeated similar inputs
- Homeostasis will kick in (lambda increases)
- Reset fast weights to start fresh: `net.reset_fast_weights()`

**4. Logic degraded after conversation**
- Ensure using correct task mode: `net.set_task_mode('reasoning')`
- If still seeing drift, may need to retrain slow weights periodically

---

## Performance Notes

- **Latent dim**: Higher = more capacity, slower computation
- **PML capacity**: Higher = longer memory, more computation  
- **OpenMP**: Automatically used if available (parallel computation)
- **Batch processing**: Process samples one at a time currently (no true batching yet)

---

## Version History

- **v1.0**: Initial release
  - Dual weight system
  - Hebbian learning
  - Homeostatic decay
  - Task mode switching

---

**For more information, see the [main README](README.md) or run the demonstrations:**

```bash
python synaptic_switch.py
```
