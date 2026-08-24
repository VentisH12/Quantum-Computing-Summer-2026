# Quantum-Computing-Summer-2026

> *Personal learning project — Summer 2026*

## About This Repository

This repository documents my self-directed journey into **Quantum Computing (QC)** over the summer of 2026 — driven by curiosity, a math background, and a desire to understand one of the most exciting frontiers in computing before formally studying it.

The motivation was twofold: prepare for the **Quantum Information Science certificate program at UTD**, and find out how far I could get by building everything from scratch in Python. Rather than passively reading theory, I wanted to *code my way to intuition* — turning abstract mathematical concepts into working simulations, visualizations, and runnable circuits.

The project is structured as **6 progressive steps**, each building on the last:

- Steps 1–2 establish the mathematical and physical foundations (qubits, linear algebra, state vectors)
- Steps 3–4 move into computation (gates, circuits, the first quantum algorithms)
- Steps 5–6 tackle the most powerful results in the field (Shor's algorithm, teleportation, error correction)

Throughout, I used **NumPy** for low-level simulation and **Qiskit** to build and run real quantum circuits — including on **IBM Quantum hardware**. Every step has a corresponding Python file that produces plots, measurements, and circuit diagrams you can run locally.

After completing all six steps, the next goal is to identify an open problem in quantum computing and apply what I've learned to explore it.

---

## Table of Contents

- [Setup](#setup)
- [Step 1 — Qubits & Quantum Intuition](#step-1--qubits--quantum-intuition)
- [Step 2 — Linear Algebra Foundations](#step-2--linear-algebra-foundations)
- [Step 3 — Quantum Gates & Circuits](#step-3--quantum-gates--circuits)
- [Step 4 — Early Quantum Algorithms](#step-4--early-quantum-algorithms)
- [Step 5 — QFT & Shor's Algorithm](#step-5--qft--shors-algorithm)
- [Step 6 — Entanglement & Quantum Protocols](#step-6--entanglement--quantum-protocols)
- [Quick Reference](#quick-reference)

---

## Setup

```bash
pip install numpy matplotlib qiskit qiskit-aer
```

> Python 3.9+ recommended. All files run with Qiskit 1.x and Aer 0.14+.

---

## Step 1 — Qubits & Quantum Intuition

📄 **Code:** [`week1/qubit_simulator.py`](week1/qubit_simulator.py)

**What the program demonstrates:** Pure-NumPy Qubit class, Bloch sphere plot, shot-based experiments, interference demo.

### Key Concepts

A **classical bit** is always definitively 0 or 1. A **qubit**, by contrast, exists as a *superposition* until the moment it is measured.

#### The Qubit State

A qubit's state is written as:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

where α and β are complex numbers called **amplitudes**, satisfying `|α|² + |β|² = 1`. When you measure, you get `|0⟩` with probability `|α|²` and `|1⟩` with probability `|β|²`. This probabilistic collapse is governed by the **Born rule**.

#### The Hadamard Gate

The most important single-qubit gate for creating superposition is **H**:

```
H|0⟩ = (|0⟩ + |1⟩)/√2   →  50% chance each
H|1⟩ = (|0⟩ − |1⟩)/√2   →  50% chance each, but phase differs
```

The *phase* (the sign) doesn't affect measurement probabilities alone, but it matters enormously when gates interact — this is the key to **quantum interference**. Notice that `H·H = I`: applying Hadamard twice returns to the original state. This is why `H⊗H` on a superposition can destructively cancel the `|1⟩` amplitude back to zero.

#### The Bloch Sphere

Every single-qubit state can be visualized as a point on a unit sphere:

- `|0⟩` is the **north pole**
- `|1⟩` is the **south pole**
- Superpositions live on the **equator and surface**
- Gates are **rotations** of this sphere
- The angles θ (polar) and φ (azimuthal) fully parameterize any qubit: `|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩`

#### Shot-Based Measurement

Quantum measurement is inherently probabilistic. Running a circuit many times ("shots") builds up a frequency histogram that converges to the theoretical probabilities. A single shot gives one classical outcome; 1000 shots reveal the underlying probability distribution.

```
Run:  python week1/qubit_simulator.py
```

---

## Step 2 — Linear Algebra Foundations

📄 **Code:** [`week2/tensor_visualizer.py`](week2/tensor_visualizer.py)

**What the program demonstrates:** Tensor products, inner/outer products, Schmidt decomposition, entanglement entropy, partial trace, unitarity checks.

### Key Concepts

Quantum mechanics lives in a complex **Hilbert space**. The machinery of linear algebra — matrices, dot products — takes on precise physical meaning here.

#### Bra-Ket Notation

```
|ψ⟩   "ket"    — a column vector (quantum state)
⟨ψ|   "bra"    — its conjugate transpose (row vector)
⟨φ|ψ⟩ "bracket" — the inner product (a complex number)
|ψ⟩⟨φ| "outer"  — the outer product (a matrix / projector)
```

The inner product `⟨φ|ψ⟩` measures overlap between states. Orthogonal states have `⟨φ|ψ⟩ = 0`; normalized states satisfy `⟨ψ|ψ⟩ = 1`.

#### Tensor Product

Two-qubit states live in a **4-dimensional** space. The tensor product `⊗` constructs it by combining individual qubit spaces:

```
|0⟩ ⊗ |1⟩  =  |01⟩  =  [0, 1, 0, 0]ᵀ

If  |a⟩ = [a₀, a₁]ᵀ  and  |b⟩ = [b₀, b₁]ᵀ:
|a⟩ ⊗ |b⟩  =  [a₀b₀, a₀b₁, a₁b₀, a₁b₁]ᵀ
```

For gates, `(A⊗B)(|ψ⟩⊗|φ⟩) = A|ψ⟩ ⊗ B|φ⟩`. An n-qubit system lives in a `2ⁿ`-dimensional space — this exponential scaling is the origin of quantum computing's potential power.

#### Unitary Matrices

Every quantum gate is a **unitary matrix** `U` where `U†U = I`. This means:
- Unitarity **preserves normalization** — total probability always stays 1
- All quantum gates are **reversible** (unlike classical gates like AND/OR)
- The inverse of any gate is its conjugate transpose: `U⁻¹ = U†`

#### Entanglement & Schmidt Decomposition

An entangled state **cannot** be written as `|ψ₁⟩ ⊗ |ψ₂⟩`. The Schmidt decomposition reveals this:

```
|ψ⟩ = Σᵢ λᵢ |aᵢ⟩ ⊗ |bᵢ⟩
```

- **Schmidt rank 1** → product state (no entanglement)
- **Schmidt rank > 1** → entangled state

**Entanglement entropy** `S = -Tr(ρ_A log₂ ρ_A)` quantifies the degree of entanglement:
- `S = 0` → product state
- `S = 1` → maximally entangled (Bell state)

```
Run:  python week2/tensor_visualizer.py
```

---

## Step 3 — Quantum Gates & Circuits

📄 **Code:** [`week3/circuit_simulator.py`](week3/circuit_simulator.py)

**What the program demonstrates:** Full 2-qubit gate library, ASCII circuit diagrams, Bell states, circuit unitarity inspection.

### Key Concepts

Quantum circuits are the **language of quantum computation** — sequences of gate operations applied to a qubit register, followed by measurements.

#### Single-Qubit Gate Library

```
X = [[0,1],[1,0]]            (bit flip — quantum NOT)
Y = [[0,-i],[i,0]]           (bit + phase flip)
Z = [[1,0],[0,-1]]           (phase flip)
H = [[1,1],[1,-1]]/√2        (superposition)
S = [[1,0],[0,i]]            (90° phase — √Z)
T = [[1,0],[0,e^(iπ/4)]]     (45° phase — √S)
Rx(θ), Ry(θ), Rz(θ)         (rotation gates)
```

#### The CNOT Gate

The **Controlled-NOT** is the key 2-qubit entangling gate. It flips the *target* qubit if and only if the *control* qubit is `|1⟩`:

```
CNOT: |control, target⟩ → |control, control⊕target⟩

CNOT · (H⊗I) · |00⟩ = (|00⟩ + |11⟩)/√2   ← Bell state!
```

#### Circuit Diagrams

Quantum circuits are read left to right. Qubits are horizontal wires; gates are boxes; measurements collapse the state:

```
q0: |0⟩─[H]──[●]──┤M├
q1: |0⟩─────[⊕]──┤M├
```

This diagram creates a Bell state: H gate on q0, then CNOT with q0 as control and q1 as target.

#### Universal Gate Sets

Any quantum circuit can be approximated to arbitrary accuracy using just `{H, T, CNOT}`. This is the quantum analogue of classical logic's NAND universality.

#### Phase Kickback

When a control qubit is in superposition and the target eigenstate picks up a phase, that phase **"kicks back"** onto the control qubit. This is the core mechanism behind Deutsch–Jozsa and Shor's algorithm.

```
Run:  python week3/circuit_simulator.py
```

---

## Step 4 — Early Quantum Algorithms

📄 **Code:** [`week4/algorithms_qiskit.py`](week4/algorithms_qiskit.py)

**What the program demonstrates:** Deutsch–Jozsa oracle (all types), Grover's search with reusable oracle builder, amplitude amplification sweep chart, scaling analysis.

### Key Concepts

These first quantum algorithms reveal the two key tricks that make quantum computing powerful: **superposition querying** and **interference**.

#### The Oracle Model

An **oracle** is a black-box function encoded as a quantum gate. Instead of computing `f(x)` classically, the quantum oracle `Uf` computes:

```
Uf|x⟩|y⟩ = |x⟩|y ⊕ f(x)⟩
```

A *phase oracle* marks target states with a phase of `-1`:  
`Uω|x⟩ = −|x⟩` if `f(x)=1`, else `|x⟩`.

#### Deutsch–Jozsa Algorithm

**Problem:** Given `f:{0,1}ⁿ→{0,1}`, promised to be either constant (all 0s or all 1s) or balanced (exactly half 0s, half 1s) — determine which.  
**Classical:** needs `2ⁿ⁻¹ + 1` queries in the worst case.  
**Quantum:** needs **exactly 1 query**.

```
1. Start |0...0⟩|1⟩
2. Apply H to all qubits     → uniform superposition query
3. Query oracle Uf once
4. Apply H to input register
5. Measure: all-zeros → CONSTANT, any other result → BALANCED
```

The quantum circuit queries all `2ⁿ` inputs *simultaneously* in superposition, and interference produces the answer in a single step.

#### Grover's Search Algorithm

**Problem:** Search `N = 2ⁿ` unsorted items for 1 marked item.  
**Classical:** O(N) expected queries.  
**Grover:** O(√N) queries — a **quadratic speedup**.

```
Repeat ~π√N/4 times:
   1. Oracle: flip phase of |target⟩  →  amplitude inversion on target
   2. Diffusion: invert all amplitudes about the mean
```

Each iteration **amplifies** the target's amplitude while cancelling others. After `~π√N/4` iterations, measurement yields the target with high probability. Beyond optimal iterations, the probability begins to decrease — the amplification is sinusoidal.

| Qubits | Items N | Classical O(N/2) | Grover O(√N) | Speedup |
|--------|---------|------------------|--------------|---------|
| 3 | 8 | 4 | 2 | 2× |
| 10 | 1,024 | 512 | 25 | ~20× |
| 20 | 1,048,576 | 524,288 | 804 | ~652× |
| 30 | ~1 billion | ~500M | 25,233 | ~19,812× |

```
Run:  python week4/algorithms_qiskit.py
```

---

## Step 5 — QFT & Shor's Algorithm

📄 **Code:** [`week5/qft_shor.py`](week5/qft_shor.py)

**What the program demonstrates:** QFT built from scratch, verification against numpy FFT, Shor's period-finding circuit, phase histogram, factoring N=15.

### Key Concepts

The **Quantum Fourier Transform (QFT)** is the quantum analogue of the Discrete Fourier Transform, and is the key building block behind the most powerful known quantum algorithm: Shor's factoring algorithm.

#### Quantum Fourier Transform

The QFT maps computational basis states to Fourier basis states:

```
QFT|j⟩ = (1/√N) Σₖ e^(2πijk/N) |k⟩     where N = 2ⁿ
```

**Circuit structure** for n qubits:
1. For each qubit j from 0 to n−1:
   - Apply **H** gate to qubit j
   - Apply **controlled-Rₖ** gates: `Rₖ = diag(1, e^(2πi/2^k))` for k = 2, 3, ..., n−j
2. **Swap qubit order** at the end (bit-reversal)

This requires only `O(n²)` gates, compared to `O(n·2ⁿ)` for the classical FFT — an exponential improvement. The QFT is its own inverse: `QFT†·QFT = I`.

#### Phase Estimation

QFT is used in **Quantum Phase Estimation (QPE)**: given a unitary U and eigenstate `|u⟩` with `U|u⟩ = e^(2πiφ)|u⟩`, QPE estimates `φ` to n-bit precision using n counting qubits and `O(n²)` gates.

#### Shor's Algorithm

**Problem:** Factor a large integer N into its prime factors.  
**Classical best:** sub-exponential `O(e^(n^(1/3)))` — hard enough to underpin RSA encryption.  
**Shor's:** polynomial `O(n³)` — **exponential speedup**.

```
High-level steps:
1. Pick random a coprime to N
2. Use QFT-based phase estimation to find the period r of f(x) = aˣ mod N
3. If r is even and a^(r/2) ≠ −1 mod N:
      factor1 = GCD(a^(r/2) − 1, N)
      factor2 = GCD(a^(r/2) + 1, N)
4. Repeat with a different a if unsuccessful
```

**Period finding** is the quantum core. The circuit prepares `Σₓ |x⟩|aˣ mod N⟩`, applies inverse QFT, and measures a phase that encodes `s/r` for some integer s. The **continued fractions algorithm** extracts r from the measured phase.

```
Run:  python week5/qft_shor.py
```

---

## Step 6 — Entanglement & Quantum Protocols

📄 **Code:** [`week6/teleportation_ecc.py`](week6/teleportation_ecc.py)

**What the program demonstrates:** All four Bell states, quantum teleportation with fidelity verification across multiple input states, superdense coding, 3-qubit bit-flip syndrome error correction code.

### Key Concepts

The final step ties everything together with quantum protocols that have **no classical analogues** — all powered by entanglement.

#### Bell States

The four **Bell states** are maximally entangled 2-qubit states (entanglement entropy = 1 ebit each):

```
|Φ⁺⟩ = (|00⟩ + |11⟩)/√2   ← default Bell pair (used in teleportation)
|Φ⁻⟩ = (|00⟩ − |11⟩)/√2
|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
|Ψ⁻⟩ = (|01⟩ − |10⟩)/√2
```

Created by: **H on qubit 0**, then **CNOT(0→1)**, starting from `|00⟩`.

#### No-Cloning Theorem

It is **impossible** to create a perfect copy of an unknown quantum state. This is a fundamental consequence of linearity of quantum mechanics, and is why quantum information behaves so differently from classical information.

#### Quantum Teleportation

Transmit an unknown qubit state `|ψ⟩` from Alice to Bob using a shared Bell pair and 2 classical bits. The state is **destroyed at the source** and **reconstructed at the destination**.

```
1. Alice & Bob share Bell pair |Φ⁺⟩ (q1 with Alice, q2 with Bob)
2. Alice entangles her message qubit |ψ⟩ with her Bell qubit (q1)
3. Alice measures (q0, q1) → gets 2 classical bits (00/01/10/11)
4. Alice sends those 2 bits to Bob (classical channel)
5. Bob applies corrections based on the bits:
       c=01 → X on q2
       c=10 → Z on q2
       c=11 → X then Z on q2
6. Bob's qubit q2 is now in state |ψ⟩
```

This is **not faster-than-light communication** — the 2 classical bits must be transmitted conventionally. The quantum channel and classical channel together enable perfect state transfer.

#### Superdense Coding

The *dual* of teleportation: send **2 classical bits** using just **1 qubit** (plus a shared Bell pair). Alice applies a local gate to her qubit to encode 2 bits; Bob performs a Bell measurement to decode them. Together, these two protocols saturate the fundamental limits of quantum communication.

#### Quantum Error Correction — 3-Qubit Bit-Flip Code

Real qubits suffer from **decoherence**: random bit flips and phase errors caused by environmental noise. Quantum error correction protects information by encoding one **logical qubit** into multiple **physical qubits**.

The **3-qubit repetition code** corrects any single bit-flip error:

```
Encoding:
    |0_L⟩ = |000⟩
    |1_L⟩ = |111⟩
    |ψ_L⟩ = α|000⟩ + β|111⟩

Syndrome measurement (non-destructive!):
    s0 = q0 ⊕ q1   (parity of qubits 0 and 1)
    s1 = q0 ⊕ q2   (parity of qubits 0 and 2)

Syndrome table:
    00 → no error
    10 → qubit 1 flipped → apply X to q1
    01 → qubit 2 flipped → apply X to q2
    11 → qubit 0 flipped → apply X to q0
```

The key insight: syndrome measurement reveals **which qubit** flipped without collapsing the logical superposition — it extracts error information while preserving quantum information.

```
Run:  python week6/teleportation_ecc.py
```

---

## Quick Reference

### Key formulas

| Concept | Formula |
|---------|---------|
| Qubit state | `\|ψ⟩ = α\|0⟩ + β\|1⟩`,  `\|α\|² + \|β\|² = 1` |
| Bloch sphere | `\|ψ⟩ = cos(θ/2)\|0⟩ + e^(iφ)sin(θ/2)\|1⟩` |
| Born rule | `P(k) = \|⟨k\|ψ⟩\|²` |
| Inner product | `⟨φ\|ψ⟩ = φ†ψ`  (complex scalar) |
| Tensor product | `\|ψ⟩ ⊗ \|φ⟩ = kron(ψ, φ)` |
| Unitarity | `U†U = I` |
| Bell state Φ⁺ | `(\|00⟩ + \|11⟩)/√2` |
| QFT | `\|j⟩ → (1/√N) Σₖ e^(2πijk/N)\|k⟩` |
| Grover iterations | `~π√N/4`  for 1 marked item out of N |
| Teleportation cost | 1 Bell pair + 2 classical bits → 1 qubit state |
| 3-qubit code | `\|0_L⟩=\|000⟩`,  `\|1_L⟩=\|111⟩`, corrects 1 bit-flip |

### Quantum speedup summary

| Algorithm | Problem | Classical | Quantum | Type |
|-----------|---------|-----------|---------|------|
| Deutsch–Jozsa | Constant vs balanced | O(2ⁿ) | O(1) | Exponential |
| Grover's search | Unstructured search | O(N) | O(√N) | Quadratic |
| Shor's factoring | Integer factorization | O(e^(n^(1/3))) | O(n³) | Exponential |
| QFT | Fourier transform | O(N log N) | O(n²) | Exponential |

### Gate quick reference

| Gate | Matrix | Effect |
|------|--------|--------|
| X | `[[0,1],[1,0]]` | Bit flip (NOT) |
| Z | `[[1,0],[0,-1]]` | Phase flip |
| H | `[[1,1],[1,-1]]/√2` | Superposition |
| S | `[[1,0],[0,i]]` | 90° phase (√Z) |
| T | `[[1,0],[0,e^(iπ/4)]]` | 45° phase (√S) |
| CNOT | 4×4 controlled-X | Entangling gate |

---

*Built with [Qiskit](https://qiskit.org/) and NumPy. Circuits run on Qiskit Aer simulator and IBM Quantum hardware.*
