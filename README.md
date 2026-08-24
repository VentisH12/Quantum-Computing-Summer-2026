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

## Setup

```bash
pip install numpy matplotlib qiskit qiskit-aer
```

> Python 3.9+ recommended. All files run with Qiskit 1.x and Aer 0.14+.

---

## Table of Contents

- [Step 1 — Qubits & Quantum Intuition](#step-1--qubits--quantum-intuition)
- [Step 2 — Linear Algebra Foundations](#step-2--linear-algebra-foundations)
- [Step 3 — Quantum Gates & Circuits](#step-3--quantum-gates--circuits)
- [Step 4 — Early Quantum Algorithms](#step-4--early-quantum-algorithms)
- [Step 5 — QFT & Shor's Algorithm](#step-5--qft--shors-algorithm)
- [Step 6 — Entanglement & Quantum Protocols](#step-6--entanglement--quantum-protocols)
- [Quick Reference](#quick-reference)

---

## Step 1 — Qubits & Quantum Intuition

📄 **Code:** [`qubit_simulator.py`](qubit_simulator.py)

### What I Did and Why

The first step was to build a qubit simulator entirely from scratch using only NumPy — no Qiskit, no quantum libraries. The reason for this was deliberate: before trusting a framework to do quantum computation, it is important to understand what a qubit actually *is* at a mathematical level and to see that the "magic" of quantum behavior is really just linear algebra playing out on complex numbers.

Think of it as learning to drive a car by first understanding how the engine works, rather than just pressing the accelerator and hoping for the best.

#### What is a qubit, in plain English?

A regular computer bit is like a light switch — it is either off (0) or on (1), always one or the other, nothing in between. A **qubit** is fundamentally different. Before you look at it, it exists in a **superposition** — a blend of both 0 and 1 simultaneously, with each possibility carrying a weight called an **amplitude**. The moment you measure it, the superposition collapses and you get a definite 0 or 1, but the *probability* of each outcome was determined by those amplitudes beforehand.

Mathematically:

```
|ψ⟩ = α|0⟩ + β|1⟩
```

Here α and β are complex numbers whose squares tell you the probabilities: `|α|²` is the chance of getting 0, and `|β|²` is the chance of getting 1. They must always add up to 1 (you always get *some* answer). This rule — probability from squared amplitude — is called the **Born rule**.

#### What is a quantum gate?

If bits are manipulated by logic gates (AND, OR, NOT), qubits are manipulated by **quantum gates** — mathematical operations that rotate the qubit's state without measuring it. The first and most important gate introduced here is the **Hadamard gate (H)**.

The Hadamard gate takes a qubit that is definitely 0 and puts it into a perfect 50/50 superposition:

```
H|0⟩ = (|0⟩ + |1⟩)/√2   →  50% chance of 0, 50% chance of 1
```

Think of it as spinning a coin that was lying flat on the table — it is now both heads and tails until it lands.

But the Hadamard gate does something even more interesting than just scrambling a qubit randomly. It preserves a hidden quantity called **phase** — the sign of the amplitude. This sign is invisible to a single measurement, but it becomes critically important when gates are combined, because phases can *cancel each other out*.

#### What is quantum interference?

This is where quantum mechanics becomes genuinely strange. If you apply the Hadamard gate *twice* to the same qubit, you would naively expect to get a random result twice and end up more scrambled. Instead, the opposite happens: you get back exactly what you started with, every single time.

```
H → H → |0⟩  always measures as 0, never as 1
```

This is **destructive interference**. The first H creates two paths (0 and 1). The second H mixes those paths again, but the negative phase on the `|1⟩` branch causes it to cancel itself out — the two "roads to 1" arrive with opposite signs and annihilate. The two "roads to 0" both arrive with the same sign and reinforce. The result is certain: you always get 0.

This is not just a mathematical curiosity — it is the central trick behind why quantum computers can solve certain problems faster than classical ones. Quantum algorithms are carefully engineered so that wrong answers cancel out and right answers reinforce.

#### The Bloch Sphere

To visualize qubit states geometrically, every possible qubit state is mapped onto the surface of a unit sphere called the **Bloch sphere**:

- `|0⟩` is the **north pole**
- `|1⟩` is the **south pole**
- `H|0⟩ = |+⟩` sits on the **equator**, pointing along the X-axis
- Applying the S or T gate rotates the state around the vertical axis, changing the phase
- Every quantum gate is a **rotation** of this sphere

This is powerful for building intuition: instead of thinking about complex algebra, you can think about where on the sphere a qubit is sitting and which direction each gate rotates it.

#### What is a shot-based experiment?

Because measurement is probabilistic, we cannot learn the full probability distribution from a single measurement. Instead, the simulator runs the same circuit hundreds or thousands of times (each run is a "shot"), resets the qubit to its initial state, and builds up a histogram of outcomes. With 1000 shots, the histogram converges closely to the true probabilities — the same way flipping a fair coin 1000 times gives you close to 50/50 heads and tails.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| Qubit state `\|ψ⟩ = α\|0⟩ + β\|1⟩` | A weighted combination of 0 and 1, with complex weights |
| Born rule `P(k) = \|α\|²` | Probability of each outcome = squared magnitude of its amplitude |
| Hadamard gate H | Creates or removes superposition; H·H = Identity |
| Phase | The sign (or complex angle) of an amplitude; invisible alone but causes interference |
| Destructive interference | Two paths to same outcome with opposite phases cancel |
| Constructive interference | Two paths with matching phases reinforce |
| Bloch sphere | Every qubit state is a point on a unit sphere; gates are rotations |
| Shot-based measurement | Repeat the circuit many times to estimate probabilities statistically |

---

### Output Analysis

#### Plot 1 — Bloch Sphere

![Bloch Sphere](images/step1_bloch_sphere.png)

This plot shows four key qubit states as arrows pointing from the center of a unit sphere outward to its surface. Each arrow represents a distinct qubit state.

**What to notice:**

- **`|0⟩` (purple arrow, pointing straight up)** — the north pole. This is the starting state: 100% probability of measuring 0. An arrow pointing directly up means no `|1⟩` component at all.

- **`|1⟩` (pink arrow, pointing straight down)** — the south pole. Flipped completely: 100% probability of measuring 1.

- **`H|0⟩ = |+⟩` (green arrow, pointing along the equator)** — after one Hadamard gate. The arrow lies perfectly on the equator, meaning equal 50/50 probability. Crucially, it points along the positive X-axis — the *phase* of the superposition is real and positive.

- **`|i⟩` (blue arrow, pointing into the page along the Y-axis)** — a state with a complex phase `e^(iπ/2) = i` on the `|1⟩` component. This state also gives 50/50 probabilities when measured, but it looks completely different from `|+⟩` on the Bloch sphere because its phase is 90° rotated. This illustrates that **phase is a real physical quantity**, not just a mathematical bookkeeping artifact.

**Key takeaway:** Two states can have identical measurement probabilities but completely different physical properties because of their phase. The Bloch sphere makes this difference visible in a way that raw numbers cannot.

---

#### Plot 2 — Measurement Histogram

![Measurement Histogram](images/step1_measurement_histogram.png)

This bar chart shows the results of running four different gate sequences on `|0⟩`, each measured 1000 times. The purple bars are the fraction of times the qubit measured as 0; the orange bars as 1.

**Reading each result left to right:**

**`H|0⟩` — Single Hadamard:** The bars are nearly equal at ~0.514 / 0.486. This is the coin flip. The qubit is in a genuine 50/50 superposition and each measurement is random. The slight deviation from exactly 50/50 is normal statistical noise from 1000 finite samples — like getting 514 heads in 1000 coin flips.

**`H·H|0⟩` — Two Hadamards:** The purple bar hits 1.000 and the orange bar is exactly 0.000. Out of 1000 measurements, not a single 1 appeared. This is the interference result — **perfectly deterministic despite going through a superposition in the middle.** This should feel counterintuitive: you put the qubit into a random state, then made it random again, and somehow got a guaranteed answer. That is destructive interference at work. The first H created equal amplitudes of 0 and 1. The second H combined those amplitudes in a way that made the `|1⟩` paths cancel to zero and the `|0⟩` paths double up to 1.

**`X|0⟩` — NOT gate:** The orange bar is 1.000, purple is 0.000. The X gate is simply a bit flip: it turns `|0⟩` into `|1⟩` with certainty. Deterministic, classical-like behavior.

**`H·Z·H|0⟩` — Sandwiched Z gate:** Again, 0.000 / 1.000 — always measures 1. This one is more subtle. The Z gate flips the *phase* of the `|1⟩` component (`+` becomes `−`) but does nothing visible if you measure immediately. However, sandwiching it between two Hadamard gates converts that invisible phase flip into a visible bit flip. This demonstrates **phase kickback** in its simplest form: the Z gate acts invisibly in the computational basis, but the Hadamard gates on either side translate its effect into something measurable. This same principle underlies Deutsch–Jozsa and Shor's algorithm — it is one of the deepest ideas in quantum computing, appearing here in its simplest form.

---

## Step 2 — Linear Algebra Foundations

📄 **Code:** [`tensor_visualizer.py`](tensor_visualizer.py)

### What We Did and Why

Step 1 built intuition about what a qubit *is*. Step 2 builds the mathematical language needed to work with *multiple* qubits — and to understand why quantum computing becomes exponentially powerful as you add more of them.

The goal was to implement the core linear algebra operations of quantum mechanics from scratch: inner products, outer products, and — most importantly — the **tensor product**, which is how you combine two separate qubit spaces into one joint system. This step also introduces the concept of **entanglement** and gives it a precise, computable measure.

#### Why bother with linear algebra?

Quantum mechanics *is* linear algebra. A qubit is a vector. A gate is a matrix multiplication. Measuring is a projection. Once you understand these operations on 1 qubit, you need the tensor product to extend them to 2, 3, or 100 qubits. There is no other way to describe multi-qubit systems — the tensor product is the mathematical foundation that everything from Bell states to Shor's algorithm is built on.

#### What is the tensor product, in plain English?

Imagine you have two separate drawers. One drawer has a ball that is either red (0) or blue (1). The other drawer also has a ball that is either red or blue. Together, the two drawers can be in four possible states: `(red, red)`, `(red, blue)`, `(blue, red)`, or `(blue, blue)`. The tensor product is the operation that combines the two separate "one-ball" descriptions into one "two-ball" description.

For qubits, combining a 2-dimensional qubit with another 2-dimensional qubit gives a **4-dimensional** space. Three qubits give 8 dimensions. Ten qubits give 1,024 dimensions. n qubits give `2ⁿ` dimensions. This exponential growth in the size of the description space is where quantum computing's power comes from — a quantum computer with 300 qubits would require more classical numbers to fully describe than there are atoms in the observable universe.

#### What is the inner product?

The inner product `⟨φ|ψ⟩` measures how much two quantum states "overlap" — how similar they are. If two states are identical, the inner product is 1. If they are completely different (orthogonal), it is 0. This is the quantum version of asking "how alike are these two states?"

The simulator confirmed: `⟨0|0⟩ = 1` (identical), `⟨0|1⟩ = 0` (orthogonal), `⟨+|−⟩ = 0` (the Hadamard basis states are also orthogonal to each other, just rotated 90° from the computational basis).

#### What makes a gate valid? Unitarity.

Not every matrix is a valid quantum gate. The constraint is **unitarity**: `U†U = I` (the gate times its conjugate transpose gives the identity). This constraint has two physical consequences. First, it guarantees that the total probability always stays at 1 — probabilities cannot leak or be created. Second, it means every gate is **reversible** — you can always run it backwards. This is unlike classical gates like AND, which throw away information.

All six gates were verified to be unitary: `H, X, Z, H⊗I, CNOT, SWAP`. Any matrix failing this test cannot be a physical quantum gate.

#### What is entanglement, precisely?

Two qubits are **entangled** if their joint state cannot be written as a product of two separate single-qubit states. The Bell state `(|00⟩ + |11⟩)/√2` is the clearest example: you cannot write it as `|something⟩ ⊗ |something else⟩`. The two qubits are genuinely correlated — measuring one immediately determines the outcome of measuring the other, regardless of distance.

The **Schmidt decomposition** makes this testable: it decomposes a 2-qubit state into its most independent form. If the decomposition requires more than one term, the state is entangled. The number of terms is the **Schmidt rank**, and the **entanglement entropy** (ranging from 0 to 1) measures the degree of entanglement. All four Bell states scored exactly 1.0; the product state `|+⟩⊗|0⟩` scored exactly 0.0.

#### What is the partial trace?

If you have two entangled qubits and you "forget" or "throw away" one of them, what does the remaining qubit look like? The partial trace computes this. For a Bell state, tracing out either qubit leaves behind a **maximally mixed state** — a 50/50 random qubit with no definite state. This means Alice, holding one qubit of a Bell pair, sees pure randomness when she measures alone. The correlations only appear when Alice and Bob compare their results — and that comparison requires a classical communication channel, which is why entanglement cannot be used for faster-than-light signalling.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| Tensor product `⊗` | Combines two qubit spaces into one joint space; n qubits → 2ⁿ dimensions |
| Inner product `⟨φ\|ψ⟩` | Overlap between two states; 1 = identical, 0 = orthogonal |
| Outer product `\|ψ⟩⟨φ\|` | Creates a matrix from two states; density matrix when `\|ψ⟩⟨ψ\|` |
| Unitarity `U†U = I` | Guarantees probability preservation and reversibility |
| Entanglement | Two qubits whose joint state cannot be factored into independent parts |
| Schmidt decomposition | Reveals the "independent structure" of a 2-qubit state |
| Entanglement entropy | 0 = no entanglement, 1 = maximally entangled (Bell state) |
| Partial trace | Describes one qubit after "forgetting" the other; always mixed for entangled states |

---

### Output Analysis

#### Plot 1 — H⊗I Gate Matrix

![Gate Matrix](images/step2_gate_matrix.png)

This heatmap shows the 4×4 matrix representing the gate `H⊗I` — Hadamard on qubit 0, identity (do nothing) on qubit 1. The left panel shows the **magnitude** of each matrix entry; the right panel shows its **phase** (angle in the complex plane).

**What to notice:**

The matrix has a clear **block structure**: the top-left and top-right 2×2 blocks both contain the Hadamard pattern (`0.707, 0.707, 0.707, -0.707`), while qubit 1 is left untouched (its dimension just repeats the pattern). This block structure is the visual signature of a tensor product gate — the Hadamard "lives in" the qubit 0 subspace and the identity "lives in" the qubit 1 subspace, and they do not interact.

The phase panel is uniformly zero (blue) everywhere except for one entry, which is `π` (red) — corresponding to the `−1/√2` entry of the Hadamard matrix. This confirms the gate is real-valued with one sign flip, exactly as expected.

**Key takeaway:** Tensor product gates are block-structured. When you see that block pattern in a matrix heatmap, you know the gate is separable — it acts independently on each qubit with no entanglement between them.

---

#### Plot 2 — Bell State Φ⁺

![Bell State Phi+](images/step2_bell_state.png)

This three-panel plot shows the state `|Φ⁺⟩ = (|00⟩ + |11⟩)/√2` — the most fundamental entangled state in quantum computing, created by H on qubit 0 followed by CNOT.

**Left panel — Probability bars:** Only `|00⟩` and `|11⟩` have nonzero probability, each exactly 0.500. The states `|01⟩` and `|10⟩` have zero probability. This is the signature of the Bell state: the two qubits are *perfectly correlated* — if qubit 0 is 0, qubit 1 is guaranteed to be 0; if qubit 0 is 1, qubit 1 is guaranteed to be 1. They are never found in opposite states.

**Middle panel — Phase diagram:** Both nonzero amplitudes (`|00⟩` and `|11⟩`) are shown as arrows pointing in the same direction along the positive real axis, both of length `1/√2 ≈ 0.707`. The fact that they point in the *same direction* means their phases are identical — this is the `+` in `Φ⁺`. If the `|11⟩` arrow pointed in the opposite direction, it would be `Φ⁻`.

**Right panel — Probability heatmap:** The 2×2 grid has probability 0.5 in the top-left (`|00⟩`) and bottom-right (`|11⟩`) corners, and zero in the other two corners. This checkerboard-like diagonal pattern is the visual signature of a maximally entangled state. A product state would show a smooth, non-diagonal distribution.

**Key takeaway:** The Bell state's probability heatmap has a distinctive diagonal pattern — only correlated outcomes appear. This cannot be produced by any pair of independent qubits, which is the mathematical definition of entanglement.

---

#### Plot 3 — Product State |+⟩⊗|0⟩

![Product State](images/step2_product_state.png)

This three-panel plot shows the state `|+⟩⊗|0⟩` — qubit 0 in superposition, qubit 1 fixed at 0. This is a **separable** (non-entangled) state with entanglement entropy = 0.

**Left panel — Probability bars:** All four basis states have nonzero probability: `|00⟩ = 0.5`, `|01⟩ = 0`, `|10⟩ = 0.5`, `|11⟩ = 0`. Wait — `|01⟩` and `|11⟩` are zero. This makes sense: qubit 1 is always `|0⟩`, so any outcome with qubit 1 = 1 has zero probability. Qubit 0 is in `|+⟩` superposition, so outcomes with qubit 0 = 0 and qubit 0 = 1 are equally likely.

**Middle panel — Phase diagram:** Both arrows for `|00⟩` and `|10⟩` point along the positive real axis at the same length, just as in the Bell state. But notice that *both* are in the same two states — qubit 1 never appears as 1. The phase structure confirms this is a real-valued product state with no complex phases.

**Right panel — Probability heatmap:** Compare this to the Bell state heatmap: instead of a diagonal pattern, we now see a **column pattern** — the left column (`q₁=0`) has probability 0.5 in both rows, and the right column (`q₁=1`) is entirely zero. This column structure shows that qubit 1's state is completely independent of qubit 0 — exactly what "product state" means. You could factor the full joint description into `P(q₀) × P(q₁)` separately.

**Key takeaway:** The contrast between these two heatmaps is one of the clearest visual demonstrations of what entanglement *is*. Product states have column/row patterns (factorizable). Entangled states have diagonal or off-diagonal patterns (non-factorizable). The Bell state heatmap simply cannot be explained by two independent coins.

---

## Step 3 — Quantum Gates & Circuits

📄 **Code:** [`circuit_simulator.py`](circuit_simulator.py)

### What We Did and Why

Steps 1 and 2 built the foundation: what a qubit is, how states are described mathematically, and how to combine multiple qubits. Step 3 puts that foundation to work by building a **full 2-qubit circuit simulator** — a system that can apply any sequence of gates to two qubits, draw the circuit as a diagram, and measure the result.

The purpose of building this from scratch rather than just using Qiskit is the same as before: you understand something much more deeply when you implement it yourself. Every gate in this simulator is a real matrix multiplication. The circuit diagram is generated from the actual sequence of operations. The measurement statistics come from genuine probabilistic sampling. There is no black box.

This step also demonstrates two ideas that bridge from bare gates to actual algorithms: **SWAP decomposition** (showing that complex gates can always be broken down into simpler ones) and **phase kickback** (the mechanism that almost every quantum speedup relies on).

#### What is a quantum circuit?

A quantum circuit is a recipe. It specifies: start with these qubits in these initial states, apply these gates in this order, then measure. It is read left to right, just like sheet music. Qubits are horizontal wires; gates are boxes placed on those wires; measurements are the final step that extract classical information.

The simplest meaningful 2-qubit circuit — the Bell state circuit — looks like this:

```
q0: |0⟩─[H]─[●]─┤M├
q1: |0⟩──────[⊕]─┤M├
```

Read it left to right: apply Hadamard to q0, then apply CNOT with q0 as control and q1 as target, then measure both. This two-gate circuit creates the most entangled state in quantum mechanics.

#### What does each gate actually do?

The simulator implements the full standard gate library. Each gate is a 2×2 or 4×4 matrix:

- **X (NOT):** Flips `|0⟩` to `|1⟩` and vice versa. The quantum version of a classical NOT gate.
- **Z (phase flip):** Leaves `|0⟩` alone but changes the sign of `|1⟩`. Invisible if you measure immediately, but has dramatic effects when combined with H.
- **H (Hadamard):** Creates or removes superposition. Moves the Bloch sphere vector from a pole to the equator or vice versa.
- **S and T (phase gates):** Rotate the qubit around the vertical axis of the Bloch sphere by 90° and 45° respectively. These fine rotations are what make the `{H, T, CNOT}` gate set universal.
- **CNOT:** The essential 2-qubit gate. It flips the *target* qubit if and only if the *control* qubit is `|1⟩`. When the control qubit is in superposition, CNOT creates entanglement.
- **Rx, Ry, Rz:** Continuous rotation gates parameterized by an angle θ. These are the workhorses of variational quantum algorithms.

#### Why does SWAP equal three CNOTs?

The SWAP gate exchanges the states of two qubits. It seems like it should be a primitive operation, but it can always be decomposed into three CNOT gates:

```
SWAP = CNOT(0→1) · CNOT(1→0) · CNOT(0→1)
```

This matters because real quantum hardware often has limited connectivity — not every qubit can interact directly with every other qubit. SWAP via CNOTs lets you route quantum information across a chip even when direct connections are not available. The simulator verified this: starting from `|10⟩`, both the direct SWAP and the three-CNOT version produced identical output `|01⟩`.

#### What is phase kickback?

This is one of the most important and counterintuitive ideas in quantum computing. When a controlled gate acts on a *target qubit that is in an eigenstate* of that gate, the eigenvalue (the phase associated with that eigenstate) transfers — "kicks back" — onto the control qubit.

The clearest demonstration: `H → Z → H` on a single qubit. The Z gate flips phase, but you cannot see a phase flip if you measure immediately. However, the H gates on either side translate the phase flip into a bit flip — the qubit that started as `|0⟩` ends up as `|1⟩`. The Z gate did its work invisibly in the superposition basis, and the H gates made it visible. This exact mechanism is what lets Deutsch–Jozsa solve its problem in one query, and what gives Shor's algorithm its power.

#### What does "universal gate set" mean?

A universal gate set is a small collection of gates that can, in combination, approximate any quantum operation to any desired precision. The set `{H, T, CNOT}` is universal for quantum computing — just as `{NAND}` is universal for classical computing. Every quantum algorithm, no matter how complex, can be compiled down to sequences of just these three gates.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| Quantum circuit | Ordered sequence of gates applied to qubits, read left to right |
| CNOT | 2-qubit controlled-NOT; flips target iff control is `\|1⟩`; creates entanglement |
| SWAP via 3 CNOTs | Any SWAP can be decomposed into three CNOT gates |
| Phase kickback | Controlled gate on eigenstate transfers phase to control qubit |
| Universal gate set | `{H, T, CNOT}` can approximate any quantum operation |
| Circuit unitary | The net 4×4 matrix of the full circuit; product of all gate matrices |
| ASCII circuit diagram | Visual notation showing gate sequence on each qubit wire |

---

### Output Analysis

#### Plot 1 — Bell Circuit Output (H → CNOT → H)

![Bell Circuit Output](images/step3_bell_circuit_output.png)

This three-panel plot shows the output of an interference circuit: Hadamard on q0, CNOT, then Hadamard on q0 again. Starting from `|00⟩`, this produces a *uniform superposition* of all four 2-qubit basis states.

**Left panel — Probability bars:** All four states (`|00⟩`, `|01⟩`, `|10⟩`, `|11⟩`) are equally likely at exactly 0.250 each. The dotted line marks the uniform distribution `1/4`. Every bar hits it precisely.

This result is interesting because it shows how interference can be used to *spread* probability uniformly — the same mechanism that Grover's algorithm starts with. Before Grover amplifies the target, it first needs this uniform superposition as its starting point.

**Middle panel — Phase diagram:** All four amplitudes are real and equal at `0.5`, pointing in the same direction along the positive real axis. No phases are different from each other. This is a "flat" state — maximum uncertainty, no structure, maximum entropy.

**Right panel — Shot-based histogram (1024 shots):** The four outcome counts are nearly equal, with small fluctuations from random sampling. This is what 1024 coin flips across four outcomes looks like. The slight unevenness is expected statistical noise — with infinite shots, all four would converge to exactly 0.25.

**Key takeaway:** The circuit `H → CNOT → H` acts as a 2-qubit "scrambler" — it takes a definite starting state and distributes probability equally across all four possibilities. This is the quantum analogue of randomizing.

---

#### Plot 2 — Circuit Unitary Heatmap

![Circuit Unitary](images/step3_circuit_unitary.png)

This heatmap shows the full 4×4 unitary matrix of the Bell-state circuit `(H⊗I) · CNOT`. The left panel is magnitudes; the right panel is phases.

**Left panel — Magnitudes:** Every entry in the matrix has the same magnitude: `0.71 ≈ 1/√2`. The matrix is completely "flat" in terms of magnitude — no entry is more important than any other. This is the signature of a circuit that spreads probability uniformly, which is exactly what we saw in the output.

**Right panel — Phases:** This is where the structure lives. The phases are not all the same — some entries are 0 (blue) and some are `π` (red, meaning a sign of `−1`). The specific pattern of phases determines exactly which quantum states interfere constructively and which interfere destructively. Change even one phase and the circuit produces a completely different output.

**Key takeaway:** In quantum computing, magnitude tells you *how much* of each state is present; phase tells you *how those states will interfere*. The magnitude panel looks uniform and featureless, but the phase panel contains the real computational content of the circuit. This is why phase is so important — and why it is so easy to underestimate.

---

## Step 4 — Early Quantum Algorithms

📄 **Code:** [`algorithms_qiskit.py`](algorithms_qiskit.py)

### What We Did and Why

This is the step where everything from the first three steps finally pays off. Steps 1–3 built the machinery — qubits, linear algebra, gates, circuits. Step 4 uses that machinery to implement the first real quantum algorithms: problems where a quantum computer provably outperforms any classical computer.

Two algorithms are implemented here using Qiskit (IBM's quantum computing framework) and run on a simulated quantum computer:

1. **Deutsch–Jozsa** — the simplest example of an exponential quantum speedup
2. **Grover's search** — the most broadly applicable quantum speedup, relevant to any search problem

The shift to Qiskit here is deliberate: after building everything from scratch, you learn to use the professional tools that let you run on real IBM quantum hardware.

#### What is an oracle?

Both algorithms use the concept of an **oracle** — a black-box function that you can query but cannot inspect directly. Think of it as a locked box with an input slot and an output light: you put a number in, and a light either turns on or stays off. You cannot open the box and look at the mechanism — you can only ask it questions.

In quantum computing, an oracle is implemented as a quantum gate that flips the *phase* of any input state where the answer is "yes":

```
Oracle|target⟩ → −|target⟩    (phase flips for "yes" inputs)
Oracle|other⟩  →  |other⟩     (unchanged for "no" inputs)
```

The phase flip is invisible to a single measurement, but when combined with interference, it becomes the signal that the algorithm amplifies.

#### Deutsch–Jozsa: the first quantum speedup

**The problem:** You are given a function that takes n-bit inputs and outputs either 0 or 1. You are promised it is one of two types: *constant* (always outputs the same answer regardless of input) or *balanced* (outputs 0 for exactly half the inputs and 1 for the other half). Which type is it?

**Classically:** In the worst case, you need to check `2ⁿ⁻¹ + 1` inputs before you can be certain. For n=100, that is more than a billion-trillion queries.

**Quantum:** Exactly **one query**, always. Regardless of n.

The trick: put all `2ⁿ` inputs into superposition simultaneously, query the oracle once, then apply interference. If the function is constant, all the "balanced" interference patterns cancel and you measure all zeros. If the function is balanced, the constant parts cancel and you measure something nonzero. The answer is in the interference pattern, not in any single query result.

The three oracle types tested were:
- `constant_0`: always returns 0
- `constant_1`: always returns 1
- `balanced`: returns 0 for half the inputs, 1 for the other half

All three gave the correct deterministic answer in a single circuit execution.

#### Grover's search: finding a needle in a quantum haystack

**The problem:** You have N unsorted items (a list, a database, a search space) and one item is "marked" as the target. Find it.

**Classically:** On average you must check N/2 items. For a billion items, that is 500 million checks.

**Quantum (Grover):** Only `~π√N/4` checks needed. For a billion items, that is about 25,000 — nearly 20,000 times faster.

The algorithm works by repeatedly doing two things:
1. **Oracle step:** Flip the phase of the target state — give it a "negative amplitude"
2. **Diffusion step:** Reflect all amplitudes about their average value

Each round, the target's amplitude grows while all other amplitudes shrink. After the optimal number of rounds, the target's amplitude is so large that measuring the register almost certainly returns the target. Going one round too many causes the amplitude to overshoot and start shrinking again — which is why the sweep plot matters.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| Oracle | Black-box function encoded as a phase-flipping quantum gate |
| Deutsch–Jozsa | Distinguishes constant from balanced functions in 1 query; exponential speedup |
| Superposition querying | Query all `2ⁿ` inputs simultaneously via superposition |
| Grover's algorithm | Finds 1 marked item in N using `O(√N)` oracle calls; quadratic speedup |
| Diffusion operator | Reflects all amplitudes about their mean; amplifies above-average amplitudes |
| Amplitude amplification | The general technique behind Grover — grow target, shrink others, via interference |
| Optimal iterations | `~π√N/4`; going past this reduces success probability sinusoidally |

---

### Output Analysis

#### Plot 1 — Deutsch–Jozsa Results

![Deutsch-Jozsa](images/step4_deutsch_jozsa.png)

This three-panel histogram shows the measurement results for all three oracle types on a 3-qubit input register (8 possible inputs: `000` through `111`).

**Left panel — `constant_0`:** 100% of measurements land on `000`. The all-zeros result is the universal signature of a constant function — the interference cancelled all nonzero outcomes completely.

**Middle panel — `constant_1`:** Again, 100% land on `000`. Even though this oracle always outputs 1 instead of 0, it is still *constant*, so the answer is still all-zeros. The algorithm does not care what the constant value is — only whether the function is constant at all.

**Right panel — `balanced`:** 100% of measurements land on `111`. Not a single shot produced `000`. The balanced function's interference pattern completely eliminates the all-zeros outcome and concentrates everything into a nonzero result.

**Key takeaway:** The contrast between panels is absolute. These are not probabilistic outcomes — they are deterministic. `000` means constant; anything else means balanced. One circuit, run once, correct answer guaranteed. No classical algorithm can match this efficiency.

---

#### Plot 2 — Grover's Search (3 qubits, target = |101⟩)

![Grover 3-qubit](images/step4_grover_3q.png)

This bar chart shows the measurement distribution after running Grover's algorithm on 3 qubits (8 items) searching for item 5, represented as the binary string `|101⟩`, with 2 iterations.

**The dominant bar:** `|101⟩` towers above all other outcomes with close to 95% probability. The 7 other possible answers each have tiny residual probabilities (around 1% each) — the amplitude amplification process was not quite perfect (2 iterations is optimal but not exact for n=3), so a small amount of probability leaked into the wrong answers.

**Classical comparison:** Without Grover, each of the 8 items would have 12.5% probability. You would need on average 4 queries. Grover found the answer in 2 queries with 95% confidence — a nearly 2× speedup for this small example. The speedup becomes dramatically larger at scale (see the scaling table above).

**Key takeaway:** This plot visually captures amplitude amplification. The "winner" bar is roughly 7.5× taller than it would be under uniform distribution — that height is built up over 2 rounds of oracle-then-diffusion, each round boosting the target at the expense of all others.

---

#### Plot 3 — Grover's Search (4 qubits, 2 targets)

![Grover 4-qubit](images/step4_grover_4q.png)

This plot shows Grover's algorithm on 4 qubits (16 items) searching for *two* simultaneous targets: items 3 (`|0011⟩`) and 11 (`|1011⟩`).

**Two spikes:** Both target states rise to high probability after 2 iterations. The other 14 states each have near-zero probability. Grover's algorithm works for multiple targets — when there are M targets in N items, the optimal number of iterations becomes `~π√(N/M)/4`, which is fewer rounds because there are more "winning" states to amplify.

**Interesting asymmetry:** The two targets may not be exactly equal in probability due to finite-shot statistical noise (this was measured with 4096 shots, and both targets were about 40–45% each, totaling ~85–90% combined).

**Key takeaway:** Grover is not limited to a single target. Any subset of "marked" items can be searched simultaneously, and the speedup scales as `O(√(N/M))` — still quadratic in the ratio of total to marked items.

---

#### Plot 4 — Amplitude Amplification Sweep

![Grover Sweep](images/step4_grover_sweep.png)

This plot shows how the probability of measuring the target state changes as a function of the number of Grover iterations, for n=3 (8 items), target = `|101⟩`.

**The sinusoidal curve (gray dashed):** This is the theoretical prediction — `P(k) = sin²((2k+1)θ)` where `θ = arcsin(1/√N)`. The probability oscillates sinusoidally with each iteration.

**The simulated points (purple circles):** The actual simulation results match the theory almost exactly, confirming that the implementation is correct.

**Key features to read from the plot:**

- At **k=0** (no iterations, just uniform superposition): probability is `1/8 = 0.125` — the baseline classical random chance
- At **k=1**: probability rises to ~0.78 — already much better than classical
- At **k=2** (optimal): probability peaks at ~0.95 — this is the best the algorithm can do in 2 steps
- At **k=3 and beyond**: the probability *drops* — the algorithm has "overshot" and the amplitudes are now rotating past the target. If you run too many iterations, you actually make things worse.
- The green dashed vertical line marks the optimal stopping point at k=2

**The red dotted horizontal line** at `1/8` marks the classical random-search baseline. Everything above this line is the quantum advantage.

**Key takeaway:** Knowing when to stop is as important as the algorithm itself. The optimal number of iterations is not "run it until it converges" — it is a precise value dictated by the geometry of the quantum state space. Too few iterations and the signal has not fully built up; too many and it starts to fade. This is unlike classical algorithms, which generally do not degrade from running "too long."

---

## Step 5 — QFT & Shor's Algorithm

📄 **Code:** [`qft_shor.py`](qft_shor.py)

### What We Did and Why

This is the most mathematically advanced step — and arguably the most consequential in terms of real-world impact. Step 5 implements the **Quantum Fourier Transform (QFT)** from scratch and uses it to demonstrate the core of **Shor's algorithm**: the quantum method for factoring large integers that, if run on a large enough quantum computer, would break most of the encryption protecting the internet today.

The QFT is built gate by gate, verified against numpy's classical FFT (they should agree mathematically), and then used to build the period-finding circuit that lies at the heart of Shor's algorithm. The target: factor 15 into 3 × 5.

#### What is the Fourier transform, in plain English?

The Fourier transform is a mathematical tool for finding hidden **periodicity** — patterns that repeat. If you hum a musical note into a microphone, the recording looks like a complicated wiggly wave. The Fourier transform decomposes that wave into its pure frequency components — it tells you "this note is mostly 440 Hz, with a bit of 880 Hz and some noise." It converts from a "time description" to a "frequency description."

The classical Discrete Fourier Transform (DFT) does this for lists of numbers. The Quantum Fourier Transform does the same thing, but for quantum states — it converts from a "computational basis description" to a "frequency basis description." And it does it with dramatically fewer operations.

#### Why is the QFT so efficient?

The classical Fast Fourier Transform (FFT) on N numbers needs `O(N log N)` operations. For N = 2ⁿ numbers, that is `O(n · 2ⁿ)` operations — it grows exponentially with the number of qubits n.

The QFT needs only `O(n²)` gates — a Hadamard and some controlled phase rotations per qubit. For n=3, that is 9 operations instead of 24. For n=50, it is 2500 operations instead of 56 trillion. This exponential advantage is real, but it comes with a catch: you cannot directly read out the Fourier coefficients (measurement collapses the state). The QFT is useful only when the Fourier information is encoded in *measurement probabilities* — exactly what Shor's algorithm exploits.

#### What is Shor's algorithm, in plain English?

**The problem:** Given a large number N (say, a 2048-bit RSA key — a number with 617 digits), find its prime factors. This is believed to be classically hard: the best classical algorithms take time that grows exponentially with the number of digits.

**Why it matters:** RSA encryption (used to secure bank transactions, email, HTTPS) relies on the fact that multiplying two large primes is easy but factoring the result is practically impossible. Shor's algorithm breaks this assumption — on a large enough quantum computer, it can factor in polynomial time.

**How it works — the key insight:** Factoring can be reduced to finding a *period*. If you pick a random number `a` and compute `a¹, a², a³, ...` modulo N, the sequence eventually repeats. The length of that cycle is the **period r**. Once you have r, a simple calculation with greatest common divisors (GCD) gives you the factors of N.

Finding the period classically is just as hard as factoring. But the QFT can find it exponentially faster: put all possible exponents into superposition, compute `aˣ mod N` for all x simultaneously, and apply the inverse QFT. The periodicity in the sequence becomes sharp peaks in the measurement histogram — and those peaks tell you r.

**For the specific example `a=7, N=15`:**

```
7¹ mod 15 = 7
7² mod 15 = 4
7³ mod 15 = 13
7⁴ mod 15 = 1   ← back to 1, period r = 4
7⁵ mod 15 = 7   ← repeating
```

Period r = 4. Then:
- `GCD(7² − 1, 15) = GCD(48, 15) = 3`
- `GCD(7² + 1, 15) = GCD(50, 15) = 5`
- Result: `15 = 3 × 5` ✓

The QFT found r = 4 from the phase peaks — then simple arithmetic gave the factors.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| QFT | Quantum Fourier Transform; converts computational basis to Fourier basis in `O(n²)` gates |
| Periodicity → peaks | A quantum state with period r produces measurement peaks at multiples of N/r |
| Phase estimation | Using the QFT to extract the phase (eigenvalue) of a quantum operation |
| Period finding | The core quantum subroutine of Shor's algorithm: find r such that `aʳ ≡ 1 mod N` |
| Shor's algorithm | Factors N in `O(n³)` quantum time; breaks RSA encryption |
| Continued fractions | Classical post-processing step that extracts r from a measured phase fraction |
| QFT verified | All 8 basis state outputs matched numpy FFT magnitudes to within `1e-6` |

---

### Output Analysis

#### Plot 1 — QFT Action on |5⟩

![QFT Action](images/step5_qft_action.png)

This four-panel figure shows what the QFT does to the single basis state `|5⟩ = |101⟩` in a 3-qubit system (N=8 possible states).

**Top-left — Before QFT:** All probability sits on `|101⟩` — a single bar at full height (1.0), everything else zero. This is a completely "certain" state: measuring before the QFT would always give `|101⟩`.

**Top-right — After QFT:** All 8 basis states now have equal probability `1/8 = 0.125`. The red dashed line marks this uniform level, and all bars hit it exactly. The QFT has spread the probability uniformly across all states — but in a very structured way. The structure is hidden in the *phases*, not the probabilities.

**Bottom-left — Real and imaginary amplitudes:** After the QFT, every state has the same magnitude `1/√8 ≈ 0.354`, but different complex phase. The Re (purple) and Im (orange) amplitudes oscillate with a sinusoidal pattern — this is the "frequency fingerprint" of the input state `|5⟩`. The frequency of this oscillation *encodes the value 5*. This is how the Fourier transform works: it converts a "point in space" (which state you're in) into a "frequency" (a sinusoidal pattern across all states).

**Bottom-right — Phase wheel:** Each of the 8 arrows represents one basis state's amplitude as a vector in the complex plane. They are evenly distributed around the unit circle — specifically rotated by `2π × 5k/8` for each state k. This is the mathematical definition of the QFT output: the phases are evenly spaced, and the spacing encodes the input value (5). Every different input state `|j⟩` would produce a different rotation pattern here, uniquely identifiable from the phase arrangement.

**Key takeaway:** The QFT does not change what you *can* measure (probabilities are all equal after QFT), but it encodes rich information in the *phases* of the amplitudes. Shor's algorithm is designed to make those phases observable through interference — the probability peaks in the next plot are a direct result of this phase structure.

---

#### Plot 2 — Shor's Phase Histogram (a=7, N=15)

![Shor Phase Histogram](images/step5_shor_phases.png)

This two-panel plot shows the measurement results from Shor's period-finding circuit for `a=7, N=15`, using 8 counting qubits (256 possible measurement outcomes).

**Left panel — Phase distribution:** The x-axis shows the measured phase (from 0 to 1), and the y-axis shows probability. The histogram is almost entirely empty — except for **four sharp spikes** at exactly `0/4, 1/4, 2/4, 3/4` (marked by the red dashed lines). These are the four peaks corresponding to `k/r` for `k = 0, 1, 2, 3` and period `r = 4`.

This is what "QFT turns periodicity into peaks" looks like in practice. The sequence `7ˣ mod 15` has period 4. The circuit put all exponents into superposition, computed the sequence, and applied the inverse QFT. The QFT detected the periodicity and converted it into four measurement spikes spaced exactly `1/4` apart on the phase axis. Nothing in between has any probability — all that amplitude was cancelled by destructive interference.

**Right panel — Top measured phases:** The 8 most-likely measurement outcomes are shown as a horizontal bar chart. The top four bars correspond to phase values `0/256, 64/256, 128/256, 192/256` — which simplify to `0, 1/4, 1/2, 3/4`. These are exactly the four expected peaks. The continued fractions algorithm takes any one of these measurements (say `64/256 = 1/4`), finds the best rational approximation with a small denominator, and extracts `r = 4` from the denominator.

**Key takeaway:** The histogram is almost entirely blank with four sharp spikes. That sparseness is the quantum advantage. A classical computer would need to compute the sequence `7¹, 7², ..., 7ⁿ mod 15` one by one and look for a repeat — potentially many operations. The quantum circuit "asked" the QFT to find the period across all exponents simultaneously, and the periodicity was revealed directly in the measurement distribution. From the peak at `64/256 = 1/4`, we extract r=4, then compute `GCD(48,15) = 3` and `GCD(50,15) = 5`, and the factorization `15 = 3 × 5` is complete.

---

## Step 6 — Entanglement & Quantum Protocols

📄 **Code:** [`teleportation_ecc.py`](teleportation_ecc.py)

### What We Did and Why

The final step brings together everything learned in Steps 1–5 and uses it to implement three protocols that have no equivalent in classical computing. Each one exploits entanglement in a different way, and each one teaches something fundamental about the nature of quantum information.

The three things implemented here are:
1. **All four Bell states** — the building blocks of quantum communication and entanglement
2. **Quantum teleportation** — transmitting a quantum state using entanglement and classical communication
3. **3-qubit error correction** — protecting quantum information from noise without ever directly measuring it

This step also addresses a practical reality: real quantum computers are noisy. Qubits get corrupted by their environment. Quantum error correction is what makes fault-tolerant quantum computation possible, and understanding even the simplest code (the 3-qubit repetition code) gives genuine insight into why this is hard and how it is solved.

#### What are Bell states, and why do they matter?

The four Bell states are the four maximally entangled 2-qubit states. "Maximally entangled" means the two qubits are as correlated as quantum mechanics allows — measuring one qubit instantly determines the outcome for the other, no matter how far apart they are.

Think of it like this: imagine two coins that are quantum-mechanically linked. Each coin, when flipped, lands heads or tails at random. But if they are in a Bell state, the two coins are *guaranteed* to always land the same way (or always opposite, depending on which Bell state). You cannot predict which way they will land, but you know with certainty that they will match.

The four Bell states differ only in *how* they are correlated:
- `|Φ⁺⟩`: both qubits always agree (both 0 or both 1)
- `|Φ⁻⟩`: both qubits always agree, but with a relative minus sign (invisible in measurement, visible in interference)
- `|Ψ⁺⟩`: both qubits always disagree (one 0 and one 1)
- `|Ψ⁻⟩`: both qubits always disagree, with a relative minus sign

#### What is quantum teleportation, in plain English?

Quantum teleportation sounds like science fiction, but it is a rigorously verified quantum protocol. The goal: Alice wants to send Bob a qubit in some unknown state `|ψ⟩`. She cannot simply measure it and tell Bob the result — measurement destroys the quantum information. She cannot copy it either — the no-cloning theorem prevents making a perfect duplicate of an unknown quantum state.

What she *can* do is use a pre-shared entangled Bell pair:

1. Alice and Bob each hold one qubit of an entangled pair, prepared in advance
2. Alice entangles her message qubit with her half of the Bell pair, then measures both
3. Her measurement gives 2 random classical bits (00, 01, 10, or 11)
4. She sends those 2 bits to Bob via any classical channel (text message, phone call — whatever)
5. Bob applies a simple correction to his qubit based on those 2 bits
6. Bob's qubit is now in exactly the state `|ψ⟩` — the *unknown* state Alice started with

The state was not "transmitted" through the Bell pair (that would require faster-than-light communication). Instead, the entanglement acted as a resource that enabled the transfer once the classical bits arrived. The simulation confirmed this works for all 8 test states — including arbitrary superpositions — with perfect fidelity = 1.0000.

#### What is quantum error correction, and why is it hard?

Real quantum computers make mistakes. Heat, electromagnetic noise, and cosmic rays all cause random errors in qubits — bit flips (0 becomes 1) and phase flips (the sign of an amplitude changes). For a quantum computer to be useful for a long computation, it needs to protect quantum information against these errors.

Classical computers handle this with redundancy: store the same bit three times (`000` for 0, `111` for 1) and use majority vote. If one bit flips, the majority still wins. But quantum mechanics has two obstacles:

1. **You cannot copy a qubit** (no-cloning theorem) — so you cannot just make three identical copies of a quantum state
2. **Measuring destroys superposition** — you cannot check whether an error occurred the classical way without collapsing the state

The 3-qubit repetition code solves this with a clever trick. Instead of copying, it *encodes* one logical qubit across three physical qubits:

```
Logical |0⟩  →  physical |000⟩
Logical |1⟩  →  physical |111⟩
Logical superposition α|0⟩ + β|1⟩  →  α|000⟩ + β|111⟩
```

If one physical qubit flips (say qubit 1: `α|010⟩ + β|101⟩`), the syndrome measurement detects *which qubit* is the odd one out — without ever measuring the logical qubit's state. The syndrome is like asking "do qubits 0 and 1 agree?" and "do qubits 0 and 2 agree?" — two binary questions whose answers pinpoint the error without revealing α or β. The right qubit is then flipped back, recovering the original logical state.

---

### Key Concepts (Technical Summary)

| Concept | What it means |
|---------|--------------|
| Bell state | Maximally entangled 2-qubit state; entropy = 1 ebit |
| No-cloning theorem | Cannot perfectly copy an unknown quantum state |
| Quantum teleportation | Transmit an unknown qubit state using 1 Bell pair + 2 classical bits |
| Fidelity | How close the received state is to the intended state; 1.0 = perfect |
| Syndrome measurement | Extracts *error location* from qubits without collapsing the logical state |
| 3-qubit code | `\|0_L⟩=\|000⟩`, `\|1_L⟩=\|111⟩`; corrects any single bit-flip |
| Logical qubit | The protected quantum information encoded across multiple physical qubits |

---

### Output Analysis

#### Plot 1 — Four Bell States

![Bell States](images/step6_bell_states.png)

This 2×2 grid shows measurement histograms for all four Bell states, each measured 4096 times. The x-axis of each panel is the 4 possible measurement outcomes (`|00⟩`, `|01⟩`, `|10⟩`, `|11⟩`); the y-axis is the probability of each outcome.

**`|Φ⁺⟩` (top-left):** Only `|00⟩` and `|11⟩` appear, each at ~0.5. The two qubits are *always in agreement* — both 0 or both 1. The correlated outcomes (`|01⟩` and `|10⟩`) have exactly zero probability.

**`|Φ⁻⟩` (top-right):** Identical distribution to `|Φ⁺⟩` — still 50/50 between `|00⟩` and `|11⟩`. You cannot tell `|Φ⁺⟩` from `|Φ⁻⟩` in the computational basis! The difference is entirely in the phase (the minus sign in front of `|11⟩`), which is invisible to direct measurement but determines how the state interferes with other operations. This is a concrete reminder that *phase matters*.

**`|Ψ⁺⟩` (bottom-left):** Only `|01⟩` and `|10⟩` appear, each at ~0.5. The qubits are *always opposite* — if one is 0 the other is 1, and vice versa. The agreed outcomes (`|00⟩` and `|11⟩`) have zero probability.

**`|Ψ⁻⟩` (bottom-right):** Again identical distribution to `|Ψ⁺⟩` — same anti-correlation pattern, phase difference invisible in measurement.

**Key takeaway:** The four Bell states form two pairs that look identical when measured, but are physically distinct because of their phases. They are the four "directions" of maximal 2-qubit entanglement. The fact that you cannot distinguish `Φ⁺` from `Φ⁻` by direct measurement is not a flaw — it is the mechanism that makes quantum cryptography and teleportation secure and functional.

---

#### Plot 2 — Quantum Teleportation Fidelity

![Teleportation Fidelity](images/step6_teleportation_fidelity.png)

This bar chart shows the **fidelity** of quantum teleportation for 8 different input states — how accurately Bob's final qubit matches what Alice started with. Fidelity of 1.000 means perfect transmission; 0.5 would mean random noise.

**All 8 bars reach 1.000.** The states tested cover the full Bloch sphere:
- `|0⟩` and `|1⟩` — the two poles (classical-like states)
- `|+⟩` and `|−⟩` — opposite equatorial states (superpositions with real amplitudes)
- `|i⟩` and `|−i⟩` — equatorial states with complex (imaginary) amplitudes
- Two arbitrary angles — states at neither pole nor equator, with both real and imaginary components

The fact that *all* of these achieve fidelity = 1.000 is significant. Teleportation does not just work for simple classical-like states — it works for any quantum state, including arbitrary superpositions with complex phases. The protocol truly transmits the full quantum information, not just classical probabilities.

The green color of all bars and the "Avg fidelity: 1.0000" label confirm what the theory predicts: when the quantum circuit is implemented correctly, teleportation is perfect. Any fidelity below 1.0 would indicate a circuit error or noise in the system.

**Key takeaway:** Teleportation's perfect fidelity is not obvious — it required a careful circuit design where Alice's measurements and Bob's corrections account for all four possible Bell-basis outcomes. The fact that it works for every input state, including arbitrary complex superpositions, is one of the most striking demonstrations in all of quantum information science.

---

#### Plot 3 — 3-Qubit Bit-Flip Error Correction

![Error Correction](images/step6_error_correction.png)

This bar chart shows the **recovery fidelity** of the 3-qubit repetition code for four scenarios: no error, and a bit-flip error on qubit 0, 1, or 2 respectively.

**All four bars reach 1.000.** In every case — including when an error was deliberately injected into the circuit — the syndrome measurement correctly identified the corrupted qubit, applied the right correction, and recovered the original logical state perfectly.

**What makes this remarkable:** The error correction succeeded *without ever measuring the logical qubit*. The syndrome measurement only asked "do these two qubits agree?" — binary parity questions that reveal the error location but nothing about the quantum state being protected. The coefficients α and β of the logical superposition `α|000⟩ + β|111⟩` were never exposed, never collapsed, and were perfectly preserved through the error-and-correction cycle.

**Compare what happens without error correction:** If qubit 0 flips and you do nothing, your logical state becomes `α|100⟩ + β|011⟩` — completely wrong. The error correction flips qubit 0 back, restoring `α|000⟩ + β|111⟩`.

**The limitation of the 3-qubit code:** This code only corrects *bit-flip* errors (X errors). It cannot handle *phase-flip* errors (Z errors), which also occur in real hardware. More sophisticated codes — like the 5-qubit perfect code or the 7-qubit Steane code — correct both types. The general theory of quantum error correction, which this simple example introduces, is one of the most active areas of quantum computing research and is the key engineering challenge for building large-scale fault-tolerant quantum computers.

**Key takeaway:** The all-1.000 bar chart represents one of the deepest achievements in quantum computing theory: the ability to protect quantum information from noise without ever observing it. This is the foundation that makes large-scale quantum computation physically possible, and the 3-qubit code demonstrated here is the simplest working instance of a concept that researchers are scaling to hundreds of physical qubits per logical qubit in state-of-the-art hardware today.

---

## Quick Reference

### Key Formulas

| Concept | Formula |
|---------|---------|
| Qubit state | `\|ψ⟩ = α\|0⟩ + β\|1⟩`,  `\|α\|² + \|β\|² = 1` |
| Bloch sphere | `\|ψ⟩ = cos(θ/2)\|0⟩ + e^(iφ)sin(θ/2)\|1⟩` |
| Born rule | `P(k) = \|⟨k\|ψ⟩\|²` |
| Inner product | `⟨φ\|ψ⟩ = φ†ψ` (complex scalar) |
| Tensor product | `\|ψ⟩ ⊗ \|φ⟩ = kron(ψ, φ)` |
| Unitarity | `U†U = I` |
| Bell state Φ⁺ | `(\|00⟩ + \|11⟩)/√2` |
| QFT | `\|j⟩ → (1/√N) Σₖ e^(2πijk/N)\|k⟩` |
| Grover iterations | `~π√N/4` for 1 marked item of N |
| Teleportation cost | 1 Bell pair + 2 classical bits → 1 qubit state |
| 3-qubit code | `\|0_L⟩=\|000⟩`, `\|1_L⟩=\|111⟩`, corrects 1 bit-flip |

### Quantum Speedup Summary

| Algorithm | Problem | Classical | Quantum | Speedup Type |
|-----------|---------|-----------|---------|--------------|
| Deutsch–Jozsa | Constant vs balanced f | O(2ⁿ) | O(1) | Exponential |
| Grover's search | Unstructured search | O(N) | O(√N) | Quadratic |
| Shor's factoring | Integer factorization | O(e^(n^(1/3))) | O(n³) | Exponential |
| QFT | Discrete Fourier transform | O(N log N) | O(n²) | Exponential |

### Gate Quick Reference

| Gate | Matrix | Effect |
|------|--------|--------|
| X | `[[0,1],[1,0]]` | Bit flip (NOT) |
| Y | `[[0,-i],[i,0]]` | Bit + phase flip |
| Z | `[[1,0],[0,-1]]` | Phase flip |
| H | `[[1,1],[1,-1]]/√2` | Superposition |
| S | `[[1,0],[0,i]]` | 90° phase (√Z) |
| T | `[[1,0],[0,e^(iπ/4)]]` | 45° phase (√S) |
| CNOT | 4×4 controlled-X | Entangling 2-qubit gate |

---

*Built with [Qiskit](https://qiskit.org/) and NumPy. Circuits validated on Qiskit Aer simulator and IBM Quantum hardware.*
