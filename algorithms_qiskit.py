"""
Week 4 — Deutsch–Jozsa & Grover's Search in Qiskit
====================================================
Implements both algorithms using Qiskit Aer, with visualisation
and a reusable oracle builder for Grover's search.
Run:  pip install qiskit qiskit-aer matplotlib
      python algorithms_qiskit.py
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from fractions import Fraction
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

sim = AerSimulator(method='statevector')


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def run_circuit(qc, shots=2048):
    """Run a Qiskit circuit and return counts dict."""
    result = sim.run(qc, shots=shots).result()
    return dict(result.get_counts())


def plot_histogram(counts_list, titles, suptitle="Results", n_qubits=None):
    """
    Plot measurement histograms for a list of (counts_dict, title) pairs.
    """
    fig, axes = plt.subplots(1, len(counts_list), figsize=(5*len(counts_list), 4))
    if len(counts_list) == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    palette = ['#7F77DD','#1D9E75','#D85A30','#3B8BD4','#BA7517','#D4537E',
               '#5DCAA5','#AFA9EC']

    for ax, counts, title in zip(axes, counts_list, titles):
        total  = sum(counts.values())
        keys   = sorted(counts.keys())
        vals   = [counts[k]/total for k in keys]
        colors = [palette[i % len(palette)] for i in range(len(keys))]
        bars   = ax.bar(keys, vals, color=colors, edgecolor='white')
        ax.set_ylim(0, 1.15); ax.set_ylabel('Probability')
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                        f'{v:.2f}', ha='center', fontsize=8)
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# DEUTSCH–JOZSA ALGORITHM
# ─────────────────────────────────────────────────────────────

def dj_oracle(n, func_type='balanced'):
    """
    Build n-qubit Deutsch–Jozsa oracle gate.

    Parameters
    ----------
    n         : number of input qubits (output is 1-qubit ancilla)
    func_type : 'constant_0' | 'constant_1' | 'balanced'

    Circuit acts on n+1 qubits: qubits 0..n-1 are input, qubit n is ancilla.
    """
    oracle = QuantumCircuit(n + 1, name=f'DJ_Oracle({func_type})')

    if func_type == 'constant_0':
        pass    # f(x) = 0 for all x: identity

    elif func_type == 'constant_1':
        oracle.x(n)     # f(x) = 1 for all x: flip ancilla unconditionally

    else:   # balanced: f(x) = parity of x (XOR of all bits)
        for i in range(n):
            oracle.cx(i, n)   # CNOT from each input qubit to ancilla

    return oracle.to_gate()


def deutsch_jozsa(n, func_type='balanced', draw=True):
    """
    Full Deutsch–Jozsa circuit on n input qubits.

    Returns (circuit, result_string)
    result_string: 'CONSTANT' or 'BALANCED'
    """
    qr = QuantumRegister(n + 1, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr)

    # Step 1: ancilla in |1⟩
    qc.x(n)

    # Step 2: Hadamard on all qubits
    for i in range(n + 1):
        qc.h(i)
    qc.barrier()

    # Step 3: Oracle query
    oracle_gate = dj_oracle(n, func_type)
    qc.append(oracle_gate, range(n + 1))
    qc.barrier()

    # Step 4: Hadamard on input register
    for i in range(n):
        qc.h(i)

    # Step 5: Measure input register
    for i in range(n):
        qc.measure(i, i)

    counts = run_circuit(qc)

    # If all-zeros measured → constant; anything else → balanced
    all_zero = '0' * n
    result = 'CONSTANT' if counts.get(all_zero, 0) / sum(counts.values()) > 0.9 else 'BALANCED'

    if draw:
        print(f"\n  DJ circuit ({n}-qubit, {func_type}): ← result = {result}")
        print(qc.draw(output='text', fold=60))

    return qc, result, counts


# ─────────────────────────────────────────────────────────────
# GROVER'S SEARCH ALGORITHM
# ─────────────────────────────────────────────────────────────

def grover_oracle(n, targets):
    """
    Phase-flip oracle: marks each target state with a phase of -1.

    Parameters
    ----------
    n       : number of qubits (searches 2^n items)
    targets : list of int — indices of target states (0 to 2^n - 1)

    Implements: Uω|x⟩ = -|x⟩ if x in targets, else |x⟩
    """
    qc = QuantumCircuit(n, name='Grover_Oracle')

    for target in targets:
        # Represent target as n-bit string
        bits = format(target, f'0{n}b')

        # Flip 0-bits so target maps to all-|1⟩
        for i, bit in enumerate(bits):
            if bit == '0':
                qc.x(i)

        # Multi-controlled Z: flips phase when all qubits are |1⟩
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            # Decompose MCZ as H · MCX · H on last qubit
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)

        # Uncompute: restore flipped qubits
        for i, bit in enumerate(bits):
            if bit == '0':
                qc.x(i)

    return qc.to_gate()


def grover_diffusion(n):
    """
    Grover diffusion operator: D = 2|s⟩⟨s| - I
    where |s⟩ = uniform superposition.

    This is equivalent to: H⊗n · (2|0⟩⟨0| - I) · H⊗n
    which maps the state to its reflection about the mean.
    """
    qc = QuantumCircuit(n, name='Diffusion')
    qc.h(range(n))         # rotate to computational basis
    qc.x(range(n))         # flip all qubits so |0...0⟩ → |1...1⟩
    # Phase flip on |1...1⟩ = controlled-Z chain
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))         # uncompute
    qc.h(range(n))         # rotate back
    return qc.to_gate()


def optimal_grover_iters(N, M):
    """Optimal number of Grover iterations: π/4 · √(N/M)."""
    return max(1, round(np.pi / 4 * np.sqrt(N / M)))


def grover_search(n, targets, num_iters=None, draw=True):
    """
    Full Grover's algorithm circuit.

    Parameters
    ----------
    n         : number of qubits
    targets   : list of int target state indices
    num_iters : number of Grover iterations (default: optimal)
    draw      : whether to print circuit diagram

    Returns (circuit, counts, num_iters_used)
    """
    N = 2 ** n
    M = len(targets)
    if num_iters is None:
        num_iters = optimal_grover_iters(N, M)

    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr)

    # Uniform superposition
    qc.h(range(n))
    qc.barrier()

    # Grover iterations
    oracle_gate = grover_oracle(n, targets)
    diff_gate   = grover_diffusion(n)

    for k in range(num_iters):
        qc.append(oracle_gate, range(n))
        qc.append(diff_gate,   range(n))
        if k < num_iters - 1:
            qc.barrier()

    qc.barrier()
    qc.measure(range(n), range(n))

    counts = run_circuit(qc, shots=4096)

    if draw:
        target_strs = [format(t, f'0{n}b') for t in targets]
        found_probs = {t: counts.get(t, 0)/4096 for t in target_strs}
        print(f"\n  Grover: n={n}, targets={targets}, iters={num_iters}")
        print(f"  Target probabilities: {found_probs}")

    return qc, counts, num_iters


def plot_grover_iteration_sweep(n, target, max_iters=None):
    """
    Show how target probability evolves over Grover iterations.
    Reveals the sinusoidal nature of amplitude amplification.
    """
    N = 2 ** n
    if max_iters is None:
        max_iters = int(np.pi / 4 * np.sqrt(N)) + 3

    target_str = format(target, f'0{n}b')
    probs = []
    iters_range = range(0, max_iters + 1)

    for k in iters_range:
        if k == 0:
            probs.append(1 / N)   # uniform superposition
        else:
            _, counts, _ = grover_search(n, [target], num_iters=k, draw=False)
            probs.append(counts.get(target_str, 0) / 4096)

    # Theoretical curve: P(k) = sin²((2k+1)·θ) where sin(θ) = 1/√N
    theta = np.arcsin(1 / np.sqrt(N))
    k_cont = np.linspace(0, max_iters, 300)
    theory = np.sin((2 * k_cont + 1) * theta) ** 2

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_cont, theory, '--', color='lightgray', linewidth=2, label='Theory sin²((2k+1)θ)')
    ax.plot(list(iters_range), probs, 'o-', color='#7F77DD', linewidth=2,
            markersize=8, label=f'Simulated P(|{target_str}⟩)')
    ax.axhline(1/N, color='#D85A30', linestyle=':', linewidth=1.5,
               label=f'Classical (1/{N} = {1/N:.3f})')
    opt = optimal_grover_iters(N, 1)
    ax.axvline(opt, color='#1D9E75', linestyle='--', linewidth=1.5,
               label=f'Optimal iters ({opt})')
    ax.set_xlabel('Number of Grover iterations')
    ax.set_ylabel('Probability of measuring target')
    ax.set_title(f"Grover amplitude amplification: n={n} qubits, target=|{target_str}⟩")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  WEEK 4 — Deutsch–Jozsa & Grover's Algorithm")
    print("=" * 60)

    # ── 1. Deutsch–Jozsa: all function types ─────────────────
    print("\n[1] Deutsch–Jozsa (n=3 input qubits)\n")
    dj_results = []
    dj_titles  = []
    for ftype in ['constant_0', 'constant_1', 'balanced']:
        _, result, counts = deutsch_jozsa(3, ftype, draw=True)
        print(f"  → Classification: {result}\n")
        dj_results.append(counts)
        dj_titles.append(f"DJ: {ftype}\n({result})")

    plot_histogram(dj_results, dj_titles, "Deutsch–Jozsa: 3-qubit results")

    # ── 2. Grover: 2-qubit (4-item) search ───────────────────
    print("\n[2] Grover's Search: 2 qubits (4 items)\n")
    for target in [0, 1, 2, 3]:
        _, counts, k = grover_search(2, [target], draw=True)
        target_str = format(target, '02b')
        p = counts.get(target_str, 0) / sum(counts.values())
        print(f"    Target |{target_str}⟩: success probability = {p:.3f}")

    # ── 3. Grover: 3-qubit (8-item) search ───────────────────
    print("\n[3] Grover's Search: 3 qubits (8 items)\n")
    qc3, counts3, k3 = grover_search(3, [5], draw=True)
    print(f"\n  Full counts: {dict(sorted(counts3.items(), key=lambda x:-x[1]))}")
    plot_histogram([counts3], [f"Grover 3q: target=|101⟩\n({k3} iteration)"],
                   "Grover's Search (3 qubits, target=5)")

    # ── 4. Grover: 4-qubit search, multiple targets ───────────
    print("\n[4] Grover's Search: 4 qubits, 2 targets\n")
    _, counts4, k4 = grover_search(4, [3, 11], draw=True)
    top5 = sorted(counts4.items(), key=lambda x:-x[1])[:5]
    print(f"  Top 5: {top5}")

    # ── 5. Amplitude amplification sweep ─────────────────────
    print("\n[5] Amplitude amplification sweep (3-qubit Grover)\n")
    plot_grover_iteration_sweep(n=3, target=5, max_iters=8)

    # ── 6. Scaling analysis ───────────────────────────────────
    print("\n[6] Grover scaling: optimal iterations vs qubit count")
    print(f"  {'n qubits':10s} {'N items':10s} {'Classical O(N)':16s} {'Grover O(√N)':14s} {'Speedup':10s}")
    for n in range(2, 10):
        N   = 2**n
        k   = optimal_grover_iters(N, 1)
        spd = N / (2*k)
        print(f"  {n:10d} {N:10d} {N//2:16d} {k:14d} {spd:10.1f}×")

    print("\nDone! ✓")
