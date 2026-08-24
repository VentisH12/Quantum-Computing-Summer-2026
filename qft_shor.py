"""
Week 5 — Quantum Fourier Transform & Shor's Algorithm
======================================================
Build QFT from scratch, verify against numpy FFT,
and implement the quantum period-finding subroutine.
Run:  pip install qiskit qiskit-aer matplotlib
      python qft_shor.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

sim = AerSimulator(method='statevector')


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def run_circuit(qc, shots=4096):
    return dict(sim.run(qc, shots=shots).result().get_counts())

def get_statevector(qc):
    """Run circuit without measurements and return statevector."""
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    job = sim.run(qc_no_meas)
    return np.array(job.result().get_statevector())


# ─────────────────────────────────────────────────────────────
# QUANTUM FOURIER TRANSFORM
# ─────────────────────────────────────────────────────────────

def qft_circuit(n, inverse=False, swap=True):
    """
    Build an n-qubit QFT (or inverse QFT) circuit from primitive gates.

    The QFT maps:
        |j⟩  →  (1/√N) Σₖ e^(2πijk/N) |k⟩    where N = 2^n

    Circuit structure:
        For qubit j (top to bottom):
          1. H gate
          2. Controlled-Rₖ gates for k=2...(n-j)
             Rₖ = diag(1, e^(2πi/2^k)) — a phase rotation
        3. Reverse qubit order (SWAP) to match DFT bit ordering

    Parameters
    ----------
    n       : number of qubits
    inverse : if True, return QFT†  (used in Shor's / phase estimation)
    swap    : if True, include final SWAP to fix bit ordering
    """
    qc = QuantumCircuit(n, name='QFT' if not inverse else 'QFT†')

    def _qft_rotation(qc, n):
        """Recursive QFT rotations (without swap)."""
        if n == 0:
            return
        n -= 1
        qc.h(n)
        for qubit in range(n):
            angle = np.pi / (2 ** (n - qubit))
            qc.cp(angle, qubit, n)   # controlled phase gate
        _qft_rotation(qc, n)

    _qft_rotation(qc, n)

    if swap:
        for i in range(n // 2):
            qc.swap(i, n - i - 1)

    if inverse:
        qc = qc.inverse()
        qc.name = 'QFT†'

    return qc


def verify_qft_vs_fft(n_qubits=3, verbose=True):
    """
    Verify our QFT circuit matches numpy's FFT on all 2^n basis states.
    The QFT and DFT should give identical results (up to bit ordering).
    """
    N = 2 ** n_qubits
    qft = qft_circuit(n_qubits)
    mismatches = 0

    if verbose:
        print(f"\n  Verifying QFT({n_qubits}) against numpy FFT:")

    for j in range(N):
        # Prepare |j⟩ by flipping appropriate bits
        init = QuantumCircuit(n_qubits)
        bits = format(j, f'0{n_qubits}b')
        for i, bit in enumerate(bits):
            if bit == '1':
                init.x(i)
        full = init.compose(qft)
        sv = get_statevector(full)

        # Classical DFT: Σₖ e^(2πijk/N) * δ_{input,j} / √N
        basis = np.zeros(N, dtype=complex)
        basis[j] = 1.0
        fft_out = np.fft.fft(basis) / np.sqrt(N)

        match = np.allclose(np.abs(sv), np.abs(fft_out), atol=1e-6)
        if not match:
            mismatches += 1
        if verbose:
            status = '✓' if match else '✗'
            print(f"  {status} |{j}⟩  →  mag match: {match}")

    if verbose:
        print(f"  {N - mismatches}/{N} states match ✓" if mismatches == 0
              else f"  {mismatches} mismatches found!")
    return mismatches == 0


def plot_qft_action(n=3):
    """
    Visualise QFT action on the |5⟩ basis state:
    show input probabilities and QFT output amplitude + phase.
    """
    N = 2 ** n
    j = 5 % N
    bits = format(j, f'0{n}b')

    init = QuantumCircuit(n)
    for i, bit in enumerate(bits):
        if bit == '1':
            init.x(i)

    sv_before = get_statevector(init)
    sv_after  = get_statevector(init.compose(qft_circuit(n)))

    labels = [format(k, f'0{n}b') for k in range(N)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"QFT action on |{j}⟩ = |{bits}⟩  (n={n} qubits, N={N})", fontsize=13)

    palette = ['#7F77DD','#1D9E75','#D85A30','#3B8BD4',
               '#BA7517','#D4537E','#5DCAA5','#AFA9EC']

    # Before: probabilities
    ax = axes[0,0]
    ax.bar(labels, np.abs(sv_before)**2, color='#7F77DD', edgecolor='white')
    ax.set_title(f'Before QFT: |{j}⟩ probabilities')
    ax.set_ylabel('P'); ax.set_ylim(0,1.2)
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    # After: probabilities (should be uniform 1/N)
    ax = axes[0,1]
    ax.bar(labels, np.abs(sv_after)**2,
           color=[palette[k % len(palette)] for k in range(N)], edgecolor='white')
    ax.axhline(1/N, color='red', linestyle='--', linewidth=1.5, label=f'1/{N}={1/N:.3f}')
    ax.set_title('After QFT: probabilities (uniform)')
    ax.set_ylabel('P'); ax.set_ylim(0, max(np.abs(sv_after)**2)*1.5 + 0.05)
    ax.legend(fontsize=9); ax.tick_params(axis='x', rotation=45, labelsize=8)

    # After: real and imaginary amplitudes
    ax = axes[1,0]
    x = np.arange(N)
    ax.bar(x - 0.2, sv_after.real, 0.4, label='Re', color='#7F77DD', edgecolor='white')
    ax.bar(x + 0.2, sv_after.imag, 0.4, label='Im', color='#D85A30', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_title('After QFT: Re and Im amplitudes')
    ax.set_ylabel('Amplitude'); ax.legend()

    # After: phases on unit circle
    ax = axes[1,1]
    th = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'lightgray', lw=0.8)
    ax.axhline(0, color='lightgray', lw=0.4); ax.axvline(0, color='lightgray', lw=0.4)
    for k in range(N):
        amp = abs(sv_after[k])
        ph  = np.angle(sv_after[k])
        if amp > 0.01:
            ax.annotate('', xy=(amp*np.cos(ph)*0.9, amp*np.sin(ph)*0.9),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color=palette[k%len(palette)], lw=1.5))
            ax.text(np.cos(ph)*1.2, np.sin(ph)*1.2, labels[k],
                    fontsize=7, ha='center', color=palette[k%len(palette)])
    ax.set_xlim(-1.5,1.5); ax.set_ylim(-1.5,1.5); ax.set_aspect('equal')
    ax.set_title('Phase of QFT amplitudes'); ax.set_xlabel('Re'); ax.set_ylabel('Im')

    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# SHOR'S PERIOD FINDING
# ─────────────────────────────────────────────────────────────

def c_amod15(a, power):
    """
    Controlled-U gate implementing f(x) = a^power mod 15.
    Hardcoded for a in {2,4,7,8,11,13} (all coprime to 15).

    Acts on 4 target qubits + 1 control qubit = 5 qubits total.
    Appended to circuit as: qc.append(c_amod15(a, p), [ctrl_qubit] + data_qubits)
    """
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:
            U.swap(2, 3); U.swap(1, 2); U.swap(0, 1)
        elif a in [7, 8]:
            U.swap(0, 1); U.swap(1, 2); U.swap(2, 3)
        elif a == 4:
            U.swap(1, 3); U.swap(0, 2)
        elif a == 11:
            U.swap(0, 1); U.swap(0, 2); U.swap(0, 3)
        # a=1 or a=14: identity (r=1, trivial)
    return U.control(1)


def shor_period_finding(a=7, N=15, n_count=8, draw=True):
    """
    Quantum phase estimation circuit for period finding in Shor's algorithm.

    The circuit estimates the period r of f(x) = a^x mod N.
    Uses n_count counting qubits (precision) and 4 data qubits (for N=15).

    Parameters
    ----------
    a       : base (must be coprime to N)
    N       : number to factor (15 supported; for others use N=21 with modifications)
    n_count : number of counting qubits — more = more precision
    """
    assert math.gcd(a, N) == 1, f"a={a} must be coprime to N={N}"

    qr_count = QuantumRegister(n_count, 'count')
    qr_data  = QuantumRegister(4, 'data')
    cr       = ClassicalRegister(n_count, 'c')

    qc = QuantumCircuit(qr_count, qr_data, cr)

    # Initialise data register in |1⟩ (= a^0 mod 15)
    qc.x(n_count)
    qc.barrier()

    # Hadamard on all counting qubits
    for q in range(n_count):
        qc.h(q)
    qc.barrier()

    # Controlled-U^(2^j) gates (Hadamard test)
    for q in range(n_count):
        exp = 2 ** q
        controlled_U = c_amod15(a, exp % 4 if a in [7,8,11,13] else exp)
        qc.append(controlled_U, [q] + list(range(n_count, n_count + 4)))

    qc.barrier()

    # Inverse QFT on counting register
    iqft = qft_circuit(n_count, inverse=True)
    qc.append(iqft, range(n_count))
    qc.barrier()

    # Measure counting register
    qc.measure(range(n_count), range(n_count))

    if draw:
        print(f"\n  Shor circuit: a={a}, N={N}, n_count={n_count}")
        print(f"  Circuit depth: {qc.depth()}")

    return qc


def continued_fraction_period(phase_int, n_count, N, max_r=20):
    """
    Use continued fractions to extract the period r from a measured phase.

    The measured phase is phase_int / 2^n_count ≈ s/r for some integer s.
    We find the best rational approximation with denominator ≤ N.
    """
    if phase_int == 0:
        return None

    phase = phase_int / (2 ** n_count)
    # Use Python's Fraction to find best rational approximation
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator

    return r if 1 < r <= max_r else None


def run_shor(a=7, N=15, n_count=8, shots=4096):
    """
    Run Shor's period finding and attempt to factor N.

    Returns dict with period, factors, and success status.
    """
    qc = shor_period_finding(a, N, n_count, draw=True)
    counts = run_circuit(qc, shots=shots)

    # Analyse top measurement results
    top = sorted(counts.items(), key=lambda x: -x[1])[:6]
    print(f"\n  Top measurements (out of 2^{n_count}={2**n_count} states):")
    periods_found = Counter()
    for state, count in top:
        phase_int = int(state, 2)
        phase = phase_int / (2**n_count)
        r = continued_fraction_period(phase_int, n_count, N)
        periods_found[r] += count
        print(f"    |{state}⟩  count={count:5d}  phase≈{phase:.4f}  "
              f"({phase_int}/{2**n_count})  r={r}")

    # Best period
    r_best = max((r for r in periods_found if r is not None),
                 key=lambda r: periods_found[r], default=None)

    print(f"\n  Best period candidate: r = {r_best}")

    result = {'a': a, 'N': N, 'period': r_best, 'factors': [], 'success': False}

    if r_best is None:
        print("  Could not determine period from measurements.")
        return result

    # Attempt to extract factors
    if r_best % 2 == 0:
        x = pow(a, r_best // 2, N)
        factor_candidates = [math.gcd(x - 1, N), math.gcd(x + 1, N)]
        for f in factor_candidates:
            if 1 < f < N:
                result['factors'].append(f)
                result['success'] = True
                other = N // f
                print(f"  ✓ Factor found: {N} = {f} × {other}")
    else:
        print(f"  Period r={r_best} is odd — retry with different a")

    return result


def plot_shor_phase_histogram(counts, n_count, a, N):
    """Visualise the phase measurement distribution from Shor's circuit."""
    total = sum(counts.values())
    N_states = 2 ** n_count

    phases = []
    probs  = []
    labels = []
    for state, count in sorted(counts.items(), key=lambda x: int(x[0], 2)):
        phase_int = int(state, 2)
        phases.append(phase_int / N_states)
        probs.append(count / total)
        labels.append(f'{phase_int}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"Shor's Period Finding: a={a}, N={N}, n_count={n_count}", fontsize=13)

    # Phase histogram
    axes[0].bar(phases, probs, width=1/N_states * 0.9,
                color='#7F77DD', edgecolor='white', align='edge')
    axes[0].set_xlabel('Measured phase (= k/N_states)')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Phase distribution')
    # Mark theoretical peaks
    from fractions import Fraction
    r_theory = {'2':4,'4':2,'7':3,'8':2,'11':2,'13':4}.get(str(a), '?')
    if isinstance(r_theory, int):
        for s in range(r_theory):
            pk = s / r_theory
            axes[0].axvline(pk, color='#D85A30', linestyle='--', linewidth=1.5, alpha=0.8)
    axes[0].set_xlim(0, 1)

    # Top-10 counts bar chart
    top10 = sorted(counts.items(), key=lambda x: -x[1])[:10]
    keys10 = [f'{int(k,2)}/{N_states}' for k,_ in top10]
    vals10 = [v/total for _,v in top10]
    axes[1].barh(range(len(keys10)), vals10,
                 color=['#7F77DD' if i==0 else '#AFA9EC' for i in range(len(keys10))],
                 edgecolor='white')
    axes[1].set_yticks(range(len(keys10))); axes[1].set_yticklabels(keys10, fontsize=9)
    axes[1].set_xlabel('Probability'); axes[1].set_title('Top-10 measured phases')
    for i, v in enumerate(vals10):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=8)

    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# Classical Shor's (for verification)
# ─────────────────────────────────────────────────────────────

def classical_period(a, N):
    """Brute-force classical period finding: find r such that a^r ≡ 1 (mod N)."""
    for r in range(1, N + 1):
        if pow(a, r, N) == 1:
            return r
    return None

def classical_shor(N):
    """Classical simulation of the full Shor algorithm."""
    print(f"\n  Classical Shor for N={N}:")
    for a in range(2, N):
        if math.gcd(a, N) != 1:
            f = math.gcd(a, N)
            print(f"  Lucky! gcd({a},{N}) = {f} directly.")
            return f, N // f
        r = classical_period(a, N)
        if r and r % 2 == 0:
            x = pow(a, r//2, N)
            f1 = math.gcd(x-1, N)
            f2 = math.gcd(x+1, N)
            if 1 < f1 < N:
                print(f"  a={a}, r={r}, gcd(a^(r/2)±1,N): {N}={f1}×{f2}")
                return f1, f2
    return None, None


from collections import Counter

# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  WEEK 5 — QFT & Shor's Period Finding")
    print("=" * 60)

    # ── 1. QFT verification ───────────────────────────────────
    print("\n[1] Verify QFT vs numpy FFT")
    for n in [2, 3, 4]:
        ok = verify_qft_vs_fft(n, verbose=(n==3))
        print(f"  QFT({n}): all basis states match FFT = {ok}")

    # ── 2. QFT action visualisation ───────────────────────────
    print("\n[2] Visualising QFT action on |5⟩")
    plot_qft_action(n=3)

    # ── 3. QFT is self-inverse ────────────────────────────────
    print("\n[3] QFT·QFT† = Identity check")
    for n in [2, 3, 4]:
        qc = QuantumCircuit(n)
        qc.compose(qft_circuit(n), inplace=True)
        qc.compose(qft_circuit(n, inverse=True), inplace=True)
        sv = get_statevector(qc)
        identity_ok = np.allclose(np.abs(sv[0]), 1.0, atol=1e-8)
        print(f"  QFT({n})·QFT†({n}) on |0⟩ → |0⟩: {identity_ok}")

    # ── 4. Classical Shor (sanity check) ──────────────────────
    print("\n[4] Classical Shor's algorithm (verification)")
    for N in [15, 21, 35]:
        f1, f2 = classical_shor(N)
        if f1:
            print(f"    {N} = {f1} × {f2}  ✓")

    # ── 5. Quantum period finding ─────────────────────────────
    print("\n[5] Quantum period finding: a=7, N=15")
    result = run_shor(a=7, N=15, n_count=8, shots=4096)

    # Visualise
    qc_vis = shor_period_finding(7, 15, 8, draw=False)
    counts_vis = run_circuit(qc_vis, shots=4096)
    plot_shor_phase_histogram(counts_vis, 8, 7, 15)

    # ── 6. Try multiple values of a ───────────────────────────
    print("\n[6] Period finding for all valid a coprime to 15")
    for a in [2, 4, 7, 8, 11, 13]:
        r_classical = classical_period(a, 15)
        result = run_shor(a=a, N=15, n_count=8, shots=2048)
        quantum_r = result['period']
        match = '✓' if quantum_r == r_classical else '~'
        print(f"\n  a={a:2d}: classical r={r_classical}, quantum r={quantum_r} {match}")
        if result['factors']:
            print(f"    Factors: {result['factors']}")

    print("\nDone! ✓")
