"""
Week 6 — Entanglement, Teleportation & Quantum Error Correction
===============================================================
Implements:
  - All four Bell states and their measurement
  - Quantum teleportation with fidelity verification
  - Superdense coding (bonus)
  - 3-qubit bit-flip repetition code with syndrome measurement
Run:  pip install qiskit qiskit-aer matplotlib
      python teleportation_ecc.py
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

sim = AerSimulator(method='statevector')


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def run_circuit(qc, shots=4096):
    return dict(sim.run(qc, shots=shots).result().get_counts())

def get_statevector(qc):
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    return np.array(sim.run(qc_no_meas).result().get_statevector())

def state_fidelity(sv1, sv2):
    """Fidelity between two pure states: F = |⟨ψ₁|ψ₂⟩|²"""
    return abs(np.conj(sv1) @ sv2) ** 2


# ─────────────────────────────────────────────────────────────
# BELL STATES
# ─────────────────────────────────────────────────────────────

def make_bell_state(which='Phi+'):
    """
    Create one of the four Bell states.

    Bell states are maximally entangled 2-qubit states:
        |Φ⁺⟩ = (|00⟩ + |11⟩)/√2   ← most common, used in teleportation
        |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
        |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
        |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2

    Created via: H on qubit 0, then CNOT(0→1), from appropriate initial state.
    """
    which_map = {
        'Phi+': ('00', []),
        'Phi-': ('10', ['z0']),    # apply Z to q0 before
        'Psi+': ('01', []),
        'Psi-': ('11', ['z0']),
    }
    # Simpler: create Φ⁺ then apply Pauli corrections
    corrections = {'Phi+':[], 'Phi-':['z0'], 'Psi+':['x1'], 'Psi-':['x1','z0']}

    qc = QuantumCircuit(2, name=f'Bell|{which}⟩')
    qc.h(0)
    qc.cx(0, 1)
    for op in corrections.get(which, []):
        if op == 'z0': qc.z(0)
        if op == 'x1': qc.x(1)
    return qc


def bell_measurement_circuit():
    """
    Bell basis measurement circuit.
    Converts Bell basis → computational basis for measurement.
    Inverse of Bell state creation: CNOT(0→1) then H on qubit 0.
    """
    qc = QuantumCircuit(2, 2, name='Bell_Meas')
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc


def demonstrate_bell_states():
    """Create and measure all four Bell states."""
    results = {}
    print("\n  Bell State Properties:")
    print(f"  {'State':8s}  {'Vector':40s}  {'Entropy':8s}  {'Shots (4096)':20s}")

    for name in ['Phi+', 'Phi-', 'Psi+', 'Psi-']:
        qc = make_bell_state(name)
        sv = get_statevector(qc)

        # Entanglement entropy via Schmidt decomposition
        M = sv.reshape(2, 2)
        _, s, _ = np.linalg.svd(M)
        s2 = s**2; s2 = s2[s2 > 1e-10]
        entropy = float(-np.sum(s2 * np.log2(s2)))

        # Measurement
        qc_meas = make_bell_state(name)
        qc_meas.measure_all()
        counts = run_circuit(qc_meas)

        sv_str = '  '.join(f'{v:.3f}|{i:02b}⟩' for i, v in enumerate(sv) if abs(v) > 0.01)
        print(f"  |{name}⟩  {sv_str:40s}  {entropy:.4f}   {counts}")
        results[name] = (sv, counts, entropy)

    return results


# ─────────────────────────────────────────────────────────────
# QUANTUM TELEPORTATION
# ─────────────────────────────────────────────────────────────

def teleportation_circuit(theta, phi, include_corrections=True):
    """
    Quantum teleportation circuit.

    Qubits:
        q0 = Alice's message qubit (state to teleport: |ψ⟩)
        q1 = Alice's half of the Bell pair
        q2 = Bob's half of the Bell pair (will become |ψ⟩ after correction)

    Classical bits:
        c0 = Alice's measurement of q0
        c1 = Alice's measurement of q1

    Protocol:
        1. Prepare |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩ on q0
        2. Create Bell pair |Φ⁺⟩ between q1 and q2
        3. Alice performs Bell measurement on (q0, q1)
        4. Alice sends 2 classical bits to Bob
        5. Bob applies X if c1=1, Z if c0=1 — recovers |ψ⟩

    Parameters
    ----------
    theta, phi         : Bloch sphere angles defining |ψ⟩
    include_corrections: if False, skip Bob's corrections (for debugging)
    """
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(2, 'alice')
    qc = QuantumCircuit(q, c)

    # ── Step 1: Prepare message qubit |ψ⟩ on q0 ──
    qc.ry(theta, 0)
    qc.rz(phi,   0)
    qc.barrier()

    # ── Step 2: Create Bell pair |Φ⁺⟩ between q1 and q2 ──
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    # ── Step 3: Alice's Bell measurement preparation ──
    qc.cx(0, 1)   # entangle message with Alice's Bell qubit
    qc.h(0)       # rotate to Bell basis
    qc.barrier()

    # ── Step 4: Alice measures q0 and q1 ──
    qc.measure(0, 0)   # c[0] ← measurement of q0
    qc.measure(1, 1)   # c[1] ← measurement of q1
    qc.barrier()

    # ── Step 5: Bob applies corrections ──
    if include_corrections:
        # If c[1]=1 (Alice measured q1 as 1) → Bob applies X
        with qc.if_test((c, 0b01)):
            qc.x(2)
        # If c[0]=1 (Alice measured q0 as 1) → Bob applies Z
        with qc.if_test((c, 0b10)):
            qc.z(2)
        # If both: XZ
        with qc.if_test((c, 0b11)):
            qc.x(2)
            qc.z(2)

    return qc


def teleportation_fidelity(theta, phi, shots=4096):
    """
    Measure teleportation fidelity by:
    1. Running teleportation
    2. Applying inverse of state preparation to Bob's qubit
    3. Measuring — should always get |0⟩ if fidelity = 1

    Returns fidelity estimate.
    """
    qc = teleportation_circuit(theta, phi, include_corrections=True)

    # Undo the state preparation on Bob's qubit (q2)
    qc.rz(-phi,   2)
    qc.ry(-theta, 2)

    # Add measurement of Bob's qubit
    c_bob = ClassicalRegister(1, 'bob')
    qc.add_register(c_bob)
    qc.measure(2, c_bob[0])

    counts = run_circuit(qc, shots=shots)

    # Count how often Bob's qubit (last bit) is 0
    bob_zero = sum(v for k, v in counts.items() if k.split(' ')[0] == '0')
    return bob_zero / shots


def run_teleportation_demo():
    """Test teleportation on a variety of input states."""
    test_states = [
        (0,           0,          "|0⟩"),
        (np.pi,       0,          "|1⟩"),
        (np.pi/2,     0,          "|+⟩"),
        (np.pi/2,     np.pi,      "|−⟩"),
        (np.pi/2,     np.pi/2,    "|i⟩"),
        (np.pi/2,    -np.pi/2,    "|−i⟩"),
        (np.pi/3,     np.pi/4,    "arbitrary-1"),
        (2*np.pi/3,   np.pi/3,    "arbitrary-2"),
    ]

    fidelities = []
    labels     = []

    print(f"\n  {'State':15s}  {'θ':8s}  {'φ':8s}  {'Fidelity':10s}  Status")
    for theta, phi, name in test_states:
        f = teleportation_fidelity(theta, phi)
        status = '✓' if f > 0.95 else '✗'
        print(f"  {name:15s}  {theta:.4f}    {phi:.4f}    {f:.4f}      {status}")
        fidelities.append(f); labels.append(name)

    # Plot fidelities
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#1D9E75' if f > 0.95 else '#D85A30' for f in fidelities]
    bars = ax.bar(range(len(labels)), fidelities, color=colors, edgecolor='white')
    ax.axhline(1.0, color='lightgray', linestyle='--', linewidth=1, label='Perfect fidelity')
    ax.axhline(0.95, color='#D85A30', linestyle=':', linewidth=1.5, label='Threshold (0.95)')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylim(0, 1.15); ax.set_ylabel('Teleportation fidelity')
    ax.set_title('Quantum Teleportation Fidelity across Input States')
    ax.legend(fontsize=9)
    for i, (bar, f) in enumerate(zip(bars, fidelities)):
        ax.text(bar.get_x() + bar.get_width()/2, f + 0.02,
                f'{f:.3f}', ha='center', fontsize=9)
    plt.tight_layout(); plt.show()

    return fidelities


# ─────────────────────────────────────────────────────────────
# SUPERDENSE CODING (bonus)
# ─────────────────────────────────────────────────────────────

def superdense_coding(message_bits='10'):
    """
    Superdense coding: send 2 classical bits using 1 qubit.

    Protocol (Alice and Bob share |Φ⁺⟩):
        '00': identity  → |Φ⁺⟩
        '01': X on q0   → |Ψ⁺⟩
        '10': Z on q0   → |Φ⁻⟩
        '11': ZX on q0  → |Ψ⁻⟩
    Bob then performs Bell measurement to recover the 2 bits.
    """
    assert len(message_bits) == 2 and all(b in '01' for b in message_bits)
    b0, b1 = message_bits

    q = QuantumRegister(2, 'q')
    c = ClassicalRegister(2, 'bob_decode')
    qc = QuantumCircuit(q, c)

    # Shared Bell pair |Φ⁺⟩
    qc.h(0); qc.cx(0, 1)
    qc.barrier()

    # Alice encodes 2 bits by applying gates to her qubit (q0)
    if b1 == '1': qc.x(0)   # bit 1
    if b0 == '1': qc.z(0)   # bit 0
    qc.barrier()

    # Bob decodes via Bell measurement
    qc.cx(0, 1); qc.h(0)
    qc.measure([0, 1], [0, 1])

    counts = run_circuit(qc)
    return counts, message_bits


# ─────────────────────────────────────────────────────────────
# QUANTUM ERROR CORRECTION: 3-qubit bit-flip code
# ─────────────────────────────────────────────────────────────

def encode_bit_flip(theta=np.pi/3, phi=np.pi/4):
    """
    Encode logical qubit into 3 physical qubits (bit-flip repetition code).

    Logical:  |0_L⟩ = |000⟩,  |1_L⟩ = |111⟩
    Encoding: |ψ_L⟩ = α|000⟩ + β|111⟩  where |ψ⟩ = α|0⟩ + β|1⟩

    The CNOT gates copy the *basis state*, not the superposition —
    the result is a logical superposition of |000⟩ and |111⟩.
    """
    qc = QuantumCircuit(3, name='Encode')
    qc.ry(theta, 0)   # prepare |ψ⟩ on qubit 0
    qc.rz(phi,   0)
    qc.cx(0, 1)       # spread to qubit 1
    qc.cx(0, 2)       # spread to qubit 2
    return qc


def bit_flip_ecc(error_qubit=None, theta=np.pi/3, phi=np.pi/4, verbose=True):
    """
    Full 3-qubit bit-flip error correction cycle.

    Steps:
        1. Encode logical qubit across q0, q1, q2
        2. (Optionally) inject a bit-flip error on one qubit
        3. Syndrome measurement: two ancilla qubits detect the error
           s0 = q0 ⊕ q1  (parity of qubits 0 and 1)
           s1 = q0 ⊕ q2  (parity of qubits 0 and 2)
           Syndrome table:
               00 → no error
               10 → q1 flipped
               01 → q2 flipped
               11 → q0 flipped
        4. Correction: flip the identified qubit
        5. Decode: un-encode to recover |ψ⟩ on q0

    Parameters
    ----------
    error_qubit : 0, 1, 2 (qubit to flip) or None (no error)
    """
    data_reg  = QuantumRegister(3, 'data')
    anc_reg   = QuantumRegister(2, 'anc')
    c_synd    = ClassicalRegister(2, 'syndrome')
    c_logical = ClassicalRegister(1, 'logical')

    qc = QuantumCircuit(data_reg, anc_reg, c_synd, c_logical)

    # ── Encoding ──
    qc.ry(theta, data_reg[0])
    qc.rz(phi,   data_reg[0])
    qc.cx(data_reg[0], data_reg[1])
    qc.cx(data_reg[0], data_reg[2])
    qc.barrier()

    # ── Error injection ──
    if error_qubit is not None:
        qc.x(data_reg[error_qubit])
    qc.barrier()

    # ── Syndrome measurement ──
    # s0: parity of data[0] ⊕ data[1]
    qc.cx(data_reg[0], anc_reg[0])
    qc.cx(data_reg[1], anc_reg[0])
    # s1: parity of data[0] ⊕ data[2]
    qc.cx(data_reg[0], anc_reg[1])
    qc.cx(data_reg[2], anc_reg[1])
    qc.measure(anc_reg[0], c_synd[0])
    qc.measure(anc_reg[1], c_synd[1])
    qc.barrier()

    # ── Classical correction ──
    # Syndrome 01 (s0=0, s1=1) → data[2] flipped
    with qc.if_test((c_synd, 0b01)):
        qc.x(data_reg[2])
    # Syndrome 10 (s0=1, s1=0) → data[1] flipped
    with qc.if_test((c_synd, 0b10)):
        qc.x(data_reg[1])
    # Syndrome 11 (s0=1, s1=1) → data[0] flipped
    with qc.if_test((c_synd, 0b11)):
        qc.x(data_reg[0])
    qc.barrier()

    # ── Decoding: un-encode to recover |ψ⟩ on data[0] ──
    qc.cx(data_reg[0], data_reg[2])
    qc.cx(data_reg[0], data_reg[1])
    # Undo state preparation on logical qubit
    qc.rz(-phi,   data_reg[0])
    qc.ry(-theta, data_reg[0])
    qc.measure(data_reg[0], c_logical[0])

    counts = run_circuit(qc, shots=4096)

    # Recovery fidelity: logical qubit = |0⟩ after un-preparation
    logical_zero = sum(v for k, v in counts.items() if k.split(' ')[-1] == '0')
    fidelity = logical_zero / 4096

    if verbose:
        label = f"q{error_qubit} flipped" if error_qubit is not None else "no error"
        status = '✓' if fidelity > 0.94 else '✗'
        syndrome_dist = Counter()
        for k, v in counts.items():
            parts = k.split(' ')
            syndrome_dist[parts[1] if len(parts) > 1 else '??'] += v
        print(f"  {status} {label:12s}  fidelity={fidelity:.4f}  syndromes={dict(syndrome_dist)}")

    return fidelity, counts


def plot_ecc_results(results):
    """Bar chart of ECC fidelity with and without errors."""
    labels, fidelities = zip(*results)
    colors = ['#1D9E75' if f > 0.94 else '#D85A30' for f in fidelities]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(labels)), fidelities, color=colors, edgecolor='white')
    ax.axhline(1.0, color='lightgray', linestyle='--', linewidth=1, label='Perfect')
    ax.axhline(0.94, color='#D85A30', linestyle=':', linewidth=1.5, label='Threshold (0.94)')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15); ax.set_ylabel('Recovery fidelity')
    ax.set_title('3-Qubit Bit-Flip Code: Error Correction Performance')
    ax.legend(fontsize=9)
    for i, (bar, f) in enumerate(zip(bars, fidelities)):
        ax.text(bar.get_x() + bar.get_width()/2, f + 0.02, f'{f:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  WEEK 6 — Entanglement, Teleportation & QEC")
    print("=" * 60)

    # ── 1. Bell states ────────────────────────────────────────
    print("\n[1] All four Bell states")
    bell_results = demonstrate_bell_states()

    # ── 2. Bell state visualisation ───────────────────────────
    print("\n[2] Visualising Bell states")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Four Bell States — Measurement Probabilities", fontsize=13)
    palette = ['#7F77DD','#1D9E75','#D85A30','#3B8BD4']
    for idx, (name, (sv, counts, ent)) in enumerate(bell_results.items()):
        ax = axes[idx // 2][idx % 2]
        keys = ['00','01','10','11']
        vals = [counts.get(k, 0)/4096 for k in keys]
        ax.bar([f'|{k}⟩' for k in keys], vals,
               color=palette, edgecolor='white')
        ax.set_ylim(0, 0.8); ax.set_title(f'|{name}⟩  (entropy={ent:.3f})')
        ax.set_ylabel('P')
    plt.tight_layout(); plt.show()

    # ── 3. Superdense coding ──────────────────────────────────
    print("\n[3] Superdense coding (2 bits over 1 qubit)")
    for msg in ['00', '01', '10', '11']:
        counts, _ = superdense_coding(msg)
        decoded = max(counts, key=counts.get)
        ok = decoded == msg[::-1]   # Qiskit reverses bit order
        print(f"  Sent: {msg}  →  Bob decoded: {decoded}  ✓" if decoded == msg else
              f"  Sent: {msg}  →  Bob decoded: {decoded}  counts={counts}")

    # ── 4. Quantum teleportation ──────────────────────────────
    print("\n[4] Quantum Teleportation")
    fidelities = run_teleportation_demo()
    avg = np.mean(fidelities)
    print(f"\n  Average fidelity: {avg:.4f}  ({'✓ Teleportation working!' if avg > 0.95 else '✗ Issues detected'})")

    # ── 5. No-corrections baseline ────────────────────────────
    print("\n[5] Teleportation without Bob's corrections (baseline)")
    base_fidelities = []
    for theta, phi, name in [(0,0,"|0⟩"),(np.pi,0,"|1⟩"),(np.pi/2,0,"|+⟩"),(np.pi/3,np.pi/4,"arb")]:
        qc = teleportation_circuit(theta, phi, include_corrections=False)
        qc.rz(-phi, 2); qc.ry(-theta, 2)
        c_bob = ClassicalRegister(1, 'bob')
        qc.add_register(c_bob); qc.measure(2, c_bob[0])
        counts = run_circuit(qc, shots=2048)
        f = sum(v for k,v in counts.items() if k.split(' ')[0]=='0') / 2048
        base_fidelities.append(f)
        print(f"  {name:8s}: fidelity without corrections = {f:.4f}  (expect ~0.5)")

    # ── 6. 3-qubit error correction ───────────────────────────
    print("\n[6] 3-Qubit Bit-Flip Error Correction")
    ecc_results = []
    for err in [None, 0, 1, 2]:
        label = f"q{err} flip" if err is not None else "no error"
        f, _ = bit_flip_ecc(error_qubit=err)
        ecc_results.append((label, f))

    plot_ecc_results(ecc_results)

    # ── 7. ECC across multiple states ─────────────────────────
    print("\n[7] ECC tested on multiple logical states")
    test_angles = [(0,0,"θ=0"),(np.pi,0,"θ=π"),(np.pi/2,0,"θ=π/2"),(np.pi/3,np.pi/4,"random")]
    for theta, phi, label in test_angles:
        fs = []
        for err in [None, 0, 1, 2]:
            f, _ = bit_flip_ecc(error_qubit=err, theta=theta, phi=phi, verbose=False)
            fs.append(f)
        avg = np.mean(fs)
        print(f"  {label:10s}: per-error fidelities={[round(f,3) for f in fs]}  avg={avg:.3f}")

    print("\nDone! ✓  You've completed the 6-week quantum computing course!")
    print("         → Next step: your mini-capstone project!")
