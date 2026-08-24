"""
Week 3 — 2-Qubit Circuit Simulator
====================================
A matrix-based 2-qubit circuit simulator with ASCII diagram printing,
gate library, shot-based measurement, and state inspection.
Run:  python circuit_simulator.py
Deps: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# ─────────────────────────────────────────────────────────────
# Gate library
# ─────────────────────────────────────────────────────────────

I2 = np.eye(2, dtype=complex)
H  = np.array([[1, 1],[1,-1]], dtype=complex) / np.sqrt(2)
X  = np.array([[0, 1],[1, 0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1, 0],[0,-1]], dtype=complex)
S  = np.array([[1, 0],[0, 1j]], dtype=complex)
T  = np.array([[1, 0],[0, np.exp(1j*np.pi/4)]], dtype=complex)

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
CNOT_10 = np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]], dtype=complex)  # ctrl=q1
CZ   = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
ISWAP= np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)


def Rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s],[-1j*s, c]], dtype=complex)

def Ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s],[s, c]], dtype=complex)

def Rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0],[0, np.exp(1j*theta/2)]], dtype=complex)

def Phase(phi):
    """Global phase gate P(φ): |0⟩→|0⟩, |1⟩→e^(iφ)|1⟩"""
    return np.array([[1,0],[0,np.exp(1j*phi)]], dtype=complex)

def kron(*mats):
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


# ─────────────────────────────────────────────────────────────
# 2-qubit circuit class
# ─────────────────────────────────────────────────────────────

class QuantumCircuit2Q:
    """
    Two-qubit quantum circuit simulator.

    Qubit ordering convention (matches Qiskit):
      State index = q0*2 + q1
      |q0 q1⟩ basis: |00⟩, |01⟩, |10⟩, |11⟩

    Gate application builds up a total unitary U_total, and
    also tracks individual operations for the ASCII diagram.
    """

    def __init__(self, init='00'):
        basis = {
            '00': np.array([1,0,0,0], dtype=complex),
            '01': np.array([0,1,0,0], dtype=complex),
            '10': np.array([0,0,1,0], dtype=complex),
            '11': np.array([0,0,0,1], dtype=complex),
        }
        if init not in basis:
            raise ValueError(f"init must be one of {list(basis.keys())}")
        self.state   = basis[init].copy()
        self.U_total = np.eye(4, dtype=complex)   # accumulated unitary
        self._ops    = []                          # (label_q0, label_q1, 4×4 mat)

    # ── Internal helpers ──────────────────────────────────────

    def _apply(self, label_q0, label_q1, mat4):
        self.state   = mat4 @ self.state
        self.U_total = mat4 @ self.U_total
        self._ops.append((label_q0, label_q1, mat4))
        return self   # allow chaining

    def _single(self, gate2, q, name):
        mat4 = kron(gate2, I2) if q == 0 else kron(I2, gate2)
        lq0  = name if q == 0 else '─'*len(name)
        lq1  = name if q == 1 else '─'*len(name)
        return self._apply(lq0, lq1, mat4)

    # ── Single-qubit gates ────────────────────────────────────

    def h(self, q):   return self._single(H, q, '[H]')
    def x(self, q):   return self._single(X, q, '[X]')
    def y(self, q):   return self._single(Y, q, '[Y]')
    def z(self, q):   return self._single(Z, q, '[Z]')
    def s(self, q):   return self._single(S, q, '[S]')
    def t(self, q):   return self._single(T, q, '[T]')
    def sdg(self, q): return self._single(S.conj().T, q, '[S†]')
    def tdg(self, q): return self._single(T.conj().T, q, '[T†]')

    def rx(self, q, theta):
        return self._single(Rx(theta), q, f'[Rx({theta:.2f})]')
    def ry(self, q, theta):
        return self._single(Ry(theta), q, f'[Ry({theta:.2f})]')
    def rz(self, q, theta):
        return self._single(Rz(theta), q, f'[Rz({theta:.2f})]')
    def p(self, q, phi):
        return self._single(Phase(phi), q, f'[P({phi:.2f})]')

    # ── Two-qubit gates ───────────────────────────────────────

    def cnot(self, ctrl=0, tgt=1):
        mat4 = CNOT if (ctrl, tgt) == (0, 1) else CNOT_10
        return self._apply('[●]', '[⊕]', mat4) if ctrl == 0 else self._apply('[⊕]','[●]', mat4)

    def cz(self):
        return self._apply('[●]', '[Z]', CZ)

    def swap(self):
        return self._apply('[✕]', '[✕]', SWAP)

    def iswap(self):
        return self._apply('[iS]', '[iS]', ISWAP)

    def cy(self):
        CY = kron(np.array([[1,0],[0,0]],dtype=complex), I2) + \
             kron(np.array([[0,0],[0,1]],dtype=complex), Y)
        return self._apply('[●]', '[Y]', CY)

    # ── Barrier (no-op, diagram only) ────────────────────────
    def barrier(self):
        self._ops.append(('|', '|', None))
        return self

    # ── Measurement & state queries ───────────────────────────

    def statevector(self):
        """Return dict {label: complex amplitude}."""
        labels = ['|00⟩','|01⟩','|10⟩','|11⟩']
        return {labels[i]: self.state[i] for i in range(4)}

    def probabilities(self):
        """Return dict {label: float probability}."""
        labels = ['|00⟩','|01⟩','|10⟩','|11⟩']
        return {labels[i]: round(float(abs(self.state[i])**2), 6) for i in range(4)}

    def measure(self, shots=1024):
        """Shot-based measurement. Returns Counter."""
        probs   = np.abs(self.state)**2
        choices = ['00','01','10','11']
        results = np.random.choice(choices, size=shots, p=probs)
        return dict(Counter(results))

    def is_unitary(self):
        return np.allclose(self.U_total.conj().T @ self.U_total, np.eye(4), atol=1e-10)

    # ── ASCII circuit diagram ─────────────────────────────────

    def draw(self, title=None):
        if title:
            print(f"\n  Circuit: {title}")
        q0_line = ['q0: |0⟩─']
        q1_line = ['q1: |0⟩─']
        for (l0, l1, _) in self._ops:
            if l0 == '|':
                q0_line.append(' ░ ')
                q1_line.append(' ░ ')
                continue
            w = max(len(l0), len(l1))
            q0_line.append(l0.center(w, '─'))
            q1_line.append(l1.center(w, '─'))
        q0_line.append('─┤M├')
        q1_line.append('─┤M├')
        print('  ' + ''.join(q0_line))
        print('  ' + ''.join(q1_line))
        print()

    def __repr__(self):
        sv = self.statevector()
        terms = [f'({a:.3f}){l}' for l, a in sv.items() if abs(a) > 1e-8]
        return '|ψ⟩ = ' + ' + '.join(terms)


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def plot_state(state_dict, title="Circuit output", shots_dict=None):
    """
    Side-by-side: statevector (amplitudes + phases) and measurement histogram.
    state_dict:  output of .statevector()
    shots_dict:  output of .measure() (optional)
    """
    labels = list(state_dict.keys())
    amps   = np.array([abs(v)    for v in state_dict.values()])
    phases = np.array([np.angle(v) for v in state_dict.values()])
    probs  = amps ** 2
    palette = ['#7F77DD','#1D9E75','#D85A30','#3B8BD4']

    ncols = 3 if shots_dict else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Amplitude bars
    axes[0].bar(labels, amps, color=palette, edgecolor='white')
    axes[0].set_ylim(0, 1.1); axes[0].set_ylabel('|amplitude|')
    axes[0].set_title('Amplitude magnitudes')
    for i, (a, lbl) in enumerate(zip(amps, labels)):
        if a > 0.01:
            axes[0].text(i, a + 0.03, f'{a:.3f}', ha='center', fontsize=9)

    # Phase wheel
    ax2 = axes[1]
    th = np.linspace(0, 2*np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), 'lightgray', lw=0.8)
    ax2.axhline(0, color='lightgray', lw=0.5); ax2.axvline(0, color='lightgray', lw=0.5)
    for i, (ph, amp, lbl) in enumerate(zip(phases, amps, labels)):
        if amp > 0.01:
            ax2.annotate('', xy=(amp*np.cos(ph), amp*np.sin(ph)), xytext=(0,0),
                         arrowprops=dict(arrowstyle='->', color=palette[i], lw=2.2))
            ax2.text(amp*np.cos(ph)*1.2, amp*np.sin(ph)*1.2, lbl,
                     color=palette[i], fontsize=9, ha='center')
    ax2.set_xlim(-1.5,1.5); ax2.set_ylim(-1.5,1.5); ax2.set_aspect('equal')
    ax2.set_title('Phase diagram'); ax2.set_xlabel('Re'); ax2.set_ylabel('Im')

    # Measurement histogram
    if shots_dict:
        ax3 = axes[2]
        total = sum(shots_dict.values())
        keys = ['00','01','10','11']
        counts = [shots_dict.get(k, 0) for k in keys]
        clabels = [f'|{k}⟩' for k in keys]
        ax3.bar(clabels, [c/total for c in counts], color=palette, edgecolor='white')
        ax3.set_ylim(0, 1.1); ax3.set_ylabel('Frequency')
        ax3.set_title(f'Measurement ({total} shots)')
        for i, c in enumerate(counts):
            if c > 0:
                ax3.text(i, c/total + 0.02, str(c), ha='center', fontsize=9)
    plt.tight_layout(); plt.show()


def plot_circuit_unitary(U, title="Circuit Unitary"):
    """Display magnitude and phase of the 4×4 circuit unitary."""
    labels = ['|00⟩','|01⟩','|10⟩','|11⟩']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for ax, data, subtitle, cmap in [
        (axes[0], np.abs(U),   'Magnitude |Uᵢⱼ|', 'Blues'),
        (axes[1], np.angle(U), 'Phase arg(Uᵢⱼ)',   'RdBu'),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(subtitle)
        for i in range(4):
            for j in range(4):
                v = data[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(v) > 0.6 else 'black')
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────
# Named circuit builders
# ─────────────────────────────────────────────────────────────

def bell_state_circuit(which='Phi+'):
    """
    Create one of the four Bell states.
    which: 'Phi+', 'Phi-', 'Psi+', 'Psi-'
    """
    init_map = {'Phi+':'00','Phi-':'10','Psi+':'01','Psi-':'11'}
    qc = QuantumCircuit2Q(init_map[which])
    qc.h(0).cnot()
    return qc

def deutsch_2qubit_oracle(f_type='balanced'):
    """
    1-qubit Deutsch oracle embedded in a 2-qubit circuit.
    Returns circuit with oracle applied (ready for H at end).
    f_type: 'constant_0','constant_1','balanced_id','balanced_not'
    """
    qc = QuantumCircuit2Q('01')   # |q_input=0, q_ancilla=1⟩ → H on both first
    qc.h(0).h(1).barrier()
    if f_type == 'constant_1':
        qc.x(1)            # flip ancilla: f(x)=1 for all x
    elif f_type == 'balanced_id':
        qc.cnot(0, 1)      # f(x) = x
    elif f_type == 'balanced_not':
        qc.x(0).cnot(0,1).x(0)  # f(x) = NOT x
    # else constant_0: do nothing
    qc.barrier().h(0)
    return qc

def swap_test_circuit(theta_A=np.pi/3, theta_B=np.pi/4):
    """
    SWAP test for measuring overlap |⟨A|B⟩|².
    (Requires 3 qubits; this is a 2-qubit approximation for demo purposes.)
    Here we just show the SWAP circuit and measure the difference.
    """
    qc = QuantumCircuit2Q('00')
    qc.ry(theta_A, 0).ry(theta_B, 1)
    qc.barrier()
    return qc


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  WEEK 3 — 2-Qubit Circuit Simulator")
    print("=" * 55)

    # ── 1. Bell states ────────────────────────────────────────
    print("\n[1] All four Bell states")
    for name, init in [('Φ⁺','00'),('Φ⁻','10'),('Ψ⁺','01'),('Ψ⁻','11')]:
        qc = QuantumCircuit2Q(init)
        qc.h(0).cnot()
        qc.draw(title=f"Bell state {name}  (init={init})")
        print(f"  {name}: {qc}")
        print(f"     Probs: {qc.probabilities()}")
        print(f"     Unitary: {qc.is_unitary()}")

    # ── 2. GHZ analogue on 2 qubits ──────────────────────────
    print("\n[2] Interference circuit: H·CNOT·H")
    qc2 = QuantumCircuit2Q('00')
    qc2.h(0).cnot().h(0)
    qc2.draw(title="H→CNOT→H")
    print(f"  State: {qc2}")
    print(f"  Probs: {qc2.probabilities()}")
    plot_state(qc2.statevector(), "H·CNOT·H on |00⟩", qc2.measure(1024))

    # ── 3. SWAP circuit ───────────────────────────────────────
    print("\n[3] SWAP circuit: swap qubit states")
    qc3 = QuantumCircuit2Q('10')   # start in |10⟩
    qc3.swap()
    qc3.draw(title="SWAP|10⟩ → |01⟩")
    print(f"  After SWAP: {qc3}")
    print(f"  Probs: {qc3.probabilities()}")

    # SWAP via 3 CNOTs
    qc3b = QuantumCircuit2Q('10')
    qc3b.cnot(0,1).cnot(1,0).cnot(0,1)
    qc3b.draw(title="SWAP via 3 CNOTs")
    print(f"  3-CNOT SWAP: {qc3b}")

    # ── 4. Phase kickback ─────────────────────────────────────
    print("\n[4] Phase kickback demo: CZ")
    qc4 = QuantumCircuit2Q('00')
    qc4.h(0).h(1).barrier().cz().barrier().h(0).h(1)
    qc4.draw(title="H⊗H → CZ → H⊗H  (phase kickback)")
    print(f"  State: {qc4}")
    print(f"  Probs: {qc4.probabilities()}  (should see all four equally)")

    # ── 5. Unitary inspection ─────────────────────────────────
    print("\n[5] Full circuit unitary")
    qc5 = QuantumCircuit2Q('00')
    qc5.h(0).cnot()
    plot_circuit_unitary(qc5.U_total, "Unitary of Bell-state circuit (H⊗I · CNOT)")

    # ── 6. Deutsch single-qubit oracle test ───────────────────
    print("\n[6] Deutsch oracle (1-qubit problem)")
    for ftype in ['constant_0','constant_1','balanced_id','balanced_not']:
        qc6 = deutsch_2qubit_oracle(ftype)
        probs = qc6.probabilities()
        # If input qubit (q0) measured as |0⟩ → constant; |1⟩ → balanced
        p_const = probs.get('|00⟩',0) + probs.get('|01⟩',0)
        result  = 'CONSTANT' if p_const > 0.9 else 'BALANCED'
        print(f"  f_type={ftype:16s}  →  P(q0=0)={p_const:.3f}  → {result}")

    # ── 7. Shot-based measurement statistics ──────────────────
    print("\n[7] Bell state Φ⁺: 4096-shot measurement")
    qc7 = QuantumCircuit2Q('00')
    qc7.h(0).cnot()
    counts = qc7.measure(4096)
    total  = 4096
    for k in ['00','01','10','11']:
        c = counts.get(k, 0)
        bar = '█' * int(c / total * 40)
        print(f"  |{k}⟩: {c:5d}  {bar}")

    print("\nDone! ✓")
