"""
Week 1 — Qubit Simulator
========================
Build a pure-Python qubit simulator with Hadamard gate and probabilistic measurement.
Run:  python qubit_simulator.py
Deps: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import mpl_toolkits.mplot3d.proj3d as proj3d


# ─────────────────────────────────────────────────────────────
# Core qubit class
# ─────────────────────────────────────────────────────────────

class Qubit:
    """Single-qubit state vector simulator using complex NumPy arrays."""

    # Standard single-qubit gates (2×2 unitaries)
    H = np.array([[1,  1],
                  [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    Y = np.array([[ 0, -1j],
                  [1j,  0]], dtype=complex)
    Z = np.array([[1,  0],
                  [0, -1]], dtype=complex)
    S = np.array([[1,  0],
                  [0, 1j]], dtype=complex)
    T = np.array([[1,  0],
                  [0,  np.exp(1j * np.pi / 4)]], dtype=complex)

    def __init__(self, state='0'):
        """
        Initialise in a computational basis state.
        state: '0' → |0⟩ = [1, 0]ᵀ
               '1' → |1⟩ = [0, 1]ᵀ
        """
        if state == '0':
            self.state = np.array([1.0 + 0j, 0.0 + 0j])
        elif state == '1':
            self.state = np.array([0.0 + 0j, 1.0 + 0j])
        else:
            raise ValueError("Initial state must be '0' or '1'")

    @classmethod
    def from_angles(cls, theta, phi):
        """
        Create qubit from Bloch sphere angles.
        |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)·sin(θ/2)|1⟩
        theta: polar angle  [0, π]
        phi:   azimuthal angle [0, 2π]
        """
        q = cls.__new__(cls)
        q.state = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2)
        ], dtype=complex)
        return q

    def apply(self, gate):
        """Apply a 2×2 unitary. Returns self for chaining."""
        if not np.allclose(gate.conj().T @ gate, np.eye(2), atol=1e-10):
            raise ValueError("Gate is not unitary!")
        self.state = gate @ self.state
        return self

    def measure(self):
        """
        Collapse the qubit via Born rule measurement.
        Returns: 0 or 1 (int)
        Post-measurement state is the collapsed basis state.
        """
        p0 = abs(self.state[0]) ** 2
        outcome = 0 if np.random.random() < p0 else 1
        self.state = (np.array([1.0, 0.0]) if outcome == 0
                      else np.array([0.0, 1.0]))
        return outcome

    def probabilities(self):
        """Return dict with measurement probabilities."""
        return {
            '|0⟩': round(float(abs(self.state[0]) ** 2), 6),
            '|1⟩': round(float(abs(self.state[1]) ** 2), 6),
        }

    def bloch_angles(self):
        """Convert state to Bloch sphere (θ, φ) in radians."""
        alpha, beta = self.state
        theta = 2 * np.arccos(np.clip(abs(alpha), 0, 1))
        phi = np.angle(beta) - np.angle(alpha)
        return theta, phi

    def bloch_vector(self):
        """Return (x, y, z) Bloch vector."""
        theta, phi = self.bloch_angles()
        return (
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )

    def __repr__(self):
        a, b = self.state
        return f"|ψ⟩ = ({a:.4f})|0⟩ + ({b:.4f})|1⟩"


# ─────────────────────────────────────────────────────────────
# Shot-based experiment runner
# ─────────────────────────────────────────────────────────────

def run_experiment(gate_sequence, shots=1000, init='0'):
    """
    Repeatedly prepare a qubit, apply gates, and measure.

    gate_sequence : list of 2×2 numpy arrays
    shots         : number of repetitions
    init          : '0' or '1'

    Returns dict {'0': count, '1': count}
    """
    counts = {'0': 0, '1': 0}
    for _ in range(shots):
        q = Qubit(init)
        for gate in gate_sequence:
            q.apply(gate)
        outcome = str(q.measure())
        counts[outcome] += 1
    return counts


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def plot_bloch_sphere(states_with_labels, title="Bloch Sphere"):
    """
    Plot one or more qubit states on the Bloch sphere.
    states_with_labels: list of (Qubit, label_str) tuples
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xs, ys, zs, color='lightblue', alpha=0.08)

    # Axes
    for (dx, dy, dz, lbl) in [(1,0,0,'+X'),(0,1,0,'+Y'),(0,0,1,'|0⟩'),
                               (-1,0,0,'-X'),(0,-1,0,'-Y'),(0,0,-1,'|1⟩')]:
        ax.plot([0, dx*1.3],[0, dy*1.3],[0, dz*1.3],
                'gray', linewidth=0.6, alpha=0.5)
        ax.text(dx*1.4, dy*1.4, dz*1.4, lbl, ha='center', fontsize=9, color='gray')

    # Equator
    theta_eq = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_eq), np.sin(theta_eq), 0, 'gray', alpha=0.3, linewidth=0.8)

    # State vectors
    colors = ['#7F77DD', '#D85A30', '#1D9E75', '#D4537E']
    for i, (q, lbl) in enumerate(states_with_labels):
        x, y, z = q.bloch_vector()
        col = colors[i % len(colors)]
        ax.quiver(0, 0, 0, x, y, z, length=1,
                  color=col, linewidth=2.5, arrow_length_ratio=0.15)
        ax.text(x*1.15, y*1.15, z*1.15, lbl, color=col, fontsize=10, fontweight='bold')

    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.4, 1.4)
    ax.set_box_aspect([1,1,1])
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.show()


def plot_measurement_histogram(experiments_dict, title="Measurement statistics"):
    """
    Bar chart comparing multiple experiments.
    experiments_dict: {experiment_label: counts_dict}
    e.g. {'H|0⟩': {'0':512,'1':488}, 'H·H|0⟩': {'0':1000,'1':0}}
    """
    labels   = list(experiments_dict.keys())
    prob_0   = [experiments_dict[l].get('0',0) /
                sum(experiments_dict[l].values()) for l in labels]
    prob_1   = [experiments_dict[l].get('1',0) /
                sum(experiments_dict[l].values()) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars0 = ax.bar(x - width/2, prob_0, width, label='P(|0⟩)',
                   color='#7F77DD', edgecolor='white')
    bars1 = ax.bar(x + width/2, prob_1, width, label='P(|1⟩)',
                   color='#D85A30', edgecolor='white')

    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='50%')
    ax.legend(fontsize=10)

    for bar, p in zip(list(bars0) + list(bars1), prob_0 + prob_1):
        if p > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.02,
                    f'{p:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  WEEK 1 — Qubit Simulator Demo")
    print("=" * 55)

    # 1. Basic state inspection
    print("\n[1] State inspection")
    q = Qubit('0')
    print(f"  Initial:      {q}")
    print(f"  Probabilities: {q.probabilities()}")

    q.apply(Qubit.H)
    print(f"\n  After H:      {q}")
    print(f"  Probabilities: {q.probabilities()}")
    print(f"  Bloch vector:  {tuple(round(v,4) for v in q.bloch_vector())}")

    # 2. Self-inverse: H·H = I
    q.apply(Qubit.H)
    print(f"\n  After H·H:    {q}  ← back to |0⟩ (H is its own inverse)")

    # 3. Phase gate exploration
    print("\n[2] Phase exploration")
    for name, gate in [('S', Qubit.S), ('T', Qubit.T), ('Z', Qubit.Z)]:
        q = Qubit('0')
        q.apply(Qubit.H).apply(gate)
        print(f"  H·{name}|0⟩  →  {q}")

    # 4. Bloch sphere visualisation
    print("\n[3] Bloch sphere — plotting 4 states...")
    states_to_plot = [
        (Qubit('0'),                                "|0⟩"),
        (Qubit('1'),                                "|1⟩"),
        (Qubit('0').apply(Qubit.H),                 "H|0⟩=|+⟩"),
        (Qubit.from_angles(np.pi/2, np.pi/2),       "|i⟩"),
    ]
    plot_bloch_sphere(states_to_plot, "Key qubit states on the Bloch sphere")

    # 5. Shot-based experiments
    print("\n[4] 1 000-shot measurement experiments")
    experiments = {
        'H|0⟩':        run_experiment([Qubit.H],               shots=1000),
        'H·H|0⟩':      run_experiment([Qubit.H, Qubit.H],      shots=1000),
        'X|0⟩':        run_experiment([Qubit.X],               shots=1000),
        'H·Z·H|0⟩':    run_experiment([Qubit.H, Qubit.Z, Qubit.H], shots=1000),
    }
    for label, counts in experiments.items():
        total = sum(counts.values())
        p0 = counts['0'] / total
        p1 = counts['1'] / total
        print(f"  {label:18s} →  P(0)={p0:.3f}  P(1)={p1:.3f}  {counts}")

    plot_measurement_histogram(experiments, "1 000-shot measurement comparison")

    # 6. Interference demo
    print("\n[5] Interference demonstration")
    print("  H|0⟩ measured 10 times:")
    for _ in range(10):
        q = Qubit('0'); q.apply(Qubit.H)
        print(f"    → {q.measure()}", end='  ')
    print()
    print("  H·H|0⟩ measured 10 times (always 0 — destructive interference):")
    for _ in range(10):
        q = Qubit('0'); q.apply(Qubit.H).apply(Qubit.H)
        print(f"    → {q.measure()}", end='  ')
    print("\n")
    print("Done! ✓  Try editing gate sequences to explore more states.")
