"""
Week 2 — Linear Algebra & Tensor Product Visualizer
=====================================================
Explore complex vectors, inner products, outer products,
tensor products, and multi-qubit state visualisation.
Run:  python tensor_visualizer.py
Deps: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product as iproduct


# ─────────────────────────────────────────────────────────────
# Basis states
# ─────────────────────────────────────────────────────────────

KET_0 = np.array([1, 0], dtype=complex)
KET_1 = np.array([0, 1], dtype=complex)

# Standard gates
I2   = np.eye(2, dtype=complex)
H    = np.array([[1,  1], [1, -1]], dtype=complex) / np.sqrt(2)
X    = np.array([[0,  1], [1,  0]], dtype=complex)
Z    = np.array([[1,  0], [0, -1]], dtype=complex)
CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)


# ─────────────────────────────────────────────────────────────
# Linear algebra primitives
# ─────────────────────────────────────────────────────────────

def tensor(*arrays):
    """
    Tensor (Kronecker) product of any number of state vectors or matrices.
    Works for both 1-D state vectors and 2-D gate matrices.

    Example:
        tensor(KET_0, KET_1)           → |01⟩  shape (4,)
        tensor(H, I2)                  → H⊗I   shape (4,4)
        tensor(KET_0, KET_1, KET_0)   → |010⟩ shape (8,)
    """
    result = arrays[0]
    for a in arrays[1:]:
        result = np.kron(result, a)
    return result


def inner(bra, ket):
    """
    Dirac inner product ⟨bra|ket⟩.
    Returns a complex scalar. Note: conjugate-linear in first argument.
    """
    return np.conj(bra) @ ket


def outer(ket, bra):
    """
    Outer product |ket⟩⟨bra|  →  matrix of shape (len(ket), len(bra)).
    Example: outer(KET_0, KET_1) = |0⟩⟨1| = [[0,1],[0,0]]
    """
    return np.outer(ket, np.conj(bra))


def norm(v):
    """Euclidean norm (should be 1 for valid quantum states)."""
    return float(np.sqrt(inner(v, v).real))


def is_normalized(v, tol=1e-10):
    return abs(norm(v) - 1.0) < tol


def is_unitary(U, tol=1e-10):
    """Check U†U ≈ I."""
    n = U.shape[0]
    return np.allclose(np.conj(U).T @ U, np.eye(n), atol=tol)


def is_hermitian(M, tol=1e-10):
    """Check M† = M."""
    return np.allclose(M, np.conj(M).T, atol=tol)


def partial_trace(rho, keep, dims):
    """
    Compute the partial trace of density matrix rho, keeping subsystem 'keep'.
    dims: list of subsystem dimensions, e.g. [2, 2] for two qubits.
    keep: index of subsystem to keep (0 = first, 1 = second).
    """
    n = len(dims)
    rho_tensor = rho.reshape(dims * 2)
    # Sum over the traced-out subsystem
    traced = np.tensordot(rho_tensor,
                          np.eye(dims[1-keep], dtype=complex),
                          axes=([1-keep, n + 1 - keep], [0, 1]))
    return traced.reshape(dims[keep], dims[keep])


def entanglement_entropy(state_2qubit):
    """
    Von Neumann entropy S(ρ_A) = -Tr(ρ_A log ρ_A) for a 2-qubit pure state.
    0 = product state (no entanglement), 1 = maximally entangled.
    """
    rho = outer(state_2qubit, state_2qubit)
    rho_A = partial_trace(rho, keep=0, dims=[2, 2])
    eigvals = np.linalg.eigvalsh(rho_A)
    eigvals = eigvals[eigvals > 1e-12]  # remove numerical zeros
    S = -np.sum(eigvals * np.log2(eigvals))
    return float(S)


def schmidt_decompose(state_2qubit):
    """
    Schmidt decomposition of a 2-qubit pure state.
    Returns (schmidt_coeffs, left_vecs, right_vecs)
    Schmidt rank = number of nonzero coefficients.
    Schmidt rank 1 → product state. Schmidt rank 2 → entangled.
    """
    M = state_2qubit.reshape(2, 2)
    U, s, Vh = np.linalg.svd(M)
    return s, U.T, Vh  # coefficients, left, right basis vectors


# ─────────────────────────────────────────────────────────────
# Basis labels utility
# ─────────────────────────────────────────────────────────────

def basis_labels(n_qubits):
    """Return list like ['|00⟩','|01⟩','|10⟩','|11⟩'] for n=2."""
    return [f"|{''.join(map(str,bits))}⟩"
            for bits in iproduct(range(2), repeat=n_qubits)]


# ─────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────

def visualize_state(state, title="State", n_qubits=None):
    """
    Three-panel visualisation of a multi-qubit state:
      - Amplitude magnitudes (bar chart)
      - Phases on a unit circle
      - Probability heatmap (2-qubit only)
    """
    if n_qubits is None:
        n_qubits = int(np.log2(len(state)))
    N = 2 ** n_qubits
    labels = basis_labels(n_qubits)
    probs  = np.abs(state) ** 2
    phases = np.angle(state)
    amps   = np.abs(state)

    palette = ['#7F77DD','#1D9E75','#D85A30','#3B8BD4',
               '#BA7517','#D4537E','#5DCAA5','#AFA9EC']

    if n_qubits == 2:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Panel 1: probability bars
    ax = axes[0]
    colors = [palette[i % len(palette)] for i in range(N)]
    bars = ax.bar(labels, probs, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 1.1); ax.set_ylabel('Probability')
    ax.set_title('Measurement probabilities')
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.02,
                    f'{p:.3f}', ha='center', fontsize=9)
    ax.axhline(1/N, color='gray', linestyle='--', linewidth=0.7,
               label=f'Uniform (1/{N})', alpha=0.7)
    ax.legend(fontsize=8); ax.tick_params(axis='x', labelsize=10)

    # Panel 2: amplitude phase wheel
    ax2 = axes[1]
    th = np.linspace(0, 2*np.pi, 300)
    ax2.plot(np.cos(th), np.sin(th), 'lightgray', linewidth=0.8)
    ax2.axhline(0, color='lightgray', linewidth=0.4)
    ax2.axvline(0, color='lightgray', linewidth=0.4)
    for i, (ph, amp, lbl) in enumerate(zip(phases, amps, labels)):
        if amp > 0.01:
            ax2.annotate('', xy=(amp*np.cos(ph), amp*np.sin(ph)),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color=palette[i%len(palette)], lw=2))
            ax2.text(amp*np.cos(ph)*1.2, amp*np.sin(ph)*1.2, lbl,
                     color=palette[i%len(palette)], fontsize=9, ha='center')
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Amplitude (phase & magnitude)')
    ax2.set_xlabel('Re'); ax2.set_ylabel('Im')

    # Panel 3 (2-qubit only): heatmap
    if n_qubits == 2:
        ax3 = axes[2]
        grid = probs.reshape(2, 2)
        im = ax3.imshow(grid, cmap='RdPu', vmin=0, vmax=1, aspect='auto')
        ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['q₁=0', 'q₁=1'])
        ax3.set_yticklabels(['q₀=0', 'q₀=1'])
        ax3.set_title('Probability heatmap')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        for r in range(2):
            for c in range(2):
                ax3.text(c, r, f'{grid[r,c]:.3f}', ha='center', va='center',
                         fontsize=11,
                         color='white' if grid[r,c] > 0.5 else 'black')

    plt.tight_layout()
    plt.show()


def visualize_matrix(M, title="Matrix", row_labels=None, col_labels=None):
    """
    Heatmap of a complex matrix — shows magnitude and phase separately.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    n = M.shape[0]
    ticks = range(n)
    lbls = row_labels or [f'{i}' for i in range(n)]

    for ax, data, subtitle, cmap in [
        (axes[0], np.abs(M),    'Magnitude |Mᵢⱼ|', 'Blues'),
        (axes[1], np.angle(M),  'Phase arg(Mᵢⱼ) [rad]', 'RdBu'),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(col_labels or lbls, fontsize=9)
        ax.set_yticklabels(lbls, fontsize=9)
        ax.set_title(subtitle)
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(val) > 0.6 else 'black')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  WEEK 2 — Linear Algebra & Tensor Products")
    print("=" * 55)

    # 1. Inner products and orthogonality
    print("\n[1] Inner products ⟨φ|ψ⟩")
    pairs = [
        (KET_0, KET_0, "⟨0|0⟩"), (KET_0, KET_1, "⟨0|1⟩"),
        (KET_1, KET_1, "⟨1|1⟩"),
    ]
    plus  = (KET_0 + KET_1) / np.sqrt(2)
    minus = (KET_0 - KET_1) / np.sqrt(2)
    pairs += [(plus, minus, "⟨+|−⟩"), (plus, plus, "⟨+|+⟩")]
    for bra, ket, label in pairs:
        val = inner(bra, ket)
        print(f"  {label} = {val:.4f}")

    # 2. Outer products (density matrices)
    print("\n[2] Outer products |ψ⟩⟨ψ| (density matrices)")
    rho_0   = outer(KET_0, KET_0)
    rho_plus = outer(plus, plus)
    print(f"  |0⟩⟨0| =\n{rho_0.real}")
    print(f"  |+⟩⟨+| =\n{rho_plus.real}")
    visualize_matrix(tensor(H, I2), "H⊗I gate matrix",
                     basis_labels(2), basis_labels(2))

    # 3. Tensor products of states
    print("\n[3] Tensor products of states")
    states = {
        '|00⟩': tensor(KET_0, KET_0),
        '|01⟩': tensor(KET_0, KET_1),
        '|10⟩': tensor(KET_1, KET_0),
        '|11⟩': tensor(KET_1, KET_1),
    }
    for label, s in states.items():
        print(f"  {label} = {s}")

    # 4. Bell states via circuits
    print("\n[4] Bell states via (H⊗I)·CNOT·|00⟩")
    HI  = tensor(H, I2)

    bell_states = {}
    for init_label, init_state in [('|00⟩', tensor(KET_0,KET_0)),
                                    ('|01⟩', tensor(KET_0,KET_1)),
                                    ('|10⟩', tensor(KET_1,KET_0)),
                                    ('|11⟩', tensor(KET_1,KET_1))]:
        bell = CNOT @ HI @ init_state
        bell_states[init_label] = bell
        ent  = entanglement_entropy(bell)
        sv, _, _ = schmidt_decompose(bell)
        print(f"  (H⊗I)·CNOT·{init_label}  →  {bell.round(4)}"
              f"  entropy={ent:.4f}  Schmidt={sv.round(4)}")

    phi_plus = bell_states['|00⟩']
    visualize_state(phi_plus, "Bell state Φ⁺ = (|00⟩+|11⟩)/√2")

    # 5. Product state (no entanglement)
    print("\n[5] Product state vs entangled state")
    product_state = tensor(plus, KET_0)   # |+⟩⊗|0⟩
    print(f"  |+⟩⊗|0⟩ entropy     = {entanglement_entropy(product_state):.6f}  (0 = product)")
    print(f"  Bell state Φ⁺ entropy = {entanglement_entropy(phi_plus):.6f}  (1 = max entangled)")
    visualize_state(product_state, "|+⟩⊗|0⟩ — product state (no entanglement)")

    # 6. Unitarity checks
    print("\n[6] Unitarity checks (U†U = I?)")
    for name, mat in [('H',H),('X',X),('Z',Z),('H⊗I',HI),('CNOT',CNOT),('SWAP',SWAP)]:
        print(f"  {name:8s} unitary: {is_unitary(mat)}")

    # 7. Partial trace (tracing out one qubit of Bell state)
    print("\n[7] Partial trace of Bell state Φ⁺")
    rho_bell = outer(phi_plus, phi_plus)
    rho_A = partial_trace(rho_bell, keep=0, dims=[2,2])
    rho_B = partial_trace(rho_bell, keep=1, dims=[2,2])
    print(f"  ρ_A = Tr_B(|Φ⁺⟩⟨Φ⁺|) =\n{rho_A.real.round(4)}")
    print(f"  (Maximally mixed — Alice sees 50/50 regardless of Bob)")
    print(f"\n  ρ_B = Tr_A(|Φ⁺⟩⟨Φ⁺|) =\n{rho_B.real.round(4)}")

    print("\nDone! ✓")
