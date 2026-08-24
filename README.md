# Quantum-Computing-Summer-2026
My learning and experimentation in the field of Quantum Computing.

Over the 2026 summer, out of interest and curiosity I started learning (in 6 steps) about Quantum Computing (QC) in preparation for the Quantum Information Science certificate courses at UTD. I wanted to apply my Python programming experience and math background to understand QC and turn concepts into visuals that represented the inner workings of QC. Upon the learning and completion of 6 steps, I wanted to see if I can tackle a research issue in QC. In addition I wanted to run the code on IBM Quantum Computer and also learn about Qiskit in the process.

**Step 1)** I learned about Qubits, State vectors, H gate, Born rule and Bloch sphere. I implemented the Qubit Simulator. The code is in qubit_simulator.py. This program was built to demonstrate Pure-NumPy Qubit class, Bloch sphere plot, shot-based experiments, interference demo.
 
  **Summary of key concepts:** 
  A classical bit is always definitively 0 or 1. A qubit, by contrast, exists as a superposition until the moment it is measured.
  A qubit's state is written as:   |ψ⟩ = α|0⟩ + β|1⟩
  where α and β are complex numbers called amplitudes, satisfying |α|² + |β|² = 1. When you measure, you get |0⟩ with probability |α|² and |1⟩ with probability |β|².

  **The Hadamard gate**
  The most important single-qubit gate for creating superposition is H:
      H|0⟩ = (|0⟩ + |1⟩)/√2 → 50% chance each H|1⟩ = (|0⟩ − |1⟩)/√2 → 50% chance each, but phase differs
  The phase (the sign) doesn't affect measurement probabilities alone, but it matters enormously when gates interact — this is the key to quantum interference.

  **The Bloch sphere**
  Every single-qubit state can be visualized as a point on a unit sphere. |0⟩ is the north pole, |1⟩ is the south pole, and superpositions live on the equator.     
  Gates are rotations of this sphere.

**Step 2)**  
