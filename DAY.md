# Daily Log

---

## 2026-05-24

### Accomplished

- Implemented [`core/base.py`](src/qaravan/core/base.py) — six ABCs (`Gate`, `Circuit`, `State`, `Simulator`, `Observable`, `NoiseModel`) that define the entire v0.2 architecture; 43 passing tests including `_embed_gate` and `to_matrix` correctness checks
- Created [`examples/design.ipynb`](examples/design.ipynb) — interactive walkthrough of the new API, ending with a 25-line inline statevector backend that runs a GHZ circuit end-to-end
- Wrote Task 4 proposal: concrete gate library (`H`, `X`, `CNOT`, `RX/Y/Z`, `RXX/YY/ZZ`, `Reset`, `Measure`, qutrits) in [`proposals/2026-05-24-task4-gates.md`](proposals/2026-05-24-task4-gates.md)
- Added ruff as dev dependency; pre-push lint check (`uv run ruff check src/ tests/`) added to TODO workflow
- Updated README with Gotchas section: big-endian qubit ordering and `Circuit.copy()` layers-vs-decomposition behaviour

### Next steps

**Faisal:**
1. Go through [`examples/design.ipynb`](examples/design.ipynb) from Section 4 (`_embed_gate`) onward — verify the non-contiguous gate embedding and the inline backend by eye
2. Review [`proposals/2026-05-24-task4-gates.md`](proposals/2026-05-24-task4-gates.md) and approve or request changes before Task 4 begins

**Claude:**
3. Implement Task 4 once proposal is approved

---

## 2026-05-26

### Accomplished

- Completed Task 6 close-out from previous session: full test suite (159 tests), ruff clean, pushed to `origin/rewrite/v0.2`
- Task 6 removed from `TODO.md`; Task 7 is now the topmost task
- Wrote and iterated Task 7 proposal [`proposals/2026-05-26-task7-statevector.md`](proposals/2026-05-26-task7-statevector.md) through three rounds of design discussion:
  - Identified `op_action` and `partial_overlap` as the two ncon primitives that all contraction code flows through
  - `project_and_renorm` and `rdm` promoted to public `Statevector` methods; `reset` wraps `project_and_renorm`
  - `measure_and_collapse` (renamed from `sample_and_collapse` in the ABC) returns `(State, str)` with full n qubits
  - `num_sites` attribute to be added to `PauliString` / `PauliSum`
  - Decided against a `tensors.py` extraction for now; SV contraction functions live in `statevector.py`
  - Confirmed `ncon_torch` stays as the contraction backend throughout

### Next steps

**Faisal:**
1. Review [`proposals/2026-05-26-task7-statevector.md`](proposals/2026-05-26-task7-statevector.md) and approve or request changes

**Claude:**
2. Implement Task 7 once proposal is approved: `Statevector` + `StatevectorSimulator` in `backends/statevector.py`, preceded by the `num_sites` attribute addition to `PauliString`/`PauliSum` and the `measure_and_collapse` rename in `core/base.py`
