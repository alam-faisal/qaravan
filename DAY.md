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
