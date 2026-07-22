# Advanced Examples

Advanced GP+ examples on the Buckling and Wing benchmark functions, covering single-fidelity and multi-fidelity setups with categorical and continuous inputs.

## Encoder runs

Each `.py` script trains twice on the same data, with encoders passed explicitly:

1. **MatrixEncoder (A)** — `MatrixEncoder(...)` for categorical and/or source columns
2. **NeuralEncoder** — `NeuralEncoder(...)` for the same columns

Metrics (and Buckling latent plots) are reported for both.

| Folder | Problem | Fidelity |
| --- | --- | --- |
| `Buckling-Function_Single-Fidelity` | Buckling | Single fidelity (HF only) |
| `Buckling-Function_Multi-Fidelity_2-sources` | Buckling | Multi-fidelity (1 HF + 1 LF) |
| `Wing-Function_Multi-Fidelity_4-sources` | Wing | Multi-fidelity (1 HF + 3 LF) |

How to run: From the repo root, run a script with your Python environment that has `gpplus` installed (e.g. `python examples/advanced/.../script.py`). Adjust seed, data generation, and kernel settings as needed.

Expected output: Model metrics and training time for Matrix and Neural encoder runs. Buckling examples also plot latent embeddings.
