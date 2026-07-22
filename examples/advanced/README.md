# Advanced Examples

Advanced GP+ examples on the Buckling and Wing benchmark functions, covering single-fidelity and multi-fidelity setups with categorical and continuous inputs.

| Folder | Problem | Fidelity |
| --- | --- | --- |
| `Single-Fidelity_Buckling-Function` | Buckling | Single fidelity (HF only) |
| `Single-Fidelity_Wing-Function` | Wing | Single fidelity (HF only) |
| `Multi-Fidelity_2-sources_Buckling-Function` | Buckling | Multi-fidelity (1 HF + 1 LF) |
| `Multi-Fidelity_4-sources_Wing-Function` | Wing | Multi-fidelity (1 HF + 3 LF) |

How to run: From the repo root, run a script with your Python environment that has `gpplus` installed (e.g. `python examples/advanced/.../script.py`). Adjust seed, data generation, and kernel/encoder settings as needed.

Expected output: Model metrics and training time. Buckling examples also plot latent embeddings for categorical encoders.
