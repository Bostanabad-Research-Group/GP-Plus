# Multi-Fidelity Buckling Function (2 Sources)

Multi-fidelity GP regression on the Buckling function with two sources: high-fidelity `s0` and low-fidelity `s1`. Uses `MVMFKernel` with source columns, continuous features, and categorical one-hot blocks.

## Encoder runs

Each `.py` script trains twice on the same data, with encoders passed explicitly:

1. **MatrixEncoder (A)** — `MatrixEncoder(...)` for cat and source
2. **NeuralEncoder** — `NeuralEncoder(...)` for cat and source

Use the notebooks to compare grouped vs separate categorical layouts for one encoder type at a time. Use the `.py` scripts to run Matrix then Neural back-to-back.

Files:
- `Buckling_MF_GP_Grouped_vs_Separate_Matrix-Encoder.ipynb` — grouped vs separate with MatrixEncoder
- `Buckling_MF_GP_Grouped_vs_Separate_NN-Encoder.ipynb` — grouped vs separate with NeuralEncoder
- `Buckling_MF_GP_Grouped_Cat_Encoder.py` — flat cat group; Matrix then Neural
- `Buckling_MF_GP_Separate_Cat_Encoder.py` — separate E / K / I cat groups; Matrix then Neural

Problem: Buckling — multi-fidelity (1 HF + 1 LF)

How to run: Open either notebook, or run either `.py` file. Feel free to adjust the seed, data generation, and model inputs.

Expected output: Model metrics (including per-source RRMSE), training time, latent embeddings, and encoder plots
