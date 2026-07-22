# Multi-Fidelity Wing Function (4 Sources)

Multi-fidelity GP regression on the Wing function with four sources (`s0`–`s3`: 1 HF + 3 LF). The input has 10 continuous numerical features plus source one-hot columns (no categorical variables).

## Encoder runs

The script trains twice on the same data, with the source encoder passed explicitly:

1. **MatrixEncoder (A)** — `MatrixEncoder(...)` for the source columns
2. **NeuralEncoder** — `NeuralEncoder(...)` for the source columns

Files:
- `Wing_MF_GP.py` — `load_data_wing_MV_MF` + `MVMFKernel` with multi-source mean/likelihood; Matrix then Neural source encoder

Problem: Wing — multi-fidelity (1 HF + 3 LF)

How to run: Run the `.py` file. Feel free to adjust the seed, data generation, and model inputs.

Expected output: Model metrics and training time for Matrix and Neural source encoders
