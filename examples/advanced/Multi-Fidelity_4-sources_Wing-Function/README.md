# Multi-Fidelity Wing Function (4 Sources)

Multi-fidelity GP regression on the Wing function with four sources (`s0`–`s3`: 1 HF + 3 LF). The input has 10 continuous numerical features plus source one-hot columns (no categorical variables).

Files:
- `Wing_MF_GP.py` — loads data via `load_data_wing_MV_MF` and fits an `MVMFKernel` GP with multi-source mean and likelihood

Problem: Wing — multi-fidelity (1 HF + 3 LF)

How to run: Run the `.py` file. Feel free to adjust the seed, data generation, and model inputs such as the combined kernel and its encoders.

Expected output: Model metrics and training time
