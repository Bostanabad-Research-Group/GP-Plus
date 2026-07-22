# Multi-Fidelity Buckling Function (2 Sources)

Multi-fidelity GP regression on the Buckling function with two sources: high-fidelity `s0` and low-fidelity `s1`. Uses `MVMFKernel` with source columns, continuous features, and categorical one-hot blocks.

Start with the notebook to see how grouped vs separate categorical encodings differ; the `.py` scripts mirror those two model setups (useful for changing `num_seeds` and aggregating metrics).

Files:
- `Buckling_MF_GP_Grouped_vs_Separate_Single_seed.ipynb` — side-by-side walkthrough of both encodings
- `Buckling_MF_GP_Grouped_Cat.py` — categorical columns as one flat group
- `Buckling_MF_GP_Separate_Cat.py` — categorical columns as separate E / K / I groups

Problem: Buckling — multi-fidelity (1 HF + 1 LF)

How to run: Open the notebook first, or run either `.py` file. Feel free to adjust the seed, data generation, and model inputs such as the combined kernel and its encoders.

Expected output: Model metrics (including per-source RRMSE), training time, latent embeddings, and encoder plots
