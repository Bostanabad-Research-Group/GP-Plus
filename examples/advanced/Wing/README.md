# Examples
These are basic example scripts for testing multifidelity operations on wing data, which has 10 numerical columns and 4 different sources (For the SF version, there is just 10 numerical columns). The script goes through using the data_gen function as well as fitting the model to the wing data.

Problem: Wing - Multifidelity (1 HF 3 LF) + Wing - Single Fidelity (HF)

How to run: File should run without errors by pressing run file. Feel free to adjust the seed, data generation, and model inputs such as the combined kernel and its encoders to test different setups.

Expected output: Model metrics, training time.