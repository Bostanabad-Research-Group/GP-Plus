# GPPlus developer setup

**Requirements**:

- Python == 3.9  
- CUDA >= 11.8 (if using GPU)  

To use GP+, you first need to install the specific versions of PyTorch. The installation process involves two steps: (1) installing the specific version of PyTorch based on your system, and (2) installing GP+.

## Installing

### (1) Install PyTorch

#### For macOS
To install PyTorch for macOS, use:

```bash
pip install torch==2.5.1 torchvision==0.16.1 torchaudio==2.5.1
```

#### For Linux and Windows
To install PyTorch for Linux and Windows, follow the steps below based on whether you have CUDA support or not.

##### For GPU Support (with CUDA)
If you have a compatible GPU and want to leverage GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch==2.5.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu118
```

##### For CPU Only
If you do not have a compatible GPU, install the CPU-only version of PyTorch:

```bash
pip install torch==2.5.1+cpu torchvision==0.16.1+cpu torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu
```

### (2) Install GP+ from source

Clone the repository and build from source.

```bash
pip install .
```