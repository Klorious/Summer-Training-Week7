# Reproducible environment

The final formal reruns were executed with:

- Python 3.12.3
- PyTorch 2.6
- Weights & Biases 0.28.0
- NVIDIA Tesla V100-SXM2-32GB
- Random seed 20260725

## Option A: use the TWCC preinstalled PyTorch environment

First verify that the container already provides the required GPU-enabled
PyTorch version:

```bash
python --version
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

Install or update the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the main imports:

```bash
python -c "import torch, torchvision, numpy, pandas, sklearn, scipy, PIL, yaml, tqdm, wandb; print('environment check passed')"
```

## Option B: create a new Conda environment

```bash
conda env create -f environment.yml
conda activate training-unit7
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

PyTorch GPU packages depend on the host driver and CUDA runtime. If the
standard installation does not detect the GPU, use the installation command
recommended by the official PyTorch selector for the current CUDA platform,
then rerun the verification command.

## W&B login

```bash
wandb login
```

Never commit a W&B API key, `.netrc`, `.env`, or the local `wandb/` directory.
