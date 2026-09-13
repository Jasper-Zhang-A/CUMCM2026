# Local execution notes

- **Python environment:** use `/home/jasper/miniconda3/envs/tslib/bin/python` for this project. It contains TiRex, PyTorch, pandas, SciPy, and the project dependencies; the base Python environment is incomplete.
- **GPU:** the host has an NVIDIA GeForce RTX 5050 Laptop GPU with CUDA 12.8. In the default sandbox, GPU access may be blocked and `torch.cuda.is_available()` can incorrectly return `False`; run GPU checks and CUDA workloads with escalated host access, then use `--device cuda:0`.
