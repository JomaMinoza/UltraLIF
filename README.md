# UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization

Official code for the ICML 2026 paper:

> **UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization**
> Jose Marie Antonio Miñoza. *ICML 2026.*
> [arXiv:2602.11206](https://arxiv.org/abs/2602.11206)

## Overview

Standard Spiking Neural Networks (SNNs) are trained with surrogate gradients that approximate the non-differentiable Heaviside spike function. UltraLIF replaces the entire neuron — membrane dynamics and spike function — with a fully differentiable formulation derived from **max-plus (tropical) algebra**.

The membrane update becomes a soft max-plus operation (logsumexp), parameterized by a learnable temperature **ε** that interpolates between hard tropical dynamics (ε→0) and linear averaging (ε→∞):

```
V(t+1) = LSE_ε(V(t) + log(τ),  I(t))          # UltraLIF  (temporal, 2-term)
V_i(t) = LSE_ε(V_{i-1}, V_i, V_{i+1}) + I_i   # UltraDLIF (spatial,  3-term)
```

No surrogate gradients. No neuromorphic hardware required.

## Model Reference

| Paper Name  | CLI Key     | Description                            |
|-------------|-------------|----------------------------------------|
| UltraLIF    | `ultratlif` | Temporal, 2-term LSE, fixed τ         |
| UltraPLIF   | `ultratplif`| Temporal, 2-term LSE, learnable τ     |
| UltraDLIF   | `ultradlif` | Spatial, 3-term LSE, fixed τ          |
| UltraDPLIF  | `ultradplif`| Spatial, 3-term LSE, learnable τ      |
| LIF         | `lif`       | Standard LIF (surrogate gradient)      |
| PLIF        | `plif`      | LIF with learnable τ                  |
| DSpike      | `dspike`    | Li et al. NeurIPS 2021                |
| DSpike+     | `dspike+`   | DSpike with learnable τ               |
| SigmaLIF    | `sigmalif`  | Sigmoid-only ablation baseline         |

## Installation

```bash
git clone https://github.com/JomaMinoza/UltraLIF.git
cd UltraLIF
pip install -r requirements.txt
```

For neuromorphic datasets, tonic is required (included in requirements.txt).

## Quick Start

```bash
# Download static datasets
python scripts/download_datasets.py --static

# Train UltraLIF (temporal) on MNIST
python experiments/train.py --model ultratlif --dataset mnist \
    --epochs 100 --hidden 64 --track-spikes

# Train UltraDLIF (spatial) on SHD
python experiments/train.py --model ultradlif --dataset shd \
    --epochs 100 --hidden 64 --timesteps 10

# Train all single-layer models on CIFAR-10
python experiments/train.py --model all --dataset cifar10 --epochs 100 --hidden 64

# ResNet18 backbone experiments
python experiments/train_resnet.py --dataset cifar10 --backbone resnet18 \
    --timesteps 1 5 10

# Epsilon ablation (Appendix B.1)
python ablations/eps_ablation.py --epochs 100 --timesteps 1 --seed 42
```

## Python API

```python
from ultralif import UltraLIF, SNN, get_dataset, train_model, set_seed

set_seed(42)
train_loader, test_loader, in_dim, n_cls = get_dataset('mnist')

neuron = UltraLIF(dim=256)
model = SNN(neuron, in_dim=in_dim, hid_dim=256, out_dim=n_cls)

best_acc, history, _ = train_model(
    model, train_loader, test_loader,
    epochs=100, lr=1e-3, device='cuda',
    track_spikes=True,
)
print(f"Best accuracy: {best_acc:.2%}")
print(f"Learned ε: {neuron.eps.item():.3f}")
```

## Supported Datasets

| Dataset      | Type          | Classes | Input dim |
|-------------|---------------|---------|-----------|
| `mnist`     | Static        | 10      | 784       |
| `fashion`   | Static        | 10      | 784       |
| `cifar10`   | Static        | 10      | 3072      |
| `nmnist`    | Neuromorphic  | 10      | 2×34×34   |
| `dvs_gesture`| Neuromorphic | 11      | 2×128×128 |
| `shd`       | Audio spike   | 20      | 700       |
| `ssc`       | Audio spike   | 35      | 700       |

## Structure

```
UltraLIF/
├── ultralif/           # Python library
│   ├── neurons/        # UltraLIF, UltraDLIF, LIF, DSpike, SigmaLIF
│   ├── networks/       # SNN, DeepSNN, ConvSNN, SpikingResNet18
│   ├── datasets/       # get_dataset(), rate_encode(), set_seed()
│   └── training/       # train_model(), TeeLogger, metrics
├── experiments/
│   ├── train.py        # Main FC/Conv/ResNet training script
│   └── train_resnet.py # ResNet backbone experiments
├── ablations/
│   └── eps_ablation.py # ε ablation (Appendix B.1)
└── scripts/
    └── download_datasets.py
```

## Citation

```bibtex
@inproceedings{minoza2026ultralif,
  title     = {UltraLIF: Fully Differentiable Spiking Neural Networks via Ultradiscretization},
  author    = {Mi{\~n}oza, Jose Marie Antonio},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
}
```
