# SILICA: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation

## Setup

1. Clone the repository:

```bash
git clone https://github.com/rtarun1/Silica
```

2. Install the required dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd silica
uv venv .venv
uv sync
```
3. Try SILICA on your images:
```bash
# minimum 5GB VRAM required
bash infer.sh
```

## Roadmap

- [ ] Inference code + Dataset
- [ ] Evaluation code
- [ ] Training code