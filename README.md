# SILICA: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation

## Setup

1. Clone the repository:

```bash
git clone https://github.com/silica-mirage/silica
```

2. Install the required dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd silica
uv venv .venv
uv pip install -r requirements.txt
```
3. Try SILICA on your images:
```bash
# minimum 5GB VRAM required
bash infer.sh
```