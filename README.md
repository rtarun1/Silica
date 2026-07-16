# SILICA: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation

## Setup

1.  **Clone the repository.**

    ```bash
    git clone https://github.com/rtarun1/Silica
    ```

2. **Install the required dependencies.**

    Using [uv](https://astral.sh/uv):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd Silica
    uv venv .venv
    source .venv/bin/activate
    uv sync
    ```

    Using pip:

    ```bash
    cd Silica
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

3.  Try SILICA on your images:

    ```bash
    # minimum 5GB VRAM required
    bash infer.sh
    ```

## ROS2 Setup (Humble)

1.  **Clone the repository.**

    ```bash
    git clone https://github.com/rtarun1/Silica

    # or add submodule to your existing workspace
    # git submodule add https://github.com/rtarun1/Silica silica
    ```

2.  Install the required dependencies.

    Using pip:

    ```bash
    cd Silica
    python3 -m pip install -e .
    ```

3.  Build the ROS2 workspace:

    ```bash
    cd <your_ros2_workspace>
    colcon build --symlink-install --packages-select silica_ros
    ```

4.  Source the workspace:

    ```bash
    source <your_ros2_workspace>/install/setup.bash
    ```

5.  Run the Silica demo launch file:

    ```bash
    ros2 launch silica_ros silica.launch.py
    ```

> [!NOTE]
> Make sure that you have a camera and that the camera is publishing RGB images. You will also need (aligned) depth images to get the glass-aware point-cloud.
> 
> The topics can be set in the config file [`config/params.yaml`](config/params.yaml).

## Roadmap

- [x] Inference code + [Mirage18k Dataset](https://huggingface.co/datasets/rtarun1/mirage18k)
- [x] ROS2 Integration
- [ ] Training & Evaluation code