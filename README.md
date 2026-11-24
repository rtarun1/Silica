<p align="center">
<h1 align="center"><strong>SILICA: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation</strong></h1>
  <p align="center">
      	<!-- <strong>Accepted at IROS 2026</strong><br> -->
        <a href="https://www.linkedin.com/in/rtarun1/" target="_blank">Tarun R</a>, 
        <a href="https://scholar.google.com/citations?user=KI2RjboAAAAJ&hl=en" target="_blank">Anuj Verma</a>, 
        <a href="https://scholar.google.com/citations?user=XCF-6gcAAAAJ&hl=en" target="_blank">Laksh Nanwani</a>, 
        <a href="https://scholar.google.com/citations?user=oVS3HHIAAAAJ&hl=en" target="_blank">Sourav Garg</a>,  
        <a href="https://scholar.google.co.in/citations?user=QDuPGHwAAAAJ&hl=en" target="_blank">Madhava Krishna</a>
    <br>
    Robotics Research Center (RRC), IIIT Hyderabad, India
    <br>
  </p>
</p>


<div id="top" align="center">

[![Project](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://silica-mirage.github.io/)
[![Paper](https://img.shields.io/badge/IEEE_Xplore-Paper-00629B?logo=ieee&logoColor=white)](https://doi.org/10.1109/IROS61738.2026.11776824)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white
)](https://arxiv.org/abs/2607.24249)
[![Model](https://img.shields.io/badge/🤗%20Silica-Model-yellow
)](https://huggingface.co/rtarun1/silica-v1-0)
[![Dataset](https://img.shields.io/badge/🤗%20Mirage18k-Dataset-green
)](https://huggingface.co/datasets/rtarun1/mirage18k/)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-blue)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
</div>

![teaser](example/assets/teaser.png)

## 🛠️ Setup

1.  **Clone the repository.**

    ```bash
    git clone https://github.com/rtarun1/Silica
    ```

2. **Install the required dependencies.**

    Using [uv](https://astral.sh/uv):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    cd Silica
    uv sync
    ```

    Using pip:

    ```bash
    cd Silica
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

## 🚀 Run inference

```bash
bash infer.sh
```

## 🏋️ Training

1. **Depth training dataset preparation**

- [Hypersim](https://github.com/apple/ml-hypersim): 
    - Download this [script](https://github.com/apple/ml-hypersim/blob/main/contrib/99991/download.py) into your `$PATH_TO_RAW_HYPERSIM_DATA` directory for data downloading.
    - Run the following command to download the data:
        ```
        cd $PATH_TO_RAW_HYPERSIM_DATA

        # Download the tone-mapped images
        python ./download.py --contains scene_cam_ --contains final_preview --contains tonemap.jpg --silent

        # Download the depth maps
        python ./download.py --contains scene_cam_ --contains geometry_hdf5 --contains depth_meters --silent
        ```
    - Download the split file from [here](https://github.com/apple/ml-hypersim/blob/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv) and put it in the `$PATH_TO_RAW_HYPERSIM_DATA` directory.
    - Process the data with the command: `bash utils/process_hypersim.sh`.

2. **Glass Segmentation training dataset preparation**

- Mirage18k dataset is available [here](https://huggingface.co/datasets/rtarun1/mirage18k/). You can download the dataset using the following command:

    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/rtarun1/mirage18k
    ```
- For [Trans10k](https://github.com/xieenze/Segment_Transparent_Objects), [GSDS](https://github.com/xavhl/GlassSemNet), [GDD](https://github.com/Mhaiyang/CVPR2020_GDNet), [3DRef](https://3dref.github.io/), and [GWD](https://github.com/ViktorLiang/GW-Depth) please follow the instructions provided in their respective repositories to download the datasets.

3. **Train the model:**

    ```bash
    bash train.sh
    ```
## 📊 Evaluation

```bash
bash eval.sh
```

## 🤖 ROS2 Humble inference

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

## 🔗 Citation
If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{r2026silica,
  author    = {R, Tarun and Verma, Anuj and Nanwani, Laksh and Garg, Sourav and Krishna, K. Madhava},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title     = {{SILICA}: Repurposing Diffusion Priors for Joint Glass Segmentation and Depth Estimation},
  year      = {2026},
  pages     = {5031--5038},
  doi       = {10.1109/IROS61738.2026.11776824}}
```


## 👏 Acknowledgement

We thank the authors of [Marigold](https://github.com/prs-eth/Marigold), [Lotus](https://github.com/EnVision-Research/Lotus) and [Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) for their valuable research and for open-sourcing the code and pretrained models that made this work possible.