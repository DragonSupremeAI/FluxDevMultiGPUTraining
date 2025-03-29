# FLUX.1-dev LoRA Fine-tuning with Multi-GPU Support

This repository contains the code for fine-tuning the [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) diffusion model using LoRA (Low-Rank Adaptation) on a custom dataset. The training script is designed to leverage multi-GPU setups (specifically tested with 4 RTX 4090s) using the `accelerate` library.

## Setup

### Prerequisites

* Python 3.8 or higher
* A compatible CUDA installation for your GPUs
* `pip` package installer

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

1.  Prepare your dataset of images and corresponding text captions.
2.  Organize your dataset in a directory structure similar to the `datasets/my_dataset` example. Each image file (e.g., `image1.jpg`) should have a corresponding text file with the same name but a `.txt` extension (e.g., `image1.txt`) containing the caption for the image.
3.  Update the `config.dataset_dir` in the `src/train.py` script to point to the location of your dataset.

### Hugging Face Token

1.  You will need a Hugging Face token to access the `FLUX.1-dev` model. If you don't have one, you can create an account on [Hugging Face](https://huggingface.co/) and generate a token from your profile settings.
2.  Replace `"your_hugging_face_token_here"` in the `config.hf_token` variable within the `src/train.py` script with your actual Hugging Face token.

## Training

To start the training process, run the `train.py` script:

```bash
python src/train.py
