```
# README.md

```markdown
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
```

The training script utilizes the `accelerate` library, which will automatically configure the training for multi-GPU usage based on your environment. Ensure you have configured `accelerate` for your multi-GPU setup if you haven't already. You can do this by running:

```bash
accelerate config
```

and following the prompts.

### Configuration

The training configuration is defined within the `TrainingConfig` class in the `src/train.py` script. You can adjust parameters such as:

* `model_name`: The pre-trained diffusion model to use (currently set to `black-forest-labs/FLUX.1-dev`).
* `dataset_dir`: Path to your training dataset.
* `output_dir`: Directory where trained LoRA models will be saved.
* `resolution`: The image resolution for training.
* `batch_size`: The batch size per GPU. Adjust this based on your GPU memory.
* `gradient_accumulation_steps`: Number of steps to accumulate gradients before performing a backward pass. Useful for increasing the effective batch size.
* `learning_rate`: The learning rate for the optimizer.
* `max_train_steps`: The total number of training steps.
* `checkpointing_steps`: Save a checkpoint every this many steps.
* `validation_steps`: Perform validation every this many steps.
* `validation_prompt`: The prompt used for validation.
* `seed`: Random seed for reproducibility.
* `mixed_precision`: Whether to use mixed precision training (`bf16` recommended for RTX 4090).

## Output

Trained LoRA checkpoints will be saved in the directory specified by `config.output_dir`.

## Multi-GPU Usage

The script is designed to work with multiple GPUs using the `accelerate` library. Ensure your environment is properly configured for multi-GPU training with `accelerate`. The script will automatically detect and utilize the available GPUs.

## License

This project is licensed under the [Specify License Here] - see the `LICENSE` file for details.

## Acknowledgements

* black-forest-labs for the FLUX.1-dev model.
* Hugging Face for the `diffusers`, `transformers`, and `accelerate` libraries.
* The open-source community for their contributions to generative AI.
```

```
# .gitignore

```
```
# Python
*.pyc
__pycache__/
.venv/
venv/
env/
*.egg-info
.DS_Store

# Output directories
output/
logs/

# Temporary files
*.tmp
*.log

# Environment files
.env

# PyTorch Lightning
lightning_logs/

# Jupyter Notebooks (optional, if you use them)
*.ipynb
.ipynb_checkpoints/
```

```
# LICENSE

```
```
MIT License

Copyright (c) [Year] [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**To use this:**

1.  Copy the text under the `# README.md` heading and save it as a file named `README.md`.
2.  Copy the text under the `# .gitignore` heading and save it as a file named `.gitignore`.
3.  Copy the text under the `# LICENSE` heading and save it as a file named `LICENSE`. **Remember to replace `[Year]` and `[Your Name]` in the MIT License example with the correct information.**

Let me know if there's anything else I can help you with!
