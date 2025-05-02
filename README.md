# Qwen2.5-VL GRPO Fine-Tuning for Skin Disease Classification

This project fine-tunes the Qwen2.5-VL-3B-Instruct model using Generative Response Policy Optimization (GRPO) for classifying skin diseases based on images. It rewards the model for correctly identifying the disease from a predefined list and adhering to a specific XML-like output format.

## Features

*   Fine-tunes Qwen2.5-VL using GRPO.
*   Utilizes 4-bit quantization (BitsAndBytes) and PEFT (LoRA) for efficient training.
*   Loads data from a JSON file containing image references and conversations.
*   Preprocesses images (resizing and padding).
*   Performs stratified sampling to balance disease classes.
*   Implements custom reward functions for accuracy and format adherence.
*   Includes a robust `GRPOTrainer` class with:
    *   Reference model maintenance.
    *   Gradient accumulation.
    *   Mixed-precision support (BF16).
    *   Detailed metrics logging.
    *   Robust checkpointing and resumption.
*   Generates plots of training metrics.

## Project Structure

QwenVL_GRPO_FineTuning/
├── src/ # Source code modules
│ ├── init.py
│ ├── config.py # Configuration (paths, hyperparameters)
│ ├── dataset.py # Data loading and preprocessing
│ ├── rewards.py # Reward function definitions
│ ├── trainer.py # GRPOTrainer class
│ └── utils.py # Utility functions
├── scripts/ # Placeholder for utility scripts
├── .gitignore # Git ignore rules
├── requirements.txt # Python dependencies
├── README.md # This file
└── train.py # Main training script


## Prerequisites

*   Python 3.8+
*   NVIDIA GPU with CUDA support (Compute Capability 8.0+ recommended for BF16)
*   Sufficient GPU Memory (depends on model size and batch settings, >16GB recommended for 3B model with LoRA)
*   Git

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/QwenVL_GRPO_FineTuning.git # Replace with your repo URL
    cd QwenVL_GRPO_FineTuning
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure your `torch` installation matches your CUDA version.*

## Data Setup

**IMPORTANT:** The dataset (JSON file and images) is typically too large to be stored in a Git repository. You need to obtain the data separately.

1.  **Download/Obtain:**
    *   The dataset JSON file (e.g., `traindataset_8cls.json`).
    *   The directory containing all referenced images (e.g., `merged_total-4874img`).

2.  **Configure Paths:**
    *   Open `src/config.py`.
    *   Modify the `DATASET_PATH` and `IMAGE_DIR` variables to point to the *actual locations* of your downloaded JSON file and image directory, respectively.
    *   Adjust `OUTPUT_DIR_BASE` to specify where training outputs (checkpoints, logs, plots) should be saved.

    *Alternatively, you can set environment variables (see comments in `src/config.py`) instead of hardcoding paths.*

## Configuration

Before running, review and adjust settings in `src/config.py`:

*   **Paths:** `DATASET_PATH`, `IMAGE_DIR`, `OUTPUT_DIR_BASE`.
*   **Resuming:** Set `CHECKPOINT_TO_RESUME_FROM` to the path of a specific checkpoint directory (e.g., `/path/to/outputs/QwenVL.../checkpoint-100`) to resume training, or set it to `None` to start fresh.
*   **Training Hyperparameters:** Modify values within the `TRAINING_ARGS` dictionary (learning rate, epochs, batch size, etc.).
*   **Sampling:** Adjust `SAMPLES_PER_DISEASE` for stratified sampling.
*   **Rewards:** Modify reward values (`REWARD_CORRECT`, etc.). You can also select which reward functions are active by editing the `ACTIVE_REWARD_FUNCTIONS` list in `src/rewards.py`.

## Running the Training

Execute the main training script from the project's root directory:

```bash
python train.py


The script will:
Load and preprocess the data.
Load the Qwen-VL model and apply LoRA.
Initialize the GRPOTrainer.
Run the training loop, logging progress and saving checkpoints periodically.
Generate a training_metrics.png plot in the run's output directory upon completion or interruption.


Output
Training outputs (checkpoints, metrics plot) will be saved in a timestamped subdirectory within the OUTPUT_DIR_BASE specified in src/config.py. Example: /kaggle/working/QwenVL_GRPO_Run_20231027-103000/.
Checkpoints are saved in directories like checkpoint-XXX, epoch-X, and potentially final_model.
