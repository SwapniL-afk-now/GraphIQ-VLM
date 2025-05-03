import torch
import time
import os

# --- Essential Paths ---
# !! IMPORTANT: Users must modify these paths !!
# Option 1: Use environment variables (Recommended for flexibility)
# DATASET_PATH = os.getenv("DERMNET_DATASET_PATH", "/path/to/your/traindataset_8cls.json")
# IMAGE_DIR = os.getenv("DERMNET_IMAGE_DIR", "/path/to/your/merged_total-4874img")
# OUTPUT_DIR_BASE = os.getenv("GRPO_OUTPUT_DIR", "/path/to/your/outputs")
# CHECKPOINT_TO_RESUME_FROM = os.getenv("GRPO_RESUME_CHECKPOINT", None) # e.g., "/path/to/outputs/Qwen.../checkpoint-100"

# Option 2: Hardcode paths (Simpler for personal use, less flexible)
# Replace with your actual paths on Kaggle or your local machine
DATASET_PATH = "/kaggle/input/selective-dermnet-for-llm/Final Training Jsons/Final Training Jsons/label/traindataset_8cls.json"
IMAGE_DIR = "/kaggle/input/selective-dermnet-for-llm/merged_total-4874img"
OUTPUT_DIR_BASE = "/kaggle/working/" # Base directory for outputs on Kaggle

# <<< IMPORTANT: SET CHECKPOINT PATH TO RESUME OR None FOR FRESH START >>>
# CHECKPOINT_TO_RESUME_FROM = "/kaggle/input/my-last-checkpoint/Qwen2.5-VL_GRPO_Run_.../checkpoint-100"
CHECKPOINT_TO_RESUME_FROM = "/kaggle/input/ckeck330/pytorch/default/1/Qwen2.5-VL_GRPO_Run_20250417-050412/checkpoint-330" # Example

# --- Model Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
VALID_DISEASES = [
    "actinic keratosis", "basal cell carcinoma", "dermatitis", "lichen planus",
    "melanoma", "psoriasis", "rosacea", "seborrheic keratosis"
]

# --- System Prompt ---
SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks the name of skin disease, and the Assistant gives the name of the disease.
The assistant first thinks about the reasoning process in the mind and then provides the user with the disease name strictly from "seborrheic keratosis", "actinic keratosis", "psoriasis", "melanoma","basal cell carcinoma", "dermatitis", "rosacea", "lichen planus". The reasoning process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., <thinking> reasoning process here </thinking><answer>  Detected disease here  </answer>.

Response Format rules:
- Always start your response with the <thinking> tag and end with the </answer> tag.
- Do not include any text or commentary before the opening <thinking> tag or after the closing </answer> tag.
- Do not include any text or commentary between the closing </thinking> tag and the opening <answer> tag.

When formulating your response, follow exactly the structure below:

<thinking>
### Analysis of the Image:
1. **Color and Pigmentation:**
2. **Texture and Surface Characteristics:**
3. **Shape and Border:**
</thinking>
<answer>
Detected disease name
</answer>
"""

# --- Device Configuration ---
if torch.cuda.is_available():
    print(f"Found {torch.cuda.device_count()} GPUs.")
    DEVICE_POLICY = "cuda:0"
    DEVICE_REF = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_POLICY
    print(f"Using {DEVICE_POLICY} (Policy), {DEVICE_REF} (Reference).")
else:
    print("Warning: No CUDA GPUs found. Using CPU. Training will be very slow.")
    DEVICE_POLICY = "cpu"
    DEVICE_REF = "cpu"

# --- GRPO Trainer Configuration ---
# GRPOConfig class moved to trainer.py, but we define the hyperparams here
TRAINING_ARGS = {
    "learning_rate": 3e-5,
    "num_train_epochs": 3,
    "num_generations": 3,
    "max_completion_length": 250,
    "gradient_accumulation_steps": 4,
    "clip_epsilon": 0.2,
    "beta": 0.001,
    "entropy_coef": 0.001,
    "logging_steps": 10,
    "save_steps": 10, # Reduced save frequency for example
    "max_steps": -1,
    "temperature": 0.7,
    "per_device_train_batch_size": 1,
    "reward_scale": 1.0,
    "max_input_ids_length": 600,
    "update_ref_steps": 10, # Reduced sync frequency for example
    "warmup_steps": 50,
    "device": DEVICE_POLICY,
    "ref_device": DEVICE_REF,
    # bf16 will be determined automatically in GRPOTrainer based on device capability
    # "bf16": True/False # Can override if needed
}

# Define a unique output directory name for this run
RUN_SUFFIX = time.strftime("%Y%m%d-%H%M%S")
RUN_OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, f"QwenVL_GRPO_Run_{RUN_SUFFIX}")

# --- Dataset Configuration ---
SAMPLES_PER_DISEASE = 100 # For stratified sampling
RANDOM_SEED = 42

# --- Reward Configuration ---
FORMAT_SCALE_FACTOR = 2.0
REWARD_CORRECT = 7.0
REWARD_INCORRECT = -1.0
REWARD_ADHERENCE = 1.0
REWARD_PENALTY = -0.5

# --- Environment Setup ---
# Set environment variables (optional, can be set externally)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # Helpful for debugging, potentially remove for performance