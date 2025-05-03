import os
import torch
import matplotlib.pyplot as plt
import traceback
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Import from our source directory
from src.config import (
    MODEL_NAME, TRAINING_ARGS, RUN_OUTPUT_DIR,
    DEVICE_POLICY, CHECKPOINT_TO_RESUME_FROM
)
from src.dataset import load_and_preprocess_data
from src.rewards import ACTIVE_REWARD_FUNCTIONS # Use the selected rewards
from src.trainer import GRPOTrainer # Import the trainer class

def main():
    """Main function to set up and run the GRPO training."""

    print("--- Starting GRPO Fine-Tuning Script ---")

    # --- 1. Load Data ---
    try:
        train_dataset = load_and_preprocess_data()
        if not train_dataset:
            print("FATAL: No training data loaded. Exiting.")
            return
    except Exception as e:
        print(f"FATAL: Failed to load dataset: {e}")
        traceback.print_exc()
        return

    # --- 2. Load Model and Processor ---
    print(f"\nLoading VLM: {MODEL_NAME}")
    # Quantization config for policy model
    bnb_config_policy = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(DEVICE_POLICY)[0] >= 8 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if processor.tokenizer.pad_token is None:
            print("Warning: Tokenizer missing pad token, setting to EOS token.")
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config_policy,
            device_map=DEVICE_POLICY, # Load directly to policy device
            trust_remote_code=True
        )
        print("Base model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load model or processor: {e}")
        traceback.print_exc()
        return

    # --- 3. Apply LoRA ---
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=8, # Example LoRA rank
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Target attention blocks
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    try:
        model = get_peft_model(model, lora_config)
        print("PEFT LoRA model created.")
        model.print_trainable_parameters()
    except Exception as e:
        print(f"FATAL: Failed to apply LoRA config: {e}")
        traceback.print_exc()
        return

    # --- 4. Setup Trainer ---
    print("\nConfiguring GRPOTrainer...")
    # Pass necessary arguments from config and loaded objects
    trainer_config_args = {
        **TRAINING_ARGS, # Unpack hyperparams from config
        "output_dir": RUN_OUTPUT_DIR # Use the generated run directory
    }

    # Handle checkpoint resuming path verification
    resume_from_path = CHECKPOINT_TO_RESUME_FROM
    if resume_from_path:
        print(f"Attempting to resume training from: {resume_from_path}")
        if not os.path.isdir(resume_from_path):
            print(f"WARNING: Checkpoint directory '{resume_from_path}' not found. Starting fresh.")
            resume_from_path = None
    else:
        print("Starting new training run (no checkpoint specified).")


    try:
        trainer = GRPOTrainer(
            model=model,
            processor=processor,
            reward_funcs=ACTIVE_REWARD_FUNCTIONS, # Use the selected list
            config_args=trainer_config_args,
            train_dataset=train_dataset,
            checkpoint_dir=resume_from_path # Pass verified path or None
        )
    except Exception as e:
        print(f"FATAL: Error initializing GRPOTrainer: {e}")
        traceback.print_exc()
        return

    # --- 5. Start Training ---
    print("\n--- Starting Training ---")
    training_metrics = None
    try:
        training_metrics = trainer.train()
        print("\n--- Training Run Finished Successfully ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User (KeyboardInterrupt) ---")
        print("Attempting to save final state...")
        trainer._save_checkpoint("checkpoint-interrupt")
        training_metrics = trainer.metrics # Get metrics collected so far
    except Exception as e:
         print(f"\n--- ERROR during training loop: {e} ---"); traceback.print_exc()
         print("Attempting to save final state after error...")
         trainer._save_checkpoint("checkpoint-error")
         training_metrics = trainer.metrics

    # --- 6. Plot Metrics ---
    if training_metrics:
        plot_metrics(training_metrics, trainer.config.output_dir)
    else:
        print("\nNo metrics collected, skipping plotting.")

    print("\n--- Script Finished ---")


def plot_metrics(metrics, output_dir):
    """Plots training metrics and saves the figure."""
    print("\nPlotting training metrics...")
    if not metrics.get('steps'):
        print("No 'steps' data found in metrics. Cannot plot.")
        return

    try:
        steps = metrics['steps']
        plt.figure(figsize=(18, 12)) # Slightly larger figure

        # Plot 1: Total Loss
        plt.subplot(2, 3, 1);
        if metrics.get('total_loss'): plt.plot(steps, metrics['total_loss'], label='Total Loss', alpha=0.8)
        plt.xlabel('Steps'); plt.ylabel('Loss'); plt.title('Total Loss'); plt.legend(); plt.grid(True)

        # Plot 2: Rewards
        plt.subplot(2, 3, 2);
        if metrics.get('mean_combined_reward'): plt.plot(steps, metrics['mean_combined_reward'], label='Mean Combined Reward', linewidth=2)
        if metrics.get('reward_components'):
             for r_name, r_vals in metrics['reward_components'].items():
                 if r_vals and len(r_vals) == len(steps): plt.plot(steps, r_vals, label=f'Reward ({r_name})', alpha=0.7, linestyle='--')
        plt.xlabel('Steps'); plt.ylabel('Reward'); plt.title('Combined Reward & Components'); plt.legend(); plt.grid(True)

        # Plot 3: KL Divergence
        plt.subplot(2, 3, 3);
        if metrics.get('kl_divergence'): plt.plot(steps, metrics['kl_divergence'], label='KL Divergence', alpha=0.8)
        plt.xlabel('Steps'); plt.ylabel('KL'); plt.title('KL Divergence'); plt.legend(); plt.grid(True)

        # Plot 4: Entropy
        plt.subplot(2, 3, 4);
        if metrics.get('entropy'): plt.plot(steps, metrics['entropy'], label='Entropy', alpha=0.8)
        plt.xlabel('Steps'); plt.ylabel('Entropy'); plt.title('Entropy'); plt.legend(); plt.grid(True)

        # Plot 5: Learning Rate
        plt.subplot(2, 3, 5);
        if metrics.get('lr'): plt.plot(steps, metrics['lr'], label='Learning Rate', alpha=0.8)
        plt.xlabel('Steps'); plt.ylabel('LR'); plt.title('Learning Rate'); plt.legend(); plt.grid(True); plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        # Plot 6: Surrogate Loss
        plt.subplot(2, 3, 6);
        if metrics.get('surrogate_loss'): plt.plot(steps, metrics['surrogate_loss'], label='Surrogate Loss', alpha=0.8)
        plt.xlabel('Steps'); plt.ylabel('Loss'); plt.title('Surrogate Loss'); plt.legend(); plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_metrics.png")
        plt.savefig(plot_path)
        print(f"Metrics plot saved to {plot_path}")
        plt.close()

    except ImportError:
        print("Matplotlib not found. Skipping metrics plot.")
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()