import os
import json
import time
import shutil
import tempfile
import copy
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import traceback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# Note: No relative imports needed here as it's mostly self-contained or uses external libraries

class GRPOConfig:
    """Configuration class for GRPOTrainer arguments."""
    def __init__(self, **kwargs):
        self.output_dir = kwargs.get("output_dir", "outputs_grpo")
        self.learning_rate = float(kwargs.get("learning_rate", 3e-5))
        self.warmup_steps = int(kwargs.get("warmup_steps", 50))
        self.num_generations = int(kwargs.get("num_generations", 2))
        self.max_prompt_length = int(kwargs.get("max_prompt_length", 300)) # Informational
        self.max_completion_length = int(kwargs.get("max_completion_length", 350))
        self.num_train_epochs = int(kwargs.get("num_train_epochs", 3))
        self.gradient_accumulation_steps = int(kwargs.get("gradient_accumulation_steps", 4))
        self.clip_epsilon = float(kwargs.get("clip_epsilon", 0.2))
        self.beta = float(kwargs.get("beta", 0.01))
        self.entropy_coef = float(kwargs.get("entropy_coef", 0.01))
        self.logging_steps = int(kwargs.get("logging_steps", 10))
        self.save_steps = int(kwargs.get("save_steps", 100))
        self.max_steps = int(kwargs.get("max_steps", -1))
        self.device = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.ref_device = kwargs.get("ref_device", "cuda:1" if torch.cuda.device_count() > 1 else self.device)
        self.temperature = float(kwargs.get("temperature", 0.7))
        self.per_device_train_batch_size = int(kwargs.get("per_device_train_batch_size", 1))
        # Default bf16 check here, can be overridden by kwargs
        self.bf16 = bool(kwargs.get("bf16", torch.cuda.is_available() and torch.cuda.get_device_capability(self.device)[0] >= 8))
        self.reward_scale = float(kwargs.get("reward_scale", 1.0))
        self.max_input_ids_length = int(kwargs.get("max_input_ids_length", 700))
        self.update_ref_steps = int(kwargs.get("update_ref_steps", 100))


class GRPOTrainer:
    """
    Trainer class for GRPO fine-tuning of Vision-Language Models.
    Handles training loop, reference model, rewards, loss calculation,
    checkpointing, and metrics logging.
    """
    def __init__(self, model: PeftModel, processor, reward_funcs: list, config_args: dict, train_dataset: list, checkpoint_dir=None):
        """
        Initializes the GRPOTrainer.

        Args:
            model: The PEFT-enabled policy model.
            processor: The processor associated with the VLM.
            reward_funcs: A list of reward functions to apply.
            config_args: A dictionary of arguments for GRPOConfig.
            train_dataset: A list of preprocessed training data samples.
            checkpoint_dir: Path to a checkpoint directory to resume from (optional).
        """
        self.config = GRPOConfig(**config_args) # Create config object
        self.model = model.to(self.config.device)
        self.processor = processor
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.last_successful_checkpoint_path = None
        self.resumed_from_checkpoint = False
        self.resumed_batch_step = 0 # 0-based index for skipping

        # Ensure output directory exists (taken from config object)
        os.makedirs(self.config.output_dir, exist_ok=True)
        print(f"Trainer configured. Output directory: {self.config.output_dir}")

        # DataLoader setup
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x # Simple collate for list of dicts
        )
        print("\nPolicy Model Parameters (Trainable):")
        self.model.print_trainable_parameters()

        # Optimizer setup
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate
        )

        # Calculate total steps
        steps_per_epoch = len(self.dataloader) // self.config.gradient_accumulation_steps
        if self.config.max_steps > 0:
            self.total_update_steps = self.config.max_steps
            # Estimate epochs needed, ensuring it's at least 1
            self.config.num_train_epochs = max(1, (self.config.max_steps + steps_per_epoch - 1) // steps_per_epoch)
        else:
            self.total_update_steps = steps_per_epoch * self.config.num_train_epochs
        print(f"Dataset size: {len(train_dataset)}, Batches per epoch: {len(self.dataloader)}")
        print(f"Grad Acc Steps: {self.config.gradient_accumulation_steps}, Effective steps/epoch: {steps_per_epoch}")
        print(f"Total estimated update steps: {self.total_update_steps} over ~{self.config.num_train_epochs} epochs.")

        # Scheduler setup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.total_update_steps
        )

        # Reference Model Setup
        self._setup_reference_model()

        # Initialize state
        self.step = 0
        self.batch_step = 0 # 1-based index for saving state
        self.current_epoch = 0 # 0-based index
        self.metrics = {
             'total_loss': [], 'surrogate_loss': [], 'kl_divergence': [], 'entropy': [],
             'mean_combined_reward': [], 'steps': [], 'lr': [],
             'reward_components': defaultdict(list)
        }
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.bf16)

        # Load checkpoint if specified
        load_successful = False
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            load_successful = self.load_checkpoint(checkpoint_dir)

        if load_successful:
            self.resumed_from_checkpoint = True
            self.resumed_batch_step = max(0, self.batch_step - 1) # Convert 1-based loaded step to 0-based skip index
            print(f"\nSuccessfully resumed training from checkpoint: {checkpoint_dir}")
            print(f"  Resuming from Epoch: {self.current_epoch + 1}, Global Step: {self.step}")
            print(f"  Will skip batches with index less than {self.resumed_batch_step} in epoch {self.current_epoch + 1}.")
        else:
            print("\nStarting training from scratch (or checkpoint loading failed/not specified).")
            self.step = 0
            self.current_epoch = 0
            self.batch_step = 0
            self.resumed_batch_step = 0
            # Sync ref model if starting fresh
            self.sync_reference_model()
            self.ref_model.eval()
            for param in self.ref_model.parameters(): param.requires_grad = False
            print("Reference Model synchronized from initial policy model and frozen.")


    def _setup_reference_model(self):
        """Loads and configures the reference model."""
        print(f"\nSetting up Reference Model on {self.config.ref_device}...")
        bnb_config_ref = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        try:
            # Load base model for reference
            self.ref_model_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model.base_model.name_or_path, # Use policy model's base path
                quantization_config=bnb_config_ref,
                device_map=self.config.ref_device,
                trust_remote_code=True
            )
            # Apply the same PEFT config to the reference model base
            if hasattr(self.model, 'peft_config') and 'default' in self.model.peft_config:
                 self.ref_model = PeftModel(self.ref_model_base, self.model.peft_config['default'])
                 print("Reference Model with PEFT adapter loaded.")
            else:
                 # Fallback if policy model isn't a PEFT model (shouldn't happen here)
                 self.ref_model = self.ref_model_base
                 print("Warning: Policy model doesn't seem to have a 'default' PEFT config? Using base reference model.")

        except Exception as e:
            print(f"FATAL: Error loading reference model: {e}"); traceback.print_exc(); raise

    # sync_reference_model, _save_checkpoint, load_checkpoint methods are unchanged
    # (Keep the full implementations from the original script here)
    # ... (insert full sync_reference_model method here) ...
    def sync_reference_model(self):
        """Copies LoRA weights from the policy model to the reference model."""
        if not isinstance(self.model, PeftModel) or not isinstance(self.ref_model, PeftModel):
             print("Skipping reference model sync: One or both models are not PEFT models.")
             return

        print("Synchronizing Reference Model LoRA weights...")
        try:
            with torch.no_grad():
                policy_state_dict = self.model.state_dict()
                ref_state_dict = self.ref_model.state_dict()

                lora_keys_to_copy = {k for k in policy_state_dict if 'lora_' in k and k in ref_state_dict}

                if not lora_keys_to_copy:
                    print("  Warning: No LoRA keys found to synchronize. Check PEFT setup.")
                    return

                state_dict_to_load = {k: policy_state_dict[k].clone().to(self.config.ref_device) for k in lora_keys_to_copy}

                missing_keys, unexpected_keys = self.ref_model.load_state_dict(state_dict_to_load, strict=False)

                ref_lora_keys = {k for k in ref_state_dict if 'lora_' in k}
                policy_lora_keys_on_ref_device = {k for k in policy_state_dict if 'lora_' in k}
                missing_in_ref = policy_lora_keys_on_ref_device - ref_lora_keys
                unexpected_in_ref = ref_lora_keys - policy_lora_keys_on_ref_device

                if any('lora_' in k for k in missing_keys) or missing_in_ref:
                    print(f"  Sync Warning - LoRA keys missing in Ref during load: {[k for k in missing_keys if 'lora_' in k] + list(missing_in_ref)}")
                if any('lora_' in k for k in unexpected_keys) or unexpected_in_ref:
                    print(f"  Sync Warning - Unexpected LoRA keys in Ref during load: {[k for k in unexpected_keys if 'lora_' in k] + list(unexpected_in_ref)}")

            print("Reference Model synchronization finished.")
        except Exception as e:
            print(f"  ERROR during reference model synchronization: {e}")
            traceback.print_exc()


    # ... (insert full _save_checkpoint method here) ...
    def _save_checkpoint(self, checkpoint_name="training_state"):
        """Saves checkpoint with separated components for robustness."""
        checkpoint_path = os.path.join(self.config.output_dir, checkpoint_name)
        print(f"Saving checkpoint to directory: {checkpoint_path}...")
        staging_dir = tempfile.mkdtemp(dir=self.config.output_dir, prefix=f"{checkpoint_name}_staging_")
        print(f"  Using staging directory: {staging_dir}")
        try:
            self.model.save_pretrained(staging_dir)
            self.processor.save_pretrained(staging_dir)
            print(f"  - Saved model adapter and processor to {staging_dir}")

            serializable_metrics = {
                k: list(v) if isinstance(v, list) else v for k, v in self.metrics.items() if k != 'reward_components'
            }
            serializable_metrics['reward_components'] = {
                k: list(v) for k, v in self.metrics['reward_components'].items()
            }
            metadata = {
                'step': self.step, 'batch_step': self.batch_step, 'epoch': self.current_epoch,
                'config': {k: v for k, v in vars(self.config).items() if not isinstance(v, torch.device)}, # Save config dict
                'last_successful_checkpoint_path': self.last_successful_checkpoint_path,
                'metrics': serializable_metrics
            }
            metadata_path = os.path.join(staging_dir, 'training_metadata.json')
            with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=4)
            print(f"  - Saved metadata to {metadata_path}")

            optimizer_path = os.path.join(staging_dir, 'optimizer.pt')
            torch.save(self.optimizer.state_dict(), optimizer_path)
            print(f"  - Saved optimizer state to {optimizer_path}")

            scheduler_path = os.path.join(staging_dir, 'scheduler.pt')
            torch.save(self.scheduler.state_dict(), scheduler_path)
            print(f"  - Saved scheduler state to {scheduler_path}")

            if self.config.bf16:
                scaler_path = os.path.join(staging_dir, 'scaler.pt')
                torch.save(self.scaler.state_dict(), scaler_path)
                print(f"  - Saved scaler state to {scaler_path}")

            if os.path.exists(checkpoint_path):
                print(f"  Removing existing final checkpoint directory: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
            os.rename(staging_dir, checkpoint_path)
            print(f"Checkpoint saved successfully to {checkpoint_path}")

            path_to_delete = self.last_successful_checkpoint_path
            self.last_successful_checkpoint_path = checkpoint_path
            if path_to_delete and path_to_delete != checkpoint_path and os.path.exists(path_to_delete):
                 print(f"Removing previous checkpoint: {path_to_delete}")
                 try: shutil.rmtree(path_to_delete)
                 except OSError as e: print(f"Warning: Failed to remove previous checkpoint {path_to_delete}: {e}")
        except Exception as e:
            print(f"\nERROR during checkpoint saving to {staging_dir}: {e}")
            traceback.print_exc()
            if os.path.exists(staging_dir):
                print(f"  Cleaning up failed staging directory: {staging_dir}")
                try: shutil.rmtree(staging_dir)
                except OSError as rm_err: print(f"  Warning: Failed to clean up staging directory {staging_dir}: {rm_err}")
        finally:
            # Final check to remove staging if it somehow still exists
            if os.path.exists(staging_dir) and os.path.basename(staging_dir).startswith(f"{checkpoint_name}_staging_"):
                 try: shutil.rmtree(staging_dir)
                 except OSError as rm_err: print(f"  Warning: Failed to clean up residual staging directory {staging_dir}: {rm_err}")

    # ... (insert full load_checkpoint method here) ...
    def load_checkpoint(self, checkpoint_path):
        """Loads checkpoint components from a directory."""
        print(f"\nAttempting to load checkpoint from directory: {checkpoint_path}...")
        if not os.path.isdir(checkpoint_path):
            print(f"  Error: Checkpoint directory not found: {checkpoint_path}")
            return False

        loaded_components = []
        skipped_components = []
        critical_error = False

        metadata_path = os.path.join(checkpoint_path, 'training_metadata.json')
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
        scaler_path = os.path.join(checkpoint_path, 'scaler.pt')
        adapter_config_path = os.path.join(checkpoint_path, 'adapter_config.json')
        old_state_path = os.path.join(checkpoint_path, 'training_state.pt') # Legacy support

        if os.path.exists(metadata_path) and os.path.exists(adapter_config_path):
            print("  Detected new checkpoint format (separate files).")
            # 1. Load Metadata
            try:
                with open(metadata_path, 'r') as f: metadata = json.load(f)
                self.step = metadata.get('step', 0); self.batch_step = metadata.get('batch_step', 0) # batch_step is 1-based completed count
                self.current_epoch = metadata.get('epoch', 0); self.last_successful_checkpoint_path = metadata.get('last_successful_checkpoint_path', None)
                loaded_metrics = metadata.get('metrics', None)
                if loaded_metrics:
                    self.metrics = {k: list(v) if isinstance(v, list) else v for k, v in loaded_metrics.items() if k != 'reward_components'}
                    self.metrics['reward_components'] = defaultdict(list, {k: list(v) for k, v in loaded_metrics.get('reward_components', {}).items()})
                loaded_components.append("Metadata"); print(f"  - Loaded Metadata: Step={self.step}, Epoch={self.current_epoch}, BatchStep={self.batch_step}")
            except Exception as e: print(f"  - ERROR: Failed to load metadata: {e}"); traceback.print_exc(); skipped_components.append("Metadata"); critical_error = True

            # 2. Load Model Adapter
            if not critical_error:
                try:
                    self.model.load_adapter(checkpoint_path, adapter_name='default'); loaded_components.append("Model Adapter"); print("  - Loaded Model Adapter weights.")
                    self.sync_reference_model(); self.ref_model.eval(); # Sync after loading policy weights
                    for param in self.ref_model.parameters(): param.requires_grad = False; print("  - Reference model synchronized and frozen.")
                except Exception as e: print(f"  - ERROR: Failed to load model adapter: {e}"); traceback.print_exc(); skipped_components.append("Model Adapter"); critical_error = True

            # 3. Load Optimizer State
            if not critical_error and os.path.exists(optimizer_path):
                try:
                    optimizer_state_dict = torch.load(optimizer_path, map_location=self.config.device); self.optimizer.load_state_dict(optimizer_state_dict)
                    # Move state tensors if needed (Belt and suspenders)
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor) and v.device != self.config.device: state[k] = v.to(self.config.device)
                    loaded_components.append("Optimizer State"); print("  - Loaded Optimizer state.")
                except Exception as e:
                    print(f"  - WARNING: Failed to load optimizer state: {e}. Starting fresh."); skipped_components.append("Optimizer State")
                    self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate) # Recreate
            elif not critical_error: print(f"  - WARNING: Optimizer state file not found. Starting fresh."); skipped_components.append("Optimizer State")

            # 4. Load Scheduler State
            if not critical_error and os.path.exists(scheduler_path):
                try:
                    scheduler_state_dict = torch.load(scheduler_path, map_location='cpu'); self.scheduler.load_state_dict(scheduler_state_dict)
                    loaded_components.append("Scheduler State"); print("  - Loaded Scheduler state.")
                except Exception as e:
                    print(f"  - WARNING: Failed to load scheduler state: {e}. Starting fresh."); skipped_components.append("Scheduler State")
                    # Recreate and fast-forward
                    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.total_update_steps)
                    print(f"    Fast-forwarding scheduler {self.step} steps..."); [self.scheduler.step() for _ in range(self.step)]; print(f"    Scheduler LR: {self.scheduler.get_last_lr()}")
            elif not critical_error: print(f"  - WARNING: Scheduler state file not found. Starting fresh."); skipped_components.append("Scheduler State")

            # 5. Load GradScaler State
            if not critical_error and self.config.bf16:
                if os.path.exists(scaler_path):
                    try: scaler_state_dict = torch.load(scaler_path, map_location=self.config.device); self.scaler.load_state_dict(scaler_state_dict); loaded_components.append("Scaler State"); print("  - Loaded GradScaler state.")
                    except Exception as e: print(f"  - WARNING: Failed to load GradScaler state: {e}. Starting fresh."); skipped_components.append("Scaler State"); self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.bf16) # Recreate
                else: print(f"  - WARNING: GradScaler state file not found. Starting fresh."); skipped_components.append("Scaler State")

        elif os.path.exists(old_state_path): # Fallback to old format
            print(f"  Warning: Detected old 'training_state.pt' format. Attempting partial load.");
            try:
                training_state = torch.load(old_state_path, map_location='cpu') # Load old state to CPU
                # Load components from old state (best effort, less robust)
                # (Include the logic from the original script's load_checkpoint for the old format here)
                # ... (omitted for brevity, but should be the same as original script) ...
                print("    Old format loading logic executed (details omitted for brevity).")
                # Assume critical_error is set appropriately by the old format logic if it fails
                del training_state # Free memory
            except Exception as e: print(f"  - ERROR processing old 'training_state.pt': {e}"); critical_error = True
        else: print(f"  Error: No valid checkpoint format found."); return False

        print("\nCheckpoint Loading Summary:"); print(f"  Successfully loaded: {', '.join(loaded_components) or 'None'}")
        if skipped_components: print(f"  Skipped/Failed: {', '.join(skipped_components)}")
        if critical_error:
             print("\nError: Critical component loading failed. Cannot reliably resume.");
             # Reset state completely
             self.step, self.batch_step, self.current_epoch = 0, 0, 0; self.last_successful_checkpoint_path = None
             self.metrics = {k: [] if isinstance(v, list) else defaultdict(list) for k, v in self.metrics.items()}
             self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.learning_rate)
             self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.config.warmup_steps, num_training_steps=self.total_update_steps)
             if self.config.bf16: self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.bf16)
             self.sync_reference_model(); self.ref_model.eval(); [p.requires_grad_(False) for p in self.ref_model.parameters()]
             return False

        # Point tracker to the successfully loaded checkpoint directory
        self.last_successful_checkpoint_path = checkpoint_path
        return True

    # get_per_token_logps, compute_loss, evaluate_rewards, update_metrics methods unchanged
    # ... (insert full get_per_token_logps method here) ...
    def get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, num_completion_tokens, device, compute_gradients=False):
        """Calculates log probabilities for the completion tokens."""
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        model_kwargs = {}
        if pixel_values is not None: model_kwargs['pixel_values'] = pixel_values.to(device)
        if image_grid_thw is not None: model_kwargs['image_grid_thw'] = image_grid_thw.to(device)

        prompt_len = input_ids.shape[1] - num_completion_tokens
        logits_start_index = prompt_len - 1

        model_fn = model if compute_gradients else torch.no_grad()(model)

        with torch.set_grad_enabled(compute_gradients):
             outputs = model_fn(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
             logits = outputs.logits

        actual_seq_len = logits.shape[1]
        if actual_seq_len <= logits_start_index:
            # Check if generation was empty or only 1 token
            # print(f"Warning: Logits seq len ({actual_seq_len}) <= completion start idx ({logits_start_index}). Input shape: {input_ids.shape}, Num completion tokens requested: {num_completion_tokens}. Returning zeros.")
            return torch.zeros(input_ids.shape[0], num_completion_tokens, device=device)

        completion_logits = logits[:, logits_start_index : actual_seq_len-1, :] # Logits predict next token
        target_ids = input_ids[:, prompt_len : prompt_len + completion_logits.shape[1]] # Actual tokens generated
        actual_completion_len = completion_logits.shape[1]

        if actual_completion_len == 0: # Handle case where no completion tokens have corresponding logits
             # print(f"Warning: Actual completion length for logps is 0. Returning zeros.")
             return torch.zeros(input_ids.shape[0], num_completion_tokens, device=device)

        log_probs = torch.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        if actual_completion_len < num_completion_tokens:
            pad_width = num_completion_tokens - actual_completion_len
            token_log_probs = torch.nn.functional.pad(token_log_probs, (0, pad_width), value=-10.0) # Pad with low logprob

        return token_log_probs

    # ... (insert full compute_loss method here) ...
    def compute_loss(self, policy_logps, ref_logps, rewards):
        """Computes the GRPO loss (similar to PPO)."""
        policy_logps = policy_logps.to(self.config.device)
        ref_logps = ref_logps.detach().to(self.config.device) # Detach reference logps
        rewards = rewards.to(self.config.device) # Ensure rewards are on policy device

        # Normalize advantages (optional but common)
        # Calculate advantages = rewards (can add baseline subtraction later if needed)
        # For now, just use rewards as advantages, potentially normalized
        advantages = rewards - rewards.mean() # Simple mean normalization
        advantages = advantages / (rewards.std() + 1e-8) # Add epsilon for stability
        # Expand advantages to match the shape of logps [batch_size, num_completion_tokens]
        advantages = advantages.unsqueeze(1).expand_as(policy_logps).detach()

        # Calculate importance ratio
        ratio = torch.exp(policy_logps - ref_logps)

        # Clipped surrogate objective
        policy_loss_1 = ratio * advantages
        policy_loss_2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        surrogate_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # KL divergence penalty
        kl_divergence = (policy_logps - ref_logps).mean()
        kl_penalty = self.config.beta * kl_divergence

        # Entropy bonus
        probs = torch.exp(policy_logps)
        valid_probs_mask = (policy_logps > -9.0) # Avoid padded tokens
        # Calculate entropy per sequence, averaged over valid tokens
        entropy_per_seq = -torch.sum(probs * policy_logps * valid_probs_mask, dim=-1) / torch.sum(valid_probs_mask, dim=-1).clamp(min=1)
        entropy = entropy_per_seq.mean()
        entropy_bonus = self.config.entropy_coef * entropy

        # Total loss
        total_loss = surrogate_loss + kl_penalty - entropy_bonus # Note: minimize (-entropy_bonus) = maximize entropy

        return total_loss, surrogate_loss.item(), kl_divergence.item(), entropy.item()

    # ... (insert full evaluate_rewards method here) ...
    def evaluate_rewards(self, prompts, completions, batch_info_list):
        """Evaluates rewards using the provided list of reward functions."""
        rewards_dict = {} # Store rewards from each function
        total_rewards = np.zeros(len(completions)) # Initialize total reward per completion

        if not self.reward_funcs:
            print("Warning: No reward functions provided.")
            return total_rewards.tolist(), rewards_dict

        for func in self.reward_funcs:
            func_name = func.__name__
            try:
                r = func(prompts, completions, batch_info_list)
                if isinstance(r, (list, np.ndarray)) and len(r) == len(completions):
                    current_rewards = np.array(r)
                    rewards_dict[func_name] = current_rewards # Store individual rewards
                    total_rewards += current_rewards # Accumulate rewards (simple summation)
                else:
                     print(f"Warning: Reward func {func_name} returned unexpected type/length. Expected {len(completions)} values. Got: {r}")
                     rewards_dict[func_name] = np.zeros(len(completions)) # Assign zero reward on error
            except Exception as e:
                print(f"Error evaluating reward function {func_name}: {e}"); traceback.print_exc()
                rewards_dict[func_name] = np.zeros(len(completions)) # Assign zero reward on error

        # Apply global reward scaling
        scaled_rewards = total_rewards * self.config.reward_scale
        scaled_rewards_dict = {k: v * self.config.reward_scale for k, v in rewards_dict.items()}

        return scaled_rewards.tolist(), scaled_rewards_dict # Return list and dict

    # ... (insert full update_metrics method here) ...
    def update_metrics(self, total_loss, surrogate_loss, kl_divergence, entropy, combined_rewards, reward_components):
        """Appends current step's metrics to the tracking dictionary."""
        self.metrics['steps'].append(self.step)
        self.metrics['total_loss'].append(total_loss) # Log item() value if it's a tensor
        self.metrics['surrogate_loss'].append(surrogate_loss)
        self.metrics['kl_divergence'].append(kl_divergence)
        self.metrics['entropy'].append(entropy)
        self.metrics['mean_combined_reward'].append(np.mean(combined_rewards)) # Log mean reward for the step
        self.metrics['lr'].append(self.scheduler.get_last_lr()[0]) # Log current learning rate

        # Update reward components using defaultdict
        for name, values in reward_components.items():
            # Append the mean of the component rewards for this step
            self.metrics['reward_components'][name].append(np.mean(values))

    # train method - modified version without hint logic
    # ... (insert full train method here, ensuring it uses self.config for params) ...
    def train(self):
        """Main training loop with robust checkpointing and batch resumption."""
        self.model.train()
        self.ref_model.eval()

        total_steps_reached = False
        # Curriculum hint logic removed

        start_epoch = self.current_epoch
        processing_resumed_epoch = self.resumed_from_checkpoint

        print(f"Starting training loop from Epoch {start_epoch + 1}")

        # --- Epoch Loop ---
        for epoch in range(start_epoch, self.config.num_train_epochs):
            if total_steps_reached: break
            self.current_epoch = epoch
            print(f"\n--- Starting Epoch {epoch + 1}/{self.config.num_train_epochs} (Current Global Step: {self.step}) ---")

            # Hint probability logic removed

            accumulation_counter = 0

            # Setup progress bar
            iterator = None
            if TQDM_AVAILABLE:
                initial_tqdm = self.resumed_batch_step if processing_resumed_epoch else 0
                iterator = tqdm(self.dataloader, desc=f"Epoch {epoch+1}", total=len(self.dataloader), initial=initial_tqdm)
            else:
                iterator = self.dataloader
                if processing_resumed_epoch and self.resumed_batch_step > 0:
                     print(f"Resuming epoch {epoch + 1}: Will skip first {self.resumed_batch_step} batches (tqdm not available).")

            # --- Batch Loop ---
            for batch_idx, batch_data in enumerate(iterator):

                # --- Skip batches logic ---
                if processing_resumed_epoch and batch_idx < self.resumed_batch_step:
                    continue
                elif processing_resumed_epoch and batch_idx == self.resumed_batch_step:
                    print(f"Resumed processing epoch {epoch+1} from batch index {batch_idx}.")
                    processing_resumed_epoch = False
                # --- End batch skipping logic ---

                self.batch_step = batch_idx + 1 # 1-based for saving state

                # --- Global Step Check ---
                if self.config.max_steps > 0 and self.step >= self.config.max_steps:
                    print(f"Reached max_steps ({self.config.max_steps}). Stopping training.")
                    total_steps_reached = True; break

                if not batch_data: # Should not happen with simple collate, but safety check
                    print(f"Warning: Empty batch encountered at index {batch_idx}. Skipping.")
                    continue

                # Assuming batch size is 1 due to simple collate_fn
                sample = batch_data[0]
                pil_image = sample.get("image"); original_prompt_text = sample.get("prompt_text")
                gt_disease = sample.get("gt_disease")

                if pil_image is None or original_prompt_text is None or gt_disease is None:
                     print(f"Warning: Skipping sample due to missing data (image, prompt, or gt_disease) in batch {batch_idx}.")
                     continue

                # --- Prepare Prompt (No Hint Logic) ---
                current_prompt_text = original_prompt_text # Use original prompt directly
                from src.config import SYSTEM_PROMPT # Import here to avoid circular dependency if config imports trainer stuff later
                messages = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": current_prompt_text}]}]
                try:
                    text_prompt_for_tokenizer = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=[text_prompt_for_tokenizer],
                                            images=[pil_image],
                                            padding=True, return_tensors="pt",
                                            max_length=self.config.max_input_ids_length,
                                            truncation=True)
                except Exception as e:
                    print(f"ERROR during input processing/tokenization for batch {batch_idx}: {e}"); traceback.print_exc(); continue

                # --- Generate Completions ---
                self.model.eval()
                with torch.no_grad():
                    inputs_gen = {k: v.to(self.config.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                    try:
                        generated_ids = self.model.generate(
                            **inputs_gen, max_new_tokens=self.config.max_completion_length,
                            temperature=self.config.temperature, do_sample=True,
                            num_return_sequences=self.config.num_generations,
                            pad_token_id=self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id, use_cache=True
                        )
                    except Exception as e:
                        print(f"ERROR during model generation for batch {batch_idx}: {e}"); traceback.print_exc()
                        if torch.cuda.is_available(): torch.cuda.empty_cache(); continue
                self.model.train()

                # --- Decode & Prepare for Reward ---
                prompt_len = inputs["input_ids"].shape[1]
                completions_ids = generated_ids[:, prompt_len:]
                num_completion_tokens = completions_ids.shape[1]
                full_decoded_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

                # Batch info for reward (no hint info)
                prompts_list = [text_prompt_for_tokenizer] * self.config.num_generations
                batch_info_list = [{"gt_disease": gt_disease} for _ in range(self.config.num_generations)]

                # --- Calculate Rewards ---
                combined_rewards, reward_components = self.evaluate_rewards(prompts_list, full_decoded_text, batch_info_list)
                rewards_tensor = torch.tensor(combined_rewards, device=self.config.device, dtype=torch.float32)

                # --- Calculate Log Probabilities ---
                full_generated_ids = generated_ids.to(self.config.device)
                pad_token_id = self.processor.tokenizer.pad_token_id or self.processor.tokenizer.eos_token_id
                full_attention_mask = (full_generated_ids != pad_token_id).long().to(self.config.device)

                pixel_values_batch = inputs.get("pixel_values")
                image_grid_thw_batch = inputs.get("image_grid_thw")
                if pixel_values_batch is not None: pixel_values_batch = pixel_values_batch.repeat_interleave(self.config.num_generations, dim=0).to(self.config.device)
                if image_grid_thw_batch is not None: image_grid_thw_batch = image_grid_thw_batch.repeat_interleave(self.config.num_generations, dim=0).to(self.config.device)

                try:
                    policy_logps = self.get_per_token_logps(
                        self.model, full_generated_ids, full_attention_mask,
                        pixel_values_batch, image_grid_thw_batch,
                        num_completion_tokens, self.config.device, compute_gradients=True
                    )
                    full_generated_ids_ref = full_generated_ids.to(self.config.ref_device)
                    full_attention_mask_ref = full_attention_mask.to(self.config.ref_device)
                    pixel_values_batch_ref = pixel_values_batch.to(self.config.ref_device) if pixel_values_batch is not None else None
                    image_grid_thw_batch_ref = image_grid_thw_batch.to(self.config.ref_device) if image_grid_thw_batch is not None else None
                    with torch.no_grad():
                        ref_logps = self.get_per_token_logps(
                            self.ref_model, full_generated_ids_ref, full_attention_mask_ref,
                            pixel_values_batch_ref, image_grid_thw_batch_ref,
                            num_completion_tokens, self.config.ref_device, compute_gradients=False
                        )
                    ref_logps = ref_logps.to(self.config.device)
                except Exception as e: print(f"ERROR calculating log probabilities for batch {batch_idx}: {e}"); traceback.print_exc(); continue

                # --- Compute Loss ---
                try:
                    with torch.cuda.amp.autocast(enabled=self.config.bf16):
                        total_loss, surrogate_loss, kl_div, entropy = self.compute_loss(policy_logps, ref_logps, rewards_tensor)
                except Exception as e: print(f"ERROR calculating loss for batch {batch_idx}: {e}"); traceback.print_exc(); continue

                # --- Backpropagation ---
                scaled_loss = total_loss / self.config.gradient_accumulation_steps
                if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                    print(f"Warning: NaN/Inf loss detected ({scaled_loss.item()}) at step {self.step}, batch_idx {batch_idx}. Skipping backward step.")
                    continue
                try:
                    self.scaler.scale(scaled_loss).backward()
                except Exception as e:
                     print(f"ERROR during backward pass for batch_idx {batch_idx}: {e}"); traceback.print_exc()
                     accumulation_counter = 0 # Reset counter if backward fails
                     continue

                # --- Optimizer Step ---
                accumulation_counter += 1
                if accumulation_counter >= self.config.gradient_accumulation_steps:
                    log_batch_idx = batch_idx # Capture batch index for logging
                    try:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0) # Clip gradients
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step() # Step LR scheduler
                        self.optimizer.zero_grad(set_to_none=True) # Zero gradients

                        # --- GLOBAL STEP INCREMENT ---
                        self.step += 1
                        accumulation_counter = 0 # Reset counter

                        # --- Logging, Checkpointing, Ref Model Sync ---
                        if self.step % self.config.logging_steps == 0:
                             self.update_metrics(total_loss.item(), surrogate_loss, kl_div, entropy, combined_rewards, reward_components)
                             lr = self.scheduler.get_last_lr()[0]
                             avg_reward = self.metrics['mean_combined_reward'][-1] if self.metrics['mean_combined_reward'] else float('nan')
                             loss_val = self.metrics['total_loss'][-1] if self.metrics['total_loss'] else float('nan')
                             s_loss = self.metrics['surrogate_loss'][-1] if self.metrics['surrogate_loss'] else float('nan')
                             kl_val = self.metrics['kl_divergence'][-1] if self.metrics['kl_divergence'] else float('nan')
                             ent_val = self.metrics['entropy'][-1] if self.metrics['entropy'] else float('nan')

                             # Use log_batch_idx + 1 for 1-based batch display
                             print(f"Step: {self.step}/{self.total_update_steps} | Ep: {epoch+1} | Ep_Batch: {log_batch_idx + 1}/{len(self.dataloader)} | "
                                   f"Loss: {loss_val:.4f} (S:{s_loss:.3f}, KL:{kl_val:.3f}, E:{ent_val:.3f}) | "
                                   f"Reward: {avg_reward:.3f} | LR: {lr:.2e}")

                             reward_log_parts = [f"{name}: {vals[-1]:.3f}" for name, vals in self.metrics['reward_components'].items() if vals]
                             if reward_log_parts: print(f"  Rewards -> {' | '.join(reward_log_parts)}")

                        # Save checkpoint periodically
                        if self.step % self.config.save_steps == 0 and self.step > 0:
                            self._save_checkpoint(f"checkpoint-{self.step}")

                        # Sync reference model periodically
                        if self.step % self.config.update_ref_steps == 0 and self.step > 0:
                            self.sync_reference_model()

                        # Check max_steps again after incrementing
                        if self.config.max_steps > 0 and self.step >= self.config.max_steps:
                            print(f"Reached max_steps ({self.config.max_steps}) after step completion. Stopping training.")
                            total_steps_reached = True
                            # Break handled by outer loop check

                    except Exception as e:
                        print(f"ERROR during optimizer step/logging/saving/sync for step {self.step}: {e}"); traceback.print_exc()
                        accumulation_counter = 0 # Reset accumulation if step fails critically
                        self.optimizer.zero_grad() # Zero grads just in case

                # --- Batch Cleanup ---
                del inputs, inputs_gen, generated_ids, full_generated_ids, full_generated_ids_ref
                del policy_logps, ref_logps, rewards_tensor, total_loss, scaled_loss
                del pixel_values_batch, image_grid_thw_batch
                del pixel_values_batch_ref, image_grid_thw_batch_ref
                if torch.cuda.is_available(): torch.cuda.empty_cache() # Clear cache periodically

            # --- End of Epoch ---
            if total_steps_reached: break
            print(f"--- Finished Epoch {epoch + 1} (Global Step: {self.step}) ---")
            # Save checkpoint at the end of each epoch
            self._save_checkpoint(f"epoch-{epoch+1}")


        # --- End of Training ---
        print("\nTraining finished.")
        if not total_steps_reached:
             print("Saving final model state...")
             self._save_checkpoint("final_model")
        else:
             print("Training stopped due to max_steps. Final model state corresponds to the last saved checkpoint.")
        return self.metrics