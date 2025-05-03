import json
import os
import random
from collections import defaultdict
from PIL import Image
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .utils import resize_and_pad_image, extract_disease # Relative imports
from .config import DATASET_PATH, IMAGE_DIR, SAMPLES_PER_DISEASE, VALID_DISEASES, RANDOM_SEED

def load_and_preprocess_data():
    """
    Loads the dataset JSON, preprocesses samples (image resizing, text extraction),
    performs stratified sampling, and returns the final list of training samples.
    """
    print("Loading and preprocessing dataset...")
    random.seed(RANDOM_SEED)

    # --- Load Raw Data ---
    try:
        with open(DATASET_PATH, "r") as f:
            raw_dataset = json.load(f)
    except FileNotFoundError:
        print(f"FATAL: Dataset file not found at {DATASET_PATH}")
        raise # Re-raise error to stop execution
    except json.JSONDecodeError:
        print(f"FATAL: Error decoding JSON from {DATASET_PATH}")
        raise # Re-raise error
    except Exception as e:
        print(f"FATAL: An unexpected error occurred loading dataset: {e}")
        raise

    print(f"Loaded {len(raw_dataset)} raw samples.")

    # --- Preprocess Samples ---
    processed_samples = []
    skipped_images = 0
    skipped_samples = 0

    iterator = tqdm(raw_dataset, desc="Preprocessing samples") if TQDM_AVAILABLE else raw_dataset

    for sample in iterator:
        image_filename = sample.get("image")
        conversations = sample.get("conversations", [])

        if not image_filename or len(conversations) < 2:
            skipped_samples += 1
            continue

        image_path = os.path.join(IMAGE_DIR, image_filename)
        if not os.path.isfile(image_path):
            # print(f"Warning: Image file not found: {image_path}") # Can be noisy
            skipped_images += 1
            continue

        # Load and resize image
        resized_img_vlm = resize_and_pad_image(image_path, size=(336, 336))
        if resized_img_vlm is None: # Skip if image loading/resizing failed
            skipped_images += 1
            continue

        # Extract text
        human_question = None
        assistant_response = None
        if conversations[0].get("from", "").lower() == "human":
            human_question = conversations[0].get("value", "").replace("<image>\n", "").strip()
        if conversations[1].get("from", "").lower() in ["gpt", "assistant"]:
            assistant_response = conversations[1].get("value", "").strip()

        if human_question and assistant_response:
            gt_disease_list = extract_disease(assistant_response)
            gt_disease = gt_disease_list[0] if gt_disease_list else "unknown"

            processed_samples.append({
                "image": resized_img_vlm,
                "image_path": image_path, # Keep for reference if needed
                "prompt_text": human_question,
                "gt_answer": assistant_response,
                "gt_disease": gt_disease,
            })
        else:
             skipped_samples += 1

    print(f"Preprocessing complete. Processed {len(processed_samples)} samples.")
    if skipped_samples > 0:
        print(f"  Skipped {skipped_samples} samples due to missing conversation data.")
    if skipped_images > 0:
        print(f"  Skipped {skipped_images} samples due to missing/invalid image files.")

    # --- Stratified Sampling ---
    print(f"\nPerforming stratified sampling (max {SAMPLES_PER_DISEASE} per class)...")
    disease_to_samples = defaultdict(list)
    unknown_samples = []

    for sample in processed_samples:
        disease = sample["gt_disease"]
        if disease != "unknown" and disease in VALID_DISEASES:
            disease_to_samples[disease].append(sample)
        else:
            unknown_samples.append(sample)

    sampled_dataset = []
    for disease, samples in disease_to_samples.items():
        count = min(len(samples), SAMPLES_PER_DISEASE)
        sampled_dataset.extend(random.sample(samples, count))
        if count < SAMPLES_PER_DISEASE:
            print(f"  Warning: Only found {count} samples for '{disease}', using all.")
        # else:
        #     print(f"  Sampled {count} for '{disease}'.")


    # Optional: Include some 'unknown' samples
    # unknown_sample_count = min(len(unknown_samples), SAMPLES_PER_DISEASE // 4)
    # if unknown_sample_count > 0:
    #     sampled_dataset.extend(random.sample(unknown_samples, unknown_sample_count))
    #     print(f"  Included {unknown_sample_count} samples with 'unknown' or non-valid GT disease.")

    random.shuffle(sampled_dataset)
    print(f"Final dataset size after sampling: {len(sampled_dataset)}")

    # --- Cleanup ---
    del raw_dataset
    del processed_samples
    del disease_to_samples
    del unknown_samples

    return sampled_dataset

# Example of how to potentially create a torch Dataset object
# from torch.utils.data import Dataset
# class DermNetDataset(Dataset):
#     def __init__(self):
#         self.data = load_and_preprocess_data()
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]