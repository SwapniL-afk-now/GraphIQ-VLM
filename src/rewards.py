import numpy as np
from .utils import extract_disease, extract_xml_tag # Relative imports
from .config import (
    VALID_DISEASES, REWARD_CORRECT, REWARD_INCORRECT,
    REWARD_ADHERENCE, REWARD_PENALTY, FORMAT_SCALE_FACTOR
)

def accuracy_reward(prompts, completions, batch_info_list):
    """
    Assigns reward based on matching ground truth disease name.
    Uses REWARD_CORRECT and REWARD_INCORRECT from config.
    """
    rewards = []
    for i, completion in enumerate(completions):
        vlm_predicted_diseases = extract_disease(completion)
        vlm_pred = vlm_predicted_diseases[0] if vlm_predicted_diseases else None
        gt_disease = batch_info_list[i].get("gt_disease")
        current_reward = REWARD_INCORRECT

        if vlm_pred is not None and gt_disease != "unknown" and gt_disease in VALID_DISEASES:
            if vlm_pred == gt_disease:
                current_reward = REWARD_CORRECT
        rewards.append(current_reward)
    return rewards

def format_adherence_reward(prompts, completions, batch_info_list):
    """
    Assigns reward for adherence to XML format and required sections.
    Uses REWARD_ADHERENCE and REWARD_PENALTY from config.
    """
    required_sections = ["Color and Pigmentation", "Texture and Surface Characteristics", "Shape and Border"]
    rewards = []
    for completion in completions:
        current_reward = 0.0
        thinking_content = extract_xml_tag(completion, "thinking")
        answer_content = extract_xml_tag(completion, "answer")

        has_thinking = bool(thinking_content)
        has_answer = bool(answer_content)
        is_well_formed = "<thinking>" in completion and "</thinking>" in completion and \
                         "<answer>" in completion and "</answer>" in completion and \
                         completion.strip().startswith("<thinking>") and completion.strip().endswith("</answer>")

        if is_well_formed and has_thinking and has_answer:
            current_reward += REWARD_ADHERENCE * 0.5
            try:
                thinking_end_idx = completion.rindex("</thinking>")
                answer_start_idx = completion.index("<answer>")
                if thinking_end_idx < answer_start_idx:
                    between_content = completion[thinking_end_idx + len("</thinking>"):answer_start_idx].strip()
                    if not between_content:
                        current_reward += REWARD_ADHERENCE * 0.2
            except ValueError:
                pass
            thinking_content_lower = thinking_content.lower()
            sections_found = sum(1 for section in required_sections if section.lower() in thinking_content_lower)
            current_reward += (sections_found / len(required_sections)) * REWARD_ADHERENCE * 0.3
        else:
            current_reward = REWARD_PENALTY

        rewards.append(min(current_reward, REWARD_ADHERENCE))
    return rewards

def scaled_format_adherence_reward(prompts, completions, batch_info_list):
    """
    Applies scaling (FORMAT_SCALE_FACTOR from config) to the format adherence reward.
    """
    original_rewards = format_adherence_reward(prompts, completions, batch_info_list)
    return [r * FORMAT_SCALE_FACTOR for r in original_rewards]

# --- Reward Function Selection ---
# Define the list of reward functions to be used by the trainer
# This makes it easy to change the active rewards by modifying this list
ACTIVE_REWARD_FUNCTIONS = [
    accuracy_reward,
    # scaled_format_adherence_reward # Uncomment to include format reward
]