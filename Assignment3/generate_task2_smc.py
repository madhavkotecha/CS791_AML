# generate_task2_smc.py
"""Task 2 — Sequential Monte Carlo (SMC) helpers.
"""
from __future__ import annotations
import os
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from api import FastRewardCalculator

def load_counts_and_reward(counts_dir: str, epsilon: float = 1e-9) -> FastRewardCalculator:
    """Initialize trigram-based reward calculator for Sequential Importance Sampling.
    
    Args:
        counts_dir: Directory path containing ngrams data with trigram_probs.pkl cache
        epsilon: Smoothing parameter - minimum probability for unseen trigrams (prevents log(0))
        
    Returns:
        FastRewardCalculator: Configured calculator for computing R(x) rewards
    """
    cache_file = os.path.join(counts_dir, "trigram_probs.pkl")
    return FastRewardCalculator(cache_file, epsilon=epsilon)


def load_model(model_name: str, hf_token: str, device: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, int]:
    """Load and configure Hugging Face model components for Sequential Importance Sampling.
    
    Args:
        model_name: Hugging Face model repository ID (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
        hf_token: Authentication token for accessing gated models
        device: Target device for model placement ("cuda:0", "cpu", etc.)
        
    Returns:
        Tuple containing:
            - tokenizer: Configured AutoTokenizer with proper padding token
            - model: AutoModelForCausalLM in evaluation mode on target device
            - eos_id: End-of-sequence token ID for generation termination
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name,token=hf_token).to(device)
    model.eval()
    return tokenizer, model, tokenizer.eos_token_id

    raise NotImplementedError("Students must implement this function.")

def cal_intermediate_target_dist(reward_calc: FastRewardCalculator, tokenizer, beta: float, full_ids: List[int]) -> float:
    """
    Args:
        reward_calc: FastRewardCalculator (token_lm.logp available).
        tokenizer: for ids→tokens conversion.
        full_ids: current full context ids (prompt + generated so far).

    Returns:
        float ΔR_t ≥ 0.
    """ 

    tokens = tokenizer.convert_ids_to_tokens(full_ids)
    return  reward_calc.calculate_reward_tokens(tokens, normalize=True)


    raise NotImplementedError("Students must implement this function.")

@torch.no_grad()
def smc_for_prompt(
    tokenizer: Any,
    model: Any,
    reward_calc: Any,
    *,
    prefix: str,
    N: int,
    max_new_tokens: int,
    eos_id: int,
    beta: float,
    k: int,
) -> Dict:
    """
    Inputs:
      tokenizer, model: HF components from load_model.
      reward_calc: FastRewardCalculator.
      prefix: full prompt string fed to the model (instruction + space + prefix).
      N: number of particles.
      max_new_tokens: continuation budget.
      eos_id: stopping id.
      beta: reward scale.
      k: top-k for proposal.

    Outputs:
      {
        "samples": [ {"text": str, "weight": float}, ... ],   
        "normalized_weights": [float, ...]
      }
    """
    device = model.device

    enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"].squeeze(0).to(device)  

    particles, prev_rewards, is_particle_done = [], [], []
    for i in range(N):
        particles.append([])
        # prev reward - at (t-1) for each particle
        prev_rewards.append(0.0)
        # track for particles generating EOS
        is_particle_done.append(False)

    weights = np.full(N, 1.0 / N, dtype=float)
    final_weights = np.zeros(N, dtype=float)

    for t in range(1, max_new_tokens + 1):
        p_llama_values, q_proposal_values, proposals, curr_rewards  = [], [], [], []

        for i in range(N):
            if is_particle_done[i]:
                proposals.append(eos_id)
                p_llama_values.append(1.0)
                q_proposal_values.append(1.0)
                curr_rewards.append(prev_rewards [i])
                continue

            if len(particles[i]) == 0:
                input_ids = prompt_ids.unsqueeze(0)
            else:
                cur = torch.tensor(particles[i], device=device, dtype=prompt_ids.dtype).unsqueeze(0)
                input_ids = torch.cat([prompt_ids.unsqueeze(0), cur], dim=1)

            # logits for next token
            with torch.no_grad():
                logits = model(input_ids).logits[:, -1, :]  # [1, V]
            full_vocab_probs = torch.softmax(logits, dim=-1).squeeze(0)  # [V]

            # top-k proposal ==> this is our Q : get top-k and then normalize
            topk_vals, topk_idx = torch.topk(full_vocab_probs, k)
            normalized_topk_probs = topk_vals / topk_vals.sum()

            sampled_xt_k = torch.multinomial(normalized_topk_probs, num_samples=1).item()
            seleted_token = int(topk_idx[sampled_xt_k].item())

            # adding P_llama and q_proposal values to list ????
            proposals.append(seleted_token)
            p_llama_values.append(float(full_vocab_probs[seleted_token].item()))
            q_proposal_values.append(float(normalized_topk_probs[sampled_xt_k].item()))

            full_ids = prompt_ids.tolist() + particles[i] + [seleted_token]
            # this reward needs complete sequence till now  
            curr_rewards.append(float(cal_intermediate_target_dist(reward_calc=reward_calc, tokenizer=tokenizer, beta=beta, full_ids=full_ids)))

            if eos_id is not None and seleted_token == eos_id:
                is_particle_done[i] = True

        # incremental weights w_t^{i} using formula (explained in report)
        incremental_weights = np.zeros(N, dtype=float)
        for i in range(N):
            # inc_wt is 1 as particle generated EOS earlier
            if is_particle_done[i] and (proposals[i] == eos_id) and (len(particles[i]) > 0 and particles[i][-1] == eos_id):
                incremental_weights[i] = 1.0
                continue

            deltaR = curr_rewards [i] - prev_rewards [i]
            
            # # division could become zero => incremenatl wt become zero 
            # if q_proposal_values[i] <= 0: q_proposal_values[i] = 1e-14
            # if p_llama_values[i] <= 0: p_llama_values[i] = 1e-14

            # TODO: can division become zero??
            incremental_weights[i] = math.exp(beta * deltaR) * (p_llama_values[i] / q_proposal_values[i])
        
        unnormalized = np.multiply(weights, incremental_weights)
        sum_w = np.sum(unnormalized)
        normalized = unnormalized / sum_w if sum_w > 0 else np.full_like(unnormalized, 1.0 / N)

        # RESAMPLE for t < max_new_tokens
        if t < max_new_tokens:
            random_indexes = np.random.choice(np.arange(N), size=N, replace=True, p=normalized)
            prev_rewards  = [curr_rewards[j] for j in random_indexes] 
            is_particle_done = [is_particle_done[j] for j in random_indexes]
            weights = np.ones(N, dtype=float) / N
            new_particles = []
            for j in random_indexes:
                seq = list(particles[j]) 
                # new chosen token of  particle j - proposal[j] - append to seq of particle[j] 
                seq.append(proposals[j])
                new_particles.append(seq)
            particles = new_particles
            
        else:
            # last step: record final weights
            final_weights = unnormalized
            for i in range(N):
                particles[i].append(proposals[i])
                prev_rewards [i] = curr_rewards[i]
            weights = normalized  

        if all(is_particle_done):
            break

    norm_final = weights / weights.sum() if final_weights.sum() <= 0.0 else final_weights / final_weights.sum()

    results  = []
    for i in range(N):
        text = tokenizer.decode(particles[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append({
            "text": text, 
            "weight": norm_final[i]
            }
        )
    return {"samples": results, "normalized_weights":  norm_final.tolist()}

    raise NotImplementedError("Students must implement this function.")
