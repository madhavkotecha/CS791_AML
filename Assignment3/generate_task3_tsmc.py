# generate_task3_tsmc_fixed.py
"""Twisted Sequential Monte Carlo (TSMC) implementation - simplified for assignment."""
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
    # same as task 2
    tokens = tokenizer.convert_ids_to_tokens(full_ids)
    return reward_calc.calculate_reward_tokens(tokens, normalize=True)
    
    raise NotImplementedError("Students must implement this function.")

def cal_twist_function(reward_calc: FastRewardCalculator, tokenizer, beta: float, seq_ids: List[int]) -> float:
    """
    Inputs:
        reward_calc: FastRewardCalculator with token_lm access.
        tokenizer: to convert ids→tokens.
        seq_ids: current full context ids (prompt + generated).

    Returns:
        Expected positive delta (float) ≥ 0.
        
    Note:
        you are allowed to define additional helper functions if needed in FastRewardCalculator class for calculation of expectation.
    """
    
    tokens = tokenizer.convert_ids_to_tokens(seq_ids)
    last_token = tokens[-1]
    T = len(tokens)
    triplet_prob_dist_map = reward_calc._tri_probs
    epsilon = reward_calc._eps


    # calculation if
    bigram_dist = reward_calc._bi_probs.get(last_token, None)
    
    if not bigram_dist:
        return 1.0
    

    # definition from report
    exp_reward = -sum(p * math.log(max(p, epsilon)) for p in bigram_dist.values())
    phi_t = math.exp(beta * exp_reward)
    if not math.isfinite(phi_t) or phi_t <= 0:
        return 1.0
    return phi_t

    raise NotImplementedError("Students must implement this function.")

@torch.no_grad()
def tsmc_for_prompt(
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
    """Run TSMC for a single prompt.

    Inputs:
      tokenizer, model: HF components from load_model.
      reward_calc: FastRewardCalculator.
      prefix: full prompt string fed to the model (instruction + space + prefix).
      N: number of particles.
      max_new_tokens: continuation budget.
      eos_id: stopping id.
      beta: reward scale.
      k: top-k for base proposal.

    Output dict (minimal for eval):
      {
        "samples": [ {"text": str, "weight": float}, ... ],   # length N
        "normalized_weights": [float, ...]                    # softmax over final log_w
      }
    """
    device = model.device
    enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    
    # from task2
    prompt_ids = enc["input_ids"].squeeze(0).to(device)  

    weights = np.full(N, 1.0 / N, dtype=float)
    final_weights = np.zeros(N, dtype=float)
    particles, prev_rewards, is_particle_done = [], [], []
    for i in range(N):
        particles.append([])
        # prev reward - at (t-1) for each particle
        prev_rewards.append(0.0)
        # track for particles generating EOS
        is_particle_done.append(False)

    # new: track phi_{t-1}
    phi_prev =  [1.0 for _ in range(N)]

    for t in range(1, max_new_tokens + 1):
        proposals, p_llama_vals, q_proposal_vals = [], [], []
        curr_rewards, phi_t_list = [], []

        for i in range(N):
            if is_particle_done[i]:
                proposals.append(eos_id)
                p_llama_vals.append(1.0)
                q_proposal_vals.append(1.0)
                curr_rewards.append(prev_rewards[i])
                phi_t_list.append(phi_prev[i])
                continue
            
            if len(particles[i]) == 0:
                input_ids = prompt_ids.unsqueeze(0)
            else:
                cur = torch.tensor(particles[i], device=device).unsqueeze(0)
                input_ids = torch.cat([prompt_ids.unsqueeze(0), cur], dim=1)
           
            with torch.no_grad():
                logits = model(input_ids).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1).squeeze(0)

            # sampled from Q -> same as Task2
            topk_vals, topk_idx = torch.topk(probs, k)
            q_topk = topk_vals / topk_vals.sum()
            sampled_xt_k = torch.multinomial(q_topk, 1).item()
            seleted_token = int(topk_idx[sampled_xt_k].item())

            p_llama = probs[seleted_token].item()
            q_prop = q_topk[sampled_xt_k].item()
            
            proposals.append(seleted_token)
            p_llama_vals.append(float(p_llama))
            q_proposal_vals.append(float(q_prop))
            

            full_ids = prompt_ids.tolist() + particles[i] + [seleted_token]
            # this reward needs complete sequence till now  
            curr_rewards.append(float(cal_intermediate_target_dist(reward_calc, tokenizer, beta, full_ids)))

            ## -- NEW LOGIC -- ##
            # calculate phi_t
            phi_t = cal_twist_function(reward_calc, tokenizer, beta, full_ids)
            phi_t_list.append(phi_t)

            if eos_id is not None and seleted_token == eos_id:
                is_particle_done[i] = True

        # incremental weights w_t^{i} 
        incremental_weights = np.zeros(N, dtype=float)
        for i in range(N):
            
            # old
            deltaR = curr_rewards[i] - prev_rewards[i]
            

            # TODO: can division become zero??
            # # new term: (phi_t / phi_previous)
            incremental_weights[i] = math.exp(beta * deltaR) * (p_llama_vals[i] / q_proposal_vals[i]) * (phi_t_list[i] / phi_prev[i])


        unnormalized = np.multiply(weights, incremental_weights)
        normalized = unnormalized / np.sum(unnormalized) if np.sum(unnormalized) > 0 else np.full_like(unnormalized, 1.0 / N)


        # RESAMPLE
        if t < max_new_tokens:
            random_indexes = np.random.choice(np.arange(N), size=N, replace=True, p=normalized)
            is_particle_done = [is_particle_done[j] for j in random_indexes]
            new_particles = []     
            for j in random_indexes:
                # new chosen token of  particle j - proposal[j] - append to seq of particle[j] 
                seq = list(particles[j]) 
                seq.append(proposals[j])
                new_particles.append(seq)
            #NEW
            phi_prev = [phi_t_list[j] for j in random_indexes]
            
            particles = new_particles
            prev_rewards  = [curr_rewards[j] for j in random_indexes] 
            weights = np.ones(N, dtype=float) / N
           
        else:
            # last step: record final weights
            final_weights = unnormalized
            for i in range(N):
                particles[i].append(proposals[i])
                prev_rewards[i] = curr_rewards[i]
                #NEW
                phi_prev[i] = phi_t_list[i]
            weights = normalized

        if all(is_particle_done):
            break

    norm_final = weights / weights.sum() if final_weights.sum() <= 0.0 else final_weights / final_weights.sum()
    results = []
    for i in range(N):
        text = tokenizer.decode(particles[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results.append({"text": text, "weight": float(norm_final[i])})

    return {"samples": results, "normalized_weights": norm_final.tolist()}


    
    
    raise NotImplementedError("Students must implement this function.")
