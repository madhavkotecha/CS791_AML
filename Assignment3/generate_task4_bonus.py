# generate_task4_bonus.py

from __future__ import annotations
import os
import math
import torch
import heapq
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

@torch.no_grad()
def beam_search_for_prompt(
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
    device = model.device

    prompt_ids = tokenizer(prefix, return_tensors="pt").to(device).input_ids
    beam = [(0.0, [])]
    final_seqn = []

    for t in range(max_new_tokens):
        candidates = []
        for curr_score, current_ids in beam:
            if current_ids and current_ids[-1] == eos_id:
                final_seqn.append((curr_score, current_ids))
                continue

            # current_ids_cat = torch.cat([prompt_ids, torch.tensor([current_ids], device=device)], dim=-1)
            # log_probs = torch.softmax(model(current_ids_cat))


            current_ids_cat = torch.cat([prompt_ids, torch.tensor([current_ids], dtype=torch.long, device=device)], dim=-1)
            outputs = model(current_ids_cat)
            logits = outputs.logits[:, -1, :]
            
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k, dim=-1)

            for i in range(k):
                updated_cont_ids = current_ids + [topk_indices[0, i].item()]
                R_t = cal_intermediate_target_dist(reward_calc, tokenizer, beta, updated_cont_ids)
                D_score = topk_log_probs[0, i].item() + (beta * R_t)

                candidates.append((curr_score + D_score, updated_cont_ids))
        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:N]           #pruning

        if all(b[1][-1] == eos_id for b in beam):
            break

    final_seqn.extend(beam)
    final_seqn.sort(key=lambda x: x[0], reverse=True)

    seen_seqs = set()
    topn_seqs = []
    for score, ids in final_seqn:
        if tuple(ids) not in seen_seqs:
            topn_seqs.append(ids)
            seen_seqs.add(tuple(ids))
        if len(topn_seqs) >= N:
            break

    final_weight = 1.0 / N
    final_norm_weights = [1.0/N] * N
    
    samples_out = [{"text": tokenizer.decode(ids, skip_special_tokens=True), "weight": final_weight} for ids in topn_seqs]

    while len(samples_out) < N and samples_out:
        samples_out.append(samples_out[0]) # duplicate best
    
    if not samples_out:
        samples_out = [{"text": "", "weight": final_weight}] * N

    return {"samples": samples_out, "normalized_weights": final_norm_weights}
