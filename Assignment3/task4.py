import argparse
from utils import set_seed, load_jsonl, save_jsonl, ensure_dir
from generate_task4_bonus import load_model, load_counts_and_reward, beam_search_for_prompt


def parse_args():
    p = argparse.ArgumentParser(
        description="Task 4 (Bonus): Beam Search for P_tharoor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run Bonus Beam Search with 8 beams on 10 prompts
    python task4.py --hf-token <token> \\
                    --counts-dir /path/to/counts --A 10 --B 8
        """
    )
    
    model_group = p.add_argument_group("Model Configuration")
    model_group.add_argument("--hf-token", type=str, required=True)
    model_group.add_argument("--device", type=str, default="cuda:0")

    reward_group = p.add_argument_group("Reward Function Parameters")
    reward_group.add_argument("--counts-dir", type=str, required=True)
    reward_group.add_argument("--epsilon", type=float, default=1e-9)

    smc_group = p.add_argument_group("Beam Search Algorithm Parameters")
    smc_group.add_argument("--beta", type=float, default=5.0)

    proposal_group = p.add_argument_group("Proposal Distribution (Beam Expansion)")
    proposal_group.add_argument("--k", type=int, default=10)

    exec_group = p.add_argument_group("Execution Configuration")
    exec_group.add_argument("--test-file", type=str, default="data/test_prompts.jsonl")
    exec_group.add_argument("--A", type=int, required=True)
    exec_group.add_argument("--B", type=int, required=True)
    exec_group.add_argument("--seed", type=int, default=123)

    output_group = p.add_argument_group("Output Configuration")
    output_group.add_argument("--out", type=str, default="data/outputs_task4_Bonus.jsonl")
    return p.parse_args()


def _method_tag(args, N: int) -> str:
    return f"Bonus[beam={N}; k_expand={args.k}; beta={args.beta}]"


def main():
    
    args = parse_args()
    set_seed(args.seed)
    ensure_dir("data")

    rows = load_jsonl(args.test_file)[: args.A]
    print(f"[Task 4] Loaded {len(rows)} prompts from {args.test_file}")

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tok, model, eos_id = load_model(model_name, args.hf_token, args.device)
    print(f"[Task 4] Loaded model: {model_name}")
    
    reward_calc = load_counts_and_reward(args.counts_dir, epsilon=args.epsilon)
    print(f"[Task 4] Loaded reward function from: {args.counts_dir}")

    out_rows = []
    print(f"[Task 4] Starting Beam Search with N={args.B} beams per prompt")
    
    for row in rows:
        prompt_id = row["prompt_id"]
        prefix = row["prefix"]
        instruction = row.get("instruction", "Continue the text.")
        
        full_prompt = f"{instruction} {prefix}"
        max_new = int(row.get("max_output_tokens", 50))
        
        print(f"[Task 4] Processing prompt {prompt_id}: '{prefix[:50]}{'...' if len(prefix) > 50 else ''}'")

        result = beam_search_for_prompt(
            tokenizer=tok,
            model=model,
            reward_calc=reward_calc,
            prefix=full_prompt,
            N=args.B,
            max_new_tokens=max_new,
            eos_id=eos_id,
            beta=args.beta,
            k=args.k,
        )

        out_rows.append({
            "prompt_id": prompt_id,
            "prefix": prefix,
            "continuations": [
                {
                    "method": _method_tag(args, args.B),
                    "samples": result["samples"],
                    "normalized_weights": result["normalized_weights"],
                }
            ]
        })
        print(f"[Task 4] Completed prompt {prompt_id}")

    save_jsonl(args.out, out_rows)
    
    total_sequences = len(out_rows) * args.B
    print(f"\n[Task 4 Complete] Successfully processed {len(out_rows)} prompts")
    print(f"[Task 4 Complete] Generated {total_sequences} total sequences")
    print(f"[Task 4 Complete] Results saved to: {args.out}")
    
    print(f"\n[Beam Search Configuration]")
    print(f"  - Beams per prompt (N): {args.B}")
    print(f"  - Top-k expansion (k): {args.k}")
    print(f"  - Reward scaling (β): {args.beta}")

    print(f"  - Model: {model_name}")
    print(f"  - Reward function: {args.counts_dir}")


if __name__ == "__main__":
    main()