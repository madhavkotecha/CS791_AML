## Students need to implement this file.
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot histogram of normalized weights from JSONL output.")
    parser.add_argument("--in_file", type=str, required=True,)
    parser.add_argument("--out_file", type=str, required=True, help="write the whole path including .png extension.")
    args = parser.parse_args()

    all_weights = []

    with open(args.in_file) as f:
        for line in f:
            record = json.loads(line)
            for block in record.get("continuations", []):
                ws = block.get("normalized_weights", [])
                all_weights.extend(ws)

    all_weights = np.array(all_weights)
    print(f"Total weights: {len(all_weights)}")

    plt.figure(figsize=(7, 5))
    plt.hist(all_weights, bins=np.linspace(0, 1, 11))
    plt.xlabel("normalized_weights")
    plt.ylabel("counts")
    plt.title("Histogram of normalized weights")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(args.out_file)
    plt.close()

if __name__ == "__main__":
    main()


'''
python plot_histogram.py --in_file out_task2/outputs_task2_SMC_A20_B8_b0p5.jsonl \
                     --out_file plot_task2/task12.png

Note: please make sure the folder is already there for saving the plots
'''