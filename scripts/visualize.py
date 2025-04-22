import re
import matplotlib.pyplot as plt
import pandas as pd

# === CONFIGURE YOUR LOG FILE PATHS HERE ===
log_paths = {
    "Postnorm": "../models/deen_transformer_post/train.log",
    "Prenorm": "../models/deen_transformer_pre/train.log",
    "Baseline": "../models/baseline.log"  # Adjust path as needed
}

# === Regex patterns for step and ppl ===
step_pattern = re.compile(r"Step:\s+(\d+)")
ppl_pattern = re.compile(r"ppl:\s+([\d.]+)")

# === Store results per model ===
ppl_data = {}

for model_name, path in log_paths.items():
    steps = []
    ppls = []
    current_step = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                step_match = step_pattern.search(line)
                if step_match:
                    current_step = int(step_match.group(1))
                ppl_match = ppl_pattern.search(line)
                if ppl_match and current_step is not None:
                    ppl = float(ppl_match.group(1))
                    steps.append(current_step)
                    ppls.append(ppl)
                    current_step = None  # reset until next step is found
    except FileNotFoundError:
        print(f" File not found: {path}")
    ppl_data[model_name] = pd.Series(data=ppls, index=steps)

# === Combine into one DataFrame ===
df = pd.DataFrame(ppl_data)
df.index.name = "Validation Step"
print("\n Perplexity Table:\n")
print(df)

# === Plotting ===
plt.figure(figsize=(10, 6))
for model_name in df.columns:
    plt.plot(df.index, df[model_name], label=model_name, marker="o")

plt.title("Validation Perplexity over Steps")
plt.xlabel("Training Step")
plt.ylabel("Perplexity")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save to file ===
output_path = "validation_perplexity.png"
plt.savefig(output_path, dpi=300)
print(f"\n Plot saved to: {output_path}")
