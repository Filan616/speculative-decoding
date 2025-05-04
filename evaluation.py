import os
import torch
import matplotlib.pyplot as plt
import time
import csv
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from speculative_generate import SpeculativeSampling

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warning for tokenizers

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

# Load HumanEval dataset
print("Loading HumanEval...")
ds = load_dataset("openai_humaneval")
test_set = ds["test"]
print(f"Loaded {len(test_set)} tasks")

# Initialize SpeculativeSampling
sampler = SpeculativeSampling(
    target_model_name="Qwen/Qwen2.5-1.5B",
    draft_model_name="Qwen/Qwen2.5-0.5B",
    device_map="auto",
    temperature=0.8
)

# Classic generation function
@torch.no_grad()
def classic_generate(prompt, max_new_tokens=128, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), output[0].shape[-1] - inputs["input_ids"].shape[-1]

# Initialize result containers
classic_results = {"pass@1": [], "pass@5": [], "pass@10": [], "latency": [], "tps": []}
speculative_results = {"pass@1": [], "pass@5": [], "pass@10": [], "latency": [], "tps": []}

print("Running experiment...")
num_runs = 1

for run in range(1, num_runs + 1):
    print(f"\n--- Run {run} ---")
    classic_predictions = []
    speculative_predictions = []
    references = []

    classic_total_time = 0
    classic_total_tokens = 0
    speculative_total_time = 0
    speculative_total_tokens = 0

    for task in test_set:
        prompt = task["prompt"]
        unit_test = task["test"]

        # Classic sampling
        classic_samples = []
        for _ in range(10):
            start = time.time()
            code, tokens = classic_generate(prompt)
            end = time.time()
            classic_total_time += (end - start)
            classic_total_tokens += tokens
            classic_samples.append(prompt + code)
        classic_predictions.append(classic_samples)

        # Speculative sampling
        speculative_samples = []
        for _ in range(10):
            start = time.time()
            code = sampler.generate(prompt, max_length=512)
            end = time.time()
            tokens = len(tokenizer(code)["input_ids"])
            speculative_total_time += (end - start)
            speculative_total_tokens += tokens
            speculative_samples.append(prompt + code)
        speculative_predictions.append(speculative_samples)

        references.append(unit_test)

    # Evaluate predictions using code_eval
    code_eval = load("code_eval")

    pass_at_k_classic, _ = code_eval.compute(
        references=references,
        predictions=classic_predictions,
        k=[1, 5, 10]
    )

    pass_at_k_speculative, _ = code_eval.compute(
        references=references,
        predictions=speculative_predictions,
        k=[1, 5, 10]
    )

    for k in [1, 5, 10]:
        classic_results[f"pass@{k}"].append(pass_at_k_classic[f"pass@{k}"])
        speculative_results[f"pass@{k}"].append(pass_at_k_speculative[f"pass@{k}"])

    classic_latency = classic_total_time / (len(test_set) * 10)
    classic_tps = classic_total_tokens / classic_total_time
    speculative_latency = speculative_total_time / (len(test_set) * 10)
    speculative_tps = speculative_total_tokens / speculative_total_time

    classic_results["latency"].append(classic_latency)
    classic_results["tps"].append(classic_tps)
    speculative_results["latency"].append(speculative_latency)
    speculative_results["tps"].append(speculative_tps)

    # Print metrics
    print("Classic Sampling:")
    for k in [1, 5, 10]:
        print(f"  pass@{k}: {pass_at_k_classic[f'pass@{k}']:.3f}")
    print(f"  latency: {classic_latency:.3f}s/token, tokens/sec: {classic_tps:.2f}")

    print("Speculative Sampling:")
    for k in [1, 5, 10]:
        print(f"  pass@{k}: {pass_at_k_speculative[f'pass@{k}']:.3f}")
    print(f"  latency: {speculative_latency:.3f}s/token, tokens/sec: {speculative_tps:.2f}")

# Save results to CSV
with open("results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Method", "pass@1", "pass@5", "pass@10", "Latency (s/token)", "Tokens/sec"])
    writer.writerow([
        "Classic",
        classic_results["pass@1"][0],
        classic_results["pass@5"][0],
        classic_results["pass@10"][0],
        classic_results["latency"][0],
        classic_results["tps"][0]
    ])
    writer.writerow([
        "Speculative",
        speculative_results["pass@1"][0],
        speculative_results["pass@5"][0],
        speculative_results["pass@10"][0],
        speculative_results["latency"][0],
        speculative_results["tps"][0]
    ])

# Plot efficiency comparison (latency and throughput)
labels = ["Latency (s/token)", "Tokens/sec"]
classic_vals = [classic_results["latency"][0], classic_results["tps"][0]]
spec_vals = [speculative_results["latency"][0], speculative_results["tps"][0]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, classic_vals, width, label="Classic")
rects2 = ax.bar(x + width/2, spec_vals, width, label="Speculative")

ax.set_ylabel("Value")
ax.set_title("Latency and Throughput Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for rect in rects1 + rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig("efficiency_bar_chart.png")
plt.close()
