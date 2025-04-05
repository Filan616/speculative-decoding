import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from speculative_generate import SpeculativeSampling

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading HumanEval...")
ds = load_dataset("openai/openai_humaneval")
test_set = ds["test"]
print(f"Loaded {len(test_set)} tasks")

sampler = SpeculativeSampling(
    target_model_name="Qwen/Qwen2.5-1.5B",
    draft_model_name="Qwen/Qwen2.5-0.5B",
    device=device,
    torch_dtype=torch.float16
)

print("Generating samples...")
predictions = []
references = []

for task in test_set:
    prompt = task["prompt"]
    unit_test = task["test"]
    samples = []
    for _ in range(10):
        code = sampler.generate(prompt, max_length=512)
        full_code = prompt + code
        samples.append(full_code)
    predictions.append(samples)
    references.append(unit_test)

print("Evaluating with pass@k...")
code_eval = load("code_eval")
pass_at_k, results = code_eval.compute(
    references=references,
    predictions=predictions,
    k=[1, 5, 10]
)

print("\n=== Pass@k result ===")
for key, val in pass_at_k.items():
    print(f"{key}: {val:.3f}")
