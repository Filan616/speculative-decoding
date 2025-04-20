import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load
from speculative_generate import SpeculativeSampling

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
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
    return tokenizer.decode(generated, skip_special_tokens=True)

# Sampling and evaluation
classic_predictions = []
speculative_predictions = []
references = []

print("Generating samples...")
for task in test_set:
    prompt = task["prompt"]
    unit_test = task["test"]

    # Classic sampling
    classic_samples = []
    for _ in range(10):
        code = classic_generate(prompt)
        classic_samples.append(prompt + code)
    classic_predictions.append(classic_samples)

    # Speculative sampling
    speculative_samples = []
    for _ in range(10):
        code = sampler.generate(prompt, max_length=512)
        speculative_samples.append(prompt + code)
    speculative_predictions.append(speculative_samples)

    references.append(unit_test)

# Evaluate using code_eval
print("Evaluating with pass@k...")
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

print("\n=== Pass@k result: Classic Sampling ===")
for k, v in pass_at_k_classic.items():
    print(f"{k}: {v:.3f}")

print("\n=== Pass@k result: Speculative Sampling ===")
for k, v in pass_at_k_speculative.items():
    print(f"{k}: {v:.3f}")
