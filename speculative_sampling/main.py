from transformers import OPTForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


# We use the opt-350M model and tokenizer

model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)

def main():
    print("Hello from speculative-sampling!")

    prompt = "Describe what a star is."
    generated_tokens = []
    for _ in range(5):  # Generate 5 tokens
        probabilities = get_probabilities(prompt, generated_tokens)
        next_token_id = torch.argmax(probabilities, dim=-1).item()
        generated_tokens.append(next_token_id)
        print("Generated so far:", tokenizer.decode(generated_tokens))



    
    

def get_probabilities(prompt, generated_tokens):
    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    # If some tokens have been generated, append them to the input
    if generated_tokens:
        input_ids = torch.cat([input_ids, torch.tensor([generated_tokens])], dim=1)

    # Get model logits
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, -1, :]  # Last token’s logits

    # Convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Get the top predicted tokens
    topk = torch.topk(probabilities, k=10)
    print("Top 10 predicted tokens and their probabilities:")
    for token_id, prob in zip(topk.indices[0].tolist(), topk.values[0].tolist()):
        token = tokenizer.decode([token_id])
        print(f"Token: {token:15} | Probability: {prob:.4f}")

    return probabilities



if __name__ == "__main__":
    main()
