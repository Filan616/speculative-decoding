from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 

# Load the main model (large model)
main_model_name = "facebook/opt-1.3b"

main_model = AutoModelForCausalLM.from_pretrained(main_model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# Load the draft model (smaller model)
draft_model_name = "facebook/opt-350m"
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, torch_dtype=torch.float16, device_map="auto")

def speculative_sampling(prompt, total_nbr_token=20, proba_limit=0.2, max_new_tokens=5, temperature=1):
    """
    arguments:
    total_nbr_token : the number of token that this function will create (the size of the output)
    proba_limit : the probability limit that will decide if the draft token is kept or not
    max_new_tokens : the number maximum of tokens created by the draft model before an evaluation by the main model.
    """
    total_text=prompt
    nbr_token=0
    while(total_nbr_token>nbr_token):
        print_result(total_text)

        #transform current text into tokens
        input_ids = tokenizer(total_text, return_tensors="pt").input_ids.to(main_model.device)
        
        # Generate with the draft model
        with torch.no_grad():
            draft_output = draft_model.generate(input_ids,repetition_penalty=1.2,temperature=temperature,  max_new_tokens=max_new_tokens)

        # Decode the draft model tokens for visibility
        draft_tokens_text = [tokenizer.decode([tok]) for tok in draft_output[0][-max_new_tokens:]]
        print_result(total_text, draft_tokens_text)

        # the main models predictions on all the tokens
        with torch.no_grad():
            main_logits = main_model(draft_output[:, :-1]).logits

        # compute probabilities for draft tokens
        accepted=[]
        first_not_accepted=-1
        for i in range(max_new_tokens):
            pos = input_ids.shape[1] - 1 + i  # Position in main_logits
            main_probs = torch.softmax(main_logits[:, pos, :], dim=-1).float()
            draft_token = draft_output[0, pos+1].item()  # Draft token at position i+1
            draft_prob = main_probs[0, draft_token].item()  # Main model's probability for draft token
            if(draft_prob<proba_limit and first_not_accepted==-1):
                first_not_accepted=pos

            #check if a token was accepted
            #TODO change way to decide if a token is accepted or not
            accepted.append(draft_prob>proba_limit)

        print_result(total_text, draft_tokens_text, accepted)

        #one of the token wasnt accepted.
        if(first_not_accepted!=-1):
            #We use the main_logits that we computed before to choose the next word
            best_token_id = temperature_sampling(main_logits[0, first_not_accepted, :], temperature=temperature)
            main_choice=tokenizer.decode(best_token_id)
            accepted_tokens=draft_tokens_text[:first_not_accepted-input_ids.shape[1] + 1]
            print_result(total_text, accepted_tokens, accepted[0:first_not_accepted-input_ids.shape[1] + 1], main_choice)
            total_text+="".join(accepted_tokens)+main_choice
            nbr_token+=first_not_accepted-input_ids.shape[1] + 2
        else:
            nbr_token+=max_new_tokens
            total_text+="".join(draft_tokens_text)
    return total_text


#different sampling ways to choose the token chosen by the main LLM
def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def top_k_sampling(logits, k=50):
    top_k_probs, top_k_indices = torch.topk(logits, k=k, dim=-1)
    probs = torch.softmax(top_k_probs, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[sampled_index].item()

def print_result(total_text, draft=[], accepted=[], main_choice=""):
    """
    prints the current text
    arguments : 

    total_text : what has been validated

    draft : what the draft model has created and not validated

    accepted : a true/false array, with true if the token in the same position of draft has been accepted.
    its value is [] if the test hasn't been made yet

    main_choice : the token chosen to be added by the main LLM
    """
    # Clear the screen and move the cursor to the top-left corner
    print(f"\033[2J\033[H\033[0m", end="", flush=True)
    if accepted==[]:
        # Print total_text and draft (in yellow)
        print(f"{total_text}\033[33m{''.join(draft)}\033[0m", end="", flush=True)
    else:
        # Build the colored text
        text_print = f"{total_text}"
        for word, is_accepted in zip(draft, accepted):
            if is_accepted:
                text_print += f"\033[32m{word}\033[0m"  # Green for accepted
            else:
                text_print += f"\033[31m{word}\033[0m"  # Red for rejected
        text_print += f"\033[36m{main_choice}\033[0m"  # Cyan for main_choice
        print(text_print, end="", flush=True)
    #the time to see what the print does, modify to choose how fast you want it to go
    time.sleep(0.5)

# Example usage
prompt = "Write a short story. The story starts by : Once upon a time there was a farmer"
output = speculative_sampling(prompt, 50, 0.1)
#print(output)