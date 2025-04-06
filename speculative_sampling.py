from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time 
import random
from datasets import load_dataset

print("Loading HumanEval...")
ds = load_dataset("openai/openai_humaneval")
test_set = ds["test"]
print(f"Loaded {len(test_set)} tasks")

COLOR = {
    "reset": "\033[0m",
    "draft": "\033[33m",
    "accepted": "\033[32m",
    "rejected": "\033[31m",
    "main": "\033[36m"
}

# Load the main model (large model)
#main_model_name = "facebook/opt-1.3b"
main_model_name ="Qwen/Qwen2.5-1.5B"

main_model = AutoModelForCausalLM.from_pretrained(main_model_name, torch_dtype=torch.float16, device_map="cuda")    
tokenizer = AutoTokenizer.from_pretrained(main_model_name)

# Load the draft model (smaller model)
#draft_model_name = "facebook/opt-350m"
draft_model_name ="Qwen/Qwen2.5-0.5B"
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, torch_dtype=torch.float16, device_map="cuda")

def speculative_sampling(prompt, total_nbr_token=20, max_new_tokens=5, temperature=1, toggle_print=True, sleep=0.5):
    """
    arguments:
    total_nbr_token : the number of token that this function will create (the size of the output)
    proba_limit : the probability limit that will decide if the draft token is kept or not
    max_new_tokens : the number maximum of tokens created by the draft model before an evaluation by the main model.
    """
    total_text=prompt
    nbr_token=0
    while(total_nbr_token>nbr_token):
        if(toggle_print):
            print_result(total_text, sleep)

        #transform current text into tokens
        input_ids = tokenizer(total_text, return_tensors="pt").input_ids.to(main_model.device)
        
        # Generate with the draft model
        draft_output_logits=[]
        draft_output=[]
        new_input=input_ids
        numb_new_tokens=max_new_tokens
        EOS_found=False
        with torch.no_grad():
            for i in range(max_new_tokens):
                draft_output_logits.append(draft_model(new_input).logits[:, -1, :])#get the logit of the last token
                next_token=temperature_sampling(draft_output_logits[i], temperature)
                draft_output.append(next_token)
                new_input = torch.cat([new_input, torch.tensor([[draft_output[i]]], device=input_ids.device)], dim=1)
                if next_token == tokenizer.eos_token_id:
                    EOS_found=True
                    numb_new_tokens=i+1
                    break
                    

        # Decode the draft model tokens for visibility
        draft_tokens_text = [tokenizer.decode([tok]) for tok in draft_output]
        if(toggle_print):
            print_result(total_text,sleep,  draft_tokens_text)

        # the main models predictions on all the tokens
        with torch.no_grad():
            main_logits = main_model(new_input).logits

        # compute probabilities for draft tokens
        accepted=[]
        first_not_accepted=-1
        for i in range(numb_new_tokens):
            pos = input_ids.shape[1] - 1 + i  # Position in main_logits
            main_probs = torch.softmax(main_logits[:, pos, :], dim=-1).float()
            draft_probs=torch.softmax(draft_output_logits[i], dim=-1).float()
            draft_token = draft_output[i]  # Draft token at position i+1
            draft_prob_main= main_probs[0, draft_token].item()  # Main model's probability for draft token
            draft_prob_draft= draft_probs[0, draft_token].item()  # draft model's probability for draft token
            r=random.random()
            if(draft_prob_main/draft_prob_draft<=r and first_not_accepted==-1):
                first_not_accepted=pos

            #check if a token was accepted
            accepted.append(draft_prob_main/draft_prob_draft>r)

        if(toggle_print):
            print_result(total_text,sleep,  draft_tokens_text, accepted)

        #one of the token wasnt accepted.
        if(first_not_accepted!=-1):
            #We use the main_logits that we computed before to choose the next word
            best_token_id = temperature_sampling(main_logits[0, first_not_accepted, :], temperature=temperature)
            main_choice=tokenizer.decode(best_token_id)
            accepted_tokens=draft_tokens_text[:first_not_accepted-input_ids.shape[1] + 1]
            if(best_token_id==tokenizer.eos_token_id):
                return total_text+"".join(accepted_tokens)
            if(toggle_print):
                print_result(total_text,sleep,  accepted_tokens, accepted[0:first_not_accepted-input_ids.shape[1] + 1], main_choice)
            total_text+="".join(accepted_tokens)+main_choice
            nbr_token+=first_not_accepted-input_ids.shape[1] + 2
        else:
            nbr_token+=max_new_tokens
            if(EOS_found==True):
                return total_text+"".join(draft_tokens_text[:-1])
            total_text+="".join(draft_tokens_text)
    return total_text


def temperature_sampling(logits, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

def top_k_sampling(logits, k=50):
    top_k_probs, top_k_indices = torch.topk(logits, k=k, dim=-1)
    probs = torch.softmax(top_k_probs, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1).item()
    return top_k_indices[sampled_index].item()

def print_result(total_text,sleep=0.5, draft=[], accepted=[], main_choice=""):
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
    print(f"\033[2J\033[H{COLOR['reset']}", end="", flush=True)
    if accepted==[]:
        # Print total_text and draft (in yellow)
        print(f"{total_text}{COLOR['draft']}{''.join(draft)}{COLOR['reset']}", end="", flush=True)
    else:
        # Build the colored text
        text_print = f"{total_text}"
        for word, is_accepted in zip(draft, accepted):
            if is_accepted:
                text_print += f"{COLOR['accepted']}{word}{COLOR['reset']}"  # Green for accepted
            else:
                text_print += f"{COLOR['rejected']}{word}{COLOR['reset']}"  # Red for rejected
        text_print += f"{COLOR['main']}{main_choice}{COLOR['reset']}"  # Cyan for main_choice
        print(text_print, end="", flush=True)
    #the time to see what the print does, modify to choose how fast you want it to go
    time.sleep(sleep)


def classic_sample(prompt, token_lenghts, model, temperature=1):
    draft_output = []
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    new_input = input_ids
    for _ in range(token_lenghts):
        logits = draft_model(new_input).logits[:, -1, :]
        next_token = temperature_sampling(logits, temperature)
        draft_output.append(next_token)
        if next_token == tokenizer.eos_token_id:
            print("BREAK")
            break
        next_token_tensor = torch.tensor([[next_token]], device=input_ids.device)
        new_input = torch.cat([new_input, next_token_tensor], dim=1)

    return "".join([tokenizer.decode([tok]) for tok in draft_output])

# Example usage
prompt = ''' 
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
'''

def time_diff(total_nbr_token=20):
    start=time.time()
    output=classic_sample(prompt, total_nbr_token, main_model)
    print(output)
    print(f"time taken with classic : {time.time()-start}")
    start=time.time()
    output=speculative_sampling(prompt,total_nbr_token=total_nbr_token, max_new_tokens=5, temperature=1, toggle_print=False, sleep=0)
    print(output)
    print(f"time taken with speculative : {time.time()-start}")

# time_diff(100)

def evaluation(prompt, test, entrypoint):
    output=speculative_sampling(prompt, 200,10, toggle_print=False)
    print(output)
    env = {}
    exec(output, env)
    exec(test, env)
    env["check"](env[entrypoint])


nbr_task_made = 0
for task in test_set.select(range(10)):
    try:
        evaluation(task["prompt"],task["test"],task["entry_point"])
        nbr_task_made+=1
        print("\n\n\ntask passed")
    except:
        print("\n\n\ntask failed")
    print(f"""total number of task passed : {nbr_task_made}""")
