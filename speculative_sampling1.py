from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
from typing import List, Optional
import warnings

# Filter Flash Attention warnings
warnings.filterwarnings("ignore", category=UserWarning)

COLOR = {
    "reset": "\033[0m",
    "draft": "\033[33m",
    "accepted": "\033[32m",
    "rejected": "\033[31m",
    "main": "\033[36m"
}


class RobustStoryGenerator:
    def __init__(self, main_model="facebook/opt-1.3b", draft_model="facebook/opt-350m",
                 device="auto", torch_dtype=torch.float16):
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model, torch_dtype=torch_dtype, device_map=device)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model, torch_dtype=torch_dtype, device_map=device)
        self.tokenizer = AutoTokenizer.from_pretrained(main_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Enhanced generation parameters
        self.config = {
            "rep_penalty": 1.8,  # Stronger repetition penalty
            "no_repeat_ngram": 4,  # Ban 4-gram repetition
            "top_k": 60,
            "top_p": 0.90,
            "temp_range": (0.4, 0.8),
            "min_new_words": 0.4  # New words ratio threshold
        }

    def generate(self, prompt, max_tokens=150, draft_len=5, threshold=0.35,
                 display=True):
        generated = prompt
        token_count = 0
        device = self.main_model.device

        while token_count < max_tokens:
            # Process inputs (add attention_mask)
            inputs = self.tokenizer(
                generated,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            ).to(device)

            try:
                # Dynamic temperature adjustment
                progress = token_count / max_tokens
                temp = self._dynamic_temperature(progress)

                # Generate draft (enhanced parameters)
                draft = self.draft_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=draft_len,
                    temperature=temp,
                    repetition_penalty=self.config["rep_penalty"],
                    no_repeat_ngram_size=self.config["no_repeat_ngram"],
                    do_sample=True,
                    top_k=self.config["top_k"],
                    top_p=self.config["top_p"],
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # Safe validation
                valid, correction = self._safe_validation(
                    inputs.input_ids,
                    draft,
                    generated,
                    threshold
                )

                # Update text
                new_text = self._decode(valid) + correction
                generated += new_text
                token_count += len(self.tokenizer.encode(new_text))

                if display:
                    self._display(generated, draft, valid, correction)
                    time.sleep(0.3)

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise

        return generated

    def _dynamic_temperature(self, progress):
        min_temp, max_temp = self.config["temp_range"]
        return max_temp - (max_temp - min_temp) * progress ** 2  # Non-linear cooling

    def _safe_validation(self, input_ids, draft, context, threshold):
        with torch.no_grad():
            logits = self.main_model(draft[:, :-1]).logits

        valid_tokens = []
        draft_tokens = draft[0][input_ids.shape[-1]:].tolist()
        context_words = set(context.split())

        for idx, token in enumerate(draft_tokens):
            if token in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                break

            # Probability check
            pos = input_ids.shape[-1] - 1 + idx
            prob = torch.softmax(logits[0, pos], -1)[token].item()
            if prob < threshold:
                break

            # Safe text generation check
            generated_part = self.tokenizer.decode(valid_tokens + [token])
            if generated_part.strip():  # Prevent empty text
                new_words = set(generated_part.split()) - context_words
                total_words = len(generated_part.split())
                if total_words == 0 or len(new_words) / total_words < self.config["min_new_words"]:
                    break

            valid_tokens.append(token)

        # Safe sampling
        correction = ""
        if len(valid_tokens) < len(draft_tokens):
            try:
                probs = torch.softmax(logits[0, pos], -1)
                correction = self.tokenizer.decode([torch.multinomial(probs, 1).item()])
            except:
                correction = ""

        return valid_tokens, correction

    def _decode(self, tokens):
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    def _display(self, base, draft, valid, correction):
        os.system('cls' if os.name == 'nt' else 'clear')
        draft_text = self._decode(draft[0][len(base):])
        valid_text = self._decode(valid)
        rejected = draft_text[len(valid_text):]

        display = (
            f"{base}"
            f"{COLOR['accepted']}{valid_text}"
            f"{COLOR['rejected']}{rejected}"
            f"{COLOR['main']}{correction}{COLOR['reset']}"
        )
        print("\033[0;0H" + display, end="", flush=True)


if __name__ == "__main__":
    # Usage example (optimized parameters)
    generator = RobustStoryGenerator()
    story = generator.generate(
        "Write a short story. The story starts by: Once upon a time there was a farmer",
        max_tokens=200,
        draft_len=6,
        threshold=0.4,
        display=True
    )
    print("\n\nFinal Story:\n", story)
