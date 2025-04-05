from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

COLOR = {
    "reset": "\033[0m",
    "draft": "\033[33m",    # Yellow - Draft
    "accepted": "\033[32m",  # Green - Accepted
    "rejected": "\033[31m",  # Red - Rejected
    "main": "\033[36m"       # Cyan - Correction by the target model
}


class SpeculativeSampling:
    def __init__(self,
                 target_model_name: str = "facebook/opt-1.3b",
                 draft_model_name: str = "facebook/opt-350m",
                 device: str = "cuda",
                 temperature: float = 0.7,
                 torch_dtype: torch.dtype = torch.float16):

        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch_dtype,
            device_map=device
        ).eval()

        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch_dtype,
            device_map=device
        ).eval()

        self.vocab_size = self.target_model.config.vocab_size
        self.temperature = temperature
        self.K = 5  # Number of draft tokens per iteration
        self.T = 100  # Maximum speculative decoding steps

    def generate(self, prompt: str, max_length: int = 200) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.target_model.device)
        full_output = prompt
        n = input_ids.size(1)

        while n < max_length and n < self.T:
            # Generate K draft tokens from the draft model
            draft_tokens = self._generate_draft(input_ids)
            draft_text = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)

            # Get target model logits for the draft tokens
            target_logits = self._get_target_logits(input_ids, draft_tokens)

            # Determine which draft tokens are accepted or rejected
            accepted, correction = self._validate_tokens(input_ids, draft_tokens, target_logits)

            # Build a color-coded display string for terminal visualization
            display_text = self._build_display(
                full_output,
                draft_tokens,
                accepted,
                correction
            )

            # Update generated tokens and input
            new_tokens = accepted + [correction] if correction is not None else accepted
            if new_tokens:
                valid_tokens = [t for t in new_tokens if t < self.vocab_size]
                new_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                full_output += new_text
                new_tokens_tensor = torch.tensor([valid_tokens], device=input_ids.device)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
                n += len(valid_tokens)

            # If all draft tokens are accepted, sample one more token
            if len(accepted) == self.K:
                extra_token = self._sample_extra_token(input_ids)
                if extra_token < self.vocab_size:
                    input_ids = torch.cat([input_ids, torch.tensor([[extra_token]], device=input_ids.device)], dim=1)
                    full_output += self.tokenizer.decode([extra_token], skip_special_tokens=True)
                    n += 1

            # Refresh display in the terminal
            self._refresh_display(display_text)
            time.sleep(0.15)

        print("\n")  # Print newline after generation completes
        return full_output

    def _build_display(self,
                       base: str,
                       draft_tokens: List[int],
                       accepted: List[int],
                       correction: Optional[int]) -> str:
        """Build a color-coded display string for terminal output."""
        accepted_text = self.tokenizer.decode(accepted, skip_special_tokens=True)
        full_draft = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)

        # Separate accepted and rejected parts from the draft
        accepted_part = full_draft[:len(accepted_text)]
        rejected_part = full_draft[len(accepted_text):]

        display_str = base
        display_str += f"{COLOR['accepted']}{accepted_part}{COLOR['reset']}"
        display_str += f"{COLOR['rejected']}{rejected_part}{COLOR['reset']}"

        if correction is not None:
            correction_text = self.tokenizer.decode([correction], skip_special_tokens=True)
            display_str += f"{COLOR['main']}{correction_text}{COLOR['reset']}"

        return display_str

    def _refresh_display(self, text: str):
        """Clear and refresh the terminal display."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\033[0;0H" + text, end="", flush=True)

    def _generate_draft(self, input_ids: torch.Tensor) -> List[int]:
        """Generate K draft tokens using the draft model."""
        draft = []
        current_input = input_ids.clone()
        for _ in range(self.K):
            with torch.no_grad():
                outputs = self.draft_model(current_input)
                logits = outputs.logits[:, -1, :]
                probs = torch.softmax(logits / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item() if next_token.item() < self.vocab_size else self.tokenizer.eos_token_id
                draft.append(token_id)
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[token_id]], device=current_input.device)
                ], dim=1)
        return draft

    def _get_target_logits(self, base_input: torch.Tensor, draft_tokens: List[int]) -> List[torch.Tensor]:
        """Get target model logits for each prefix extended with draft tokens."""
        sequences = [base_input]
        for token in draft_tokens:
            token = token if token < self.vocab_size else self.tokenizer.eos_token_id
            token_tensor = torch.tensor([[token]], device=base_input.device)
            seq = torch.cat([sequences[-1], token_tensor], dim=1)
            sequences.append(seq)

        all_logits = []
        with torch.no_grad():
            for seq in sequences:
                logits = self.target_model(seq).logits[:, -1, :]
                all_logits.append(logits)
        return all_logits

    def _validate_tokens(self,
                         base_input: torch.Tensor,
                         draft_tokens: List[int],
                         target_logits: List[torch.Tensor]) -> (List[int], Optional[int]):
        """Use rejection sampling to determine which draft tokens to accept."""
        accepted = []
        current_input = base_input.clone()

        for t in range(len(draft_tokens)):
            draft_token = draft_tokens[t] if draft_tokens[t] < self.vocab_size else self.tokenizer.eos_token_id
            with torch.no_grad():
                p = torch.softmax(self.draft_model(current_input).logits[:, -1, :], dim=-1)
            q = torch.softmax(target_logits[t], dim=-1)

            try:
                ratio = q[0, draft_token] / p[0, draft_token]
            except IndexError:
                ratio = torch.tensor(0.0)

            accept_prob = min(1.0, ratio.item())

            if torch.rand(1).item() < accept_prob:
                accepted.append(draft_token)
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[draft_token]], device=current_input.device)
                ], dim=1)
            else:
                correction_probs = torch.clamp(q - p, min=0)
                correction_probs[:, self.vocab_size:] = 0
                correction_probs /= correction_probs.sum(dim=-1, keepdim=True)
                correction = torch.multinomial(correction_probs[0], 1).item()
                return accepted, correction

        return accepted, None

    def _sample_extra_token(self, input_ids: torch.Tensor) -> int:
        """Sample one extra token from the target model if all drafts were accepted."""
        with torch.no_grad():
            logits = self.target_model(input_ids).logits[:, -1, :]
            logits[:, self.vocab_size:] = -float('inf')
            probs = torch.softmax(logits / self.temperature, dim=-1)
            return torch.multinomial(probs[0], 1).item()


if __name__ == "__main__":
    generator = SpeculativeSampling(
    target_model_name="Qwen/Qwen2.5-1.5B",
    draft_model_name="Qwen/Qwen2.5-0.5B",
    temperature=0.8
)
    prompt = "In a magical library where books can rewrite reality"
    result = generator.generate(prompt, max_length=300)
    print("\nFinal Story:\n" + COLOR["main"] + result + COLOR["reset"])
