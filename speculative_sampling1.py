from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os
import matplotlib.pyplot as plt

COLOR = {
    "reset": "\033[0m",
    "draft": "\033[33m",
    "accepted": "\033[32m",
    "rejected": "\033[31m",
    "main": "\033[36m"
}

TEMPERATURE = 0.7  # Temperature for sampling

class SpeculativeSampling:
    def __init__(self,
                 target_model_name: str = "facebook/opt-1.3b",
                 draft_model_name: str = "facebook/opt-350m",
                 device: str = "cuda",
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
        self.K = 5  # initial K, will be updated dynamically
        self.T = 100

    def generate(self, prompt: str, max_length: int = 200) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.target_model.device)
        full_output = prompt
        n = input_ids.size(1)
        self.accept_history = []

        while n < max_length and n < self.T:
            if len(self.accept_history) >= 3:
                avg_alpha = sum(self.accept_history[-3:]) / 3
                if avg_alpha >= 0.999:
                    avg_alpha = 0.999
                self.K, _ = self._optimal_gamma(avg_alpha)
                print(f"[Info] Dynamic K updated to: {self.K:.0f}, estimated α ≈ {avg_alpha:.2f}")

            draft_tokens = self._generate_draft(input_ids)
            target_logits = self._get_target_logits(input_ids, draft_tokens)
            accepted, correction = self._validate_tokens(input_ids, draft_tokens, target_logits)
            self.accept_history.append(len(accepted) / self.K if self.K > 0 else 0)

            display_text = self._build_display(full_output, draft_tokens, accepted, correction)

            new_tokens = accepted + [correction] if correction is not None else accepted
            if new_tokens:
                valid_tokens = [t for t in new_tokens if t < self.vocab_size]
                new_text = self.tokenizer.decode(valid_tokens, skip_special_tokens=True)
                full_output += new_text
                new_tokens_tensor = torch.tensor([valid_tokens], device=input_ids.device)
                input_ids = torch.cat([input_ids, new_tokens_tensor], dim=1)
                n += len(valid_tokens)

            if len(accepted) == self.K:
                extra_token = self._sample_extra_token(input_ids)
                if extra_token < self.vocab_size:
                    input_ids = torch.cat([input_ids, torch.tensor([[extra_token]], device=input_ids.device)], dim=1)
                    full_output += self.tokenizer.decode([extra_token], skip_special_tokens=True)
                    n += 1

            self._refresh_display(display_text)
            time.sleep(0.15)

        print("\n")
        self._plot_acceptance_rate()
        return full_output

    def _plot_acceptance_rate(self):
        if not self.accept_history:
            return
        plt.figure(figsize=(8, 4))
        plt.plot(self.accept_history, marker='o')
        plt.title("Draft Token Acceptance Rate per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Acceptance Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("acceptance_rate_curve.png")
        print("[Info] Acceptance rate curve saved as 'acceptance_rate_curve.png'")

    def _optimal_gamma(self, alpha: float, cost_ratio: float = 0.1, max_gamma: int = 16):
        def r_gamma(alpha, gamma):
            return (1 - alpha**(gamma + 1)) / (1 - alpha)

        best_gamma = 1
        best_speedup = 0.0
        for gamma in range(1, max_gamma + 1):
            try:
                r = r_gamma(alpha, gamma)
                total_cost = (gamma + 1) + cost_ratio * gamma
                speedup = r / total_cost
                if speedup > best_speedup:
                    best_gamma = gamma
                    best_speedup = speedup
            except ZeroDivisionError:
                continue
        return best_gamma, best_speedup

    def _build_display(self, base: str, draft_tokens: List[int], accepted: List[int], correction: Optional[int]) -> str:
        accepted_text = self.tokenizer.decode(accepted, skip_special_tokens=True)
        full_draft = self.tokenizer.decode(draft_tokens, skip_special_tokens=True)
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
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\033[0;0H" + text, end="", flush=True)

    def _temperature_sample(self, probs: torch.Tensor, temperature: float = TEMPERATURE) -> int:
        adjusted = torch.log(probs + 1e-10) / temperature
        adjusted_probs = torch.softmax(adjusted, dim=-1)
        return torch.multinomial(adjusted_probs, num_samples=1).item()

    def _generate_draft(self, input_ids: torch.Tensor) -> List[int]:
        draft = []
        current_input = input_ids.clone()
        for _ in range(self.K):
            with torch.no_grad():
                outputs = self.draft_model(current_input)
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                token_id = self._temperature_sample(probs[0])
                token_id = token_id if token_id < self.vocab_size else self.tokenizer.eos_token_id
                draft.append(token_id)
                current_input = torch.cat([current_input, torch.tensor([[token_id]], device=current_input.device)], dim=1)
        return draft

    def _get_target_logits(self, base_input: torch.Tensor, draft_tokens: List[int]) -> List[torch.Tensor]:
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
                current_input = torch.cat([current_input, torch.tensor([[draft_token]], device=current_input.device)], dim=1)
            else:
                correction_probs = torch.clamp(q - p, min=0)
                correction_probs[:, self.vocab_size:] = 0
                correction_probs /= correction_probs.sum(dim=-1, keepdim=True)
                correction = torch.multinomial(correction_probs[0], 1).item()
                return accepted, correction
        return accepted, None

    def _sample_extra_token(self, input_ids: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.target_model(input_ids).logits[:, -1, :]
            logits[:, self.vocab_size:] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            return self._temperature_sample(probs[0])


if __name__ == "__main__":
    generator = SpeculativeSampling()
    prompt = "In a magical library where books can rewrite reality"
    result = generator.generate(prompt, max_length=300)
    print("\nFinal Story:\n" + COLOR["main"] + result + COLOR["reset"])
