from typing import Dict, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import logging
from config import MISTRAL_MAX_LENGTH, MODEL_CACHE_DIR, BASE_ENCODER_MODEL

logger = logging.getLogger(__name__)


class Encoder:
    def __init__(self, model_name_or_path: str, device: str = "cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, device_map="auto", cache_dir=MODEL_CACHE_DIR
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", cache_dir=MODEL_CACHE_DIR
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _mistral_encode(self, text: Union[List[str], str]) -> torch.Tensor:
        self.model.eval()
        max_length = MISTRAL_MAX_LENGTH
        with torch.no_grad():
            batch_dct = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
                return_attention_mask=False,
            ).to(self.device)

            embeddings = self.model(**batch_dct)[0].detach().cpu()

        return embeddings

    def _generate_completion_transformers(
        self, text: List[Dict[str, str]], max_length: int = 512, answer_prefix: str = ""
    ) -> torch.Tensor:
        messages = (
            self.tokenizer.apply_chat_template(
                text, add_generation_prompt=True, tokenize=False
            )
            + answer_prefix
        )

        model_inputs = self.tokenizer(
            messages, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)

        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            return_dict_in_generate=True,
        )
        with torch.no_grad():
            generated_outputs = self.model.generate(
                **model_inputs,
                generation_config=generation_config,
            )
        generated_tokens = generated_outputs["sequences"][:, model_inputs["input_ids"].shape[1] :]  # type: ignore
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return generated_text

    def _generate_completion_openai(
        self, text: List[Dict[str, str]], max_length: int = 512, answer_prefix: str = ""
    ) -> torch.Tensor:
        raise NotImplementedError(
            "OpenAI generation is not implemented in this encoder."
        )

    def generate_completion(
        self,
        text: List[Dict[str, str]],
        max_length: int = 512,
        answer_prefix: str = "",
        framework: str = "transformers",
    ) -> torch.Tensor:
        if framework == "transformers":
            return self._generate_completion_transformers(
                text, max_length, answer_prefix
            )
        elif framework == "openai":
            return self._generate_completion_openai(text, max_length, answer_prefix)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def encode(self, text: str, model_type: str = "mistral") -> torch.Tensor:
        if model_type == "mistral":
            return self._mistral_encode(text)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return self.encode(*args, **kwds)


if __name__ == "__main__":
    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    text = [{"role": "user", "content": "How are you doing?"}]
    embeddings = encoder.encode(text[0]["content"])
    print(embeddings)
    completion = encoder.generate_completion(
        text, max_length=50, answer_prefix="Answer: "
    )
    print(completion)
