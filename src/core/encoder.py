from typing import Dict, List, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils import logger
import os
from config import MISTRAL_MAX_LENGTH, MODEL_CACHE_DIR

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None


class Encoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "",
        framework: str = "transformers",
        api_key: str = "",
    ):
        self.framework = framework
        self.model_name_or_path = model_name_or_path
        self.embedding_dim = None

        if framework == "transformers":
            self.device = (
                device
                if device
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, device_map=self.device, cache_dir=MODEL_CACHE_DIR
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=self.device, cache_dir=MODEL_CACHE_DIR
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.embedding_dim = self.model.config.hidden_size
        elif framework in ["openai", "openrouter"]:
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI library is required for OpenAI/OpenRouter support"
                )

            self.api_key = api_key or os.getenv(
                "OPENAI_API_KEY" if framework == "openai" else "OPENROUTER_API_KEY"
            )
            if not self.api_key:
                raise ValueError(
                    f"API key is required for {framework}. Set {framework.upper()}_API_KEY environment variable or pass api_key parameter."
                )

            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=(
                    "https://openrouter.ai/api/v1"
                    if framework == "openrouter"
                    else None
                ),
            )
            self.device = device
            self.embedding_dim = 1024
        else:
            raise ValueError(f"Unsupported framework: {framework}")

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
                return_attention_mask=True,
            ).to(self.device)

            outputs = self.model(
                **batch_dct,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [batch, seq, hidden]
            attn = (
                batch_dct["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            )  # [batch, seq, 1]
            masked = last_hidden * attn
            lengths = attn.sum(dim=1).clamp(min=1.0)  # [batch, 1]
            embeddings = (masked.sum(dim=1) / lengths).detach().cpu()  # [batch, hidden]

        return embeddings

    def _generate_completion_transformers(
        self, text: List[Dict[str, str]], max_length: int = 512, answer_prefix: str = ""
    ) -> List[str]:
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
    ) -> List[str]:
        """Generate completion using OpenAI or OpenRouter API."""
        try:
            # Convert messages to OpenAI format
            messages = []
            for msg in text:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                else:
                    # Handle legacy format
                    messages.append({"role": "user", "content": str(msg)})

            # Add answer prefix if provided
            if answer_prefix:
                messages[-1]["content"] += answer_prefix

            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                max_tokens=max_length,
                temperature=0.1,
                top_p=0.9,
            )

            return [response.choices[0].message.content.strip()]

        except Exception as e:
            logger.error(f"Error in OpenAI/OpenRouter generation: {e}")
            return [f"Error generating response: {str(e)}"]

    def generate_completion(
        self,
        text: List[Dict[str, str]],
        max_length: int = 512,
        answer_prefix: str = "",
    ) -> List[str]:
        """Generate completion using the specified or default framework."""
        framework = self.framework

        if framework == "transformers":
            return self._generate_completion_transformers(
                text, max_length, answer_prefix
            )
        elif framework in ["openai", "openrouter"]:
            return self._generate_completion_openai(text, max_length, answer_prefix)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def encode(self, text, model_type: str = "mistral") -> torch.Tensor:
        """Encode text to embeddings. Only available for transformers framework."""
        if self.framework != "transformers":
            raise ValueError(
                f"Encoding is only supported for transformers framework, not {self.framework}"
            )

        if model_type == "mistral":
            return self._mistral_encode(text)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def __call__(self, *args, **kwds) -> torch.Tensor:
        return self.encode(*args, **kwds)
