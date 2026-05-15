import os
import numpy as np
import ml_dtypes
from safetensors.numpy import load_file
from transformers import AutoTokenizer
import tiny_math

from layers import TransformerBlock

class TinyLlamaEngine:
    def __init__(self, model_dir):
        print("Loading dictionary...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.eos_token_id = self.tokenizer.eos_token_id

        print("Loading weights to RAM...")
        self.weights = load_file(os.path.join(model_dir, "model.safetensors"))

        self.embed_tokens = self.weights["model.embed_tokens.weight"].astype(np.float32)
        self.lm_head = self.weights["lm_head.weight"].astype(np.float32).T
        self.norm_weight = self.weights["model.norm.weight"].astype(np.float32)

        print("Constructing 22 transformer layers...")
        self.layers = [TransformerBlock(i, self.weights) for i in range(22)]
        print("Engine online!\n" + "="*40)

    def generate(self, prompt, max_tokens=30):
        tokens = self.tokenizer.encode(prompt)
        print(f"\nUser: {prompt}")
        print("AI: ", end="", flush=True)

        printed_text = self.tokenizer.decode(tokens)

        for _ in range(max_tokens):
            x = self.embed_tokens[tokens]

            for layer in self.layers:
                x = layer.forward(x)

            x = tiny_math.rmsnorm(x, self.norm_weight)
            last_token_vector = x[-1:] 
            logits = tiny_math.matmul(last_token_vector, self.lm_head)
            next_token_id = int(np.argmax(logits))
            
            if next_token_id == self.eos_token_id:
                break
                
            tokens.append(next_token_id)
            full_text = self.tokenizer.decode(tokens)
            
            new_text = full_text[len(printed_text):]
            print(new_text, end="", flush=True)
            
            printed_text = full_text
            
        print("\n" + "-"*40)