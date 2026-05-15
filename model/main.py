import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine import TinyLlamaEngine

if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tinyllama_files'))
    
    llm = TinyLlamaEngine(model_path)
    while True:
        input_text = input("\n write something ")
        llm.generate(input_text, max_tokens=20)
        