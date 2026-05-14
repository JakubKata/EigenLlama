from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    allow_patterns=["*.safetensors", "*.model", "*.json"],
    local_dir="./tinyllama_files"
)
print("ack")