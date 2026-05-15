import numpy as np
import tiny_math

class LlamaMLP:
    def __init__(self, prefix, weights):
        self.gate_proj = weights[f"{prefix}.gate_proj.weight"].astype(np.float32).T
        self.up_proj   = weights[f"{prefix}.up_proj.weight"].astype(np.float32).T
        self.down_proj = weights[f"{prefix}.down_proj.weight"].astype(np.float32).T

    def forward(self, x):
        gate = tiny_math.matmul(x, self.gate_proj)
        up = tiny_math.matmul(x, self.up_proj)
        
        gate_safe = np.clip(gate, -50.0, 50.0)
        silu_gate = gate_safe * (1.0 / (1.0 + np.exp(-gate_safe)))
        activated = silu_gate * up
        
        return tiny_math.matmul(activated, self.down_proj)

class LlamaAttention:
    def __init__(self, prefix, weights):
        self.q_proj = weights[f"{prefix}.q_proj.weight"].astype(np.float32).T
        self.k_proj = weights[f"{prefix}.k_proj.weight"].astype(np.float32).T
        self.v_proj = weights[f"{prefix}.v_proj.weight"].astype(np.float32).T
        self.o_proj = weights[f"{prefix}.o_proj.weight"].astype(np.float32).T

        self.n_heads = 32
        self.n_kv_heads = 4
        self.head_dim = 2048 // 32

    def forward(self, x):
        seq_len = x.shape[0]

        q = tiny_math.matmul(x, self.q_proj)
        k = tiny_math.matmul(x, self.k_proj)
        v = tiny_math.matmul(x, self.v_proj)

        q = q.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)

        k = np.repeat(k, self.n_heads // self.n_kv_heads, axis=0)
        v = np.repeat(v, self.n_heads // self.n_kv_heads, axis=0)

        inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.head_dim, 2).astype(np.float32) / self.head_dim))
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq)
        
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)
        
        def rotate_half(tensor):
            half = tensor.shape[-1] // 2
            x1 = tensor[..., :half]
            x2 = tensor[..., half:]
            return np.concatenate((-x2, x1), axis=-1)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask

        scores_safe = scores - np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores_safe)
        probs = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)

        out = np.matmul(probs, v)
        out = out.transpose(1, 0, 2).reshape(seq_len, -1)

        return tiny_math.matmul(out, self.o_proj)

class TransformerBlock:
    def __init__(self, layer_id, weights):
        prefix = f"model.layers.{layer_id}"
        self.input_layernorm_weight = weights[f"{prefix}.input_layernorm.weight"].astype(np.float32)
        self.post_attention_layernorm_weight = weights[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32)
        
        self.attention = LlamaAttention(f"{prefix}.self_attn", weights)
        self.mlp = LlamaMLP(f"{prefix}.mlp", weights)

    def forward(self, x):
        norm_x = tiny_math.rmsnorm(x, self.input_layernorm_weight)
        x = x + self.attention.forward(norm_x)
        
        norm_x2 = tiny_math.rmsnorm(x, self.post_attention_layernorm_weight)
        x = x + self.mlp.forward(norm_x2)
        
        return x