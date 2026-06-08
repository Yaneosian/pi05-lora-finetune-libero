# Changes made to openpi/src/openpi/models/gemma.py
# Two modifications:
#   1. Variant type definition (add 5 new rank variants)
#   2. get_config() function (add 5 new if branches)



# ============ Modification 1: Variant type definition ============
# Original:
# Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]
#
# Updated:
Variant = Literal[
    "dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora",
    "gemma_2b_lora_r4", "gemma_2b_lora_r8", "gemma_2b_lora_r16",
    "gemma_2b_lora_r32", "gemma_2b_lora_r64",
]



# ============ Modification 2: New if branches in get_config() ============
# Convention: alpha=rank, so effective scaling factor alpha/rank=1
# Ensures rank comparisons are not confounded by scaling differences

if variant == "gemma_2b_lora_r4":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=4,  alpha=4.0),
            "ffn":  lora.LoRAConfig(rank=4,  alpha=4.0),
        },
    )
if variant == "gemma_2b_lora_r8":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=8,  alpha=8.0),
            "ffn":  lora.LoRAConfig(rank=8,  alpha=8.0),
        },
    )
if variant == "gemma_2b_lora_r16":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=16, alpha=16.0),
            "ffn":  lora.LoRAConfig(rank=16, alpha=16.0),
        },
    )
if variant == "gemma_2b_lora_r32":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=32, alpha=32.0),
            "ffn":  lora.LoRAConfig(rank=32, alpha=32.0),
        },
    )
if variant == "gemma_2b_lora_r64":
    return Config(
        width=2048, depth=18, mlp_dim=16_384,
        num_heads=8, num_kv_heads=1, head_dim=256,
        lora_configs={
            "attn": lora.LoRAConfig(rank=64, alpha=64.0),
            "ffn":  lora.LoRAConfig(rank=64, alpha=64.0),
        },
    )