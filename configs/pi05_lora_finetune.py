# LoRA fine-tuning config for π0.5 on LIBERO
# Usage: add this TrainConfig into openpi/src/openpi/training/config.py
# Then run: python scripts/train.py pi05_libero_low_mem_finetune

TrainConfig(
    name="pi05_libero_low_mem_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",        # PaliGemma backbone with LoRA (rank=16)
        action_expert_variant="gemma_300m_lora",  # Action expert with LoRA (rank=32)
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=64,
    num_train_steps=30_000,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=5e-5,
        decay_steps=30_000,
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    save_interval=2_500,
    keep_period=5_000,
),