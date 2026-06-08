# Zero-shot evaluation config for π0.5 base model on LIBERO
# Add this TrainConfig into openpi/src/openpi/training/config.py
# Usage: python scripts/eval.py --config pi05_base_eval_libero

TrainConfig(
    name="pi05_base_eval_libero",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi05_libero/assets",
            asset_id="physical-intelligence/libero",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    num_train_steps=1,
    batch_size=1,
    ema_decay=None,
),