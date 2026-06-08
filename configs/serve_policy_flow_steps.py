# Changes made to openpi/scripts/serve_policy.py
# Adds --num_steps argument to control flow matching denoising steps at inference time
# Default=10 ensures backward compatibility when --num_steps is not specified
#
# Usage:
#   python scripts/serve_policy.py --num_steps 5   # 5-step inference
#   python scripts/serve_policy.py --num_steps 1   # single-step inference
#   python scripts/serve_policy.py                 # default 10-step (unchanged)



# ============ Modification 1: Add num_steps to Args dataclass ============
# Location: ~line 47-53 in serve_policy.py
# Add the following line before "record: bool = False"

@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8000
    # Number of denoising steps for flow matching inference.
    num_steps: int = 10              # default=10, backward compatible
    # Record the policy's behavior for debugging.
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)




# ============ Modification 2: Pass num_steps into create_policy ============
# Location: ~line 93 in serve_policy.py
# Original:
#   case Checkpoint():
#       return _policy_config.create_trained_policy(
#           _config.get_config(args.policy.config),
#           args.policy.dir,
#           default_prompt=args.default_prompt,
#       )
#
# Updated:
    case Checkpoint():
        return _policy_config.create_trained_policy(
            _config.get_config(args.policy.config),
            args.policy.dir,
            default_prompt=args.default_prompt,
            sample_kwargs={"num_steps": args.num_steps},  # pass num_steps through
        )