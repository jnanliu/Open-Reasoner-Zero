"""
Qwen2.5-7B base model + ppo

debug running command in single node:

DEBUG_MODE=True python -m playground.orz_7b_grpo

"""


import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from loguru import logger
from omegaconf.listconfig import ListConfig

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExpConfig
from playground.orz_7b_ppo import PPOExp

DEBUG_MODE = False if os.environ.get("DEBUG_MODE", "False") == "False" else True  # Global debug flag

file_name = f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"

executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    total_num_nodes: int = 8 if not DEBUG_MODE else 8

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "Qwen/Qwen2.5-7B-Instruct" # TODO: or put your downloaded model path here!
    reward_pretrain: Optional[str] = None
    save_interval: int = 20
    ckpt_path: str = f"orz_ckpt/{file_name}/Qwen2.5-7B_orz-grpo-filtered-data_lr3e-6_rbs32_ng64_len2048+16384"
    save_path: str = f"orz_ckpt/{file_name}/Qwen2.5-7B_orz-grpo-filtered-data_lr3e-6_rbs32_ng64_len2048+16384"
    tensorboard_log_dir: str = f"orz_logs/{file_name}/Qwen2.5-7B_orz-grpo-filtered-data_lr3e-6_rbs32_ng64_len2048+16384"
    wandb_name: str = 'Qwen2.5-7B_orz-grpo-filtered-data_lr3e-6_rbs32_ng64_len2048+16384'

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    prompt_data: ListConfig = ListConfig(
        [
            "data/orz_math_filtered.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [
            "data/eval_data/math500.json",
            "data/eval_data/aime2024.json",
            "data/eval_data/gpqa_diamond.json",
        ]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 2048
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 1
    max_steps: int = 200
    rollout_batch_size: int = 32 if not DEBUG_MODE else 16
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 32

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True
    eval_interval: int = 10

    # generate related settings
    packing_max_len: int = 16384 + 2048
    generate_max_len: int = 16384  # TODO: change to larger later
    max_len: int = 16384 + 2048  # TODO: change to larger later
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = True

    gpu_memory_utilization: float = 0.3 if not DEBUG_MODE else 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
