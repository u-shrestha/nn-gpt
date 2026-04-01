import os


def _clear_distributed_env() -> None:
    # Reward workers must behave like isolated single-process evaluators.
    # Inheriting torchrun/accelerate env can make nested formal eval pick up
    # the outer distributed topology and bind work to the wrong GPU.
    keys = (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "NODE_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_USE_AGENT_STORE",
        "ACCELERATE_PROCESS_INDEX",
        "ACCELERATE_LOCAL_PROCESS_INDEX",
        "ACCELERATE_NUM_PROCESSES",
    )
    for key in keys:
        os.environ.pop(key, None)


def reward_worker_main(conn, assigned_gpu=None, assigned_cuda_visible_device=None) -> None:
    _clear_distributed_env()
    if assigned_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        visible_device = (
            str(assigned_cuda_visible_device)
            if assigned_cuda_visible_device is not None
            else str(int(assigned_gpu))
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from ab.gpt.util import Reward as RewardUtil

    RewardUtil._persistent_eval_worker_entry(conn, assigned_gpu, assigned_cuda_visible_device)
