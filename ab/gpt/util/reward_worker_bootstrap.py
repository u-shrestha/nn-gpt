import os


def reward_worker_main(conn, assigned_gpu=None, assigned_cuda_visible_device=None) -> None:
    if assigned_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        visible_device = (
            str(assigned_cuda_visible_device)
            if assigned_cuda_visible_device is not None
            else str(int(assigned_gpu))
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    from ab.gpt.util import Reward as RewardUtil

    RewardUtil._persistent_eval_worker_entry(conn, assigned_gpu, assigned_cuda_visible_device)
