import os


def reward_worker_main(conn, assigned_gpu=None) -> None:
    if assigned_gpu is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(assigned_gpu))
    from ab.gpt.util import Reward as RewardUtil

    RewardUtil._persistent_eval_worker_entry(conn, assigned_gpu)
