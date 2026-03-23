import os


def reward_worker_main(conn) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from ab.gpt.util import Reward as RewardUtil

    RewardUtil._persistent_eval_worker_entry(conn)
