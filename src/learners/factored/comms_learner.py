from src.learners import REGISTRY as le_REGISTRY
from src.components.episode_buffer import EpisodeBatch
from src.controllers.factored import CommsMAC


class CommsLearner:
    def __init__(self, mac: CommsMAC, scheme, logger, args):

        # you only want to pass the env mac to the env learner here
        self.env_learner = le_REGISTRY[args.learner](mac.env_mac, scheme, logger, args)
        self.mac = mac

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        self.env_learner.train(batch, t_env, episode_num)

    def cuda(self) -> None:
        self.env_learner.cuda()

    def save_models(self, path: str) -> None:
        self.env_learner.save_models(path)

    def optimize_hl_agent(self, hlmdp, success_rate_spec: float):
        self.mac.comms_agent.optimize_policy(hlmdp, success_rate_spec)
