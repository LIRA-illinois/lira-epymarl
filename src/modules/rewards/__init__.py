from .dr_reinforceR_centralized_reward import CentralizedRewardNetwork
from .dr_reinforceR_independent_reward import IndependentRewardNetwork


REGISTRY = {}

REGISTRY["dr_reinforceR_centralized_reward"] = CentralizedRewardNetwork
REGISTRY["dr_reinforceR_independent_reward"] = IndependentRewardNetwork
