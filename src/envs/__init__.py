import os
import sys
import gymnasium as gym
from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .smaclite_wrapper import SMACliteWrapper

# Additional environments
from .gym_multigrid_wrapper import GymMultiGridWrapper
from .basic_gymnasium_wrapper import BasicGymnasiumWrapper, SUPPORTED_ENVS as basic_supported_envs

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


# registering both smac and smacv2 causes a pysc2 error
# --> dynamically register the needed env
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn


REGISTRY = {}


def smaclite_fn(**kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return SMACliteWrapper(**kwargs)


REGISTRY["smaclite"] = smaclite_fn


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY["gymma"] = gymma_fn


# Additional environments
def gym_multigrid_fn(**kwargs) -> gym.Env:
    return GymMultiGridWrapper(**kwargs)


REGISTRY["gym_multigrid"] = gym_multigrid_fn


def gym_env_fn(common_reward=None, reward_scalarisation=None, **env_args) -> gym.Env:
    return BasicGymnasiumWrapper(env_args=env_args)

for env in basic_supported_envs:
    REGISTRY[env] = gym_env_fn
