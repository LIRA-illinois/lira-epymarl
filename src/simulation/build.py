# sim_build.py
from typing import Any, Optional
from types import SimpleNamespace as SN
import torch as th

from ..components.episode_buffer import ReplayBuffer
from ..components.transforms import OneHot
from ..controllers import REGISTRY as mac_REGISTRY
from ..controllers.factored import REGISTRY as factored_mac_REGISTRY
from ..learners import REGISTRY as le_REGISTRY
from ..learners.factored import REGISTRY as factored_le_REGISTRY
from ..runners import REGISTRY as r_REGISTRY


def build_sim(
    args: SN,
    logger,
    agent_state_dict: Optional[dict] = None,
) -> tuple[SN, Any, ReplayBuffer, Any]:
    # update env args with comms values to be used in HLMDP model
    if hasattr(args, "message_budget_per_agent") and args.env_args.get("hierarchical") is not None:
        args.env_args["message_budget_per_agent"] = args.message_budget_per_agent

    args, runner, env_info, scheme, groups, preprocess = _build_env_spec(args, logger)

    buffer = _build_replay_buffer(
        scheme=scheme,
        groups=groups,
        env_info=env_info,
        preprocess=preprocess,
        args=args,
    )

    # build controller and learner
    # buffer.scheme has preprocess in it, needed to init these objects
    if hasattr(args, "factored_hierarchical_policy"):
        mac = factored_mac_REGISTRY[args.factored_mac](buffer.scheme, groups, args)
        learner = factored_le_REGISTRY[args.factored_learner](
            mac, buffer.scheme, logger, args
        )
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()
        mac.cuda()

    if agent_state_dict is not None:
        mac.load_models(agent_state_dict)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    return args, runner, buffer, learner


def _build_env_spec(args: SN, logger):
    """Create runner, query env info, and build base scheme/groups/preprocess."""
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}

    if hasattr(args, "factored_hierarchical_policy"):
        scheme["hl_state"] = {"vshape": env_info["hl_state_shape"]}

    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    return args, runner, env_info, scheme, groups, preprocess


def _build_replay_buffer(
    scheme: dict,
    groups: dict,
    env_info: dict,
    preprocess: dict,
    args: SN,
) -> ReplayBuffer:

    device = "cpu" if args.buffer_cpu_only else args.device

    return ReplayBuffer(
        scheme=scheme,
        groups=groups,
        buffer_size=args.buffer_size,
        max_seq_length=env_info["episode_limit"] + 1,
        preprocess=preprocess,
        device=device,
    )
