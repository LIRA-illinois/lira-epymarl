from numbers import Integral
from types import SimpleNamespace

import numpy as np


def policy_context_dim(args: SimpleNamespace) -> int:
    context_dim = int(getattr(args, "condition_on_msg_budget", False))
    if getattr(args, "condition_on_subtask", False):
        contexts = getattr(args, "subtask_contexts", None)
        if not contexts:
            raise ValueError(
                "subtask_contexts must be configured when condition_on_subtask is enabled."
            )
        normalized_contexts = [_normalize_subtask(context) for context in contexts]
        if len(normalized_contexts) != len(set(normalized_contexts)):
            raise ValueError("subtask_contexts must not contain duplicates.")
        context_dim += len(normalized_contexts)
    return context_dim


def build_policy_context(
    args: SimpleNamespace, reset_options: dict | None = None
) -> np.ndarray | None:
    if policy_context_dim(args) == 0:
        return None

    reset_options = reset_options or {}
    context: list[float] = []

    if getattr(args, "condition_on_msg_budget", False):
        budget = reset_options.get("msg_budget_per_agent")
        if budget is None:
            configured_budgets = getattr(args, "msg_budget_per_agent", None)
            if configured_budgets is None or len(configured_budgets) != 1:
                raise ValueError(
                    "A single msg_budget_per_agent must be specified for each "
                    "conditioned rollout."
                )
            budget = configured_budgets[0]

        max_budget = args.n_agents - 1
        if not isinstance(budget, Integral):
            raise ValueError(
                f"msg_budget_per_agent must be an integer from 0 to {max_budget}."
            )
        budget = int(budget)
        if not 0 <= budget <= max_budget:
            raise ValueError(
                f"msg_budget_per_agent must be an integer from 0 to {max_budget}."
            )
        context.append(float(budget / max_budget) if max_budget else 0.0)

    if getattr(args, "condition_on_subtask", False):
        contexts = [_normalize_subtask(value) for value in args.subtask_contexts]
        subtask = reset_options.get("hl_task", reset_options.get("subtask_idx"))
        if subtask is None:
            subtask = getattr(args, "hl_task", None)
        if subtask is None:
            subtask = args.env_args.get("subtask_idx")
        if subtask is None:
            raise ValueError(
                "Each conditioned rollout must specify hl_task or subtask_idx."
            )

        subtask = _normalize_subtask(subtask)
        try:
            subtask_index = contexts.index(subtask)
        except ValueError as error:
            raise ValueError(
                f"Subtask {subtask!r} is not listed in subtask_contexts."
            ) from error

        context.extend(
            float(index == subtask_index) for index in range(len(contexts))
        )

    return np.asarray(context, dtype=np.float32)


def _normalize_subtask(value: int | list | tuple) -> int | tuple[int, int]:
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(state, Integral) for state in value
    ):
        return int(value[0]), int(value[1])
    raise ValueError(
        "Subtask contexts must be integer indices or [from_state, to_state] pairs."
    )
