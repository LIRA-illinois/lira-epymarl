from typing import Callable
import ast
import sys


def mp_kwargs_wrapper(kwargs: dict):
    """helper method to parallelize with mp.pool.map() a list of kwargs dicts
    https://stackoverflow.com/questions/49901824/python-pool-starmap-with-list-of-dictionaries

    Parameters
    ----------
    kwargs : dict
        kwargs dict with a "function" key with value as the function to
        be executed in the using mp.Pool.map() with the remaining items in the dict as kwargs
    """
    fn = kwargs.pop("function")
    return fn(**kwargs)


def string_inputs_to_list(config: dict, key: str, output_type: Callable) -> dict:
    """Converts space-delimited string list to list of "type" for the given key in the config dictionary."""
    if isinstance(config[key][0], output_type):
        return config

    if isinstance(config[key], str):
        # config[key] takes this form in main.py after the run command is formatted in grid_search_experiment
        eval_str = config[key].split("=")[0][1:-1]
    elif isinstance(config[key], list):
        # in grid_search_experiment, this is a list with a single space-delimited string with the parameter values
        eval_str = config[key][0]

    param_values: list[float] = [output_type(x) for x in eval_str.split(" ")]
    config[key] = param_values

    return config


def is_debugger_active() -> bool:
    # Returns True if a debugger trace is actively running
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

# argument parsing, taken from the Sacred library
def get_config_updates(updates):
    """
    Parse the UPDATES given on the commandline.

    Parameters
    ----------
        updates (list[str]):
            list of update-strings of the form NAME=LITERAL or just NAME.

    Returns
    -------
        (dict, list):
            Config updates and named configs to use

    """
    config_updates = {}
    named_configs = []
    if not updates:
        return config_updates, named_configs
    for upd in updates:
        if upd == "":
            continue
        path, sep, value = upd.partition("=")
        if sep == "=":
            path = path.strip()  # get rid of surrounding whitespace
            value = value.strip()  # get rid of surrounding whitespace
            set_by_dotted_path(config_updates, path, _convert_value(value))
        else:
            named_configs.append(path)
    return config_updates, named_configs


def set_by_dotted_path(d, path, value) -> None:
    """
    Set an entry in a nested dict using a dotted path.

    Will create dictionaries as needed.

    Examples
    --------
    >>> d = {'foo': {'bar': 7}}
    >>> set_by_dotted_path(d, 'foo.bar', 10)
    >>> d
    {'foo': {'bar': 10}}
    >>> set_by_dotted_path(d, 'foo.d.baz', 3)
    >>> d
    {'foo': {'bar': 10, 'd': {'baz': 3}}}

    """
    split_path = path.split(".")
    current_option = d
    for p in split_path[:-1]:
        if p not in current_option:
            current_option[p] = dict()
        current_option = current_option[p]
    current_option[split_path[-1]] = value


def _convert_value(value):
    """Parse string as python literal if possible and fallback to string."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # use as string if nothing else worked
        return value
