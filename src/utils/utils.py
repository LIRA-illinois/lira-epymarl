import ast


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


def set_by_dotted_path(d, path, value):
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
