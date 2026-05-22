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
