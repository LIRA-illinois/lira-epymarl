from collections import namedtuple


def convert(dictionary) -> convert.GenericDict:
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
