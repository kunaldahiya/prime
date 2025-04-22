from argparse import Namespace
from deepxml.libs.utils import save_predictions


def filter_params(args: Namespace, prefix: str) -> Namespace:
    """
    Filter the arguments as per a prefix from a given namespace
    """
    out = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return Namespace(**out)
