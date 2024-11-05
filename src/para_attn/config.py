import os  # noqa: C101
import sys

# import torch


_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
}


class attention:
    allow_reduced_precision_compute = os.getenv("PARA_ATTN_ALLOW_REDUCED_PRECISION_COMPUTE") == "1"

    fast_math = os.getenv("PARA_ATTN_FAST_MATH") == "1"


try:
    from torch.utils._config_module import install_config_module
except ImportError:
    # torch<2.2.0
    from torch._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
