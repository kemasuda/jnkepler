__all__ = []

__uri__ = "https://github.com/kemasuda/jnkepler"
__author__ = "Kento Masuda"
__email__ = "kmasuda@ess.sci.osaka-u.ac.jp"
__license__ = "MIT"
__description__ = "JAX code for modeling nearly-Keplerian orbits"

from .jnkepler_version import __version__
from . import jaxttv
from . import nbodytransit
from . import nbodyrv
from . import infer
from . import information
from . import keplerian

import os
import warnings
import jax

from ._cpu_performance import _cpu_performance_message

_cpu_advice = _cpu_performance_message(
    getattr(jax, "__version_info__", None), os.environ.get("XLA_FLAGS", ""),
)
if _cpu_advice:
    warnings.warn(_cpu_advice, UserWarning)
