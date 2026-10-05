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

if jax.__version_info__ >= (0, 7):
    warnings.warn(
        f'JAX {jax.__version__} is outside jnkepler\'s current jax<0.7 dependency '
        'requirement. Some jnkepler CPU workloads are substantially slower with '
        'the default XLA CPU runtime. See the README CPU performance note and '
        'issue #30 (https://github.com/kemasuda/jnkepler/issues/30) for current guidance.',
        UserWarning,
    )
elif "--xla_cpu_use_thunk_runtime=false" not in os.environ.get("XLA_FLAGS", ""):
    warnings.warn(
        'For best tested CPU performance with JAX < 0.7, set '
        'XLA_FLAGS="--xla_cpu_use_thunk_runtime=false" before importing JAX '
        'to use the legacy CPU runtime. See the README CPU performance note '
        'for details.',
        UserWarning,
    )
