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

if "--xla_cpu_use_thunk_runtime=false" not in os.environ.get("XLA_FLAGS", ""):
    warnings.warn(
        'For best CPU performance on JAX >= 0.4.32, please disable the thunk runtime.\n'
        '\n'
        'You can do this by setting the environment variable BEFORE importing jax:\n'
        '    export XLA_FLAGS="--xla_cpu_use_thunk_runtime=false"\n'
        '\n'
        'Or inside Python, add these lines before importing jax:\n'
        '    import os\n'
        '    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"\n'
        '    import jax\n'
        '\n'
        'Without this, CPU execution may be significantly slower, especially when computing gradients.',
        UserWarning,
    )
