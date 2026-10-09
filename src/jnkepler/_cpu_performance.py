"""CPU performance advice without initializing JAX or changing settings."""

import re
import shlex


def _flag_values(xla_flags):
    if not isinstance(xla_flags, str):
        return {}
    try:
        tokens = shlex.split(xla_flags)
    except ValueError:
        return {}
    values = {}
    for token in tokens:
        name, separator, value = token.partition("=")
        if separator and name.startswith("--"):
            values[name[2:]] = value
    return values


def _explicit_cpu_threshold(flags):
    # Backend options are comma-separated key=value entries, not top-level flags.
    options = {}
    for entry in flags.get("xla_backend_extra_options", "").split(","):
        key, separator, value = entry.partition("=")
        if separator:
            options[key] = value
    value = options.get("xla_cpu_small_while_loop_byte_threshold", "")
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        return False
    try:
        return -(2**63) <= int(value) < 2**63
    except ValueError:
        return False


def _cpu_performance_message(version_info, xla_flags):
    """Return advice, or None for an explicit setting / unknown JAX version.

    Recognizing an environment setting does not verify backend application.
    """
    try:
        major, minor = version_info[:2]
    except (TypeError, ValueError):
        return None
    if any(type(part) is not int or part < 0 for part in (major, minor)):
        return None

    flags = _flag_values(xla_flags)
    if (major, minor) < (0, 7):
        if flags.get("xla_cpu_use_thunk_runtime", "").lower() in ("false", "0"):
            return None
        setting = "--xla_cpu_use_thunk_runtime=false"
        series = "JAX < 0.7"
    else:
        if _explicit_cpu_threshold(flags):
            return None
        setting = (
            "--xla_backend_extra_options="
            "xla_cpu_small_while_loop_byte_threshold=65536"
        )
        series = "JAX >= 0.7"
    return (
        f"For CPU performance with {series}, set XLA_FLAGS\n"
        "before importing JAX or jnkepler.\n"
        "\n"
        "In a shell:\n"
        f'    export XLA_FLAGS="{setting}"\n'
        "\n"
        "Or in Python:\n"
        "    import os\n"
        f'    os.environ["XLA_FLAGS"] = "{setting}"\n'
        "    import jax\n"
        "    import jnkepler\n"
        "\n"
        "See the README CPU performance note for details."
    )
