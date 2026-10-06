import importlib
import os
import warnings

import jax
import jax.numpy as jnp
import pytest
import jnkepler
from jnkepler._cpu_performance import _cpu_performance_message


LEGACY = "--xla_cpu_use_thunk_runtime=false"
THRESHOLD = "xla_cpu_small_while_loop_byte_threshold"
MODERN = f"--xla_backend_extra_options={THRESHOLD}=65536"


@pytest.mark.parametrize("version_info, flags, setting", [
    ((0, 6, 2), None, LEGACY),
    ((0, 6, 2), "", LEGACY),
    ((0, 6, 2), "--xla_cpu_use_thunk_runtime=true", LEGACY),
    ((0, 6, 2), LEGACY, None),
    ((0, 6, 2), f"--xla_force_host_platform_device_count=2 {LEGACY}", None),
    ((0, 6, 2), "--xla_cpu_use_thunk_runtime=0", None),
    ((0, 6, 2), "--xla_cpu_use_thunk_runtime=falsehood", LEGACY),
    ((0, 6, 2), "--other=--xla_cpu_use_thunk_runtime=false", LEGACY),
    ((0, 6, 2), f"{LEGACY} --xla_cpu_use_thunk_runtime=true", LEGACY),
    ((0, 6, 2), MODERN, LEGACY),
    ((0, 7, 0), "", MODERN),
    ((0, 7, 0), MODERN, None),
    ((0, 10, 2), "", MODERN),
    ((0, 11, 2), None, MODERN),
    ((0, 11, 2), LEGACY, MODERN),
    ((0, 11, 2), f"{LEGACY} {MODERN}", None),
    ((0, 11, 2), f"--xla_cpu_multi_thread_eigen=false {MODERN}", None),
    ((0, 11, 2), f'--xla_backend_extra_options="other=1,{THRESHOLD}=4096"', None),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=131072", None),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=0", None),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=-1", None),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=+4096", None),
    ((0, 11, 2), f"--xla_backend_extra_options=other_{THRESHOLD}=65536", MODERN),
    ((0, 11, 2), f"--other=xla_backend_extra_options={THRESHOLD}=65536", MODERN),
    ((0, 11, 2), f"--{THRESHOLD}=65536", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=garbage", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=65536suffix", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=65536.0", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}={2**63}", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}={'9'*5000}", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=4096,{THRESHOLD}=bad", MODERN),
    ((0, 11, 2), f"--xla_backend_extra_options={THRESHOLD}=bad,{THRESHOLD}=4096", None),
    ((0, 11, 2), '"unterminated', MODERN),
    ((0, 11, 2), 123, MODERN),
    ((0, 11, 2), [MODERN], MODERN),
    ((1, 0, 0), "", MODERN),
])
def test_cpu_advice(version_info, flags, setting):
    before = dict(os.environ)
    message = _cpu_performance_message(version_info, flags)
    assert dict(os.environ) == before
    if setting is None:
        assert message is None
    else:
        assert f'XLA_FLAGS="{setting}"' in message
        assert "CPU performance" in message
        assert "before importing JAX or jnkepler" in message
        assert "README" in message
        assert "outside" not in message
        assert "dependency requirement" not in message


@pytest.mark.parametrize("version_info", [None, (), (0,), "garbage", ("0", "11"), (-1, 7)])
def test_unknown_version_does_not_break_advice(version_info):
    assert _cpu_performance_message(version_info, MODERN) is None


@pytest.mark.parametrize("version, version_info, flags, expect_warning", [
    ("0.6.2", (0, 6, 2), None, True),
    ("0.6.2", (0, 6, 2), LEGACY, False),
    ("0.7.0", (0, 7, 0), None, True),
    ("0.7.0", (0, 7, 0), MODERN, False),
    ("0.7.0.dev20261006", (0, 7, 0), MODERN, False),
    ("0.11.2+cpu", (0, 11, 2), MODERN, False),
    ("0.11.2", (0, 11, 2), LEGACY, True),
    ("0.11.2", (0, 11, 2), '"unterminated', True),
])
def test_import_advice_has_no_setting_or_backend_side_effects(
    monkeypatch, version, version_info, flags, expect_warning,
):
    monkeypatch.setattr(jax, "__version__", version)
    monkeypatch.setattr(jax, "__version_info__", version_info)
    if flags is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", flags)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU advice must not initialize a backend or change JAX settings")

    for name in ("devices", "local_devices", "device_count", "local_device_count", "default_backend"):
        monkeypatch.setattr(jax, name, forbidden)
    monkeypatch.setattr(jax.config, "update", forbidden)
    monkeypatch.setattr(jnp, "array", forbidden)
    monkeypatch.setattr(jnp, "asarray", forbidden)
    before = dict(os.environ)
    # Scientific submodules are cached; this exercises the actual initializer's advice.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(jnkepler)
    assert dict(os.environ) == before
    assert len(caught) == int(expect_warning)
    if expect_warning:
        assert caught[0].category is UserWarning
        assert str(caught[0].message) == _cpu_performance_message(version_info, flags)
