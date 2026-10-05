import importlib
import warnings

import jax
import pytest
import jnkepler


def _import_warnings(monkeypatch, version, version_info, xla_flags):
    monkeypatch.setattr(jax, "__version__", version)
    monkeypatch.setattr(jax, "__version_info__", version_info)
    if xla_flags is None:
        monkeypatch.delenv("XLA_FLAGS", raising=False)
    else:
        monkeypatch.setenv("XLA_FLAGS", xla_flags)

    # Reload the initializer while keeping the scientific submodules cached.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(jnkepler)
    return caught


@pytest.mark.parametrize("xla_flags", [
    None,
    "--xla_cpu_use_thunk_runtime=true",
    "--xla_force_host_platform_device_count=2",
])
def test_old_jax_recommends_legacy_runtime(monkeypatch, xla_flags):
    caught = _import_warnings(monkeypatch, "0.6.2", (0, 6, 2), xla_flags)

    assert len(caught) == 1
    assert caught[0].category is UserWarning
    message = str(caught[0].message)
    assert "best tested CPU performance" in message
    assert '--xla_cpu_use_thunk_runtime=false' in message
    assert "before importing JAX" in message
    assert "outside" not in message


@pytest.mark.parametrize("xla_flags", [
    "--xla_cpu_use_thunk_runtime=false",
    "--xla_force_host_platform_device_count=2 --xla_cpu_use_thunk_runtime=false",
])
def test_old_jax_with_legacy_runtime_is_silent(monkeypatch, xla_flags):
    caught = _import_warnings(monkeypatch, "0.6.2", (0, 6, 2), xla_flags)

    assert not caught


@pytest.mark.parametrize("version, version_info", [
    ("0.7.0", (0, 7, 0)),
    ("0.11.2", (0, 11, 2)),
    ("0.7.0.dev20261005", (0, 7, 0)),
    ("0.11.2+cpu", (0, 11, 2)),
])
@pytest.mark.parametrize("xla_flags", [
    None,
    "--xla_cpu_use_thunk_runtime=false",
    "--xla_backend_extra_options=xla_cpu_small_while_loop_byte_threshold=65536",
])
def test_new_jax_points_to_current_guidance(
    monkeypatch, version, version_info, xla_flags,
):
    caught = _import_warnings(monkeypatch, version, version_info, xla_flags)

    assert len(caught) == 1
    assert caught[0].category is UserWarning
    message = str(caught[0].message)
    assert f"JAX {version}" in message
    assert "outside" in message
    assert "jax<0.7 dependency requirement" in message
    assert "substantially slower" in message
    assert "default XLA CPU runtime" in message
    assert "README" in message
    assert "issue #30 (https://github.com/kemasuda/jnkepler/issues/30)" in message
    assert "--xla_cpu_use_thunk_runtime=false" not in message
    assert "65536" not in message
