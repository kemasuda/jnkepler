import inspect
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


def _load_infer_module():
    src_root = Path(__file__).resolve().parents[3] / "src" / "jnkepler"
    package_name = "_jnkepler_testpkg"
    jaxttv_name = f"{package_name}.jaxttv"
    infer_name = f"{jaxttv_name}.infer"

    package = types.ModuleType(package_name)
    package.__path__ = [str(src_root)]
    sys.modules.setdefault(package_name, package)

    jaxttv_package = types.ModuleType(jaxttv_name)
    jaxttv_package.__path__ = [str(src_root / "jaxttv")]
    sys.modules.setdefault(jaxttv_name, jaxttv_package)

    spec = importlib.util.spec_from_file_location(
        infer_name,
        src_root / "jaxttv" / "infer.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[infer_name] = module

    missing = object()
    old_scipy_modules = {
        name: sys.modules.get(name, missing)
        for name in ("scipy", "scipy.special", "scipy.optimize")
    }

    scipy_module = types.ModuleType("scipy")
    scipy_special = types.ModuleType("scipy.special")
    scipy_optimize = types.ModuleType("scipy.optimize")
    scipy_special.gammaln = math.lgamma

    def _least_squares_placeholder(*args, **kwargs):
        raise AssertionError("least_squares should be monkeypatched in this test")

    scipy_optimize.least_squares = _least_squares_placeholder
    scipy_module.special = scipy_special
    scipy_module.optimize = scipy_optimize

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.special"] = scipy_special
    sys.modules["scipy.optimize"] = scipy_optimize

    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in old_scipy_modules.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    return module


infer = _load_infer_module()


def _param_bounds():
    return {
        "period": [np.array([1.0]), np.array([2.0])],
        "ecosw": [np.array([-0.1]), np.array([0.1])],
        "esinw": [np.array([-0.1]), np.array([0.1])],
        "cosi": [np.array([-0.1]), np.array([0.1])],
        "lnode": [np.array([-0.1]), np.array([0.1])],
        "tic": [np.array([0.0]), np.array([1.0])],
        "lnpmass": [np.array([-20.0]), np.array([-10.0])],
    }


def _fake_jttv():
    return SimpleNamespace(
        transit_time_method="fast",
        nplanet=1,
        tcobs_flatten=np.array([0.0]),
        errorobs_flatten=np.array([1.0]),
        tcobs=[np.array([0.0])],
    )


def _run_stubbed_lsq(monkeypatch, **kwargs):
    captured = {}

    def fake_get_cached_residual_functions(
        jttv,
        npl,
        keys,
        transit_orbit_idx=None,
        transit_time_method=None,
        jac=False,
        diff_mode="fwd",
    ):
        captured["transit_time_method"] = transit_time_method
        captured["diff_mode"] = diff_mode

        def resid(p):
            return jnp.array([p[0] - 1.5])

        def jac_resid(p):
            return jnp.zeros((1, p.shape[0])).at[0, 0].set(1.0)

        return {
            "resid": resid,
            "jac_resid": jac_resid if jac else None,
            "is_warmed": False,
        }

    def fake_least_squares(
        fun,
        p0,
        jac,
        bounds,
        method,
        loss,
        f_scale,
        max_nfev,
    ):
        captured["least_squares_jac"] = jac
        return SimpleNamespace(x=np.asarray(p0), cost=0.0, nfev=1)

    def fake_get_transit_times_obs_with_method(*args, **kwargs):
        return jnp.array([0.0]), jnp.array(0.0)

    monkeypatch.setattr(
        infer,
        "_get_cached_residual_functions",
        fake_get_cached_residual_functions,
    )
    monkeypatch.setattr(
        infer,
        "_get_transit_times_obs_with_method",
        fake_get_transit_times_obs_with_method,
    )
    monkeypatch.setattr(infer, "least_squares", fake_least_squares)

    jttv = _fake_jttv()
    infer.ttv_optim_least_squares(
        jttv,
        _param_bounds(),
        jac=True,
        plot=False,
        **kwargs,
    )
    captured["jttv_transit_time_method"] = jttv.transit_time_method
    return captured


def test_ttv_optim_least_squares_defaults_to_newton_and_jacfwd():
    sig = inspect.signature(infer.ttv_optim_least_squares)

    assert sig.parameters["transit_time_method"].default == "newton"
    assert sig.parameters["diff_mode"].default == "fwd"


def test_ttv_optim_least_squares_default_modes(monkeypatch):
    captured = _run_stubbed_lsq(monkeypatch)

    assert captured["transit_time_method"] == "newton"
    assert captured["diff_mode"] == "fwd"
    assert captured["jttv_transit_time_method"] == "fast"


def test_ttv_optim_least_squares_explicit_modes_are_preserved(monkeypatch):
    captured = _run_stubbed_lsq(
        monkeypatch,
        transit_time_method="fast",
        diff_mode="rev",
    )

    assert captured["transit_time_method"] == "fast"
    assert captured["diff_mode"] == "rev"


def test_ttv_optim_least_squares_explicit_auto_mode_is_preserved(monkeypatch):
    captured = _run_stubbed_lsq(
        monkeypatch,
        transit_time_method="newton",
        diff_mode="auto",
    )

    assert captured["transit_time_method"] == "newton"
    assert captured["diff_mode"] == "rev"
