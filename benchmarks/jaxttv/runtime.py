r"""Synchronized, steady-state JaxTTV chi-square benchmark for Kepler-51.

Run in a fresh process from the repository root to benchmark this checkout::

    PYTHONPATH=src python benchmarks/jaxttv/runtime.py --warmup 3 --repeat 20

Optionally compare forward-mode gradients of the same scalar chi-square::

    PYTHONPATH=src python benchmarks/jaxttv/runtime.py --jacfwd

Without PYTHONPATH=src, the script benchmarks the installed jnkepler package.
Set XLA_FLAGS externally before starting the process; this script only reports it.
The canonical read_testdata_tc() loader supplies the complete problem setup.
Compilation, data loading, and sanity checks are outside measured regions.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jaxlib
import numpy as np

import jnkepler
from jnkepler.tests import read_testdata_tc


def synchronize(value):
    """Wait for every asynchronous array in an arbitrary result pytree."""
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def benchmark(function, *args, warmup=3, repeat=20):
    """Return one elapsed time in seconds per fully synchronized invocation."""
    if warmup < 1 or repeat < 1:
        raise ValueError("warmup and repeat must both be at least 1")

    for _ in range(warmup):
        synchronize(function(*args))

    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function(*args)
        synchronize(result)
        elapsed.append(time.perf_counter() - start)
    return elapsed


def check_finite(name, value):
    """Check all output leaves on the host, outside the timed invocations."""
    synchronize(value)
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or any(not np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise RuntimeError(f"{name} returned an empty pytree or non-finite values")


def report_environment(jttv, pdic, args):
    print("JaxTTV steady-state runtime benchmark", flush=True)
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine architecture: {platform.machine()}")
    print(f"JAX: {jax.__version__}")
    print(f"jaxlib: {jaxlib.__version__}")
    print(f"jnkepler: {jnkepler.__version__}")
    print(f"jnkepler package: {jnkepler.__file__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"jax_enable_x64: {jax.config.jax_enable_x64}")
    flags = os.environ.get("XLA_FLAGS")
    print(f"XLA_FLAGS: {flags!r}" if flags is not None else "XLA_FLAGS: <unset>")
    print(f"Problem: kep51 (read_testdata_tc), planets={jttv.nplanet}")
    print(f"Initial periods (days): {jttv.p_init.tolist()}")
    print(
        f"Integration: dt={jttv.dt} days, "
        f"t_start={jttv.t_start}, t_end={jttv.t_end}"
    )
    print(f"Integration time samples: {len(jttv.times)}")
    print(f"Observed transits per planet: {[len(tc) for tc in jttv.tcobs]}")
    print(f"Total modeled transit times: {len(jttv.tcobs_flatten)}")
    print(
        f"Transit method: {jttv.transit_time_method}, "
        f"nitr_kepler={jttv.nitr_kepler}, nitr_transit={jttv.nitr_transit}"
    )
    print(f"Parameter keys: {', '.join(sorted(pdic))}")
    print(f"Parameter dtypes: {', '.join(sorted({str(v.dtype) for v in pdic.values()}))}")
    print(f"Warmup calls per function: {args.warmup}")
    print(f"Measured repeats per function: {args.repeat}")
    print(f"Forward-mode diagnostic (--jacfwd): {args.jacfwd}")
    print("Compiling and warming up each function before timing...", flush=True)


def report_timings(name, elapsed):
    mean = statistics.mean(elapsed)
    median = statistics.median(elapsed)
    print(
        f"{name}: mean={mean * 1000:.3f} ms, "
        f"median={median * 1000:.3f} ms, "
        f"stddev={statistics.pstdev(elapsed) * 1000:.3f} ms "
        f"(population), repeats={len(elapsed)}"
    )
    return mean, median


def result_record(jttv, pdic, args, chi2, timings, ratios):
    """Build a JSON-compatible record outside the measured regions."""
    return {
        "schema_version": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "jax_version": jax.__version__,
            "jaxlib_version": jaxlib.__version__,
            "jnkepler_version": jnkepler.__version__,
            "jnkepler_package_path": jnkepler.__file__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            # Preserve unset (null), set-but-empty (""), and exact flag text.
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        },
        "problem": {
            "name": "kep51",
            "loader": "jnkepler.tests.read_testdata_tc",
            "number_of_planets": int(jttv.nplanet),
            "initial_periods_days": jttv.p_init.tolist(),
            "integration_timestep_days": float(jttv.dt),
            "integration_start": float(jttv.t_start),
            "integration_end": float(jttv.t_end),
            "integration_time_samples": len(jttv.times),
            "observed_transits_per_planet": [len(tc) for tc in jttv.tcobs],
            "number_of_modeled_transit_times": len(jttv.tcobs_flatten),
            "transit_method": jttv.transit_time_method,
            "nitr_kepler": int(jttv.nitr_kepler),
            "nitr_transit": int(jttv.nitr_transit),
            "parameter_keys": sorted(pdic),
            "parameter_dtypes": sorted({str(v.dtype) for v in pdic.values()}),
            "chisquare_definition": (
                "sum(((tcobs_flatten - tc_model) / errorobs_flatten) ** 2)"
            ),
        },
        "benchmark_configuration": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "jacfwd": args.jacfwd,
            "standard_deviation": "population",
            "runtime_ratio_statistics": ["mean", "median"],
        },
        "chi_square": float(chi2),
        "benchmarks": {
            name: {
                "elapsed_seconds": [float(value) for value in elapsed],
                "mean_seconds": statistics.mean(elapsed),
                "median_seconds": statistics.median(elapsed),
                "stddev_seconds": statistics.pstdev(elapsed),
                "repeats": len(elapsed),
            }
            for name, elapsed in timings.items()
        },
        "runtime_ratios": ratios,
    }


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--warmup", type=positive_int, default=3,
        help="synchronized warmup calls per function (default: 3)",
    )
    parser.add_argument(
        "--repeat", type=positive_int, default=20,
        help="synchronized measured calls per function (default: 20)",
    )
    parser.add_argument(
        "--jacfwd", action="store_true",
        help="also benchmark the forward-mode gradient of scalar chi-square",
    )
    parser.add_argument(
        "--output", type=Path, metavar="PATH",
        help="save benchmark results as JSON, creating parent directories as needed",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    jttv, _, _, pdic = read_testdata_tc()
    synchronize((jttv.times, pdic))
    report_environment(jttv, pdic, args)

    def transit_times(params):
        return jttv.get_transit_times_obs(params)[0]

    def chisquare(params):
        tc_model = transit_times(params)
        residual = (jttv.tcobs_flatten - tc_model) / jttv.errorobs_flatten
        return jnp.sum(residual**2)

    value_fn = jax.jit(chisquare)
    reverse_fn = jax.jit(jax.value_and_grad(chisquare))
    value_elapsed = benchmark(
        value_fn, pdic, warmup=args.warmup, repeat=args.repeat
    )
    reverse_elapsed = benchmark(
        reverse_fn, pdic, warmup=args.warmup, repeat=args.repeat
    )

    if args.jacfwd:
        def chisquare_with_aux(params):
            chi2 = chisquare(params)
            return chi2, chi2

        forward_gradient = jax.jacfwd(chisquare_with_aux, has_aux=True)

        @jax.jit
        def forward_fn(params):
            gradient, chi2 = forward_gradient(params)
            return chi2, gradient

        forward_elapsed = benchmark(
            forward_fn, pdic, warmup=args.warmup, repeat=args.repeat
        )

    # Re-evaluate and validate outside every measured region.
    chi2 = value_fn(pdic)
    reverse_chi2, reverse_gradient = reverse_fn(pdic)
    check_finite("Chi-square", chi2)
    check_finite("Reverse-mode value and gradient", (reverse_chi2, reverse_gradient))
    np.testing.assert_allclose(reverse_chi2, chi2, rtol=1e-8, atol=1e-8)

    if args.jacfwd:
        forward_chi2, forward_gradient_value = forward_fn(pdic)
        check_finite(
            "Forward-mode value and gradient", (forward_chi2, forward_gradient_value)
        )
        np.testing.assert_allclose(forward_chi2, chi2, rtol=1e-8, atol=1e-8)
        if jax.tree_util.tree_structure(reverse_gradient) != jax.tree_util.tree_structure(
            forward_gradient_value
        ):
            raise RuntimeError("Forward- and reverse-mode gradient pytrees differ")
        reverse_flat, _ = ravel_pytree(reverse_gradient)
        forward_flat, _ = ravel_pytree(forward_gradient_value)
        reverse_flat = np.asarray(reverse_flat)
        forward_flat = np.asarray(forward_flat)
        max_gradient_difference = np.max(np.abs(forward_flat - reverse_flat))
        gradient_scale = max(
            np.max(np.abs(reverse_flat)), np.max(np.abs(forward_flat))
        )
        normalized_gradient_difference = (
            max_gradient_difference / gradient_scale if gradient_scale else 0.0
        )

    modes = "reverse- and forward-mode" if args.jacfwd else "reverse-mode"
    print(f"Sanity checks: chi-square and all {modes} gradient leaves are finite")
    print("Chi-square values agree across paths (rtol=1e-8, atol=1e-8)")
    print(f"Chi-square: {float(chi2):.12g}")
    value_mean, value_median = report_timings("chi-square value", value_elapsed)
    reverse_mean, reverse_median = report_timings(
        "chi-square value_and_grad (reverse)", reverse_elapsed
    )
    print(f"reverse gradient / value: {reverse_mean / value_mean:.3f}x (ratio of means)")
    print(
        f"reverse gradient / value: {reverse_median / value_median:.3f}x (ratio of medians)"
    )
    if args.jacfwd:
        forward_mean, forward_median = report_timings(
            "chi-square value + grad (jacfwd)", forward_elapsed
        )
        print(f"forward gradient / value: {forward_mean / value_mean:.3f}x (ratio of means)")
        print(
            f"forward gradient / value: {forward_median / value_median:.3f}x (ratio of medians)"
        )
        print(
            "forward gradient / reverse gradient: "
            f"{forward_mean / reverse_mean:.3f}x (ratio of means)"
        )
        print(
            "forward gradient / reverse gradient: "
            f"{forward_median / reverse_median:.3f}x (ratio of medians)"
        )
        print(
            "max |grad_chi2(jacfwd) - grad_chi2(reverse)|: "
            f"{max_gradient_difference:.6e}"
        )
        print(
            "max gradient difference / max gradient magnitude: "
            f"{normalized_gradient_difference:.6e}"
        )

    if args.output is not None:
        timings = {
            "chi_square_value": value_elapsed,
            "chi_square_value_and_grad_reverse": reverse_elapsed,
        }
        ratios = {
            "reverse_gradient_over_value": {
                "ratio_of_means": reverse_mean / value_mean,
                "ratio_of_medians": reverse_median / value_median,
            }
        }
        if args.jacfwd:
            timings["chi_square_value_and_grad_jacfwd"] = forward_elapsed
            ratios.update({
                "forward_gradient_over_value": {
                    "ratio_of_means": forward_mean / value_mean,
                    "ratio_of_medians": forward_median / value_median,
                },
                "forward_gradient_over_reverse_gradient": {
                    "ratio_of_means": forward_mean / reverse_mean,
                    "ratio_of_medians": forward_median / reverse_median,
                },
            })
        result = result_record(jttv, pdic, args, chi2, timings, ratios)
        if args.jacfwd:
            result["gradient_agreement"] = {
                "max_absolute_difference": float(max_gradient_difference),
                "max_gradient_magnitude": float(gradient_scale),
                "normalized_max_difference": float(normalized_gradient_difference),
                "chi_square_reverse": float(reverse_chi2),
                "chi_square_forward": float(forward_chi2),
                "chi_square_values_agree": True,
                "chi_square_rtol": 1e-8,
                "chi_square_atol": 1e-8,
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"JSON output: {args.output}")


if __name__ == "__main__":
    main()
