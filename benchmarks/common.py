"""Shared harness for the benchmark scripts in this directory.

Import this before jax in every benchmark script. It sets the XLA flags,
which only take effect ahead of the first jax import, and enables float64
so that timing runs match the precision jnkepler uses in practice.
"""
import os

# Apply the thunk-runtime flag jnkepler recommends on import, rather than
# only warning about it: it is worth roughly a factor of three here, more
# than anything these scripts measure. Append so existing flags survive,
# and echo the result, since runs with and without it are not comparable.
_THUNK_FLAG = "--xla_cpu_use_thunk_runtime=false"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_cpu_use_thunk_runtime" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} {_THUNK_FLAG}".strip()
print(f"XLA_FLAGS = {os.environ['XLA_FLAGS']}")

import jax
jax.config.update('jax_enable_x64', True)

import time
import statistics
import numpy as np
import jax.numpy as jnp
from jax import checkpoint
from jax.lax import scan
from jnkepler.jaxttv.symplectic import (
    integrate_xv, _compute_ki, real_to_mapTO, kepler_step, nbody_kicks)
from jnkepler.jaxttv.findtransit import find_transit_times_fast
from jnkepler.jaxttv.utils import initialize_jacobi_xv
from jnkepler.jaxttv.conversion import G


def timeit(fn, args, N=50, warmup=10):
    """Mean wall-clock ms per call, after warmup."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    return _timed_mean(fn, args, N)


def _timed_mean(fn, args, N):
    """Mean wall-clock ms per call, assuming the function is already warm."""
    t0 = time.perf_counter()
    for _ in range(N):
        jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / N * 1000


def timeit_paired(fn_a, args_a, fn_b, args_b, N=30, warmup=6, repeat=20):
    """Time two callables alternately, returning a list of (a, b) ms pairs.

    Each repeat measures both callables back to back, so slow drift such as
    thermal throttling stays common to the pair and largely cancels in their
    ratio. Compilation and warmup happen once, which makes extra repeats
    cost only the timed calls themselves.

    Repeats within a single process do not see variation between processes,
    so treat the resulting spread as a lower bound on run-to-run scatter.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn_a(*args_a))
        jax.block_until_ready(fn_b(*args_b))
    return [(_timed_mean(fn_a, args_a, N), _timed_mean(fn_b, args_b, N))
            for _ in range(repeat)]


def median_iqr(xs):
    """Median and interquartile range, as a robust center and spread."""
    xs = sorted(xs)
    n = len(xs)
    return statistics.median(xs), xs[(3 * n) // 4] - xs[n // 4]


# Parameters differentiated in HMC for the Kepler-51 test system
KEP51_DIFF_KEYS = ['period', 'ecosw', 'esinw', 'tic', 'lnode', 'cosi', 'pmass']


def load_kep51(integrate_fn=None, finder_fn=None):
    """Kepler-51 full-NLL pipeline differentiable w.r.t. the HMC parameters.

    Returns (nll, grad_fn, p0, info) where nll/grad_fn take a param dict
    with KEP51_DIFF_KEYS and info holds (jttv, pdic).
    """
    from jnkepler.tests import read_testdata_tc
    integrate_fn = integrate_fn or integrate_xv
    finder_fn = finder_fn or find_transit_times_fast

    jttv, _, _, pdic = read_testdata_tc()
    oidx = jnp.asarray(jttv.pidx.astype(int) - 1)
    tc1d = jnp.asarray(jttv.tcobs_flatten)
    err1d = jnp.asarray(jttv.errorobs_flatten)
    times = jttv.times
    t_start = jttv.t_start
    nitr = jttv.nitr_kepler

    p0 = {k: jnp.asarray(pdic[k]) for k in KEP51_DIFF_KEYS}
    pfix = {k: v for k, v in pdic.items() if k not in KEP51_DIFF_KEYS}

    @jax.jit
    def nll(p):
        pd = {**pfix, **p}
        x0, v0, masses = initialize_jacobi_xv(pd, t_start)
        t, xv = integrate_fn(x0, v0, masses, times, nitr=nitr)
        tc = finder_fn(oidx, tc1d, t, xv, masses)
        return 0.5 * jnp.sum(((tc1d - tc) / err1d) ** 2)

    grad_fn = jax.jit(jax.grad(nll))
    return nll, grad_fn, p0, (jttv, pdic)


def load_kep51_integration_only(integrate_fn=None):
    """Integration-only scalar loss (sum of trajectory) for Kepler-51."""
    from jnkepler.tests import read_testdata_tc
    integrate_fn = integrate_fn or integrate_xv

    jttv, _, _, pdic = read_testdata_tc()
    times = jttv.times
    t_start = jttv.t_start
    nitr = jttv.nitr_kepler
    p0 = {k: jnp.asarray(pdic[k]) for k in KEP51_DIFF_KEYS}
    pfix = {k: v for k, v in pdic.items() if k not in KEP51_DIFF_KEYS}

    @jax.jit
    def loss(p):
        pd = {**pfix, **p}
        x0, v0, masses = initialize_jacobi_xv(pd, t_start)
        t, xv = integrate_fn(x0, v0, masses, times, nitr=nitr)
        return jnp.sum(xv)

    return loss, jax.jit(jax.grad(loss)), p0


def make_system_raw(nplanets, baseline_years, dt=1.0):
    """Synthetic circular transiting system: raw pieces.

    Returns dict with x0, v0, masses, times, tcobs (observed, noisy),
    errobs, pidxarr.

    Observations are always generated with the library integrate_xv, never
    with the integrate_fn under test, so that every arm of a comparison fits
    identical data.
    """
    periods = [20., 35., 55., 80., 120.][:nplanets]
    masses = jnp.array([1.0] + [3e-6] * nplanets)

    x_jac = np.zeros((nplanets, 3))
    v_jac = np.zeros((nplanets, 3))
    for i in range(nplanets):
        a = (G * jnp.sum(masses[:i+2]) * (periods[i] / (2 * np.pi))**2)**(1./3.)
        gm = G * jnp.sum(masses[:i+2])
        x_jac[i, 0] = float(a)
        v_jac[i, 1] = float(jnp.sqrt(gm / a))

    x_jac, v_jac = jnp.array(x_jac), jnp.array(v_jac)
    baseline_days = baseline_years * 365.25
    times = jnp.arange(0., baseline_days, dt)

    t_out, xv = integrate_xv(x_jac, v_jac, masses, times, nitr=10)

    tcobs_list, pidx_list = [], []
    for j in range(nplanets):
        n_transits = int(baseline_days / periods[j])
        tc_approx = jnp.arange(n_transits) * periods[j] + periods[j] * 0.5
        tc_approx = tc_approx[tc_approx < baseline_days - periods[j]]
        tcobs_list.append(tc_approx)
        pidx_list.append(jnp.full(len(tc_approx), j, dtype=jnp.int32))

    tcobs = jnp.concatenate(tcobs_list)
    pidxarr = jnp.concatenate(pidx_list)
    order = jnp.argsort(tcobs)
    tcobs, pidxarr = tcobs[order], pidxarr[order]

    tc_model = find_transit_times_fast(pidxarr, tcobs, t_out, xv, masses)
    tcobs_true = tc_model + 1e-4 * jax.random.normal(
        jax.random.PRNGKey(42), tc_model.shape)
    errobs = jnp.full_like(tcobs_true, 0.001)

    return dict(x0=x_jac, v0=v_jac, masses=masses, times=times,
                tcobs=tcobs_true, errobs=errobs, pidxarr=pidxarr)


def make_system(nplanets, baseline_years, dt=1.0, integrate_fn=None,
                finder_fn=None):
    """Synthetic system NLL differentiable w.r.t. masses.

    Returns (nll, grad_fn, masses, nsteps, ntransits).
    """
    integrate_fn = integrate_fn or integrate_xv
    finder_fn = finder_fn or find_transit_times_fast
    S = make_system_raw(nplanets, baseline_years, dt)
    x_jac, v_jac, times = S['x0'], S['v0'], S['times']
    tcobs_true, errobs, pidxarr = S['tcobs'], S['errobs'], S['pidxarr']

    @jax.jit
    def nll(m):
        t, xv_ = integrate_fn(x_jac, v_jac, m, times, nitr=10)
        tc = finder_fn(pidxarr, tcobs_true, t, xv_, m)
        return 0.5 * jnp.sum(((tcobs_true - tc) / errobs) ** 2)

    return nll, jax.jit(jax.grad(nll)), S['masses'], len(times), len(tcobs_true)


def make_integrate(policy):
    """Build an integrate_xv equivalent using the given checkpoint policy.

    This mirrors integrate_xv rather than calling it, so that the scan step
    can be rebuilt under a chosen policy. Pass the result to load_kep51 or
    make_system as integrate_fn. Being a copy, it drifts if integrate_xv
    changes, so dots_policy_ab.py checks it against the library function.

    Passing the integrator in explicitly is the supported way to select a
    policy here. Patching symplectic.integrate_xv has no effect, because
    the from-import above binds the function into this module's namespace
    and the builders resolve that binding, not the attribute on symplectic.

    A policy of None gives the default full rematerialization, which these
    scripts label "bare" in their output, as against "dots" for
    dots_saveable.
    """
    def integrate(x, v, masses, times, nitr=10):
        ki = _compute_ki(masses)
        dtarr = jnp.diff(times)
        dt0 = dtarr[0]
        x, v = real_to_mapTO(x, v, ki, masses, dt0)
        x, v = kepler_step(x, v, ki, dt0 * 0.5, nitr=nitr)

        def step(xvin, dt):
            x, v = xvin
            x, v = nbody_kicks(x, v, ki, masses, dt)
            xo, vo = kepler_step(x, v, ki, dt, nitr=nitr)
            return [xo, vo], jnp.array([xo, vo])

        step = checkpoint(step, policy=policy) if policy else checkpoint(step)
        _, xv = scan(step, [x, v], dtarr)
        return times[1:] + 0.5 * dt0, xv
    return integrate


def active_policy(nplanets):
    """Report the checkpoint policy integrate_xv selects for this system.

    Mirrors the condition in symplectic.integrate_xv, where the mass array
    holds the star as well as the planets.
    """
    return "dots_saveable" if nplanets + 1 > 4 else "bare"


def grad_maxreldiff(g1, g2):
    """Max relative difference between two gradient pytrees."""
    l1 = jax.tree_util.tree_leaves(g1)
    l2 = jax.tree_util.tree_leaves(g2)
    out = 0.0
    for a, b in zip(l1, l2):
        denom = jnp.maximum(jnp.abs(a), 1e-30)
        out = max(out, float(jnp.max(jnp.abs(a - b) / denom)))
    return out
