"""Compare the bare and dots_saveable checkpoint policies for integrate_xv.

Checks first that the policy does not change the answer, then measures both
the isolated NLL gradient and the post-warmup NUTS cost per leapfrog step,
on Kepler-51 (3 planets) and on a synthetic 5-planet system where the policy
should matter most. Reporting both timings matters because an isolated
gradient speedup only reaches MCMC wall time to the extent that NUTS is
dominated by gradient evaluation rather than its own overhead.

The two integrators are passed in explicitly rather than patched onto the
symplectic module, which would have no effect here. See make_integrate in
common.py for why.
"""
import common as C
import time
import jax
import jax.numpy as jnp
from jax import checkpoint_policies
from jnkepler.jaxttv.findtransit import find_transit_times_fast

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from jax import random

print(f"JAX {jax.__version__}")

integrate_bare = C.make_integrate(None)
integrate_dots = C.make_integrate(checkpoint_policies.dots_saveable)

# ---------- agreement ----------
# The speedup is only worth having if the policy leaves the answer alone,
# so establish that before reporting any timing. One arm is also checked
# against the library integrate_xv, which catches make_integrate drifting
# away from the function it copies.
print("\n--- gradient agreement ---")
TOL = 1e-12
for label, npl, build in [
        ("kep51", 3, lambda integ: C.load_kep51(integrate_fn=integ)[1:3]),
        ("synth5pl", 5, lambda integ: C.make_system(5, 10, integrate_fn=integ)[1:3])]:
    grad_bare, p = build(integrate_bare)
    grad_dots, _ = build(integrate_dots)
    grad_lib, _ = build(None)
    g_bare, g_dots, g_lib = grad_bare(p), grad_dots(p), grad_lib(p)

    d_policy = C.grad_maxreldiff(g_bare, g_dots)
    # Compare the library against whichever arm already shares its policy at
    # this planet count, so that a failure here isolates a drifted copy. The
    # other pairing would vary implementation and policy at the same time.
    arm = "dots" if C.active_policy(npl) == "dots_saveable" else "bare"
    d_copy = C.grad_maxreldiff(g_dots if arm == "dots" else g_bare, g_lib)
    print(f"{label:<9s} bare vs dots {d_policy:.3e}   "
          f"{arm} vs library {d_copy:.3e}")
    assert d_policy < TOL, f"{label}: policy changed the gradient"
    assert d_copy < TOL, f"{label}: make_integrate has drifted from integrate_xv"

# ---------- isolated gradients ----------
REPEAT = 20
print(f"\n--- isolated NLL gradients, median over {REPEAT} paired repeats ---")
iso = {}
for label, build in [
        ("kep51", lambda integ: C.load_kep51(integrate_fn=integ)[1:3]),
        ("synth5pl", lambda integ: C.make_system(5, 10, integrate_fn=integ)[1:3])]:
    grad_bare, p = build(integrate_bare)
    grad_dots, _ = build(integrate_dots)
    pairs = C.timeit_paired(grad_bare, (p,), grad_dots, (p,), repeat=REPEAT)
    t_bare, s_bare = C.median_iqr([q[0] for q in pairs])
    t_dots, s_dots = C.median_iqr([q[1] for q in pairs])
    r, s_r = C.median_iqr([q[0] / q[1] for q in pairs])
    iso[label] = (t_bare, t_dots, r)
    print(f"{label:<9s} bare {t_bare:6.3f}+-{s_bare:5.3f}  "
          f"dots {t_dots:6.3f}+-{s_dots:5.3f} ms  speedup {r:5.3f}+-{s_r:5.3f}")

# ---------- NUTS ms/leapfrog on synth 5pl ----------
print("\n--- NUTS post-warmup ms/leapfrog, synth 5pl/10yr (masses only) ---")
S = C.make_system_raw(5, 10)
x0, v0, masses0 = S['x0'], S['v0'], S['masses']
times, tcobs, errobs, pidx = S['times'], S['tcobs'], S['errobs'], S['pidxarr']
lo, hi = masses0[1:] * 0.3, masses0[1:] * 3.0


def make_model(integ):
    def model():
        pm = numpyro.sample('pmass', dist.Uniform(lo, hi).to_event(1))
        masses = jnp.concatenate([masses0[:1], pm])
        t, xv = integ(x0, v0, masses, times, nitr=10)
        tc = find_transit_times_fast(pidx, tcobs, t, xv, masses)
        numpyro.sample('obs', dist.Normal(tc, errobs), obs=tcobs)
    return model


NUM_WARMUP, NUM_SAMPLES = 200, 200
mcmc_res = {}
for name, integ in [("bare", integrate_bare), ("dots", integrate_dots)]:
    kernel = NUTS(make_model(integ),
                  init_strategy=init_to_value(values={'pmass': masses0[1:]}),
                  max_tree_depth=8, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=1, progress_bar=False)
    t0 = time.perf_counter()
    mcmc.warmup(random.PRNGKey(0), extra_fields=('num_steps',))
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(1), extra_fields=('num_steps',))
    t_samp = time.perf_counter() - t0
    total_steps = int(jnp.sum(mcmc.get_extra_fields()['num_steps']))
    ms_lf = t_samp / total_steps * 1000
    mcmc_res[name] = ms_lf
    print(f"{name:<5s} warmup {t_warm:6.1f}s  sample {t_samp:6.1f}s  "
          f"leapfrogs {total_steps:>6d}  ms/leapfrog {ms_lf:6.2f}")

# The NUTS numbers come from one chain per arm, since repeating the sample
# phase costs minutes rather than the seconds a gradient repeat costs. Read
# them alongside the gradient rows above, which do carry a spread.
print(f"\nNUTS ms/leapfrog speedup (dots): {mcmc_res['bare']/mcmc_res['dots']:.3f}x"
      f"  (single run, no repeat spread)")
print(f"isolated grad speedup (dots)   : {iso['synth5pl'][2]:.3f}x")

# The two costs below are not the same measurement, so the ratio is a sanity
# check rather than an overhead figure. A leapfrog runs inside one compiled
# scan, while the isolated gradient is dispatched from Python once per call
# and carries that dispatch in its number. The in-loop cost is therefore the
# smaller of the two, and the speedups above agreeing is the real result.
print(f"in-loop leapfrog vs per-call grad (bare): "
      f"{mcmc_res['bare']/iso['synth5pl'][0]:.3f}x")
