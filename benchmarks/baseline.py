"""Baseline cost of the core evaluation paths, for comparison across branches.

Prints forward and gradient wall-clock cost for the Kepler-51 negative log
likelihood, for Kepler-51 integration alone, and for two synthetic systems.

Everything here runs through the library integrate_xv as the current tree
defines it, so each row is labeled with the checkpoint policy that applies
at that planet count. These are not the controlled bare and dots arms that
dots_policy_ab.py builds, and the rows should not be read as such.
"""
import common as C
import jax
import jax.numpy as jnp

print(f"JAX {jax.__version__}")
print("=" * 60)

# Kepler-51 full NLL (3pl, 2795 steps, 53 transits, 21 params)
nll, grad_fn, p0, _ = C.load_kep51()
v = nll(p0)
g = grad_fn(p0)
assert not any(jnp.any(jnp.isnan(x)) for x in jax.tree_util.tree_leaves(g))
tf = C.timeit(nll, (p0,))
tg = C.timeit(grad_fn, (p0,))
print(f"kep51 full NLL       [{C.active_policy(3):<14s}]: fwd {tf:7.3f} ms   "
      f"grad {tg:7.3f} ms   (nll={float(v):.4f})")

# Kepler-51 integration only
loss_i, grad_i, p0i = C.load_kep51_integration_only()
tfi = C.timeit(loss_i, (p0i,))
tgi = C.timeit(grad_i, (p0i,))
print(f"kep51 integration    [{C.active_policy(3):<14s}]: fwd {tfi:7.3f} ms   "
      f"grad {tgi:7.3f} ms")

# Synthetic systems (grad w.r.t. masses)
for npl, yr in [(3, 10), (5, 10)]:
    nll_s, grad_s, m, nstep, ntr = C.make_system(npl, yr)
    tfs = C.timeit(nll_s, (m,), N=30, warmup=6)
    tgs = C.timeit(grad_s, (m,), N=30, warmup=6)
    print(f"synth {npl}pl/{yr}yr [{C.active_policy(npl):<14s}] "
          f"({nstep} steps, {ntr} tr): fwd {tfs:7.3f} ms   grad {tgs:7.3f} ms")
