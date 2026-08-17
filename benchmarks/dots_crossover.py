"""Find the planet count at which dots_saveable begins to pay off.

Times the mass gradient of a synthetic system under both checkpoint
policies, sweeping planet count, since the benefit depends on how much of
the step is dot products relative to the rest of the Kepler solve.

The sweep covers the point where integrate_xv switches policy, so it also
checks that the two policies agree numerically at every count, including
the boundary itself. A speedup at a count where the gradients disagree
would not be a speedup at all.

Each row is a median over repeated paired measurements, with the spread
quoted as an interquartile range. The spread matters most at low planet
counts, where the effect is around one percent and comparable to the noise.
"""
import common as C
import jax
from jax import checkpoint_policies


TOL = 1e-12
REPEAT = 20

print(f"JAX {jax.__version__}")
print(f"median and IQR over {REPEAT} paired repeats")
for npl in [2, 3, 4, 5]:
    _, grad_bare, m, _, _ = C.make_system(
        npl, 10, integrate_fn=C.make_integrate(None))
    _, grad_dots, _, _, _ = C.make_system(
        npl, 10, integrate_fn=C.make_integrate(checkpoint_policies.dots_saveable))

    d = C.grad_maxreldiff(grad_bare(m), grad_dots(m))
    assert d < TOL, f"{npl}pl: policies disagree on the gradient"

    pairs = C.timeit_paired(grad_bare, (m,), grad_dots, (m,), repeat=REPEAT)
    t_bare, s_bare = C.median_iqr([p[0] for p in pairs])
    t_dots, s_dots = C.median_iqr([p[1] for p in pairs])
    # Ratio per repeat rather than of the medians, so the spread reflects
    # how much the speedup itself moves rather than how much the machine does.
    r, s_r = C.median_iqr([p[0] / p[1] for p in pairs])
    print(f"{npl}pl  bare {t_bare:6.3f}+-{s_bare:5.3f}  "
          f"dots {t_dots:6.3f}+-{s_dots:5.3f} ms  "
          f"speedup {r:5.3f}+-{s_r:5.3f}  maxreldiff {d:.1e}")
