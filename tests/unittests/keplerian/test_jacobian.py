import jax.numpy as jnp
import numpy as np
from jnkepler.keplerian.jacobian import *
from jnkepler.jaxttv.conversion import G


key_sets = {
    "am":   ("a", "ecc", "inc", "lnode", "omega", "M"),
    "atau": ("a", "ecc", "inc", "lnode", "omega", "tau"),
    "Pm":   ("period", "ecc", "inc", "lnode", "omega", "M"),
    "Ptau": ("period", "ecc", "inc", "lnode", "omega", "tau"),
}

key_sets_2D = {
    "am":   ("a", "ecc", "omega", "M"),
    "Pm":   ("period", "ecc", "omega", "M"),
}

params = dict(
    period=100.,
    ecc=0.2,
    inc=0.3,
    omega=0.5,
    lnode=1.0,
    tau=10.0,
    mass=1.0,
    t_ref=0.,
)

params['a'] = (G * params['mass'] /
               (2 * jnp.pi / params['period'])**2)**(1./3.)
params['M'] = 2 * jnp.pi * (params['t_ref'] - params['tau']) / params['period']
params['n'] = 2 * jnp.pi / params['period']


def analytic_det(params, keys):
    ks = tuple(keys)
    if ks == key_sets["am"]:
        return det_jkep_am(params)
    if ks == key_sets["atau"]:
        return det_jkep_atau(params)
    if ks == key_sets["Pm"]:
        return det_jkep_pm(params)
    if ks == key_sets["Ptau"]:
        return det_jkep_ptau(params)


def analytic_det2D(params, keys):
    ks = tuple(keys)
    if ks == key_sets_2D["am"]:
        return det_jkep2D_am(params)
    if ks == key_sets_2D["Pm"]:
        return det_jkep2D_pm(params)


def test_det_jkep():
    for _, keys in key_sets.items():
        sign_jax, logabs_jax = slogdet_jkep_jax(params, keys)

        det_ana = analytic_det(params, keys)
        sign_ana = jnp.sign(det_ana)
        logabs_ana = jnp.log(jnp.abs(det_ana) + 1e-300)

        assert np.allclose(sign_jax, sign_ana)
        assert np.allclose(logabs_jax, logabs_ana)


def test_det_jkep2D():
    for _, keys in key_sets_2D.items():
        sign_jax, logabs_jax = slogdet_jkep2D_jax(params, keys)

        det_ana = analytic_det2D(params, keys)
        sign_ana = jnp.sign(det_ana)
        logabs_ana = jnp.log(jnp.abs(det_ana) + 1e-300)

        assert np.allclose(sign_jax, sign_ana)
        assert np.allclose(logabs_jax, logabs_ana)


if __name__ == '__main__':
    test_det_jkep()
    test_det_jkep2D()
