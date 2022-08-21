#%%
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan
from .markley import get_E
from jax.config import config
config.update('jax_enable_x64', True)

#%%
BIG_G = 2.959122082855911e-4

#%%
@jit
def reduce_angle(M):
    Mmod = M % (2*jnp.pi)
    Mred = jnp.where(Mmod >= jnp.pi, Mmod-2*jnp.pi, Mmod) # M in -pi to pi
    return Mred

@jit
def tic_to_u(tic, period, ecc, omega, t_epoch):
    tanw2 = jnp.tan(0.5 * omega)
    uic = 2 * jnp.arctan( jnp.sqrt((1.-ecc)/(1.+ecc)) * (1.-tanw2)/(1.+tanw2) ) # u at t=tic
    M_epoch = 2 * jnp.pi / period * (t_epoch - tic) + uic - ecc * jnp.sin(uic) # M at t=0
    u_epoch = get_E(reduce_angle(M_epoch), ecc)
    return u_epoch

@jit
def tic_to_m(tic, period, ecc, omega, t_epoch):
    tanw2 = jnp.tan(0.5 * omega)
    uic = 2 * jnp.arctan( jnp.sqrt((1.-ecc)/(1.+ecc)) * (1.-tanw2)/(1.+tanw2) ) # u at t=tic
    M_epoch = 2 * jnp.pi / period * (t_epoch - tic) + uic - ecc * jnp.sin(uic) # M at t=0
    return M_epoch

@jit
def elements_to_xvrel(porb, ecc, inc, omega, lnode, u, mass):
    """ convert elements to position/velocity

        Args:
            mass -> should be ki?

    """
    cosu, sinu = jnp.cos(u), jnp.sin(u)
    cosw, sinw, cosO, sinO, cosi, sini = jnp.cos(omega), jnp.sin(omega), jnp.cos(lnode), jnp.sin(lnode), jnp.cos(inc), jnp.sin(inc)

    n = 2 * jnp.pi / porb
    na = (n * BIG_G * mass) ** (1./3.)
    R = 1.0 - ecc * cosu

    Pvec = jnp.array([cosw*cosO - sinw*sinO*cosi, cosw*sinO + sinw*cosO*cosi, sinw*sini])
    Qvec = jnp.array([-sinw*cosO - cosw*sinO*cosi, -sinw*sinO + cosw*cosO*cosi, cosw*sini])
    x, y = cosu - ecc, jnp.sqrt(1.-ecc*ecc) * sinu
    vx, vy = -sinu, jnp.sqrt(1.-ecc*ecc) * cosu

    xrel = (na / n) * (Pvec * x + Qvec * y)
    vrel = (na / R) * (Pvec * vx + Qvec * vy)

    return xrel, vrel

def xv_to_elements(x, v, ki):
    """ convert position/velocity to elements

        Args:
            x, v: position and velocity (Norbit, Nxyz)
            ki: 'GM' in Kepler's 3rd law (Norbit); depends on what x/v mean (Jacobi, astrocentric, ...)

        Returns:
            sma, mean motion, ecc, inclination, \omega, \Omega, mean anomaly

    """
    r0 = jnp.sqrt(jnp.sum(x*x, axis=1))
    v0s = jnp.sum(v*v, axis=1)
    u = jnp.sum(x*v, axis=1)
    a = 1. / (2./r0 - v0s/ki)
    n = jnp.sqrt(ki / (a*a*a))
    ecosE0, esinE0 = 1. - r0 / a, u / (n*a*a)
    e = jnp.sqrt(ecosE0**2 + esinE0**2)
    E = jnp.arctan2(esinE0, ecosE0)

    hx = x[:,1] * v[:,2] - x[:,2] * v[:,1]
    hy = x[:,2] * v[:,0] - x[:,0] * v[:,2]
    hz = x[:,0] * v[:,1] - x[:,1] * v[:,0]
    hnorm = jnp.sqrt(hx**2 + hy**2 + hz**2)
    inc = jnp.arccos(hz / hnorm)

    P = (jnp.cos(E) / r0)[:,None] * x - (jnp.sqrt(a / ki) * jnp.sin(E))[:,None] * v
    Q = (jnp.sin(E) / r0 / jnp.sqrt(1 - e*e))[:,None] * x + (jnp.sqrt(a / ki) * (jnp.cos(E)-e) / jnp.sqrt(1 - e*e))[:,None] * v
    PQz = jnp.sqrt(P[:,2]**2 + Q[:,2]**2)
    omega = jnp.where(PQz!=0., jnp.arctan2(P[:,2], Q[:,2]), 0.)
    coslnode = (P[:,0] * Q[:,2] - P[:,2] * Q[:,0]) / PQz
    sinlnode = (P[:,1] * Q[:,2] - P[:,2] * Q[:,1]) / PQz
    lnode = jnp.where(PQz!=0., jnp.arctan2(sinlnode, coslnode), 0.)

    return jnp.array([a, 2*jnp.pi/n, e, inc, omega, lnode, E - esinE0])

@jit
def initialize_from_elements(elements, masses, t_epoch):
    """ compute initial position/velocity from JaxTTV elements
        here the elements are interpreted as Jacobi using the total interior mass (see Rein & Tamayo 2015)

        Args:
            elements: orbital elements (period, ecosw, esinw, cosi, \Omega, T_inf_conjunction)
            masses: masses of the bodies (Nbody,)
            t_epoch: epoch at which elements are defined

        Returns:
            Jacobi position/velocity at t_epoch

    """
    xrel, vrel = [], []
    for j in range(len(elements)):
        #porb, ecc, inc, omega, lnode, tic = elements[j]
        porb, ecosw, esinw, cosi, lnode, tic = elements[j]
        ecc = jnp.sqrt(ecosw**2 + esinw**2)
        omega = jnp.arctan2(esinw, ecosw)
        inc = jnp.arccos(cosi)

        u = tic_to_u(tic, porb, ecc, omega, t_epoch)
        xr, vr = elements_to_xvrel(porb, ecc, inc, omega, lnode, u, jnp.sum(masses[:j+2]))
        xrel.append(xr)
        vrel.append(vr)

    xrel, vrel = jnp.array(xrel), jnp.array(vrel)
    return xrel, vrel

@jit
def jacobi_to_astrocentric(xrel_j, vrel_j, masses):
    nbody = len(masses)
    mmat = jnp.eye(nbody-1) + jnp.tril(jnp.tile(masses[1:] / jnp.cumsum(masses)[1:], (nbody-1,1)), k=-1)
    return mmat@xrel_j, mmat@vrel_j

@jit
def astrocentric_to_cm(xrel_ast, vrel_ast, masses):
    mtot = jnp.sum(masses)
    xcm_ast = jnp.sum(masses[1:][:,None] * xrel_ast, axis=0) / mtot
    vcm_ast = jnp.sum(masses[1:][:,None] * vrel_ast, axis=0) / mtot
    xcm = jnp.vstack([-xcm_ast, xrel_ast - xcm_ast])
    vcm = jnp.vstack([-vcm_ast, vrel_ast - vcm_ast])
    return xcm, vcm

@jit
def cm_to_astrocentric(x, v, a, j):
    xastj = x[:,j,:] - x[:,0,:]
    vastj = v[:,j,:] - v[:,0,:]
    aastj = a[:,j,:] - a[:,0,:]
    return xastj, vastj, aastj

# compute total energy of the system in CM frame unit: Msun*(AU/day)^2 */
@jit
def get_energy(x, v, masses):
    K = jnp.sum(0.5 * masses * jnp.sum(v*v, axis=1))
    X = x[:,None] - x[None,:]
    M = masses[:,None] * masses[None,:]
    U = -BIG_G * jnp.sum(M * jnp.tril(1./jnp.sqrt(jnp.sum(X*X, axis=2)), k=-1))
    return K + U

get_energy_vmap = jit(vmap(get_energy, (0,0,None), 0))

@jit
def get_ediff(xva, masses):
    _xva = jnp.array([xva[0,:,:,:], xva[-1,:,:,:]])
    etot = get_energy_vmap(_xva[:,0,:,:], _xva[:,1,:,:], masses)
    return etot[1]/etot[0] - 1.

@jit
def get_acm(x, masses):
    xjk = jnp.transpose(x[:,None] - x[None, :], axes=[0,2,1])
    x2jk = jnp.sum(xjk * xjk, axis=1)[:,None,:]
    x2jk = jnp.where(x2jk!=0., x2jk, jnp.inf)
    x2jkinv = 1. / x2jk

    x2jkinv1p5 = x2jkinv * jnp.sqrt(x2jkinv)
    Xjk = - xjk * x2jkinv1p5

    a = BIG_G * jnp.dot(Xjk, masses)

    return a
