r"""
Module for handling Yukawa potential.

Potential
*********

The Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distant :math:`r` is defined as

.. math::
    U_{ab}(r) = \frac{q_a q_b}{4 \pi \epsilon_0} \frac{e^{- \kappa r} }{r}.

where :math:`\kappa = 1/\lambda` is the screening parameter.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = q_iq_j^2/(4 pi eps0)
    pot_matrix[1] = 1/lambda
    pot_matrix[2] = Ewald screening parameter

"""
from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import exp, inf, pi, sqrt, zeros
from scipy.integrate import quad
from scipy.special import gamma

from warnings import warn


@jit(nopython=True)
def yukawa_force_pppm(r_in, pot_matrix):
    """
    Numba'd function to calculate Potential and Force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (4, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Potential value

    f_r : float
        Force between two particles calculated using eq.(22) in :cite:`Dharuman2017`.

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 0.5, 0.25,  0.0001])
    >>> yukawa_force_pppm(r, pot_matrix)
    (0.16287410244138842, 0.18025091684402375)

    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]  # Ewald parameter alpha

    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[-1]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    u_r = (
        pot_matrix[0]
        * (0.5 / r)
        * (exp(kappa_r) * erfc(alpha_r + 0.5 * kappa_alpha) + exp(-kappa_r) * erfc(alpha_r - 0.5 * kappa_alpha))
    )
    # Derivative of the exponential term and 1/r
    f1 = (0.5 / r) * exp(kappa_r) * erfc(alpha_r + 0.5 * kappa_alpha) * (1.0 / r - kappa)
    f2 = (0.5 / r) * exp(-kappa_r) * erfc(alpha_r - 0.5 * kappa_alpha) * (1.0 / r + kappa)
    # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
    f3 = (alpha / sqrt(pi) / r) * (
        exp(-((alpha_r + 0.5 * kappa_alpha) ** 2)) * exp(kappa_r)
        + exp(-((alpha_r - 0.5 * kappa_alpha) ** 2)) * exp(-kappa_r)
    )
    f_r = pot_matrix[0] * (f1 + f2 + f3)

    return u_r, f_r


@jit(nopython=True)
def yukawa_force(r_in, pot_matrix):
    """
    Numba'd function to calculate Potential and Force between two particles.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)


    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([ 1.0, 1.0, 0.0001])
    >>> yukawa_force(r, pot_matrix)
    (0.06766764161830635, 0.10150146242745953)

    """
    # Short-range cutoff to deal with divergence of the Coulomb potential
    rs = pot_matrix[-1]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    u_r = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
    f_r = u_r * (1.0 / r + pot_matrix[1])

    return u_r, f_r


def force_deriv(r, pot_matrix):
    """Calculate the second derivative of the potential.

    Parameters
    ----------
    r : float
        Distance between particles

    pot_matrix : numpy.ndarray
        Values of the potential constants. \n
        Shape = (3, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    Raises
    ------
       : DeprecationWarning
    """

    warn(
        "Deprecated feature. It will be removed in a future release. \n" "Use potential_derivatives.",
        category=DeprecationWarning,
    )

    _, _, d2v_dr2 = potential_derivatives(r, pot_matrix)

    return d2v_dr2


def potential_derivatives(r, pot_matrix):
    """Calculate the first and second derivative of the potential.

    Parameters
    ----------
    r_in : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables.

    Returns
    -------
    U : float, numpy.ndarray
        Potential value.

    dv_dr : float, numpy.ndarray
        First derivative of the potential.

    d2v_dr2 : float, numpy.ndarray
        Second derivative of the potential.

    """
    kappa = pot_matrix[1]
    kappa_r = kappa * r
    u_r = exp(-kappa_r) / r
    du_dr = -(1.0 + kappa_r) * u_r / r
    d2u_dr2 = -(1.0 / r + kappa) * du_dr + u_r / r**2

    u_r *= pot_matrix[0]
    du_dr *= pot_matrix[0]
    d2u_dr2 *= pot_matrix[0]

    return u_r, du_dr, d2u_dr2


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"kappa = {potential.a_ws / potential.screening_length:.4f}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)


def update_params(potential, species):
    """
    Assign potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """
    # species[-1] is the electronic background
    potential.matrix = zeros((len(species) - 1, len(species) - 1, 4))
    potential.matrix[:, :, 1] = 1.0 / potential.screening_length

    if not hasattr(potential, "kappa") or potential.kappa is None:
        potential.kappa = potential.matrix[0,0, 1] * potential.a_ws

    for i, sp1 in enumerate(species[:-1]): # species[-1] is the electronic background
        q1 = sp1.charge
        for j, sp2 in enumerate(species[:-1]):
            q2 = sp2.charge
            potential.matrix[i, j, 0] = q1 * q2 / potential.fourpie0

    potential.matrix[:, :, -1] = potential.a_rs

    potential.potential_derivatives = potential_derivatives

    if potential.method == "pp":

        potential.force = yukawa_force
        potential.calc_force_error_quad = calc_force_error_quad

    elif potential.method == "pppm":
        potential.force = yukawa_force_pppm
        potential.matrix[:, :, 2] = potential.pppm_alpha_ewald

def calc_force_error_quad(potential):
    r"""
    Calculate the force error by integrating the square modulus of the force over the neglected volume.\n
    The force error is calculated from

    .. math::
        \Delta F =  \left [ 4 \pi \int_{r_c}^{\infty} dr \, r^2  \left ( \frac{d\phi(r)}{r} \right )^2 ]^{1/2}

    where :math:`\phi(r)` is only the radial part of the potential, :math:`r_c` is the cutoff radius, and :math:`r` is scaled by the input parameter `a`.\n
    The integral is calculated using `scipy.integrate.quad`. The derivative of the potential is obtained from :meth:`potential_derivatives`.

    Parameters
    ----------
    a : float
        Rescaling length. Usually it is the Wigner-Seitz radius.

    rc : float
        Cutoff radius to be used as the lower limit of the integral. The lower limit is actually `rc /a`.

    pot_matrix: numpy.ndarray
        Slice of the `sarkas.potentials.Potential.matrix` containing the parameters of the potential. It must be a 1D-array.

    Returns
    -------
    f_err: float
        Force error. It is the sqrt root of the integral. It is calculated using `scipy.integrate.quad`  and :func:`potential_derivatives`.

    Examples
    --------
    >>> import numpy as np
    >>> potential_matrix = np.zeros(2)
    >>> a = 1.0 # Wigner-seitz radius
    >>> kappa = 2.0 # in units of a_ws
    >>> potential_matrix[1] = kappa
    >>> rc = 6.0 # in units of a_ws
    >>> calc_force_error_quad(a, rc, potential_matrix)
    6.636507826720378e-06

    """

    params = potential.matrix.copy()
    # Rescale the potential parameters so that quad does not fail.
    params[:, :, 0] /= potential.matrix[:, :, 0] 
    params[:, :, 1] *= potential.a_ws  # kappa
    params[:, :, 2] *= potential.a_ws  # Ewald parameter    
    params[:, :, -1] /= potential.a_ws # Short-range cutoff

    r_c = potential.rc / potential.a_ws

    # Solid angle integral
    solid_angle = 2 * pi**(potential.dimensions / 2) / gamma(potential.dimensions / 2)

    integrand = lambda r: solid_angle * r**(potential.dimensions - 1) * ( potential.force(r, params[0,0])[1])**2
    result, _ = quad( integrand, a=r_c, b=inf)

    # Rescaling constant = Q^2 sqrt(N/V) = Q^2 sqrt(3 /(4 pi)), with V = L^3 = (4pi /  N)^3
    QFactor = potential.QFactor / (potential.matrix[0, 0, 0] * potential.total_num_ptcls)
    f_err = sqrt( result * 3 / (4 * pi)) * QFactor

    return f_err

