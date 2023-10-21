r"""
Module for handling Lennard-Jones interaction.

Potential
*********

The generalized Lennard-Jones potential is defined as

.. math::
    U_{\mu\nu}(r) = k \epsilon_{\mu\nu} \left [ \left ( \frac{\sigma_{\mu\nu}}{r}\right )^m -
    \left ( \frac{\sigma_{\mu\nu}}{r}\right )^n \right ],

where

.. math::
    k = \frac{n}{m-n} \left ( \frac{n}{m} \right )^{\frac{m}{n-m}}.

In the case of multispecies liquids we use the `Lorentz-Berthelot <https://en.wikipedia.org/wiki/Combining_rules>`_
mixing rules

.. math::
    \epsilon_{12} = \sqrt{\epsilon_{11} \epsilon_{22}}, \quad \sigma_{12} = \frac{\sigma_{11} + \sigma_{22}}{2}.

Force Error
***********

The force error for the LJ potential is given by

.. math::
    \Delta F = \frac{k\epsilon}{ \sqrt{2\pi n}} \left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}}
    + \frac{n^2 \sigma^{2n}}{2n - 1} \frac{1}{r_c^{2n -1}} \
    -\frac{2 m n \sigma^{m + n}}{m + n - 1} \frac{1}{r_c^{m + n -1}} \
    \right ]^{1/2}

which we approximate with the first term only

.. math::
    \Delta F \approx \frac{k\epsilon} {\sqrt{2\pi n} }
    \left [ \frac{m^2 \sigma^{2m}}{2m - 1} \frac{1}{r_c^{2m -1}} \right ]^{1/2}

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block::

    pot_matrix[0] = epsilon_12 * lj_constant
    pot_matrix[1] = sigmas
    pot_matrix[2] = highest power
    pot_matrix[3] = lowest power
    pot_matrix[4] = short-range cutoff

"""
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, pi, sqrt, zeros, inf
from scipy.integrate import quad
from sarkas.potentials.core import Potential

from ..utilities.maths import force_error_analytic_lcl


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def lj_force(r, pot_matrix):
    """
    Numba'd function to calculate the PP force between particles using Lennard-Jones Potential.

    Parameters
    ----------
    r_in : float
        Particles' distance.

    pot_matrix : numpy.ndarray
        LJ potential parameters. \n
        Shape = (5, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force.

    Examples
    --------
    >>> pot_const = 4.0 * 1.656e-21 # 4*epsilon in [J] (mks units)
    >>> sigma = 3.4e-10   # [m] (mks units)
    >>> high_pow, low_pow = 12., 6.
    >>> short_cutoff = 0.0001 * sigma
    >>> pot_mat = array([pot_const, sigma, high_pow, low_pow, short_cutoff])
    >>> r = 15.0 * sigma  # particles' distance in [m]
    >>> lj_force(r, pot_mat)
    (-5.815308131440668e-28, -6.841538377536503e-19)

    """

    # rs = pot_matrix[4]
    # # Branchless programming
    # r = r_in * (r_in >= rs) + rs * (r_in < rs)

    epsilon = pot_matrix[0]
    sigma = pot_matrix[1]
    s_over_r = sigma / r
    s_over_r_high = s_over_r ** pot_matrix[2]
    s_over_r_low = s_over_r ** pot_matrix[3]

    u_r = epsilon * (s_over_r_high - s_over_r_low)
    f_r = epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

    return u_r, f_r


class LennardJones(Potential):

    def potential_derivatives(self, r, pot_matrix):
        """Calculate the first and second derivatives of the potential.

        Parameters
        ----------
        r_in : float
            Distance between two particles.

        pot_matrix : numpy.ndarray
            It contains potential dependent variables.

        Returns
        -------
        u_r : float, numpy.ndarray
            Potential value.

        dv_dr : float, numpy.ndarray
            First derivative of the potential.

        d2v_dr2 : float, numpy.ndarray
            Second derivative of the potential.


        """

        epsilon = pot_matrix[0]
        sigma = pot_matrix[1]
        s_over_r = sigma / r
        s_over_r_high = s_over_r ** pot_matrix[2]
        s_over_r_low = s_over_r ** pot_matrix[3]

        r2 = r * r

        u_r = epsilon * (s_over_r_high - s_over_r_low)
        dv_dr = -epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

        d2v_dr2 = (
            epsilon
            * (pot_matrix[2] * (pot_matrix[2] + 1) * s_over_r_high - pot_matrix[3] * (pot_matrix[3] + 1) * s_over_r_low)
            / r2
        )

        return u_r, dv_dr, d2v_dr2

    def pot_pretty_print(self):
        """
        Print potential specific parameters in a user-friendly way.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Class handling potential form.

        """

        rho = self.sigma_avg**self.dimensions * self.total_num_density
        tau = self.kB * self.T_desired / self.epsilon_tot
        msg = ( 
            f"epsilon_tot = {self.epsilon_tot/self.eV2J:.6e} {self.units_dict['electron volt']} = {self.epsilon_tot:6e} {self.units_dict['energy']}\n"
            f"sigma_avg = {self.sigma_avg/self.a_ws:.6e} a_ws =  {self.sigma_avg:6e} {self.units_dict['length']}\n"
            f"reduced density = {rho:.6e}\n"
            f"reduced temperature = {tau:.6e}\n"
            f"inverse reduced temperature = {1.0/tau:.6e}"
            )
        print(msg)

    def update_params(self, species = None):
        """
        Assign potential dependent simulation's parameters.

        Parameters
        ----------
        species : list, None
            List of species data (:class:`sarkas.plasma.Species`). 
            This is not used. It is there for compatibility with methods of other potentials.

        """
        self.matrix = zeros((self.num_species, self.num_species, 5))
        # See Lima Physica A 391 4281 (2012) for the following definitions
        if not hasattr(self, "powers"):
            self.powers = array([12, 6])

        exponent = self.powers[0] / (self.powers[1] - self.powers[0])
        lj_constant = self.powers[1] / (self.powers[0] - self.powers[1])
        lj_constant *= (self.powers[1] / self.powers[0]) ** exponent

        # Use the Lorentz-Berthelot mixing rules.
        # Lorentz: sigma_12 = 0.5 * (sigma_1 + sigma_2)
        # Berthelot: epsilon_12 = sqrt( eps_1 eps2)
        self.epsilon_tot = 0.0
        for i, sp1 in enumerate(species[:-1]): # -1 so that you don't look at the electron backround
            for j, sp2 in enumerate(species[:-1]):
                self.matrix[i, j, 0] = lj_constant * sqrt(sp1.epsilon) * sqrt(sp2.epsilon)
                self.matrix[i, j, 1] = 0.5 *(sp1.sigma + sp2.sigma)
                self.matrix[i, j, 2] = self.powers[0]
                self.matrix[i, j, 3] = self.powers[1]

                self.epsilon_tot += sp1.charge * sp2.charge
            # self.sigma_avg += sp1.sigma
        
        # self.sigma_avg = 0.0
        self.sigma_avg = self.matrix[:,:, 1].trace()/len(species[:-1])
        self.matrix[:, :, 4] = self.a_rs
    
        # if self.method == "pp":
        self.force_error = self.calc_force_error_quad(self.a_ws, self.rc, self.matrix[0, 0])
        # elif self.method == "brute":
        #     self.force_error = 0.0

    def force_error_integrand(self, r, pot_matrix):
        r"""Auxiliary function to be used in `scipy.integrate.quad` to calculate the integrand.

        Parameters
        ----------
        r_in : float
            Distance between two particles.

        pot_matrix : numpy.ndarray
            Slice of the `sarkas.potentials.Potential.matrix` containing the necessary potential parameters.

        Returns
        -------
        _ : float
            Integrand :math:`4\pi r^2 ( d r\phi(r)/dr )^2`

        """

        _, dv_dr, _ = self.potential_derivatives(r, pot_matrix)

        return 4.0 * pi * r**2 * dv_dr**2

    def calc_force_error_quad(self, a, rc, pot_matrix):
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
            Slice of the `sarkas.potentials.Potential.matrix` containing the parameters of the self. It must be a 1D-array.

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

        params = pot_matrix.copy()
        params[0] = 1
        # Un-dimensionalize the screening length.
        params[1] *= a
        r_c = rc / a
        result, _ = quad(self.force_error_integrand, a=r_c, b=inf, args=(params,))

        f_err = sqrt(result)

        return f_err

    @staticmethod
    @jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
    def force(r, pot_matrix):
        """
        Numba'd function to calculate the PP force between particles using Lennard-Jones Potential.

        Parameters
        ----------
        r_in : float
            Particles' distance.

        pot_matrix : numpy.ndarray
            LJ potential parameters. \n
            Shape = (5, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

        Returns
        -------
        u_r : float
            Potential.

        f_r : float
            Force.

        Examples
        --------
        >>> pot_const = 4.0 * 1.656e-21 # 4*epsilon in [J] (mks units)
        >>> sigma = 3.4e-10   # [m] (mks units)
        >>> high_pow, low_pow = 12., 6.
        >>> short_cutoff = 0.0001 * sigma
        >>> pot_mat = array([pot_const, sigma, high_pow, low_pow, short_cutoff])
        >>> r = 15.0 * sigma  # particles' distance in [m]
        >>> lj_force(r, pot_mat)
        (-5.815308131440668e-28, -6.841538377536503e-19)

        """

        # rs = pot_matrix[4]
        # # Branchless programming
        # r = r_in * (r_in >= rs) + rs * (r_in < rs)

        epsilon = pot_matrix[0]
        sigma = pot_matrix[1]
        s_over_r = sigma / r
        s_over_r_high = s_over_r ** pot_matrix[2]
        s_over_r_low = s_over_r ** pot_matrix[3]

        u_r = epsilon * (s_over_r_high - s_over_r_low)
        f_r = epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

        return u_r, f_r
