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
from warnings import warn

from .core import Potential
from ..utilities.maths import force_error_analytic_lcl, force_error_analytic_pp



class Yukawa_SR(Potential):

    @staticmethod
    @jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
    def force(r_in, pot_matrix):
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


    def force_deriv(self, r, pot_matrix):
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

        _, _, d2v_dr2 = self.potential_derivatives(r, pot_matrix)

        return d2v_dr2


    def potential_derivatives(self, r, pot_matrix):
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
        dv_dr = -(1.0 + kappa_r) * u_r / r
        d2v_dr2 = -(1.0 / r + kappa) * dv_dr + u_r / r**2

        u_r *= pot_matrix[0]
        dv_dr *= pot_matrix[0]
        d2v_dr2 *= pot_matrix[0]

        return u_r, dv_dr, d2v_dr2


    def pot_pretty_print(self):
        """
        Print potential specific parameters in a user-friendly way.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Class handling potential form.

        """
        msg = (
            f"screening type : {self.screening_length_type}\n"
            f"screening length = {self.screening_length:.6e} {self.units_dict['length']}\n"
            f"kappa = {self.a_ws / self.screening_length:.4f}\n"
            f"Gamma_eff = {self.coupling_constant:.2f}"
        )
        print(msg)


    def update_params(self, species):
        """
        Assign potential dependent simulation's parameters.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Class handling potential form.

        species: list,
            List of species in the simulation. Unused.

        """

        self.calc_screening_length(species)

        self.matrix = zeros((self.num_species, self.num_species, 3))

        self.matrix[:, :, 1] = 1.0 / self.screening_length

        # self.matrix[:, :, 0] = self.species_charges.reshape((len(self.species_charge), 1))
        # * self.species_charges / self.fourpie0
        # the above line is the Python version of the for loops below. I believe that the for loops are easier to understand
        for i, q1 in enumerate(self.species_charges):
            for j, q2 in enumerate(self.species_charges):
                self.matrix[i, j, 0] = q1 * q2 / self.fourpie0

        self.matrix[:, :, -1] = self.a_rs

        # self.force_error = force_error_analytic_lcl(
        #     self.type, self.rc, self.matrix, sqrt(3.0 * self.a_ws / (4.0 * pi))
        # )
        self.force_error = self.calc_force_error_quad(self.a_ws, self.rc, self.matrix[0, 0])

        # # Force error calculated from eq.(43) in Ref.[1]_
        # self.force_error = sqrt( TWOPI / self.electron_TF_wavelength) * exp(- self.rc / self.electron_TF_wavelength)
        # # Renormalize
        # self.force_error *= self.a_ws ** 2 * sqrt(self.total_num_ptcls / self.pbox_volume)


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


        params = pot_matrix.copy()
        params[0] = 1
        # Un-dimensionalize the screening length.
        params[1] *= a
        r_c = rc / a
        result, _ = quad(self.force_error_integrand, a=r_c, b=inf, args=(params,))

        f_err = sqrt(result)

        return f_err


class Yukawa_LR(Potential):

    @staticmethod
    @jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
    def force(r_in, pot_matrix):
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
        f1 = (0.5 / r) * exp(kappa * r) * erfc(alpha_r + 0.5 * kappa_alpha) * (1.0 / r - kappa)
        f2 = (0.5 / r) * exp(-kappa * r) * erfc(alpha_r - 0.5 * kappa_alpha) * (1.0 / r + kappa)
        # Derivative of erfc(a r) = 2a/sqrt(pi) e^{-a^2 r^2}* (x/r)
        f3 = (alpha / sqrt(pi) / r) * (
            exp(-((alpha_r + 0.5 * kappa_alpha) ** 2)) * exp(kappa_r)
            + exp(-((alpha_r - 0.5 * kappa_alpha) ** 2)) * exp(-kappa_r)
        )
        f_r = pot_matrix[0] * (f1 + f2 + f3)

        return u_r, f_r

    def force_deriv(self, r, pot_matrix):
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

        _, _, d2v_dr2 = self.potential_derivatives(r, pot_matrix)

        return d2v_dr2

    def potential_derivatives(self, r, pot_matrix):
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
        dv_dr = -(1.0 + kappa_r) * u_r / r
        d2v_dr2 = -(1.0 / r + kappa) * dv_dr + u_r / r**2

        u_r *= pot_matrix[0]
        dv_dr *= pot_matrix[0]
        d2v_dr2 *= pot_matrix[0]

        return u_r, dv_dr, d2v_dr2

    def pot_pretty_print_info(self):
        """
        Print potential specific parameters in a user-friendly way.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Class handling potential form.

        """
        msg = (
            f"screening type : {self.screening_length_type}\n"
            f"screening length = {self.screening_length:.6e} {self.units_dict['length']}\n"
            f"kappa = {self.a_ws / self.screening_length:.4f}\n"
            f"Gamma_eff = {self.coupling_constant:.2f}"
        )
        print(msg)

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

        params = pot_matrix.copy()
        params[0] = 1
        # Un-dimensionalize the screening length.
        params[1] *= a
        r_c = rc / a
        result, _ = quad(self.force_error_integrand, a=r_c, b=inf, args=(params,))

        f_err = sqrt(result)

        return f_err

    def update_params(self, species):
        """
        Assign potential dependent simulation's parameters.

        Parameters
        ----------
        potential : :class:`sarkas.potentials.core.Potential`
            Class handling potential form.

        """        
        self.calc_screening_length(species)
        
        self.matrix = zeros((self.num_species, self.num_species, 4))
        self.matrix[:, :, 1] = 1.0 / self.screening_length

        # self.matrix[:, :, 0] = self.species_charges.reshape((len(self.species_charge), 1))
        # * self.species_charges / self.fourpie0
        # the above line is the Python version of the for loops below. I believe that the for loops are easier to understand
        for i, q1 in enumerate(self.species_charges):
            for j, q2 in enumerate(self.species_charges):
                self.matrix[i, j, 0] = q1 * q2 / self.fourpie0

        self.matrix[:, :, 2] = self.pppm_alpha_ewald
        self.matrix[:, :, -1] = self.a_rs

        rescaling_constant = sqrt(self.total_num_ptcls) * self.a_ws**2 / sqrt(self.pbox_volume)
        self.pppm_pp_err = force_error_analytic_pp(
            self.type, self.rc, self.screening_length, self.pppm_alpha_ewald, rescaling_constant
        )

        # PP force error calculation. Note that the equation was derived for a single component plasma.
        # kappa_over_alpha = -0.25 * (self.matrix[0, 0, 1] / self.matrix[0, 0, 2]) ** 2
        # alpha_times_rcut = -((self.matrix[0, 0, 2] * self.rc) ** 2)
        # self.pppm_pp_err = 2.0 * exp(kappa_over_alpha + alpha_times_rcut) / sqrt(self.rc)
        # self.pppm_pp_err *= sqrt(self.total_num_ptcls) * self.a_ws ** 2 / sqrt(self.pbox_volume)
