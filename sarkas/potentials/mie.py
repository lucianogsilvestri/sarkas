r"""
Module for handling Lennard-Jones interaction.

Potential
*********

The generalized Lennard-Jones potential, aka Mie potential, is defined as

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

The elements of the :attr:`sarkas.potentials.base.PotentialBase.matrix` are:

.. code-block::

    pot_matrix[0] = epsilon_12 * lj_constant
    pot_matrix[1] = sigmas
    pot_matrix[2] = highest power
    pot_matrix[3] = lowest power
    pot_matrix[4] = short-range cutoff

"""
from typing import Optional, Any
from numba import jit
from numpy import array, sqrt, zeros

from ..utilities.maths import force_error_analytic_lcl
from .base import PotentialBase

@jit(nopython=True)
def lj_force(r_in, pot_matrix):
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

    rs = pot_matrix[4]
    # Branchless programming
    r = r_in * (r_in >= rs) + rs * (r_in < rs)

    epsilon = pot_matrix[0]
    sigma = pot_matrix[1]
    s_over_r = sigma / r
    s_over_r_high = s_over_r ** pot_matrix[2]
    s_over_r_low = s_over_r ** pot_matrix[3]

    u_r = epsilon * (s_over_r_high - s_over_r_low)
    f_r = epsilon * (pot_matrix[2] * s_over_r_high - pot_matrix[3] * s_over_r_low) / r

    return u_r, f_r


class MiePotential(PotentialBase):
    """
    Mie potential class.
    """
    def __init__(self):
        super().__init__()
        self.type = "lj"
        self.screening_length_type = "unscreened"
        self.params = None
        self.powers = array([12, 6])
        self.epsilon_tot = 0.0
        self.sigma_avg = 0.0
        self.force = lj_force
        self.force_error = None

    def pretty_print_info(self):
        """Print potential specific parameters in a user-friendly way."""

        msg = f"Potential type: {self.type}\n"
        msg += f"epsilon_tot = {self.epsilon_tot:.6e} [eV] = {self.epsilon_tot:6e} {self.units_dict['energy']}"
        msg += f"sigma_avg = {self.sigma_avg:.6e} a_ws =  {self.sigma_avg:6e} {self.units_dict['length']}"
        return msg
    
    def initialize_potential_parameters(self, species_list: list[Any]) -> None:
        """
        Initialize potential-specific parameters.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        if self.params is not None:
            return
        num_species = len(species_list)

        # See Lima Physica A 391 4281 (2012) for the following definitions
        exponent = self.powers[0] / (self.powers[1] - self.powers[0])
        lj_constant = self.powers[1] / (self.powers[0] - self.powers[1])
        lj_constant *= (self.powers[1] / self.powers[0]) ** exponent

        # Use the Lorentz-Berthelot mixing rules.
        # Lorentz: sigma_ij = 0.5 * (sigma_i + sigma_j)
        # Berthelot: epsilon_ij = sqrt( eps_i eps_j)
        self.params = zeros((num_species, num_species, 4))
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list):
                self.params[i, j, 0] = lj_constant * sqrt(sp1.epsilon * sp2.epsilon)
                self.params[i, j, 1] = 0.5 * (sp1.sigma + sp2.sigma)
                self.params[i, j, 2] = self.powers[0]
                self.params[i, j, 3] = self.powers[1]
                
                self.epsilon_tot += sp1.epsilon * sp2.epsilon
            self.sigma_avg += sp1.sigma
        self.epsilon_tot = sqrt(self.epsilon_tot)
        self.sigma_avg = self.sigma_avg / num_species
        
    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create the parameter matrix for species interactions.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """

        num_species = len(species_list)
        self.matrix = zeros((num_species, num_species, 5))
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list):
                self.matrix[i, j, :4] = self.params[i][j]
                self.matrix[i, j, 4] = self.a_rs

    def set_force_function(self) -> None:
        """
        Set the force calculation function.
        """
        self.force = lj_force

    def set_algorithm_parameters(self, algorithm_type: Optional[str], **kwargs):
        """
        Set algorithm-specific parameters.
        """
        if algorithm_type is not None:
            self.algorithm_type = algorithm_type
        
        if 'a_rs' in kwargs:
            self.a_rs = kwargs['a_rs']

    
    def potential_derivatives(self, r_in, pot_matrix):
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
        
        rs = pot_matrix[-1]
        r = r_in * (r_in >= rs) + rs * (r_in < rs)

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