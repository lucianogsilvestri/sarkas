"""
Coulomb potential implementation.

The Coulomb potential for charged particle interactions is given by:

.. math::
    U_{ij}(r) = \frac{q_i q_j}{4\pi \epsilon_0 r}

For PPPM, the short-range part uses the complementary error function:

.. math::
    U_{ij}^{\text{short}}(r) = \frac{q_i q_j}{4\pi \epsilon_0 r} \text{erfc}(\alpha r)


"""

from math import erfc
from numba import jit
from numpy import exp, inf, pi, sqrt, zeros, array
from typing import Any, Callable, Optional, List
from warnings import warn

from .base import PotentialBase

@jit(nopython=True)
def coulomb_force(r_in: float, pot_matrix: Any) -> tuple[float, float]:
    """
    Calculate the pure Coulomb potential and force.

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters for a specific species pair.
        pot_matrix[0] = q_i * q_j / (4 * pi * eps0)
        pot_matrix[1] = 0.0 # alpha_ewald (Ewald parameter)
        pot_matrix[2] = a_rs (short-range cutoff)

    Returns
    -------
    u_r : float
        Potential value.
    f_r : float
        Force magnitude.
    """
    pot_matrix_copy = pot_matrix.copy()
    pot_matrix_copy[-2] = 0.0 # Set Ewald parameter to 0.0
    return coulomb_force_pppm(r_in, pot_matrix_copy)

@jit(nopython=True)
def coulomb_force_pppm(r_in: float, pot_matrix: Any) -> tuple[float, float]:
    """
    Calculate the short-range part of the Coulomb potential and force (PPPM/Ewald).

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters for a specific species pair.
        pot_matrix[0] = q_i * q_j / (4 * pi * eps0)
        pot_matrix[1] = alpha_ewald (Ewald parameter)
        pot_matrix[2] = a_rs (short-range cutoff)

    Returns
    -------
    u_r : float
        Short-range potential value.
    f_r : float
        Short-range force magnitude.
    """
    alpha = pot_matrix[1]
    rs = pot_matrix[2]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    u_r = pot_matrix[0] * erfc(alpha * r) / r
    f_r = pot_matrix[0] * (
        erfc(alpha * r) / r**2 + (2.0 * alpha / (sqrt(pi) * r)) * exp(-(alpha * r) ** 2)
    )
    return u_r, f_r


class Coulomb(PotentialBase):
    """
    Coulomb potential implementation.

    Attributes
    ----------
    coulomb_params : list or ndarray
        coulomb_params[i][j][0] = q_i * q_j / (4π ε₀)
        (User can override before setup)
    pppm_alpha_ewald : float
        Algorithm-specific parameter for PPPM.
    a_rs : float
        Algorithm-specific short-range cutoff.

    Usage Example
    -------------
    >>> c = Coulomb()
    >>> # Optionally override potential-specific parameters before setup:
    >>> c.coulomb_params = [[[1.0], [1.0]], [[1.0], [1.0]]]  # shape (num_species, num_species, 1)
    >>> c.pppm_alpha_ewald = 0.25
    >>> c.a_rs = 0.1
    >>> c.setup(params, species_list)
    """
    def __init__(self) -> None:
        super().__init__()
        self.type = "coulomb"
        self.pppm_alpha_ewald: Optional[float] = None
        self.algorithm_type: str = "pppm"
        self.params = None
        self.force = coulomb_force_pppm

    def initialize_potential_parameters(self, species_list: list[Any]) -> None:
        """
        Initialize Coulomb-specific parameters for all species pairs.
        If self.params is set by the user, use those values.
        Otherwise, compute defaults.
        """
        if self.params is not None:
            return
        num_species = len(species_list)
        fourpie0 = self.fourpie0 if self.fourpie0 is not None else 1.0
        self.params = array([
            [
                [sp1.charge * sp2.charge / fourpie0]
                for sp2 in species_list
            ]
            for sp1 in species_list
        ])

    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create parameter matrix for Coulomb interactions from self.params and algorithm-specific parameters.
        """
        num_species = len(species_list)

        self.matrix = zeros((num_species, num_species, 3))
        for i in range(num_species):
            for j in range(num_species):
                self.matrix[i, j, 0] = self.params[i][j][0]
                self.matrix[i, j, 1] = self.pppm_alpha_ewald
                self.matrix[i, j, 2] = self.a_rs

    def set_force_function(self) -> None:
        """
        Set the appropriate force function based on algorithm.
        """
        if self.algorithm_type == "pppm":
            self.force_function = coulomb_force_pppm
        else:
            raise NotImplementedError("Coulomb potential is only implemented for the PPPM method.")

    def set_algorithm_parameters(self, algorithm_type: Optional[str], **kwargs: Any) -> None:
        """
        Set algorithm-specific parameters.

        Parameters
        ----------
        algorithm_type : str
            Type of algorithm ('pp', 'pppm', 'fmm')
        **kwargs : dict
            Algorithm-specific parameters:
            - alpha_ewald : float (for PPPM)
        """
        if algorithm_type is not None:
            self.algorithm_type = algorithm_type

        if self.algorithm_type == "pppm":
            if 'pppm_alpha_ewald' in kwargs:
                self.pppm_alpha_ewald = kwargs['pppm_alpha_ewald']
            else:
                raise ValueError("pppm_alpha_ewald must be provided in kwargs")

        if 'a_rs' in kwargs:
            self.a_rs = kwargs['a_rs']

    def pretty_print_info(self) -> str:
        """
        Print potential-specific parameters in a user-friendly way.
        """
        msg = (
            f"Potential type: {self.type}\n"
            f"Algorithm: {self.algorithm_type}\n"
            f"Ewald parameter (alpha): {self.pppm_alpha_ewald}\n"
            f"Short-range cutoff (a_rs): {self.a_rs}\n"
        )
        return msg

    def potential_derivatives(self, r: float, pot_matrix: Any) -> tuple[float, float, float]:
        """
        Calculate the first and second derivatives of the Coulomb potential.

        Parameters
        ----------
        r : float
            Distance between particles.
        pot_matrix : numpy.ndarray
            Potential parameters.

        Returns
        -------
        u_r : float
            Potential value.
        dv_dr : float
            First derivative of the potential.
        d2u_dr2 : float
            Second derivative of the potential (not implemented, returns 0.0).
        """
        u_r = pot_matrix[0] / r
        du_dr = -pot_matrix[0] / r**2
        d2u_dr2 = 2.0 * pot_matrix[0] / r**3
        return u_r, du_dr, d2u_dr2

