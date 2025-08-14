r"""
Module for handling Quantum Statistical Potentials.

Potential
*********

Quantum Statistical Potentials are defined by three terms

.. math::
    U(r) = U_{\rm pauli}(r) + U_{\rm coul} + U_{\rm diff} (r)

where

.. math::
    U_{\rm pauli}(r) = k_BT \ln (2)  e^{ - 4\pi r^2/ \Lambda_{ab}^2 }

is due to the Pauli exclusion principle,

.. math::
    U_{\rm coul}(r) = \frac{q_iq_j}{4\pi \epsilon_0} \frac{1}{r}

is the usual Coulomb interaction, and :math:`U_{\rm diff}(r)` is a diffraction term.

There are two possibilities for the diffraction term. The most common is the Deutsch Potential

.. math::
    U_{\rm deutsch}(r) = \frac{q_aq_b}{4\pi \epsilon_0} \frac{e^{- 2 \pi r/\Lambda_{ab}} }{r}.

The second most common form is the Kelbg potential

.. math::
    U_{\rm kelbg}(r) = - \frac{q_aq_b}{4\pi \epsilon_0} \frac{1}{r} \left [  e^{- 2 \pi r^2/\Lambda_{ab}^2 }
    - \sqrt{2} \pi \dfrac{r}{\Lambda_{ab}} \textrm{erfc} \left ( \sqrt{ 2\pi}  r/ \Lambda_{ab} \right )
    \right ].

In the above equations the screening length :math:`\Lambda_{ab}` is the thermal de Broglie wavelength
between the two charges defined as

.. math::
   \Lambda_{ab} = \sqrt{\frac{2\pi \hbar^2}{\mu_{ab} k_BT}}, \quad  \mu_{ab} = \frac{m_a m_b}{m_a + m_b}


Note that in Ref. :cite:`Hansen1981` the DeBroglie wavelength is defined as

.. math::
   \Lambda_{ee} = \sqrt{ \dfrac{\hbar^2}{2 \pi \mu_{ee} k_{B} T}},

while in statistical physics textbooks is defined as

.. math::
   \Lambda_{ee} = \sqrt{ \dfrac{2 \pi \hbar^2}{\mu_{ee} k_{B} T}} .

The latter will be used in Sarkas. The difference is in the factor of :math:`2\pi`, i.e. the difference between
a wave number and wave length.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = qi*qj/4*pi*eps0
    pot_matrix[1] = 2pi/deBroglie
    pot_matrix[2] = e-e Pauli term factor
    pot_matrix[3] = e-e Pauli term exponent term
    pot_matrix[4] = Ewald parameter
    pot_matrix[5] = Short-range cutoff

"""

from math import erfc
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, exp, inf, log, ndarray, pi, sqrt, unravel_index, zeros, isclose
from warnings import warn
from scipy.integrate import quad
from scipy.special import gamma

from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import force_error_analytic_pp, TWOPI


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def deutsch_force(r, pot_matrix):
    """
    Calculate Deutsch QSP Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Potential.

    f_r : float
        Force between two particles.


    """

    A = pot_matrix[0]
    C = pot_matrix[1]
    D = pot_matrix[2]
    F = pot_matrix[3]
    alpha = pot_matrix[4]

    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    u_ewald = A * erfc(alpha * r) / r
    f_ewald = u_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / sqrt(pi)) * exp(-a2 * r2) / r  # erfc derivative

    # Diffraction potential and force term
    u_diff = -A * exp(-C * r) / r
    f_diff = u_diff * (1.0 / r + C)  # 1/r derivative

    # Pauli Term
    u_pauli, f_pauli = pauli_force(r, pot_matrix)

    u_r = u_ewald + E * u_diff + u_pauli
    f_r = f_ewald + E * f_diff + f_pauli

    return u_r, f_r


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def pauli_force(r, pot_matrix):
    """
    Calculate Pauli term of the QSP potential

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Pauli Potential.

    f_r : float
        Pauli Force between two particles.


    """
    D = pot_matrix[2]
    F = pot_matrix[3]

    r2 = r * r

    # Pauli Term
    u_r = D * log(1.0 - 0.5 * exp(-F * r2))
    f_r = - D * ( r * F * exp(-F * r2)) / (1.0 - 0.5 * exp(-F * r2))

    return u_r, f_r


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def hansen_force(r, pot_matrix):
    """
    Calculate Deutsch QSP Force between two particles.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    U : float
        Potential.

    force : float
        Force between two particles.


    """

    A = pot_matrix[0]
    C = pot_matrix[1]
    D = pot_matrix[2]
    F = pot_matrix[3]
    alpha = pot_matrix[4]

    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * erfc(alpha * r) / r
    f_ewald = U_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / sqrt(pi)) * exp(-a2 * r2) / r  # erfc derivative

    # Diffraction potential and force term
    U_diff = -A * exp(-C * r) / r
    f_diff = U_diff / r  # 1/r derivative
    f_diff += -A * C * exp(-C * r) / r  # exp derivative

    # Pauli potential and force terms
    U_pauli = D * exp(-F * r2)
    f_pauli = 2.0 * r * D * F * exp(-F * r2)

    U = U_ewald + U_diff + U_pauli
    force = f_ewald + f_diff + f_pauli

    return U, force


@jit(UniTuple(float64, 2)(float64, float64[:]), nopython=True)
def kelbg_force(r, pot_matrix):
    """
    Calculates the QSP Force between two particles when the pppm algorithm is chosen.

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Potential.

    force : float
        Force between two particles.

    Notes
    -----
    The Kelbg potential is defined as

    .. math::
        U_{\rm kelbg}(r) = - \frac{q_aq_b}{4\pi \epsilon_0} \frac{1}{r} \left [  e^{- 2 \pi r^2/\Lambda_{ab}^2 }
        - \sqrt{2} \pi \dfrac{r}{\Lambda_{ab}} \textrm{erfc} \left ( \sqrt{ 2\pi}  r/ \Lambda_{ab} \right )
        \right ].

    where :math:`\Lambda_{ab}` is the thermal de Broglie wavelength between the two charges. The `pot_matrix` should have the following elements
    
    pot_matrix[0] = qi*qj/4*pi*eps0
    pot_matrix[1] = sqrt(2pi)/deBroglie
    pot_matrix[2] = e-e Pauli term factor (O or 1)
    pot_matrix[3] = e-e Pauli term exponent term
    pot_matrix[4] = Ewald parameter
    pot_matrix[5] = Short-range cutoff
    """

    A = pot_matrix[0]  # qi*qj/4*pi*eps0
    C = pot_matrix[1]  # sqrt(2pi)/deBroglie
    D = pot_matrix[2]  # e-e Pauli term factor
    F = pot_matrix[3]
    E = pot_matrix[4] # flag for diffraction term 
    alpha = pot_matrix[5]

    C2 = C * C
    a2 = alpha * alpha
    r2 = r * r

    # Ewald short-range potential and force terms
    U_ewald = A * erfc(alpha * r) / r
    f_ewald = U_ewald / r  # 1/r derivative
    f_ewald += A * (2.0 * alpha / sqrt(pi) / r) * exp(-a2 * r2)  # erfc derivative

    # potential
    erfc_argument = C * r 
    u_r_diff = A * C * sqrt(pi) * erfc(erfc_argument)  # C = sqrt(2pi)/deBroglie hence C * sqrt(pi) = sqrt(2)/deBroglie * pi 
    u_r_diff_1 = -A * exp(-C2 * r2) / r
    # Force
    dvdr_diff = A * 2.0 * C2 * exp(-C2 * r2)   # erfc derivative
    dvdr_diff_1 = u_r_diff_1 * (1.0 / r + 2.0 * C2 * r)  # exp(r^2)/r derivative

    # Pauli Term
    U_pauli, f_pauli = pauli_force(r, pot_matrix)

    u_r = U_ewald + E * (u_r_diff + u_r_diff_1) + U_pauli
    force = f_ewald + E * (dvdr_diff + dvdr_diff_1) + f_pauli

    return u_r, force


def pauli_term_derivatives(r, pot_matrix):
    """
    Calculate the Pauli term of the QSP potential

    Parameters
    ----------
    r : float
        Distance between two particles.

    pot_matrix : numpy.ndarray
        It contains potential dependent variables. \n
        Shape = (6, :attr:`sarkas.core.Parameters.num_species`, :attr:`sarkas.core.Parameters.num_species`)

    Returns
    -------
    u_r : float
        Pauli Potential.
    
    dvdr : float
        Pauli Force between two particles.
    
    d2v_dr2 : float
        Pauli Force between two particles.
    
    """
    D = pot_matrix[2]
    F = pot_matrix[3]
    A = pot_matrix[4]

    r2 = r * r
    F2 = F * F

    # Pauli Term
    u_r = D * log(1.0 - 0.5 * A * exp(-F2 * r2))
    dvdr = - D *  r * F2 * A * exp(-F2 * r2) / (1.0 - 0.5 * A * exp(- F2 * r2) )
    denom = (1.0 - 0.5 * A * exp(- F2 * r2) - 0.5)**2
    d2v_dr2 = D * F2 * ( A * exp(-F2 * r2) *( 1 - 2.0 * F2 * r**2 ) - 0.5 * A * exp(- 2.0 * F2 * r2) ) / denom

    return u_r, dvdr, d2v_dr2


def deutsch_potential_derivatives(r, pot_matrix):
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

    A = pot_matrix[0]  # qi*qj/4*pi*eps0
    C = pot_matrix[1]  # 2pi/deBroglie
    E = pot_matrix[5]  # flag for diffraction term

    r2 = r * r
    r3 = r2 * r

    # Pauli term. Note that D = 0 if Pauli is false
    # u_r_pauli = D * log(1.0 - 0.5 * exp(-F * r2))
    # dvdr_pauli = r * F / (exp(F * r2) - 0.5)
    # d2v_dr2_pauli = -2.0 * F * (exp(F * r2) * (4 * F * r2 - 2) + 1) / (2.0 * exp(F * r2) - 1.0) ** 2
    u_r_pauli, dvdr_pauli, d2v_dr2_pauli = pauli_term_derivatives(r, pot_matrix)

    # Diffraction potential and force term
    u_r_diff = -A * exp(-C * r) / r
    dvdr_diff = -u_r_diff * (1.0 / r + C)  # 1/r derivative
    d2v_dr2_diff = u_r_diff / r2 + dvdr_diff * (1.0 / r + C)

    # Coulomb part
    u_r_coul = A / r
    dvdr_coul = -A / r2
    d2v_dr2_coul = 2.0 * A / r3

    u_r = u_r_coul + E * u_r_diff + u_r_pauli
    dv_dr = dvdr_coul + E * dvdr_diff + dvdr_pauli
    d2v_dr2 = d2v_dr2_coul + E * d2v_dr2_diff + d2v_dr2_pauli

    return u_r, dv_dr, d2v_dr2


def hansen_potential_derivatives(r, pot_matrix):
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

    A = pot_matrix[0]  # qi*qj/4*pi*eps0
    C = pot_matrix[1]  # 2pi/deBroglie
    D = pot_matrix[2]  # e-e Pauli term factor
    F = pot_matrix[3]  # e-e Pauli term exponent term
    E = pot_matrix[5]  # flag for diffraction term

    r2 = r * r
    r3 = r2 * r

    # Pauli potential and force terms. Note that D = 0 if Pauli is false
    u_r_pauli = D * exp(-F * r2)
    dvdr_pauli = 2.0 * F * r * u_r_pauli
    d2v_dr2_pauli = 2.0 * F * u_r_pauli + dvdr_pauli * 2.0 * F * r

    # Diffraction potential and force term
    u_r_diff = -A * exp(-C * r) / r
    dvdr_diff = -u_r_diff * (1.0 / r + C)  # 1/r derivative
    d2v_dr2_diff = u_r_diff / r2 + dvdr_diff * (1.0 / r + C)

    # Coulomb part
    u_r_coul = A / r
    dvdr_coul = -A / r2
    d2v_dr2_coul = 2.0 * A / r3

    u_r = u_r_coul + E * u_r_diff + u_r_pauli
    dv_dr = dvdr_coul + E * dvdr_diff + dvdr_pauli
    d2v_dr2 = d2v_dr2_coul + E * d2v_dr2_diff + d2v_dr2_pauli

    return u_r, dv_dr, d2v_dr2


def kelbg_potential_derivatives(r, pot_matrix):
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

    A = pot_matrix[0]  # qi*qj/4*pi*eps0
    C = pot_matrix[1]  # 2pi/deBroglie
    E = pot_matrix[5]  # flag for diffraction term

    r2 = r * r
    r3 = r2 * r

    # Pauli term. Note that D = 0 if Pauli is false
    u_r_pauli, dvdr_pauli, d2v_dr2_pauli = pauli_term_derivatives(r, pot_matrix)

    # Diffraction potential and force term
    C2 = C * C
    # potential
    u_r_diff = A * C * sqrt(pi) * r * erfc(C * r / sqrt(pi))
    u_r_diff_1 = -A * exp(-C2 * r2 / pi) / r
    # Force
    dvdr_diff = -2.0 * A * C2 * exp(-C2 * r2 / pi) / pi  # erfc derivative
    dvdr_diff_1 = -u_r_diff_1 * (1.0 / r + 2.0 * C2 * r / pi)  # exp(r)/r derivative
    # 
    d2v_dr2_diff = dvdr_diff * (-2.0 * C2 * r / pi)
    d2v_dr2_diff_1 = u_r_diff_1 * (1.0 / r2 - 2.0 * C2 / pi) + (1.0 / r + 2.0 * C2 * r / pi) * dvdr_diff_1

    u_r_diff += u_r_diff_1
    dvdr_diff += dvdr_diff_1
    d2v_dr2_diff += d2v_dr2_diff_1

    # Coulomb part
    u_r_coul = A / r
    dvdr_coul = -A / r2
    d2v_dr2_coul = 2.0 * A / r3

    u_r = u_r_coul + E * u_r_diff + u_r_pauli
    dv_dr = dvdr_coul + E * dvdr_diff + dvdr_pauli
    d2v_dr2 = d2v_dr2_coul + E * d2v_dr2_diff + d2v_dr2_pauli

    return u_r, dv_dr, d2v_dr2


def pretty_print_info(potential):
    """
    Print potential specific parameters in a user-friendly way.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.

    """

    ii_scr_len = 1.0 / potential.matrix[1, 1, 1]
    ei_scr_len = 1.0 / potential.matrix[0, 1, 1]
    ee_scr_len = 1.0 / potential.matrix[0, 0, 1]
    e_deBroglie_lambda = sqrt(2.0) * pi / potential.matrix[0, 0, 1]
    i_deBroglie_lambda = sqrt(2.0) * pi / potential.matrix[1, 1, 1]
    a_ws = potential.a_ws

    info_str = f"QSP type: {potential.qsp_type}\n"
    info_str += f"Pauli term: {potential.qsp_pauli}\n"
    info_str += f"e de Broglie wavelength = {e_deBroglie_lambda / a_ws:.4e} a_ws = {e_deBroglie_lambda:.6e} {potential.units_dict['length']}\n"
    info_str += f"ion de Broglie wavelength  = {i_deBroglie_lambda / a_ws:.4e} a_ws = {i_deBroglie_lambda:.6e} {potential.units_dict['length']}\n"
    info_str += (
        f"In the following screening length/kappa refers to the argument in the exponential of the diffraction term.\n"
    )
    info_str += (
        f"e-e screening length = {ee_scr_len / a_ws:.4e} a_ws = {ee_scr_len:.6e} {potential.units_dict['length']}\n"
    )
    info_str += f"e-e screening kappa = {potential.matrix[0, 0, 1] * a_ws:.4e}\n"
    info_str += (
        f"i-i screening length = {ii_scr_len / a_ws:.4e} a_ws = {ii_scr_len:.6e} {potential.units_dict['length']}\n"
    )
    info_str += f"i-i screening kappa = {potential.matrix[1, 1, 1] * a_ws:.4e}\n"
    info_str += (
        f"e-i screening length = {ei_scr_len / a_ws:.4e} a_ws = {ei_scr_len:.6e} {potential.units_dict['length']}\n"
    )
    info_str += f"e-i coupling constant = {potential.coupling_constant:.4e}\n"
    info_str += f"e-i screening kappa = a_i/lambda_TF = {potential.ai / potential.screening_length:.4e}"

    print(info_str)


def update_params(potential, species):
    """
    Create potential dependent simulation's parameters.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.
        Can include custom diffractive lengths via:
        - potential.ee_diffractive_length: Custom e-e diffractive length
        - potential.ei_diffractive_length: Custom e-i diffractive length(s)
                                         Can be a single value or a list matching the number of ion species

    species : list,
        List of species data (:class:`sarkas.plasma.Species`).
    """
    # Do a bunch of checks
    # pppm algorithm only
    if potential.method != "pppm":
        raise ValueError("QSP interaction can only be calculated using pppm algorithm.")

    # Check for neutrality
    if ~isclose(potential.total_net_charge, 0.0):
        warn("Total net charge is not zero.", category=AlgorithmWarning)

    # Default attributes
    if not hasattr(potential, "qsp_type"):
        potential.qsp_type = "deutsch"
    if not hasattr(potential, "qsp_pauli"):
        potential.qsp_pauli = True

    # Enforce consistency
    potential.qsp_type = potential.qsp_type.lower()

    four_pi = 2.0 * TWOPI
    log_2 = log(2.0)

    # Redefine ion temperatures and ion total number density
    total_ion_temperature = 0.0
    total_ion_number_density = 0.0
    for _, sp1 in enumerate(species[1:]):
        total_ion_temperature += sp1.concentration * sp1.temperature
        total_ion_number_density += sp1.number_density

    # Calculate the total and ion Wigner-Seitz Radius from the total density
    potential.ai = (3.0 / (four_pi * total_ion_number_density)) ** (1.0 / 3.0)  # Ion WS

    theta = species[0].degeneracy_parameter
    # This is the Jones and Murillo definition of the de Broglie wavelength
    deBroglie_const = TWOPI * potential.hbar**2 / potential.kB

    potential.matrix = zeros((potential.num_species, potential.num_species, 8))

    # Check for custom diffractive lengths
    has_custom_ee = hasattr(potential, "ee_diffractive_length")
    has_custom_ei = hasattr(potential, "ei_diffractive_length")
    
    # If ei_diffractive_length is provided, verify it's either a single value
    # or a list matching the number of ion species
    if has_custom_ei:
        num_ion_species = potential.num_species - 1  # total minus electrons
        if isinstance(potential.ei_diffractive_length, (list, tuple, ndarray)):
            if len(potential.ei_diffractive_length) != num_ion_species:
                raise ValueError(f"ei_diffractive_length list must have {num_ion_species} elements (one per ion species)")
        else:
            # Convert single value to list for consistent handling
            potential.ei_diffractive_length = [potential.ei_diffractive_length] * num_ion_species

    for i, sp1 in enumerate(species):
        m1 = sp1.mass
        q1 = sp1.charge

        for j, sp2 in enumerate(species):
            m2 = sp2.mass
            q2 = sp2.charge

            reduced = (m1 * m2) / (m1 + m2)

            if sp1.name == "e" or sp2.name == "e":
                # Handle electron-related interactions (e-e and e-i)
                if sp1.name == sp2.name and sp1.name == "e":  # e-e interaction
                    if has_custom_ee:
                        # Use user-provided e-e diffractive length
                        lambda_deB = potential.ee_diffractive_length
                    else:
                        # Calculate e-e diffractive length
                        lambda_deB = sqrt(deBroglie_const / (reduced * species[0].temperature))
                    
                    if potential.qsp_type == "hansen":
                        potential.matrix[i, j, 2] = log_2 * potential.kB * sp1.temperature
                        potential.matrix[i, j, 3] = sqrt(four_pi / (log_2 * lambda_deB**2))
                    else:
                        # See eq.(44), (45) Jones and Murillo, HEDP (2007)
                        a1, a2, a3 = 0.2975, 6.090, 1.541
                        b1, b2, b3, b4 = 0.0842, 0.1027, 1.096, 1.359
                        A_theta = 1.0 + a1 / (1.0 + a2 * theta**a3)
                        B_theta = 1.0 + b1 * exp(-b2 * theta**b3) / theta**b4

                        potential.matrix[i, j, 2] = -potential.kB * sp1.temperature
                        potential.matrix[i, j, 3] = sqrt(TWOPI * B_theta) / lambda_deB
                        potential.matrix[i, j, 4] = A_theta
                else:  # e-i interaction
                    if has_custom_ei:
                        # Determine which ion species we're working with to get the right diffractive length
                        if sp1.name == "e":
                            ion_index = j - 1  # Offset because e is at index 0
                        else:
                            ion_index = i - 1  # Offset because e is at index 0
                        
                        # Use user-provided e-i diffractive length
                        lambda_deB = potential.ei_diffractive_length[ion_index]
                    else:
                        # Calculate e-i diffractive length
                        lambda_deB = sqrt(deBroglie_const / (reduced * species[0].temperature))

                potential.matrix[i, j, 5] = 1.0
            else:  # i-i interaction
                # Use ion temperature in i-i interactions only
                lambda_deB = sqrt(deBroglie_const / (reduced * total_ion_temperature))
                potential.matrix[i, j, 5] = 0.0   # No diffraction term for i-i interactions
                potential.matrix[i, j, 2] = 0.0   # No Pauli term for i-i interactions
                potential.matrix[i, j, 3] = 0.0   # No Pauli term for i-i interactions
                potential.matrix[i, j, 4] = 0.0   # No Pauli term for i-i interactions

            potential.matrix[i, j, 0] = q1 * q2 / potential.fourpie0
            potential.matrix[i, j, 1] = sqrt(TWOPI) / lambda_deB if potential.qsp_type == "kelbg" else TWOPI / lambda_deB

    if not potential.qsp_pauli:
        potential.matrix[:, :, 2] = 0.0
        potential.matrix[:, :, 3] = 0.0
        potential.matrix[:, :, 4] = 0.0

    potential.matrix[:, :, 6] = potential.pppm_alpha_ewald
    potential.matrix[:, :, 7] = potential.a_rs

    if potential.qsp_type == "deutsch":
        potential.force = deutsch_force
        potential.potential_derivatives = deutsch_potential_derivatives

    elif potential.qsp_type == "hansen":
        potential.force = hansen_force
        potential.potential_derivatives = hansen_potential_derivatives

    elif potential.qsp_type == "kelbg":
        potential.force = kelbg_force
        potential.potential_derivatives = kelbg_potential_derivatives


def calc_force_error_quad(potential):

    pot_matrix = potential.matrix.copy()

    # Rescale the q_iq_j term with e-e value
    pot_matrix[:, :, 0] /= potential.matrix[0, 0, 0] 
    
    # Rescale the diffraction lengths by the WS radius
    pot_matrix[:, :, 1] *= potential.a_ws

    # Rescale the Pauli term
    pot_matrix[:, :, 2] /= potential.matrix[0, 0, 0]
    pot_matrix[:, :, 3] *= potential.a_ws

    # Rescale the Ewald parameter and the short-range cutoff
    pot_matrix[:, :, 5] *= potential.a_ws
    pot_matrix[:, :, 6] /= potential.a_ws

    r_c = potential.rc / potential.a_ws

    # Solid angle integral
    solid_angle = 2.0 * pi**(potential.dimensions / 2) / gamma(potential.dimensions / 2)

    integrand = lambda r: solid_angle * r**(potential.dimensions - 1) * ( potential.force(r, pot_matrix[0,0])[1])**2
    f_err_a, _ = quad( integrand, a=r_c, b=inf)

    # Force Error =  QFactor/sqrt(N V) f_err
    # QFactor = Sum_s q_s^2 N_s / (4 * pi * epsilon_0),  s indicates species
    # Constant = QFactor/N * sqrt(N/V) = QFactor / N * sqrt( 3/ (4 pi a_ws*3) )
    # Rescale by e^2/(4 pi eps0 a_{ws}^2)
    # Constant = QFactor / (N * e^2/(4 pi eps0) ) * sqrt(3/ (4 pi)) * ( e^2 / sqrt(a_ws^3))
    # f_err rescaled by a_ws is
    # f_err = f_err_a * (1 / sqrt(a_ws)) ,
    # Force Error = QFactor / (N * e^2/(4 pi eps0) ) * sqrt(3/ (4 pi)) * f_err_a * ( e^2 / a_ws^2))

    QFactor = potential.QFactor / (potential.matrix[0, 0, 0] * potential.total_num_ptcls)
    f_err = sqrt( f_err_a * 3.0 / (4.0 * pi)) * QFactor

    return f_err