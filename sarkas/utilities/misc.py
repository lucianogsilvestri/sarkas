"""Miscellaneous routines."""

from astropy import constants as ast_c
from astropy import units as ast_u
from pandas import concat, Series


def add_col_to_df(df, data, column_name):
    """Routine to add a column of data to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to which data has to be added.

    data: numpy.ndarray
        Data to be added.

    column_name: str
        Name of the column to be added.

    Returns
    -------
    _ : pandas.DataFrame
        Original `df` concatenated with the `data`.

    Note
    ----
        It creates a `pandas.Series` from `data` using `df.index`. Then it uses `concat` to add the column.

    """
    col_data = Series(data, index=df.index)

    return concat([df, col_data.rename(column_name)], axis=1)


def calculate_beta(temperature, units="mks", k_B=None):
    """
    Calculate the inverse temperature :math:`\\beta = 1 / (k_B T)`.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    units : str, optional
        Units of the temperature. Default is "mks". Possible values are "mks/SI" and "cgs" and "eV.

    k_B : float, optional
        Boltzmann constant. Default is None.

    Returns
    -------
    float
        Inverse temperature.

    Notes
    -----
    The inverse temperature is calculated as :math:`\\beta = 1 / (k_B T)`, where :math:`k_B` is the Boltzmann constant and :math:`T` is the temperature.
    The units of the temperature are specified by the `units` parameter.
    The default value of :math:`k_B` is taken from the `astropy.constants` module.

    Example
    -------
    >>> from sarkas.tools.observables.Thermodynamics import calculate_beta
    >>> calculate_beta(300)
    2.41432350534664e+20
    >>> calculate_beta(300, units = "SI")
    2.41432350534664e+20
    >>> calculate_beta(300, units = "cgs")
    24143235053466.4
    >>> calculate_beta(300, units = "eV")
    38.681727071833606
    >>> calculate_beta(300, k_B = 1.380649e-23)  # SI, mks
    2.41432350534664e+20
    >>> calculate_beta(300, k_B = 1.380649e-16) # cgs
    24143235053466.4
    >>> calculate_beta(300, k_B = 8.617333262145179e-05) # eV
    38.681727071833606

    """

    if k_B is None:
        if units == "cgs":
            k_B = ast_c.k_B.cgs.value
        elif units == "eV":
            k_B = (ast_c.k_B * ast_u.J.to(ast_u.eV)).value
        elif units == "SI":
            k_B = ast_c.k_B.value
        else:
            k_B = ast_c.k_B.value

    return 1.0 / (k_B * temperature)
