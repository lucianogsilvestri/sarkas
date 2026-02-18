# File: your_package/plotting/styles.py
"""
Custom style templates for matplotlib and Plotly.
This module provides MSUstyle and PUBstyle for consistent styling across both plotting libraries.
"""

import os
import plotly.graph_objects as go
import plotly.io as pio

# MSU Color palette
MSU_COLORS = [
    "#18453b",
    "#f08521",
    "#008183",
    "#6e005f",
    "#d1de3f",
    "#0db14b",
    "#c89a58",
    "#535054",
    "#909ab7",
    "#e8d9b5",
    "#94ae4a",
    "#cb5a28",
]

# PUB Color palette (standard matplotlib colors)
PUB_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _create_msu_template():
    """
    Create the MSU Plotly template.

    Returns
    -------
    plotly.graph_objects.layout.Template
        MSU custom template for Plotly.
    """
    template = go.layout.Template()

    template.layout = go.Layout(
        # Font settings
        font=dict(size=16, family="sans-serif"),
        # Color cycle
        colorway=MSU_COLORS,
        # Background colors
        plot_bgcolor="rgb(235, 235, 240, 0.4)",  # [0.92, 0.92, 0.94]
        paper_bgcolor="white",
        # X-axis default settings
        xaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.2)",
            gridwidth=0.8,
            griddash="dash",
            showgrid=True,
            linecolor="black",
            linewidth=0.8,
            mirror=True,
            ticks="outside",
            tickfont=dict(size=14),
            title=dict(font=dict(size=15)),
            showline=True,
        ),
        # Y-axis default settings
        yaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.2)",
            gridwidth=0.8,
            griddash="dash",
            showgrid=True,
            linecolor="black",
            linewidth=0.8,
            mirror=True,
            ticks="outside",
            tickfont=dict(size=14),
            title=dict(font=dict(size=15)),
            showline=True,
        ),
        # Title settings
        title=dict(font=dict(size=18), x=0.5, xanchor="center"),
        # Legend settings
        legend=dict(bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=14)),
        # Hover label settings
        hoverlabel=dict(font_size=14),
    )

    # Trace-specific settings
    template.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=6))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0.8, color="black")))]
    template.data.heatmap = [go.Heatmap(colorscale="Viridis")]
    template.data.contour = [go.Contour(colorscale="Viridis")]

    return template


def _create_pub_template():
    """
    Create the PUB Plotly template.

    Returns
    -------
    plotly.graph_objects.layout.Template
        PUB custom template for Plotly.
    """
    template = go.layout.Template()

    template.layout = go.Layout(
        # Font settings
        font=dict(size=14, family="sans-serif"),
        # Color cycle
        colorway=PUB_COLORS,
        # Background colors - white for publication style
        plot_bgcolor="white",
        paper_bgcolor="white",
        # X-axis default settings
        xaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.2)",  # grid.alpha: 0.3
            gridwidth=0.8,
            griddash="dash",
            showgrid=True,
            linecolor="black",
            linewidth=0.8,
            mirror=True,
            ticks="outside",
            tickfont=dict(size=14),
            title=dict(font=dict(size=15)),
            showline=True,
        ),
        # Y-axis default settings
        yaxis=dict(
            gridcolor="rgba(0, 0, 0, 0.2)",  # grid.alpha: 0.3
            gridwidth=0.8,
            griddash="dash",
            showgrid=True,
            linecolor="black",
            linewidth=0.8,
            mirror=True,
            ticks="outside",
            tickfont=dict(size=14),
            title=dict(font=dict(size=15)),
            showline=True,
        ),
        # Title settings
        title=dict(font=dict(size=16), x=0.5, xanchor="center"),
        # Legend settings
        legend=dict(bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=14)),
        # Hover label settings
        hoverlabel=dict(font_size=14),
    )

    # Trace-specific settings
    template.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=6))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0.8, color="black")))]
    template.data.heatmap = [go.Heatmap(colorscale="Viridis")]
    template.data.contour = [go.Contour(colorscale="Viridis")]

    return template


def register_all_styles():
    """
    Register both MSUstyle and PUBstyle templates with Plotly.
    This function is automatically called when the module is imported.
    """
    if "MSUstyle" not in pio.templates:
        pio.templates["MSUstyle"] = _create_msu_template()

    if "PUBstyle" not in pio.templates:
        pio.templates["PUBstyle"] = _create_pub_template()


def use_msu_style():
    """
    Set MSUstyle as the default Plotly template for the current session.

    Examples
    --------
    >>> from your_package.plotting.styles import use_msu_style
    >>> use_msu_style()
    >>> # Now all plots will use MSUstyle automatically
    """
    register_all_styles()
    pio.templates.default = "MSUstyle"


def use_pub_style():
    """
    Set PUBstyle as the default Plotly template for the current session.

    Examples
    --------
    >>> from your_package.plotting.styles import use_pub_style
    >>> use_pub_style()
    >>> # Now all plots will use PUBstyle automatically
    """
    register_all_styles()
    pio.templates.default = "PUBstyle"


def get_msu_colors():
    """
    Get the MSU color palette.

    Returns
    -------
    list of str
        List of hex color codes for the MSU color palette.

    Examples
    --------
    >>> from your_package.plotting.styles import get_msu_colors
    >>> colors = get_msu_colors()
    >>> print(colors[0])
    '#18453b'
    """
    return MSU_COLORS.copy()


def get_pub_colors():
    """
    Get the PUB color palette.

    Returns
    -------
    list of str
        List of hex color codes for the PUB color palette.

    Examples
    --------
    >>> from your_package.plotting.styles import get_pub_colors
    >>> colors = get_pub_colors()
    >>> print(colors[0])
    '#1f77b4'
    """
    return PUB_COLORS.copy()


# Automatically register both templates when module is imported
register_all_styles()


# Paths to matplotlib style files
MSU_MPLSTYLE_PATH = os.path.join(os.path.dirname(__file__), "mplstyles", "MSUstyle.mplstyle")
PUB_MPLSTYLE_PATH = os.path.join(os.path.dirname(__file__), "mplstyles", "PUBstyle.mplstyle")


def use_msu_style_matplotlib():
    """
    Apply MSU style to matplotlib.

    Examples
    --------
    >>> from your_package.plotting.styles import use_msu_style_matplotlib
    >>> import matplotlib.pyplot as plt
    >>> use_msu_style_matplotlib()
    >>> # Now all matplotlib plots will use MSUstyle
    """
    import matplotlib.pyplot as plt

    if os.path.exists(MSU_MPLSTYLE_PATH):
        plt.style.use(MSU_MPLSTYLE_PATH)
    else:
        raise FileNotFoundError(f"MSU matplotlib style file not found at {MSU_MPLSTYLE_PATH}")


def use_pub_style_matplotlib():
    """
    Apply PUB style to matplotlib.

    Examples
    --------
    >>> from your_package.plotting.styles import use_pub_style_matplotlib
    >>> import matplotlib.pyplot as plt
    >>> use_pub_style_matplotlib()
    >>> # Now all matplotlib plots will use PUBstyle
    """
    import matplotlib.pyplot as plt

    if os.path.exists(PUB_MPLSTYLE_PATH):
        plt.style.use(PUB_MPLSTYLE_PATH)
    else:
        raise FileNotFoundError(f"PUB matplotlib style file not found at {PUB_MPLSTYLE_PATH}")
