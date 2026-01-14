import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Page Configuration
st.set_page_config(
    page_title="Advanced Tensor Analyzer with LaTeX",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MATHEMATICAL FOUNDATION SECTION (MAIN PAGE) - ENHANCED LaTeX
# ============================================================================
st.markdown(r"""
# 🧮 Advanced Mathematical Framework for Orthotropic Tensor Analysis

## 📐 Complete Mathematical Formulation

This tool visualizes and analyzes directional dependence of permeability and porous resistivity tensors 
for anisotropic porous materials like Polydimethylsiloxane (PDMS) sponges using rigorous tensor mathematics.

### 1. **Porosity Tensor Representation**

For orthotropic materials with three orthogonal symmetry planes, porosity is direction-dependent:

$$
\phi_i = \lim_{V \to 0} \frac{V_{\text{pore},i}}{V_{\text{total}}}, \quad i = x, y, z
$$

where $\phi_i$ represents the porosity in direction $i$, defined as the ratio of pore volume to total volume in that specific orientation.

### 2. **Permeability Power Law Model**

The principal permeability components follow an empirical power-law relationship:

$$
\boxed{\kappa_{ii} = \beta_i \cdot (\phi_i)^{m_i}}, \quad i = x, y, z
$$

**Variables:**
- $\kappa_{ii}$: Permeability tensor diagonal component in direction $i$ (m²)
- $\beta_i$: Material-specific empirical constant (m²)
- $\phi_i$: Directional porosity (dimensionless, $0 < \phi_i < 1$)
- $m_i$: Empirical exponent characterizing pore connectivity (dimensionless)

### 3. **Second-Order Permeability Tensor**

For orthotropic materials with principal axes aligned with Cartesian coordinates:

$$
\boldsymbol{\kappa} = 
\begin{bmatrix}
\kappa_{xx} & 0 & 0 \\
0 & \kappa_{yy} & 0 \\
0 & 0 & \kappa_{zz}
\end{bmatrix}
\in \mathbb{R}^{3 \times 3} \quad \text{[m²]}
$$

This is a **symmetric positive definite** tensor, representing hydraulic conductivity in porous media.

### 4. **Porous Resistivity Tensor Definition**

The resistivity tensor $\mathbf{R}$ is defined as the **inverse** of the permeability tensor:

$$
\mathbf{R} = \boldsymbol{\kappa}^{-1} = 
\begin{bmatrix}
R_{xx} & 0 & 0 \\
0 & R_{yy} & 0 \\
0 & 0 & R_{zz}
\end{bmatrix}
\quad \text{[m⁻²]}
$$

where $R_{ii} = 1/\kappa_{ii}$ represents the resistance to flow in direction $i$.

### 5. **Extended Navier-Stokes-Brinkman Equation**

The momentum equation for incompressible flow through anisotropic porous media:

$$
\rho\left(\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v}\right) = 
-\nabla p + \mu\nabla^2\mathbf{v} - \mu\mathbf{R}\mathbf{v}
$$

**Terms:**
- $\mathbf{v}(\mathbf{x}, t)$: Velocity vector field (m/s)
- $p(\mathbf{x}, t)$: Pressure field (Pa)
- $\rho$: Fluid density (kg/m³)
- $\mu$: Dynamic viscosity (Pa·s)
- $\mathbf{R}$: Anisotropic resistivity tensor (m⁻²)

The term $-\mu\mathbf{R}\mathbf{v}$ represents the **Darcy-Forchheimer drag force** specific to orthotropic media.

### 6. **Tensor Quadric Representation Surface**

The visualization displays a **quadric surface** representing:

$$
\mathbf{n}^T \boldsymbol{\kappa} \mathbf{n} = 1, \quad \|\mathbf{n}\| = 1
$$

where $\mathbf{n} = [\cos\theta\sin\phi, \sin\theta\sin\phi, \cos\phi]^T$ is a unit direction vector. This ellipsoid's semi-axes are proportional to $\sqrt{\kappa_{ii}}$, providing intuitive visualization of directional permeability.

### 7. **Anisotropy Metrics**

**Anisotropy Ratio:**
$$
A_{\kappa} = \frac{\max(\kappa_{xx}, \kappa_{yy}, \kappa_{zz})}{\min(\kappa_{xx}, \kappa_{yy}, \kappa_{zz})}
$$

**Condition Number:**
$$
\kappa(\boldsymbol{\kappa}) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$
where $\lambda_i$ are eigenvalues of $\boldsymbol{\kappa}$.

### 8. **Physical Interpretation**

- **Spherical Tensor** ($A_{\kappa} \approx 1$): Isotropic material, uniform flow
- **Prolate Tensor** ($A_{\kappa} \gg 1$): Uniaxial anisotropy, preferential flow
- **Oblate Tensor**: Planar anisotropy, laminar flow dominance
- **Triaxial Tensor**: Fully anisotropic, complex flow patterns
""")

st.markdown("---")

# ============================================================================
# DEFAULT VALUES FOR POROUS PDMS MATERIAL
# ============================================================================
PDMS_DEFAULTS = {
    "phi_x": 0.60,
    "phi_y": 0.60,
    "phi_z": 0.59,
    "Bx": 2.1e-9,
    "By": 2.9e-9,
    "Bz": 2.2e-9,
    "mx": 1.85,
    "my": 1.85,
    "mz": 1.85
}

# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================
def compute_permeability_matrix(phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz):
    """Compute orthotropic permeability tensor using power law."""
    k11 = Bx * (phi_x)**mx
    k22 = By * (phi_y)**my
    k33 = Bz * (phi_z)**mz
    
    # Ensure positive definiteness
    permeability_matrix = np.array([[k11, 0.0, 0.0],
                                    [0.0, k22, 0.0],
                                    [0.0, 0.0, k33]])
    permeability_matrix = np.maximum(permeability_matrix, 1e-30)  # Avoid zero
    
    return permeability_matrix, (k11, k22, k33)

def compute_ellipsoid_points(a, b, c, n_points=100):
    """Generate points for an ellipsoid with given semi-axes."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    
    # Use meshgrid for more accurate sampling
    U, V = np.meshgrid(u, v)
    
    x = a * np.cos(U) * np.sin(V)
    y = b * np.sin(U) * np.sin(V)
    z = c * np.cos(V)
    
    return x, y, z

def compute_tensor_magnitude(x, y, z, tensor_matrix, scaling_factors):
    """Compute tensor magnitude at each point on ellipsoid."""
    # Normalize coordinates by scaling factors
    sf_x, sf_y, sf_z = scaling_factors
    
    # Avoid division by zero
    sf_x = max(sf_x, 1e-10)
    sf_y = max(sf_y, 1e-10)
    sf_z = max(sf_z, 1e-10)
    
    x_norm = x / sf_x
    y_norm = y / sf_y
    z_norm = z / sf_z
    
    # Quadratic form: v^T * K * v
    magnitude = (tensor_matrix[0,0] * x_norm**2 + 
                 tensor_matrix[1,1] * y_norm**2 + 
                 tensor_matrix[2,2] * z_norm**2)
    
    # Ensure positive values
    magnitude = np.maximum(magnitude, 0)
    
    return magnitude

def matplotlib_to_plotly_colorscale(cmap, n=256):
    """Convert matplotlib colormap to Plotly colorscale."""
    # Sample the colormap
    samples = cmap(np.linspace(0, 1, n))
    
    colorscale = []
    for i, (r, g, b, a) in enumerate(samples):
        colorscale.append([
            i / (n - 1),
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        ])
    
    return colorscale

# ============================================================================
# STREAMLIT INTERFACE - SIDEBAR WITH LaTeX
# ============================================================================
st.sidebar.header(r"⚙️ Material Parameters $\phi_i, \beta_i, m_i$")

# Quick preset selection
preset = st.sidebar.radio("Material Preset:", 
                          ["Custom", "Porous PDMS (Paper Default)", "Isotropic", "Highly Anisotropic"])

if preset == "Porous PDMS (Paper Default)":
    phi_x, phi_y, phi_z = PDMS_DEFAULTS["phi_x"], PDMS_DEFAULTS["phi_y"], PDMS_DEFAULTS["phi_z"]
    Bx, By, Bz = PDMS_DEFAULTS["Bx"], PDMS_DEFAULTS["By"], PDMS_DEFAULTS["Bz"]
    mx, my, mz = PDMS_DEFAULTS["mx"], PDMS_DEFAULTS["my"], PDMS_DEFAULTS["mz"]
elif preset == "Isotropic":
    phi_x = phi_y = phi_z = 0.5
    Bx = By = Bz = 1.0e-9
    mx = my = mz = 2.0
elif preset == "Highly Anisotropic":
    phi_x, phi_y, phi_z = 0.8, 0.3, 0.6
    Bx, By, Bz = 5.0e-9, 0.5e-9, 2.0e-9
    mx, my, mz = 1.5, 2.5, 2.0
else:
    # Custom inputs in expandable sections with LaTeX
    with st.sidebar.expander(r"Porosity Values $\phi_i$", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            phi_x = st.number_input(r"$\phi_x$", 0.01, 0.99, 0.2, 0.01,
                                   help="Porosity normal to YZ plane", key="phi_x_input")
        with col2:
            phi_y = st.number_input(r"$\phi_y$", 0.01, 0.99, 0.3, 0.01,
                                   help="Porosity normal to XZ plane", key="phi_y_input")
        with col3:
            phi_z = st.number_input(r"$\phi_z$", 0.01, 0.99, 0.4, 0.01,
                                   help="Porosity normal to XY plane", key="phi_z_input")
    
    with st.sidebar.expander(r"Power Law Parameters $\beta_i, m_i$", expanded=True):
        st.markdown(r"**Base Constants $\beta_i$ [m²]**")
        col1, col2, col3 = st.columns(3)
        with col1:
            Bx = st.number_input(r"$\beta_x$", 1.0e-12, 1.0e-5, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for X-direction", key="Bx_input")
        with col2:
            By = st.number_input(r"$\beta_y$", 1.0e-12, 1.0e-5, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for Y-direction", key="By_input")
        with col3:
            Bz = st.number_input(r"$\beta_z$", 1.0e-12, 1.0e-5, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for Z-direction", key="Bz_input")
        
        st.markdown(r"**Exponents $m_i$**")
        col1, col2, col3 = st.columns(3)
        with col1:
            mx = st.number_input(r"$m_x$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for X-direction", key="mx_input")
        with col2:
            my = st.number_input(r"$m_y$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for Y-direction", key="my_input")
        with col3:
            mz = st.number_input(r"$m_z$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for Z-direction", key="mz_input")

# Visualization options
st.sidebar.header("🎨 Visualization Options")

visualization_option = st.sidebar.radio(
    "Tensor to Visualize:",
    (r'$\boldsymbol{\kappa}$ (Permeability)', r'$\mathbf{R}$ (Resistivity)', 'Both'),
    index=0
)

# Visualization library selection
viz_library = st.sidebar.radio(
    "Visualization Library:",
    ('Plotly (Interactive 3D)', 'Matplotlib (Static)'),
    index=0
)

# Enhanced colormap selection
all_colormaps = sorted(plt.colormaps())

# Create categorized colormaps
cmap_categories = {
    "Perceptually Uniform": ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    "Sequential": ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                   'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                   'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                   'summer', 'autumn', 'winter', 'spring', 'cool', 'Wistia',
                   'hot', 'afmhot', 'gist_heat', 'copper', 'bone', 'pink'],
    "Diverging": ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
    "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
    "Qualitative": ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'],
    "Classic (Includes Rainbow/Jet/Turbo)": ['rainbow', 'jet', 'turbo', 'flag', 'prism', 'ocean', 
                              'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 
                              'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow',
                              'nipy_spectral', 'gist_ncar']
}

selected_category = st.sidebar.selectbox("Colormap Category:", list(cmap_categories.keys()))
cmap_name = st.sidebar.selectbox(f"Select colormap:", sorted(cmap_categories[selected_category]))

# Additional visualization parameters
with st.sidebar.expander("Advanced Settings"):
    resolution = st.slider("Mesh Resolution", 30, 200, 80, 10)
    show_axes = st.checkbox("Show Coordinate Axes", True)
    show_colorbar = st.checkbox("Show Colorbar", True)
    transparency = st.slider("Surface Transparency", 0.0, 1.0, 0.8, 0.1)
    if viz_library == 'Plotly (Interactive 3D)':
        show_contour = st.checkbox("Show Contour Lines", False)
        lighting_effects = st.checkbox("Enable Lighting Effects", True)
        aspect_mode = st.selectbox("Aspect Ratio", ['data', 'cube', 'auto'], index=0)
    else:
        show_wireframe = st.checkbox("Show Wireframe", False)
        projection_type = st.selectbox("Projection", ['persp', 'ortho'], index=0)

# ============================================================================
# COMPUTE TENSORS
# ============================================================================
permeability_matrix, (k11, k22, k33) = compute_permeability_matrix(
    phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz
)

# Calculate porous resistivity (inverse of permeability)
try:
    porous_resistivity_matrix = np.linalg.inv(permeability_matrix)
    R11, R22, R33 = porous_resistivity_matrix[0,0], porous_resistivity_matrix[1,1], porous_resistivity_matrix[2,2]
except np.linalg.LinAlgError:
    st.error("Singular matrix encountered. Please adjust parameters.")
    porous_resistivity_matrix = np.diag([1e10, 1e10, 1e10])
    R11 = R22 = R33 = 1e10

# ============================================================================
# DISPLAY TENSOR VALUES WITH LaTeX
# ============================================================================
st.markdown(r"## 📊 Computed Tensor Components")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(r"$\kappa_{xx}$ (X-direction)", f"{k11:.2e} m²")
    st.metric(r"$R_{xx}$ (X-direction)", f"{R11:.2e} m⁻²")
    st.latex(rf"\kappa_{{xx}} = \beta_x \cdot \phi_x^{{m_x}} = {Bx:.2e} \cdot {phi_x:.2f}^{{{mx:.2f}}}")

with col2:
    st.metric(r"$\kappa_{yy}$ (Y-direction)", f"{k22:.2e} m²")
    st.metric(r"$R_{yy}$ (Y-direction)", f"{R22:.2e} m⁻²")
    st.latex(rf"\kappa_{{yy}} = \beta_y \cdot \phi_y^{{m_y}} = {By:.2e} \cdot {phi_y:.2f}^{{{my:.2f}}}")

with col3:
    st.metric(r"$\kappa_{zz}$ (Z-direction)", f"{k33:.2e} m²")
    st.metric(r"$R_{zz}$ (Z-direction)", f"{R33:.2e} m⁻²")
    st.latex(rf"\kappa_{{zz}} = \beta_z \cdot \phi_z^{{m_z}} = {Bz:.2e} \cdot {phi_z:.2f}^{{{mz:.2f}}}")

# Display full tensors with LaTeX
st.markdown("### Full Tensor Representations")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown(r"**Permeability Tensor $\boldsymbol{\kappa}$:**")
    st.latex(rf"""
    \boldsymbol{{\kappa}} = 
    \begin{{bmatrix}}
        {k11:.2e} & 0 & 0 \\
        0 & {k22:.2e} & 0 \\
        0 & 0 & {k33:.2e}
    \end{{bmatrix}} \text{{ m}}^2
    """)

with col_t2:
    st.markdown(r"**Resistivity Tensor $\mathbf{R} = \boldsymbol{\kappa}^{-1}$:**")
    st.latex(rf"""
    \mathbf{{R}} = 
    \begin{{bmatrix}}
        {R11:.2e} & 0 & 0 \\
        0 & {R22:.2e} & 0 \\
        0 & 0 & {R33:.2e}
    \end{{bmatrix}} \text{{ m}}^{{-2}}
    """)

# Anisotropy metrics
anisotropy_k = max(k11, k22, k33) / min(k11, k22, k33) if min(k11, k22, k33) > 0 else float('inf')
anisotropy_R = max(R11, R22, R33) / min(R11, R22, R33) if min(R11, R22, R33) > 0 else float('inf')

st.info(rf"""
**Anisotropy Analysis:**
- Permeability Anisotropy Ratio: $A_\kappa = {anisotropy_k:.2f}$
- Resistivity Anisotropy Ratio: $A_R = {anisotropy_R:.2f}$
""")

# ============================================================================
# VISUALIZATION SECTION - WITH PROPER LaTeX IN PLOTLY
# ============================================================================
st.markdown(r"## 🎯 3D Tensor Visualization: $\mathbf{n}^T \boldsymbol{\kappa} \mathbf{n} = 1$")

if viz_library == 'Plotly (Interactive 3D)':
    # Get matplotlib colormap
    matplotlib_cmap = cm.get_cmap(cmap_name)
    plotly_colorscale = matplotlib_to_plotly_colorscale(matplotlib_cmap)
    
    # Function to create Plotly surface with FIXED colorbar
    def create_plotly_surface(tensor_matrix, title, colorbar_title, is_resistivity=False):
        if is_resistivity:
            # For resistivity: scaling from tensor diagonal
            diag_vals = np.diag(tensor_matrix)
            a, b, c = np.sqrt(np.maximum(diag_vals, 1e-30))
            scaling_factors = (a, b, c)
        else:
            # For permeability: scaling from normalized values
            a = np.sqrt(k11 / max(Bx, 1e-30))
            b = np.sqrt(k22 / max(By, 1e-30))
            c = np.sqrt(k33 / max(Bz, 1e-30))
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Create colorbar configuration
        colorbar_config = None
        if show_colorbar:
            colorbar_config = dict(
                title=dict(
                    text=colorbar_title,
                    side="right"
                ),
                thickness=20,
                len=0.75,
                tickfont=dict(size=10)
            )
        
        # Create surface trace - FIXED COLORBAR SYNTAX
        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=magnitude,
            colorscale=plotly_colorscale,
            opacity=transparency,
            colorbar=colorbar_config,  # Fixed: Use proper dict format
            contours=dict(
                x=dict(show=show_contour, color='gray', width=1),
                y=dict(show=show_contour, color='gray', width=1),
                z=dict(show=show_contour, color='gray', width=1)
            ),
            lighting=dict(
                ambient=0.3,
                diffuse=0.9 if lighting_effects else 0.6,
                specular=0.2 if lighting_effects else 0.1,
                roughness=0.5,
                fresnel=0.2
            ),
            name=title,
            showscale=show_colorbar
        )
        
        return surface, X, Y, Z
    
    # Create visualization based on option
    if visualization_option == 'Both':
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=(
                r'$\boldsymbol{\kappa}\; \text{Permeability Tensor}$',
                r'$\mathbf{R}\; \text{Resistivity Tensor}$'
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Permeability surface
        surf_perm, X_perm, Y_perm, Z_perm = create_plotly_surface(
            permeability_matrix, 
            r'$\boldsymbol{\kappa}$', 
            r'$\kappa\;[\mathrm{m}^2]$', 
            False
        )
        fig.add_trace(surf_perm, row=1, col=1)
        
        # Resistivity surface
        surf_res, X_res, Y_res, Z_res = create_plotly_surface(
            porous_resistivity_matrix, 
            r'$\mathbf{R}$', 
            r'$R\;[\mathrm{m}^{-2}]$', 
            True
        )
        fig.add_trace(surf_res, row=1, col=2)
        
        # Calculate aspect ratios
        aspect_perm = [np.ptp(X_perm), np.ptp(Y_perm), np.ptp(Z_perm)]
        aspect_res = [np.ptp(X_res), np.ptp(Y_res), np.ptp(Z_res)]
        
        # Normalize aspect ratios
        max_perm = max(aspect_perm)
        max_res = max(aspect_res)
        if max_perm > 0:
            aspect_perm = [a/max_perm for a in aspect_perm]
        if max_res > 0:
            aspect_res = [a/max_res for a in aspect_res]
        
        scene1 = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=aspect_perm[0], y=aspect_perm[1], z=aspect_perm[2]),
            aspectmode='manual' if aspect_mode == 'data' else aspect_mode,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        scene2 = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=aspect_res[0], y=aspect_res[1], z=aspect_res[2]),
            aspectmode='manual' if aspect_mode == 'data' else aspect_mode,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        fig.update_layout(
            scene=scene1,
            scene2=scene2,
            width=1400,
            height=600,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            title_text=r"Orthotropic Tensor Visualization",
            title_x=0.5
        )
        
    else:
        # Single tensor visualization
        if 'Permeability' in visualization_option:
            tensor_matrix = permeability_matrix
            title = r'$\boldsymbol{\kappa}\; \text{Permeability Tensor}$'
            colorbar_title = r'$\kappa\;[\mathrm{m}^2]$'
            is_resistivity = False
        else:
            tensor_matrix = porous_resistivity_matrix
            title = r'$\mathbf{R}\; \text{Resistivity Tensor}$'
            colorbar_title = r'$R\;[\mathrm{m}^{-2}]$'
            is_resistivity = True
        
        surf, X, Y, Z = create_plotly_surface(
            tensor_matrix, title, colorbar_title, is_resistivity
        )
        
        fig = go.Figure(data=[surf])
        
        # Calculate aspect ratio
        aspect = [np.ptp(X), np.ptp(Y), np.ptp(Z)]
        max_aspect = max(aspect)
        if max_aspect > 0:
            aspect = [a/max_aspect for a in aspect]
        
        scene = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=aspect[0], y=aspect[1], z=aspect[2]),
            aspectmode='manual' if aspect_mode == 'data' else aspect_mode,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        fig.update_layout(
            scene=scene,
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False,
            title=dict(
                text=title,
                x=0.5,
                y=0.95,
                xanchor='center'
            )
        )
    
    # Add coordinate axes if requested
    if show_axes:
        # Determine axis length from data
        if visualization_option == 'Both':
            all_X, all_Y, all_Z = X_perm, Y_perm, Z_perm
        else:
            all_X, all_Y, all_Z = X, Y, Z
        
        axis_length = max(np.max(np.abs(all_X)), np.max(np.abs(all_Y)), np.max(np.abs(all_Z))) * 1.3
        
        # Add axes with arrows using cones for better visualization
        fig.add_trace(go.Cone(
            x=[axis_length], y=[0], z=[0],
            u=[axis_length * 0.2], v=[0], w=[0],
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            name=r'$x$'
        ))
        
        fig.add_trace(go.Cone(
            x=[0], y=[axis_length], z=[0],
            u=[0], v=[axis_length * 0.2], w=[0],
            colorscale=[[0, 'green'], [1, 'green']],
            showscale=False,
            name=r'$y$'
        ))
        
        fig.add_trace(go.Cone(
            x=[0], y=[0], z=[axis_length],
            u=[0], v=[0], w=[axis_length * 0.2],
            colorscale=[[0, 'blue'], [1, 'blue']],
            showscale=False,
            name=r'$z$'
        ))
        
        # Add axis lines
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines',
            line=dict(color='red', width=4),
            showlegend=False,
            name='X-axis'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines',
            line=dict(color='green', width=4),
            showlegend=False,
            name='Y-axis'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines',
            line=dict(color='blue', width=4),
            showlegend=False,
            name='Z-axis'
        ))
    
    st.plotly_chart(fig, use_container_width=True, config={'responsive': True})

else:  # Matplotlib visualization
    # Create figure based on visualization option
    if visualization_option == 'Both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        titles = [r'$\boldsymbol{\kappa}$ Permeability Tensor', 
                 r'$\mathbf{R}$ Resistivity Tensor']
        tensors = [permeability_matrix, porous_resistivity_matrix]
        colorbar_titles = [r'$\kappa$ [m²]', r'$R$ [m⁻²]']
        is_resistivity_list = [False, True]
    else:
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        axes = [ax]
        
        if 'Permeability' in visualization_option:
            titles = [r'$\boldsymbol{\kappa}$ Permeability Tensor']
            tensors = [permeability_matrix]
            colorbar_titles = [r'$\kappa$ [m²]']
            is_resistivity_list = [False]
        else:
            titles = [r'$\mathbf{R}$ Resistivity Tensor']
            tensors = [porous_resistivity_matrix]
            colorbar_titles = [r'$R$ [m⁻²]']
            is_resistivity_list = [True]
    
    # Set projection
    for ax in axes:
        ax.set_proj_type(projection_type)
    
    cmap = cm.get_cmap(cmap_name)
    
    for idx, (ax, tensor_matrix, title, colorbar_title, is_resistivity) in enumerate(zip(
        axes, tensors, titles, colorbar_titles, is_resistivity_list)):
        
        if is_resistivity:
            diag_vals = np.diag(tensor_matrix)
            a, b, c = np.sqrt(np.maximum(diag_vals, 1e-30))
            scaling_factors = (a, b, c)
        else:
            a = np.sqrt(k11 / max(Bx, 1e-30))
            b = np.sqrt(k22 / max(By, 1e-30))
            c = np.sqrt(k33 / max(Bz, 1e-30))
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Normalize for coloring
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:
            norm = plt.Normalize(mag_min, mag_max)
        else:
            norm = plt.Normalize(0, 1)
        
        # Create surface
        if show_wireframe:
            ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5, rstride=5, cstride=5)
        
        surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(magnitude)),
                              rstride=2, cstride=2, alpha=transparency,
                              linewidth=0.1, antialiased=True, shade=True)
        
        # Add coordinate axes
        if show_axes:
            axis_length = max(a, b, c) * 1.5
            ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
            ax.text(axis_length, 0, 0, r'$x$', fontsize=12, color='red', ha='center', va='center')
            ax.text(0, axis_length, 0, r'$y$', fontsize=12, color='green', ha='center', va='center')
            ax.text(0, 0, axis_length, r'$z$', fontsize=12, color='blue', ha='center', va='center')
        
        # Set labels and title
        ax.set_xlabel(r'$x$', fontsize=12, labelpad=10)
        ax.set_ylabel(r'$y$', fontsize=12, labelpad=10)
        ax.set_zlabel(r'$z$', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Set aspect ratio
        ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
        
        # Add colorbar
        if show_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label(colorbar_title, fontsize=12, rotation=270, labelpad=15)
        
        # Style improvements
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.xaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.yaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
        ax.zaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.2)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================================
# DATA DOWNLOAD SECTION
# ============================================================================
st.markdown("---")
st.subheader("📥 Download Results")

col1, col2, col3 = st.columns(3)

with col1:
    df_k = pd.DataFrame(permeability_matrix, 
                       index=['X', 'Y', 'Z'], 
                       columns=['X', 'Y', 'Z'])
    csv_k = df_k.to_csv()
    st.download_button(
        label="Download Permeability Tensor (CSV)",
        data=csv_k,
        file_name="permeability_tensor.csv",
        mime="text/csv",
        type="primary"
    )

with col2:
    df_r = pd.DataFrame(porous_resistivity_matrix,
                       index=['X', 'Y', 'Z'],
                       columns=['X', 'Y', 'Z'])
    csv_r = df_r.to_csv()
    st.download_button(
        label="Download Resistivity Tensor (CSV)",
        data=csv_r,
        file_name="resistivity_tensor.csv",
        mime="text/csv",
        type="primary"
    )

with col3:
    summary = {
        "parameters": {
            "phi_x": phi_x, "phi_y": phi_y, "phi_z": phi_z,
            "Bx": Bx, "By": By, "Bz": Bz,
            "mx": mx, "my": my, "mz": mz,
            "preset": preset
        },
        "permeability": {
            "kappa_xx": float(k11), "kappa_yy": float(k22), "kappa_zz": float(k33)
        },
        "resistivity": {
            "R_xx": float(R11), "R_yy": float(R22), "R_zz": float(R33)
        },
        "anisotropy": {
            "permeability_ratio": float(anisotropy_k),
            "resistivity_ratio": float(anisotropy_R)
        }
    }
    
    json_str = json.dumps(summary, indent=2)
    st.download_button(
        label="Download Summary (JSON)",
        data=json_str,
        file_name="tensor_analysis_summary.json",
        mime="application/json",
        type="primary"
    )

# ============================================================================
# LaTeX REPORT GENERATION
# ============================================================================
st.markdown("### 📝 LaTeX Report Generation")

with st.expander("Generate LaTeX Report", expanded=False):
    latex_code = rf"""
\documentclass{{article}}
\usepackage{{amsmath}}
\usepackage{{amsfonts}}
\usepackage{{amssymb}}
\usepackage{{bm}}
\usepackage{{graphicx}}
\usepackage{{geometry}}
\usepackage{{booktabs}}

\geometry{{a4paper, margin=1in}}

\title{{Orthotropic Tensor Analysis Report}}
\author{{Tensor Analysis Tool}}
\date{{\today}}

\begin{{document}}

\maketitle

\section{{Material Parameters}}
\begin{{align*}}
    \phi_x &= {phi_x:.3f}, &\quad \phi_y &= {phi_y:.3f}, &\quad \phi_z &= {phi_z:.3f} \\
    \beta_x &= {Bx:.2e}\,\text{{m}}^2, &\quad \beta_y &= {By:.2e}\,\text{{m}}^2, &\quad \beta_z &= {Bz:.2e}\,\text{{m}}^2 \\
    m_x &= {mx:.2f}, &\quad m_y &= {my:.2f}, &\quad m_z &= {mz:.2f}
\end{{align*}}

\section{{Computed Tensors}}

\subsection{{Permeability Tensor $\boldsymbol{{\kappa}}$}}
\begin{{equation}}
    \boldsymbol{{\kappa}} = 
    \begin{{bmatrix}}
        {k11:.2e} & 0 & 0 \\
        0 & {k22:.2e} & 0 \\
        0 & 0 & {k33:.2e}
    \end{{bmatrix}}
    \,\text{{m}}^2
\end{{equation}}

\subsection{{Resistivity Tensor $\mathbf{{R}} = \boldsymbol{{\kappa}}^{{-1}}$}}
\begin{{equation}}
    \mathbf{{R}} = \boldsymbol{{\kappa}}^{{-1}} = 
    \begin{{bmatrix}}
        {R11:.2e} & 0 & 0 \\
        0 & {R22:.2e} & 0 \\
        0 & 0 & {R33:.2e}
    \end{{bmatrix}}
    \,\text{{m}}^{{-2}}
\end{{equation}}

\section{{Anisotropy Analysis}}
\begin{{align*}}
    A_\kappa &= \frac{{\max(\kappa_{{xx}}, \kappa_{{yy}}, \kappa_{{zz}})}}{{\min(\kappa_{{xx}}, \kappa_{{yy}}, \kappa_{{zz}})}} = {anisotropy_k:.2f} \\
    A_R &= \frac{{\max(R_{{xx}}, R_{{yy}}, R_{{zz}})}}{{\min(R_{{xx}}, R_{{yy}}, R_{{zz}})}} = {anisotropy_R:.2f}
\end{{align*}}

\section{{Component Calculations}}

\subsection{{X-direction}}
\begin{{align*}}
    \kappa_{{xx}} &= \beta_x \cdot \phi_x^{{m_x}} = {Bx:.2e} \cdot ({phi_x:.3f})^{{{mx:.2f}}} = {k11:.2e}\,\text{{m}}^2 \\
    R_{{xx}} &= \frac{{1}}{{\kappa_{{xx}}}} = {R11:.2e}\,\text{{m}}^{{-2}}
\end{{align*}}

\subsection{{Y-direction}}
\begin{{align*}}
    \kappa_{{yy}} &= \beta_y \cdot \phi_y^{{m_y}} = {By:.2e} \cdot ({phi_y:.3f})^{{{my:.2f}}} = {k22:.2e}\,\text{{m}}^2 \\
    R_{{yy}} &= \frac{{1}}{{\kappa_{{yy}}}} = {R22:.2e}\,\text{{m}}^{{-2}}
\end{{align*}}

\subsection{{Z-direction}}
\begin{{align*}}
    \kappa_{{zz}} &= \beta_z \cdot \phi_z^{{m_z}} = {Bz:.2e} \cdot ({phi_z:.3f})^{{{mz:.2f}}} = {k33:.2e}\,\text{{m}}^2 \\
    R_{{zz}} &= \frac{{1}}{{\kappa_{{zz}}}} = {R33:.2e}\,\text{{m}}^{{-2}}
\end{{align*}}

\section{{Interpretation}}
The material exhibits {('isotropic' if anisotropy_k < 1.1 else 'moderate anisotropy' if anisotropy_k < 3 else 'strong anisotropy')} properties with permeability anisotropy ratio $A_\kappa = {anisotropy_k:.2f}$.

\end{{document}}
"""
    
    st.code(latex_code, language="latex")
    
    st.download_button(
        label="Download LaTeX Report",
        data=latex_code,
        file_name="tensor_analysis_report.tex",
        mime="text/plain",
        type="secondary"
    )

# ============================================================================
# FOOTER WITH LaTeX
# ============================================================================
st.markdown("---")
st.markdown(r"""
<div style="text-align: center; color: gray; font-size: 0.9em;">
<p><strong>Advanced Orthotropic Tensor Analyzer</strong> • $\kappa_{ii} = \beta_i \cdot (\phi_i)^{m_i}$</p>
<p>Based on: <em>"Microstructural contributions of PDMS sponges to their water treatment efficacy"</em></p>
<p>Tensor mathematics: $\mathbf{R} = \boldsymbol{\kappa}^{-1}$, $\mathbf{n}^T\boldsymbol{\kappa}\mathbf{n} = 1$</p>
</div>
""", unsafe_allow_html=True)
