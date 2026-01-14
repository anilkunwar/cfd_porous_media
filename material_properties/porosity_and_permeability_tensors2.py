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
from scipy.linalg import eig, norm
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Advanced Tensor Analysis Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MATHEMATICAL FOUNDATION SECTION (MAIN PAGE)
# ============================================================================
st.markdown("""
# 🧮 Advanced Tensor Analysis for Orthotropic Porous Materials

This comprehensive tool visualizes and analyzes directional dependence of permeability 
and porous resistivity tensors for anisotropic porous materials like Polydimethylsiloxane (PDMS) sponges.

## 📐 Mathematical Foundation

### 1. **Directional Porosity Tensor**
For orthotropic materials, porosity varies with direction:
$$
\phi_i = \text{Porosity in direction } i \quad (i = x, y, z)
$$
where $\phi_i$ represents porosity normal to the plane perpendicular to direction $i$.

### 2. **Permeability Power Law**
The principal permeability components follow a power-law relationship:
$$
\boxed{\kappa_{ii} = \beta_i \times (\phi_i)^{m_i}}
$$
- $\kappa_{ii}$: Permeability component in direction $i$ (m²)
- $\beta_i$: Empirical constant (m²)
- $\phi_i$: Directional porosity (dimensionless, $0 < \phi_i < 1$)
- $m_i$: Empirical exponent

### 3. **Permeability Tensor**
For orthotropic materials with principal axes aligned with coordinate axes:
$$
\boldsymbol{\kappa} = 
\begin{bmatrix}
\kappa_{xx} & 0 & 0 \\\\
0 & \kappa_{yy} & 0 \\\\
0 & 0 & \kappa_{zz}
\end{bmatrix}
\quad \text{[m²]}
$$

### 4. **Porous Resistivity Tensor**
The resistivity tensor is the inverse of the permeability tensor:
$$
\mathbf{R} = \boldsymbol{\kappa}^{-1} = 
\begin{bmatrix}
R_{xx} & 0 & 0 \\\\
0 & R_{yy} & 0 \\\\
0 & 0 & R_{zz}
\end{bmatrix}
\quad \text{[m⁻²]}
$$
where $R_{ii} = 1/\kappa_{ii}$.

### 5. **Navier-Stokes Extension for Porous Media**
The momentum equation for incompressible flow through porous media:
$$
\rho \frac{\partial \mathbf{v}}{\partial t} + \rho (\mathbf{v} \cdot \nabla) \mathbf{v} = 
-\nabla p + \mu \nabla^2 \mathbf{v} - \mu \mathbf{R} \mathbf{v}
$$
where:
- $\mathbf{v}$: Velocity vector field
- $p$: Pressure field
- $\rho$: Fluid density
- $\mu$: Dynamic viscosity
- $\mathbf{R}$: Resistivity tensor representing porous drag

### 6. **Tensor Representation Surface**
The visualization shows a quadric surface (ellipsoid) representing:
$$
\mathbf{n}^T \boldsymbol{\kappa} \mathbf{n} = 1
$$
where $\mathbf{n}$ is a unit direction vector, and the surface semi-axes are proportional to $\sqrt{\kappa_{ii}}$.
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
    "mz": 1.85,
    "name": "Porous PDMS (Paper Default)"
}

# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================
def compute_permeability_matrix(phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz):
    """Compute orthotropic permeability tensor using power law with numerical stability."""
    # Ensure porosities are in valid range with small epsilon
    phi_x = np.clip(phi_x, 1e-6, 0.999999)
    phi_y = np.clip(phi_y, 1e-6, 0.999999)
    phi_z = np.clip(phi_z, 1e-6, 0.999999)
    
    # Compute components with safe exponentiation
    k11 = Bx * (phi_x)**mx if Bx > 0 else 1e-20
    k22 = By * (phi_y)**my if By > 0 else 1e-20
    k33 = Bz * (phi_z)**mz if Bz > 0 else 1e-20
    
    # Ensure positive definiteness
    k11 = max(k11, 1e-20)
    k22 = max(k22, 1e-20)
    k33 = max(k33, 1e-20)
    
    permeability_matrix = np.array([[k11, 0.0, 0.0],
                                    [0.0, k22, 0.0],
                                    [0.0, 0.0, k33]])
    
    return permeability_matrix, (k11, k22, k33)

def compute_tensor_analysis(tensor_matrix):
    """Compute tensor properties: eigenvalues, eigenvectors, condition number."""
    try:
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eig(tensor_matrix)
        
        # Ensure eigenvalues are real (should be for symmetric positive definite)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort by eigenvalue magnitude
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute condition number (ratio of largest to smallest eigenvalue)
        condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'condition_number': condition_number,
            'principal_directions': eigenvectors.T
        }
    except:
        # Fallback if eigendecomposition fails
        return {
            'eigenvalues': np.diag(tensor_matrix),
            'eigenvectors': np.eye(3),
            'condition_number': 1.0,
            'principal_directions': np.eye(3)
        }

def compute_ellipsoid_points(a, b, c, n_points=100):
    """Generate points for an ellipsoid with given semi-axes (numerically stable)."""
    # Ensure positive axes
    a = max(a, 1e-6)
    b = max(b, 1e-6)
    c = max(c, 1e-6)
    
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    
    U, V = np.meshgrid(u, v)
    
    x = a * np.cos(U) * np.sin(V)
    y = b * np.sin(U) * np.sin(V)
    z = c * np.cos(V)
    
    return x, y, z

def compute_tensor_magnitude(x, y, z, tensor_matrix, scaling_factors):
    """Compute tensor magnitude at each point on ellipsoid."""
    # Safe normalization
    sf_x, sf_y, sf_z = scaling_factors
    sf_x = max(sf_x, 1e-6)
    sf_y = max(sf_y, 1e-6)
    sf_z = max(sf_z, 1e-6)
    
    x_norm = x / sf_x
    y_norm = y / sf_y
    z_norm = z / sf_z
    
    # Quadratic form: v^T * K * v
    magnitude = (tensor_matrix[0,0] * x_norm**2 + 
                 tensor_matrix[1,1] * y_norm**2 + 
                 tensor_matrix[2,2] * z_norm**2)
    
    # Ensure positive
    magnitude = np.maximum(magnitude, 0)
    
    return magnitude

def matplotlib_to_plotly_colorscale(cmap, n=256):
    """
    Convert ANY matplotlib colormap (Listed or LinearSegmented)
    to a Plotly-compatible colorscale.
    """
    # Sample the colormap
    samples = cmap(np.linspace(0, 1, n))
    
    colorscale = []
    for i, (r, g, b, a) in enumerate(samples):
        colorscale.append([
            i / (n - 1),
            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        ])
    
    return colorscale

# ============================================================================
# STREAMLIT INTERFACE - SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Material Parameters")

# Material preset system
PRESETS = {
    "Custom": {},
    "Porous PDMS (Paper Default)": PDMS_DEFAULTS,
    "Isotropic Material": {
        "phi_x": 0.5, "phi_y": 0.5, "phi_z": 0.5,
        "Bx": 1.0e-9, "By": 1.0e-9, "Bz": 1.0e-9,
        "mx": 2.0, "my": 2.0, "mz": 2.0,
        "name": "Isotropic Material"
    },
    "Highly Anisotropic": {
        "phi_x": 0.8, "phi_y": 0.3, "phi_z": 0.6,
        "Bx": 5.0e-9, "By": 0.5e-9, "Bz": 2.0e-9,
        "mx": 1.5, "my": 2.5, "mz": 2.0,
        "name": "Highly Anisotropic"
    },
    "Layered Composite": {
        "phi_x": 0.4, "phi_y": 0.7, "phi_z": 0.4,
        "Bx": 0.8e-9, "By": 3.0e-9, "Bz": 0.8e-9,
        "mx": 1.8, "my": 1.8, "mz": 1.8,
        "name": "Layered Composite"
    }
}

preset = st.sidebar.selectbox("Material Preset:", list(PRESETS.keys()))

if preset != "Custom":
    preset_data = PRESETS[preset]
    phi_x, phi_y, phi_z = preset_data["phi_x"], preset_data["phi_y"], preset_data["phi_z"]
    Bx, By, Bz = preset_data["Bx"], preset_data["By"], preset_data["Bz"]
    mx, my, mz = preset_data["mx"], preset_data["my"], preset_data["mz"]
else:
    # Custom inputs with validation
    with st.sidebar.expander("Porosity Values (φ)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            phi_x = st.number_input(r"$\phi_x$ (YZ plane)", 0.01, 0.99, 0.2, 0.01,
                                   help="Porosity normal to YZ plane", key="phi_x")
        with col2:
            phi_y = st.number_input(r"$\phi_y$ (XZ plane)", 0.01, 0.99, 0.3, 0.01,
                                   help="Porosity normal to XZ plane", key="phi_y")
        with col3:
            phi_z = st.number_input(r"$\phi_z$ (XY plane)", 0.01, 0.99, 0.4, 0.01,
                                   help="Porosity normal to XY plane", key="phi_z")
    
    with st.sidebar.expander("Power Law Parameters", expanded=True):
        st.markdown(r"**Base Permeability Constants $\beta$ [m²]**")
        col1, col2, col3 = st.columns(3)
        with col1:
            Bx = st.number_input(r"$\beta_x$", 1.0e-12, 1.0e-4, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for X-direction", key="Bx")
        with col2:
            By = st.number_input(r"$\beta_y$", 1.0e-12, 1.0e-4, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for Y-direction", key="By")
        with col3:
            Bz = st.number_input(r"$\beta_z$", 1.0e-12, 1.0e-4, 1.0e-9, 1.0e-10, 
                                format="%.2e", help="Base constant for Z-direction", key="Bz")
        
        st.markdown(r"**Exponents $m$**")
        col1, col2, col3 = st.columns(3)
        with col1:
            mx = st.number_input(r"$m_x$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for X-direction", key="mx")
        with col2:
            my = st.number_input(r"$m_y$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for Y-direction", key="my")
        with col3:
            mz = st.number_input(r"$m_z$", 0.1, 10.0, 2.0, 0.1, 
                                help="Power law exponent for Z-direction", key="mz")

# Visualization options
st.sidebar.header("🎨 Visualization Options")

visualization_option = st.sidebar.radio(
    "Tensor to Visualize:",
    ('Permeability', 'Porous Resistivity', 'Both'),
    index=0
)

viz_library = st.sidebar.radio(
    "Visualization Library:",
    ('Plotly (Interactive)', 'Matplotlib (Static)'),
    index=0
)

# Enhanced colormap selection
all_colormaps = sorted(plt.colormaps())

# Categorize colormaps properly
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
    "Classic (Jet/Rainbow)": ['jet', 'rainbow', 'turbo', 'flag', 'prism', 'ocean', 
                              'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 
                              'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow',
                              'nipy_spectral', 'gist_ncar']
}

selected_category = st.sidebar.selectbox("Colormap Category:", list(cmap_categories.keys()))
cmap_name = st.sidebar.selectbox(f"Select colormap:", sorted(cmap_categories[selected_category]))

# Advanced settings
with st.sidebar.expander("Advanced Settings", expanded=False):
    resolution = st.slider("Mesh Resolution", 30, 200, 80, 10)
    show_axes = st.checkbox("Show Coordinate Axes", True)
    show_colorbar = st.checkbox("Show Colorbar", True)
    transparency = st.slider("Surface Transparency", 0.0, 1.0, 0.8, 0.1)
    
    if viz_library == 'Plotly (Interactive)':
        show_contour = st.checkbox("Show Contour Lines", False)
        lighting_effects = st.checkbox("Enable Lighting Effects", True)
        show_eigenvectors = st.checkbox("Show Principal Directions", True)
    else:
        show_wireframe = st.checkbox("Show Wireframe", False)

# ============================================================================
# COMPUTE TENSORS AND ANALYSIS
# ============================================================================
permeability_matrix, (k11, k22, k33) = compute_permeability_matrix(
    phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz
)

# Compute tensor analysis
k_analysis = compute_tensor_analysis(permeability_matrix)

# Calculate porous resistivity
try:
    porous_resistivity_matrix = np.linalg.inv(permeability_matrix)
    R11, R22, R33 = porous_resistivity_matrix[0,0], porous_resistivity_matrix[1,1], porous_resistivity_matrix[2,2]
except np.linalg.LinAlgError:
    # Fallback for singular matrices
    porous_resistivity_matrix = np.diag(1.0 / np.array([k11, k22, k33]))
    R11, R22, R33 = porous_resistivity_matrix[0,0], porous_resistivity_matrix[1,1], porous_resistivity_matrix[2,2]

R_analysis = compute_tensor_analysis(porous_resistivity_matrix)

# Compute anisotropy metrics
anisotropy_k = max(k11, k22, k33) / min(k11, k22, k33)
anisotropy_R = max(R11, R22, R33) / min(R11, R22, R33)

# ============================================================================
# DISPLAY TENSOR VALUES AND ANALYSIS
# ============================================================================
st.markdown("## 📊 Tensor Properties")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(r"$\kappa_{xx}$ (X-direction)", f"{k11:.2e} m²")
    st.metric(r"$R_{xx}$ (X-direction)", f"{R11:.2e} m⁻²")

with col2:
    st.metric(r"$\kappa_{yy}$ (Y-direction)", f"{k22:.2e} m²")
    st.metric(r"$R_{yy}$ (Y-direction)", f"{R22:.2e} m⁻²")

with col3:
    st.metric(r"$\kappa_{zz}$ (Z-direction)", f"{k33:.2e} m²")
    st.metric(r"$R_{zz}$ (Z-direction)", f"{R33:.2e} m⁻²")

# Anisotropy and condition number display
st.markdown("### 🔍 Tensor Analysis")
ana_col1, ana_col2, ana_col3, ana_col4 = st.columns(4)

with ana_col1:
    st.metric("Permeability Anisotropy", f"{anisotropy_k:.2f}")
with ana_col2:
    st.metric("Resistivity Anisotropy", f"{anisotropy_R:.2f}")
with ana_col3:
    st.metric(r"$\kappa$ Condition Number", f"{k_analysis['condition_number']:.2f}")
with ana_col4:
    st.metric(r"$R$ Condition Number", f"{R_analysis['condition_number']:.2f}")

# Show eigenvalues
with st.expander("Eigenvalue Analysis", expanded=False):
    col_eig1, col_eig2 = st.columns(2)
    
    with col_eig1:
        st.markdown("**Permeability Eigenvalues**")
        eig_df_k = pd.DataFrame({
            'Eigenvalue': k_analysis['eigenvalues'],
            'Direction': ['Principal 1', 'Principal 2', 'Principal 3']
        })
        st.dataframe(eig_df_k, use_container_width=True)
    
    with col_eig2:
        st.markdown("**Resistivity Eigenvalues**")
        eig_df_r = pd.DataFrame({
            'Eigenvalue': R_analysis['eigenvalues'],
            'Direction': ['Principal 1', 'Principal 2', 'Principal 3']
        })
        st.dataframe(eig_df_r, use_container_width=True)

# ============================================================================
# VISUALIZATION SECTION
# ============================================================================
st.markdown("## 🎯 3D Tensor Visualization")

if viz_library == 'Plotly (Interactive)':
    # Get matplotlib colormap
    matplotlib_cmap = cm.get_cmap(cmap_name)
    plotly_colorscale = matplotlib_to_plotly_colorscale(matplotlib_cmap)
    
    # Function to create Plotly surface
    def create_plotly_surface(tensor_matrix, title, colorbar_title, is_resistivity=False, show_vectors=False):
        """Create a Plotly surface with robust error handling."""
        
        # Safe scaling factors
        diag = np.diag(tensor_matrix)
        diag = np.maximum(diag, 1e-30)
        
        if is_resistivity:
            a, b, c = np.sqrt(diag)
            scaling_factors = (a, b, c)
        else:
            # Safe division
            Bx_safe = max(Bx, 1e-30)
            By_safe = max(By, 1e-30)
            Bz_safe = max(Bz, 1e-30)
            
            a = np.sqrt(k11 / Bx_safe) if k11 > 0 else 0.1
            b = np.sqrt(k22 / By_safe) if k22 > 0 else 0.1
            c = np.sqrt(k33 / Bz_safe) if k33 > 0 else 0.1
            
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Safe normalization
        mag_min, mag_max = magnitude.min(), magnitude.max()
        if mag_max > mag_min:
            magnitude_norm = (magnitude - mag_min) / (mag_max - mag_min)
        else:
            magnitude_norm = np.zeros_like(magnitude)
        
        # Create surface trace
        surface = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=magnitude,
            colorscale=plotly_colorscale,
            opacity=transparency,
            colorbar=dict(
                title=f"${colorbar_title}$",
                titleside="right",
                titlefont=dict(size=12)
            ),
            contours=dict(
                x=dict(show=show_contour, color='gray', width=1),
                y=dict(show=show_contour, color='gray', width=1),
                z=dict(show=show_contour, color='gray', width=1)
            ) if show_contour else {},
            lighting=dict(
                ambient=0.4,
                diffuse=0.9 if lighting_effects else 0.5,
                specular=0.3 if lighting_effects else 0.1,
                roughness=0.4,
            ),
            name=title,
            showscale=show_colorbar
        )
        
        # Add eigenvectors if requested
        vector_traces = []
        if show_vectors and show_eigenvectors:
            analysis = R_analysis if is_resistivity else k_analysis
            eigenvectors = analysis['eigenvectors']
            eigenvalues = analysis['eigenvalues']
            
            max_axis = max(a, b, c)
            scale_factor = max_axis * 1.2
            
            colors = ['#FF0000', '#00FF00', '#0000FF']  # RGB for principal directions
            
            for i in range(3):
                ev = eigenvectors[:, i]
                ev_norm = ev / np.linalg.norm(ev) if np.linalg.norm(ev) > 0 else ev
                
                # Scale by eigenvalue
                ev_scaled = ev_norm * scale_factor * (eigenvalues[i] / eigenvalues[0])
                
                vector_traces.append(
                    go.Scatter3d(
                        x=[0, ev_scaled[0]],
                        y=[0, ev_scaled[1]],
                        z=[0, ev_scaled[2]],
                        mode='lines+markers',
                        line=dict(color=colors[i], width=4),
                        marker=dict(size=4, color=colors[i]),
                        name=f'Principal {i+1}',
                        showlegend=True
                    )
                )
        
        return surface, vector_traces, X, Y, Z
    
    # Create visualization based on option
    if visualization_option == 'Both':
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=(
                r"$\boldsymbol{\kappa}$ Permeability Tensor",
                r"$\mathbf{R} = \boldsymbol{\kappa}^{-1}$ Resistivity Tensor"
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.1
        )
        
        # Permeability surface
        surf_perm, vectors_perm, X_perm, Y_perm, Z_perm = create_plotly_surface(
            permeability_matrix, 
            r'$\boldsymbol{\kappa}$', 
            r'\kappa\;[\mathrm{m}^2]', 
            is_resistivity=False,
            show_vectors=True
        )
        fig.add_trace(surf_perm, row=1, col=1)
        
        # Add vectors to permeability plot
        for trace in vectors_perm:
            fig.add_trace(trace, row=1, col=1)
        
        # Resistivity surface
        surf_res, vectors_res, X_res, Y_res, Z_res = create_plotly_surface(
            porous_resistivity_matrix, 
            r'$\mathbf{R}$', 
            r'R\;[\mathrm{m}^{-2}]', 
            is_resistivity=True,
            show_vectors=True
        )
        fig.add_trace(surf_res, row=1, col=2)
        
        # Add vectors to resistivity plot
        for trace in vectors_res:
            fig.add_trace(trace, row=1, col=2)
        
        # Update scene layouts
        scene1 = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=np.ptp(X_perm), y=np.ptp(Y_perm), z=np.ptp(Z_perm)),
            aspectmode='manual',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        scene2 = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=np.ptp(X_res), y=np.ptp(Y_res), z=np.ptp(Z_res)),
            aspectmode='manual',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        fig.update_layout(
            scene=scene1,
            scene2=scene2,
            width=1400,
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            title=dict(x=0.5, y=0.95, xanchor='center')
        )
        
    else:
        # Single tensor visualization
        if visualization_option == 'Permeability':
            tensor_matrix = permeability_matrix
            title = r'$\boldsymbol{\kappa}$ Permeability Tensor'
            colorbar_title = r'\kappa\;[\mathrm{m}^2]'
            is_resistivity = False
        else:
            tensor_matrix = porous_resistivity_matrix
            title = r'$\mathbf{R}$ Resistivity Tensor'
            colorbar_title = r'R\;[\mathrm{m}^{-2}]'
            is_resistivity = True
        
        surf, vectors, X, Y, Z = create_plotly_surface(
            tensor_matrix, title, colorbar_title, is_resistivity, show_vectors=True
        )
        
        fig = go.Figure(data=[surf] + vectors)
        
        scene = dict(
            xaxis_title=r'$x$',
            yaxis_title=r'$y$',
            zaxis_title=r'$z$',
            aspectratio=dict(x=np.ptp(X), y=np.ptp(Y), z=np.ptp(Z)),
            aspectmode='manual',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )
        
        fig.update_layout(
            scene=scene,
            width=900,
            height=700,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
            title=dict(text=title, x=0.5, y=0.95, xanchor='center')
        )
    
    # Add coordinate axes if requested
    if show_axes:
        axis_length = max(np.max(X), np.max(Y), np.max(Z)) * 1.5
        
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
                 r'$\mathbf{R} = \boldsymbol{\kappa}^{-1}$ Resistivity Tensor']
        tensors = [permeability_matrix, porous_resistivity_matrix]
        colorbar_titles = [r'$\kappa$ [m²]', r'$R$ [m⁻²]']
        is_resistivity_list = [False, True]
    else:
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        axes = [ax]
        
        if visualization_option == 'Permeability':
            titles = [r'$\boldsymbol{\kappa}$ Permeability Tensor']
            tensors = [permeability_matrix]
            colorbar_titles = [r'$\kappa$ [m²]']
            is_resistivity_list = [False]
        else:
            titles = [r'$\mathbf{R}$ Resistivity Tensor']
            tensors = [porous_resistivity_matrix]
            colorbar_titles = [r'$R$ [m⁻²]']
            is_resistivity_list = [True]
    
    cmap = cm.get_cmap(cmap_name)
    
    for idx, (ax, tensor_matrix, title, colorbar_title, is_resistivity) in enumerate(zip(
        axes, tensors, titles, colorbar_titles, is_resistivity_list)):
        
        # Scaling
        if is_resistivity:
            a, b, c = np.sqrt(np.diag(tensor_matrix))
            scaling_factors = (a, b, c)
        else:
            scaling_factors = (
                np.sqrt(k11 / max(Bx, 1e-30)),
                np.sqrt(k22 / max(By, 1e-30)),
                np.sqrt(k33 / max(Bz, 1e-30))
            )
            a, b, c = scaling_factors
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Normalize for coloring
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        
        # Create surface
        if 'show_wireframe' in locals() and show_wireframe:
            ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)
        
        surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(magnitude)),
                              rstride=2, cstride=2, alpha=transparency,
                              linewidth=0.1, antialiased=True, shade=True)
        
        # Add coordinate axes
        if show_axes:
            axis_length = max(a, b, c) * 1.5
            ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
            ax.text(axis_length, 0, 0, r'$x$', fontsize=12, color='red', ha='center')
            ax.text(0, axis_length, 0, r'$y$', fontsize=12, color='green', ha='center')
            ax.text(0, 0, axis_length, r'$z$', fontsize=12, color='blue', ha='center')
        
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
        ax.grid(True, alpha=0.3)
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
# DATA EXPORT AND REPORT GENERATION
# ============================================================================
st.markdown("---")
st.markdown("## 📥 Export Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Download permeability tensor
    df_k = pd.DataFrame(permeability_matrix, 
                       index=['X', 'Y', 'Z'], 
                       columns=['X', 'Y', 'Z'])
    csv_k = df_k.to_csv()
    st.download_button(
        label="Download κ Tensor (CSV)",
        data=csv_k,
        file_name="permeability_tensor.csv",
        mime="text/csv",
        type="primary"
    )

with col2:
    # Download resistivity tensor
    df_r = pd.DataFrame(porous_resistivity_matrix,
                       index=['X', 'Y', 'Z'],
                       columns=['X', 'Y', 'Z'])
    csv_r = df_r.to_csv()
    st.download_button(
        label="Download R Tensor (CSV)",
        data=csv_r,
        file_name="resistivity_tensor.csv",
        mime="text/csv",
        type="primary"
    )

with col3:
    # Generate comprehensive report
    report = {
        "analysis_metadata": {
            "tool": "Orthotropic Tensor Analyzer",
            "version": "2.0",
            "preset": preset
        },
        "input_parameters": {
            "porosity": {"phi_x": phi_x, "phi_y": phi_y, "phi_z": phi_z},
            "base_constants": {"Bx": Bx, "By": By, "Bz": Bz},
            "exponents": {"mx": mx, "my": my, "mz": mz}
        },
        "permeability_results": {
            "tensor": permeability_matrix.tolist(),
            "components": {"kappa_xx": k11, "kappa_yy": k22, "kappa_zz": k33},
            "eigenvalues": k_analysis['eigenvalues'].tolist(),
            "eigenvectors": k_analysis['eigenvectors'].tolist(),
            "condition_number": float(k_analysis['condition_number'])
        },
        "resistivity_results": {
            "tensor": porous_resistivity_matrix.tolist(),
            "components": {"R_xx": R11, "R_yy": R22, "R_zz": R33},
            "eigenvalues": R_analysis['eigenvalues'].tolist(),
            "eigenvectors": R_analysis['eigenvectors'].tolist(),
            "condition_number": float(R_analysis['condition_number'])
        },
        "anisotropy_analysis": {
            "permeability_ratio": float(anisotropy_k),
            "resistivity_ratio": float(anisotropy_R),
            "classification": "Isotropic" if anisotropy_k < 1.1 else 
                            "Moderately Anisotropic" if anisotropy_k < 3 else 
                            "Highly Anisotropic"
        }
    }
    
    json_report = json.dumps(report, indent=2)
    st.download_button(
        label="Download Full Report (JSON)",
        data=json_report,
        file_name="tensor_analysis_report.json",
        mime="application/json",
        type="primary"
    )

# Generate LaTeX representation
st.markdown("### 📝 LaTeX Export")
with st.expander("LaTeX Tensor Representations", expanded=False):
    latex_code = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{bm}}

\\begin{{document}}

\\section*{{Tensor Analysis Report}}

\\subsection*{{Input Parameters}}
\\begin{{align*}}
    \\phi_x &= {phi_x:.3f}, \\quad \\phi_y = {phi_y:.3f}, \\quad \\phi_z = {phi_z:.3f} \\\\
    \\beta_x &= {Bx:.2e}\\,\\text{{m}}^2, \\quad \\beta_y = {By:.2e}\\,\\text{{m}}^2, \\quad \\beta_z = {Bz:.2e}\\,\\text{{m}}^2 \\\\
    m_x &= {mx:.2f}, \\quad m_y = {my:.2f}, \\quad m_z = {mz:.2f}
\\end{{align*}}

\\subsection*{{Permeability Tensor}}
\\begin{{equation}}
    \\bm{{\\kappa}} = 
    \\begin{{bmatrix}}
        {k11:.2e} & 0 & 0 \\\\
        0 & {k22:.2e} & 0 \\\\
        0 & 0 & {k33:.2e}
    \\end{{bmatrix}}
    \\,\\text{{m}}^2
\\end{{equation}}

\\subsection*{{Resistivity Tensor}}
\\begin{{equation}}
    \\mathbf{{R}} = \\bm{{\\kappa}}^{{-1}} = 
    \\begin{{bmatrix}}
        {R11:.2e} & 0 & 0 \\\\
        0 & {R22:.2e} & 0 \\\\
        0 & 0 & {R33:.2e}
    \\end{{bmatrix}}
    \\,\\text{{m}}^{{-2}}
\\end{{equation}}

\\subsection*{{Anisotropy Analysis}}
\\begin{{align*}}
    \\text{{Permeability Anisotropy Ratio}} &= {anisotropy_k:.2f} \\\\
    \\text{{Resistivity Anisotropy Ratio}} &= {anisotropy_R:.2f} \\\\
    \\text{{Condition Number }}\\kappa &= {k_analysis['condition_number']:.2f} \\\\
    \\text{{Condition Number }}R &= {R_analysis['condition_number']:.2f}
\\end{{align*}}

\\end{{document}}
"""
    
    st.code(latex_code, language="latex")
    
    st.download_button(
        label="Download LaTeX Report",
        data=latex_code,
        file_name="tensor_analysis.tex",
        mime="text/plain",
        type="secondary"
    )

# ============================================================================
# INTERPRETATION AND APPLICATION GUIDE
# ============================================================================
with st.expander("🔍 Advanced Interpretation Guide", expanded=False):
    st.markdown("""
    ### 📊 Tensor Ellipsoid Interpretation
    
    #### **Shape Analysis**
    1. **Spherical Ellipsoid** ($a ≈ b ≈ c$):
       - **Interpretation**: Isotropic material
       - **Flow behavior**: Uniform in all directions
       - **Application**: Ideal for uniform filtration
    
    2. **Prolate Ellipsoid** ($a > b ≈ c$, cigar-shaped):
       - **Interpretation**: Uniaxial anisotropy
       - **Flow behavior**: Preferential flow along major axis
       - **Application**: Directional flow control
    
    3. **Oblate Ellipsoid** ($a ≈ b > c$, disc-shaped):
       - **Interpretation**: Planar anisotropy
       - **Flow behavior**: Enhanced flow in plane, restricted normal to plane
       - **Application**: Membrane filters, thin films
    
    4. **Triaxial Ellipsoid** ($a ≠ b ≠ c$):
       - **Interpretation**: Full anisotropy
       - **Flow behavior**: Complex directional dependence
       - **Application**: Engineered porous media
    
    #### **Color Mapping Interpretation**
    - **Warm colors** (red, orange, yellow): Higher tensor values
    - **Cool colors** (blue, purple): Lower tensor values
    - **Gradients**: Indicate rate of change with direction
    
    ### 🎯 PDMS-Specific Insights
    
    Based on experimental data:
    - **Y-direction dominance**: Typically highest permeability in fabricated PDMS
    - **Template effect**: Sugar template method creates directional porosity
    - **Functionalization impact**: Surface treatments modify directional properties
    
    ### ⚙️ Engineering Applications
    
    1. **Filter Design Optimization**:
       - Align flow with high permeability directions
       - Use anisotropy to enhance contact time
       - Design graded porosity for staged filtration
    
    2. **Material Characterization**:
       - Anisotropy ratio > 3: Consider directional design
       - Condition number > 10: Numerical challenges in simulation
       - Eigenvector alignment: Material processing effects
    
    3. **Performance Prediction**:
       - Higher permeability: Faster flow, shorter residence time
       - Higher resistivity: Slower flow, better adsorption
       - Anisotropy: Direction-dependent efficiency
    
    ### 🔬 Advanced Metrics
    
    1. **Tensor Condition Number**:
       - $CN > 100$: Ill-conditioned, numerical instability
       - $CN < 10$: Well-conditioned, stable computations
       - Critical for finite element simulations
    
    2. **Principal Direction Analysis**:
       - Eigenvectors show natural flow directions
       - Alignment with material axes indicates fabrication quality
       - Deviation suggests microstructural defects
    
    3. **Anisotropy Classification**:
       - **Mild**: 1.0-1.5 (nearly isotropic)
       - **Moderate**: 1.5-3.0 (noticeable directionality)
       - **Strong**: 3.0-10.0 (highly directional)
       - **Extreme**: >10.0 (essentially 1D/2D material)
    """)

# ============================================================================
# FOOTER AND CITATIONS
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
<p><strong>Advanced Orthotropic Tensor Analyzer v2.0</strong></p>
<p>Based on: <em>"Microstructural contributions of PDMS sponges to their water treatment efficacy"</em></p>
<p>Tensor model: $\kappa_{ii} = \beta_i \times (\phi_i)^{m_i}$ • PDMS parameters from experimental characterization</p>
<p>Includes: Rainbow, Jet, Turbo colormaps • Plotly interactive visualization • Matplotlib static rendering</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# PERFORMANCE METRICS (Optional)
# ============================================================================
with st.sidebar.expander("Performance Metrics", expanded=False):
    import time
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    
    elapsed = time.time() - st.session_state.start_time
    st.metric("Computation Time", f"{elapsed:.3f} s")
    st.metric("Tensor Size", "3×3")
    st.metric("Visualization Points", f"{resolution}²")
    
    # Reset timer
    if st.button("Reset Timer"):
        st.session_state.start_time = time.time()
        st.rerun()
