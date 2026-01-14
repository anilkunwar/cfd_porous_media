import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json

# Page Configuration
st.set_page_config(
    page_title="Orthotropic Tensor Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MATHEMATICAL FOUNDATION SECTION (MAIN PAGE)
# ============================================================================
st.markdown("""
# 🧮 Mathematical Foundations of Orthotropic Tensor Analysis

This tool visualizes the directional dependence of permeability and porous resistivity tensors 
for anisotropic porous materials like Polydimethylsiloxane (PDMS) sponges.

## 📐 Mathematical Formulation

### 1. **Directional Porosity**
Porosity varies with direction in orthotropic materials:
$$
\\phi_i = \\text{Porosity in direction } i \\quad (i = x, y, z)
$$
where $\\phi_i$ represents porosity normal to the plane perpendicular to direction $i$.

### 2. **Permeability Power Law**
The principal permeability components follow a power-law relationship:
$$
\\boxed{\\kappa_{ii} = \\beta_i \\times (\\phi_i)^{m_i}}
$$
- $\\kappa_{ii}$: Permeability component in direction $i$ (m²)
- $\\beta_i$: Empirical constant (m²)
- $\\phi_i$: Directional porosity (dimensionless, 0–1)
- $m_i$: Empirical exponent

### 3. **Permeability Tensor**
For orthotropic materials, the permeability tensor is diagonal:
$$
\\boldsymbol{\\kappa} = 
\\begin{bmatrix}
\\kappa_{xx} & 0 & 0 \\\\
0 & \\kappa_{yy} & 0 \\\\
0 & 0 & \\kappa_{zz}
\\end{bmatrix}
\\quad \\text{(m²)}
$$

### 4. **Porous Resistivity Tensor**
The resistivity tensor is the inverse of the permeability tensor:
$$
\\mathbf{R} = \\boldsymbol{\\kappa}^{-1} = 
\\begin{bmatrix}
R_{xx} & 0 & 0 \\\\
0 & R_{yy} & 0 \\\\
0 & 0 & R_{zz}
\\end{bmatrix}
\\quad \\text{(m⁻²)}
$$
where $R_{ii} = 1/\\kappa_{ii}$ for the diagonal components.

### 5. **Navier-Stokes Extension for Porous Media**
The momentum equation for flow through porous media includes a resistivity term:
$$
\\rho \\frac{\\partial \\mathbf{v}}{\\partial t} + \\rho (\\mathbf{v} \\cdot \\nabla) \\mathbf{v} = 
-\\nabla p + \\mu \\nabla^2 \\mathbf{v} - \\mu \\mathbf{R} \\mathbf{v}
$$
where $\\mathbf{R}$ is the resistivity tensor representing drag forces from the porous structure.

### 6. **Tensor Representation Surface**
The visualization shows a quadric surface representing the tensor magnitude in all directions:
$$
\\mathbf{n}^T \\boldsymbol{\\kappa} \\mathbf{n} = \\text{constant}
$$
where $\\mathbf{n}$ is a unit direction vector.
""")

# Add a separator
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
    permeability_matrix = np.array([[k11, 0, 0],
                                    [0, k22, 0],
                                    [0, 0, k33]])
    permeability_matrix = np.maximum(permeability_matrix, 1e-30)  # Avoid zero
    
    return permeability_matrix, (k11, k22, k33)

def compute_ellipsoid_points(a, b, c, n_points=100):
    """Generate points for an ellipsoid with given semi-axes."""
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

def compute_tensor_magnitude(x, y, z, tensor_matrix, scaling_factors):
    """Compute tensor magnitude at each point on ellipsoid."""
    # Normalize coordinates by scaling factors
    x_norm = x / scaling_factors[0]
    y_norm = y / scaling_factors[1]
    z_norm = z / scaling_factors[2]
    
    magnitude = (tensor_matrix[0,0] * x_norm**2 + 
                 tensor_matrix[1,1] * y_norm**2 + 
                 tensor_matrix[2,2] * z_norm**2)
    
    return magnitude

# ============================================================================
# STREAMLIT INTERFACE - SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Material Parameters")

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
    # Custom inputs in expandable sections
    with st.sidebar.expander("Porosity Values (φ)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            phi_x = st.number_input("φₓ", 0.01, 0.99, 0.2, 0.01,
                                   help="Porosity normal to YZ plane")
        with col2:
            phi_y = st.number_input("φᵧ", 0.01, 0.99, 0.3, 0.01,
                                   help="Porosity normal to XZ plane")
        with col3:
            phi_z = st.number_input("φ_z", 0.01, 0.99, 0.4, 0.01,
                                   help="Porosity normal to XY plane")
    
    with st.sidebar.expander("Power Law Parameters", expanded=True):
        st.markdown("**Base Permeability Constants β (m²)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            Bx = st.number_input("βₓ", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
        with col2:
            By = st.number_input("βᵧ", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
        with col3:
            Bz = st.number_input("β_z", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
        
        st.markdown("**Exponents m**")
        col1, col2, col3 = st.columns(3)
        with col1:
            mx = st.number_input("mₓ", 0.1, 5.0, 2.0, 0.1)
        with col2:
            my = st.number_input("mᵧ", 0.1, 5.0, 2.0, 0.1)
        with col3:
            mz = st.number_input("m_z", 0.1, 5.0, 2.0, 0.1)

# Visualization options
st.sidebar.header("🎨 Visualization Options")

visualization_option = st.sidebar.radio(
    "Tensor to Visualize:",
    ('Permeability', 'Porous Resistivity', 'Both'),
    index=0
)

# Visualization library selection
viz_library = st.sidebar.radio(
    "Visualization Library:",
    ('Plotly (Interactive)', 'Matplotlib (Static)'),
    index=0
)

# Enhanced colormap selection
all_colormaps = sorted(plt.colormaps())

# Create categorized colormaps
cmap_categories = {
    "Sequential": ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                   'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                   'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                   'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                   'summer', 'autumn', 'winter', 'spring', 'cool', 'Wistia',
                   'hot', 'afmhot', 'gist_heat', 'copper', 'bone', 'pink'],
    
    "Diverging": ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'],
    
    "Cyclic": ['twilight', 'twilight_shifted', 'hsv'],
    
    "Qualitative": ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'],
    
    "Perceptually Uniform": ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    
    "Classic": ['rainbow', 'jet', 'turbo', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow',
                'nipy_spectral', 'gist_ncar', 'gist_yarg', 'gist_gray', 'gray', 'binary']
}

# Ensure rainbow, jet, turbo are included
for cmap in ['rainbow', 'jet', 'turbo']:
    if cmap not in cmap_categories["Classic"]:
        cmap_categories["Classic"].append(cmap)

selected_category = st.sidebar.selectbox("Colormap Category:", list(cmap_categories.keys()))
cmap_name = st.sidebar.selectbox(f"Select {selected_category} colormap:", 
                                 sorted(cmap_categories[selected_category]))

# Additional visualization parameters
with st.sidebar.expander("Advanced Settings"):
    resolution = st.slider("Mesh Resolution", 30, 200, 80, 10)
    show_axes = st.checkbox("Show Coordinate Axes", True)
    show_colorbar = st.checkbox("Show Colorbar", True)
    transparency = st.slider("Surface Transparency", 0.0, 1.0, 0.8, 0.1)
    if viz_library == 'Plotly (Interactive)':
        show_contour = st.checkbox("Show Contour Lines", False)
        lighting_effects = st.checkbox("Enable Lighting Effects", True)
    else:
        show_wireframe = st.checkbox("Show Wireframe", False)

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
    porous_resistivity_matrix = np.eye(3) * 1e10
    R11 = R22 = R33 = 1e10

# ============================================================================
# DISPLAY TENSOR VALUES
# ============================================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("κₓₓ (X-direction)", f"{k11:.2e} m²")
    st.metric("Rₓₓ (X-direction)", f"{R11:.2e} m⁻²")

with col2:
    st.metric("κᵧᵧ (Y-direction)", f"{k22:.2e} m²")
    st.metric("Rᵧᵧ (Y-direction)", f"{R22:.2e} m⁻²")

with col3:
    st.metric("κ_zz (Z-direction)", f"{k33:.2e} m²")
    st.metric("R_zz (Z-direction)", f"{R33:.2e} m⁻²")

# Anisotropy metrics
anisotropy_k = max(k11, k22, k33) / min(k11, k22, k33)
anisotropy_R = max(R11, R22, R33) / min(R11, R22, R33)

st.info(f"**Anisotropy Ratio**: Permeability = {anisotropy_k:.2f}, Resistivity = {anisotropy_R:.2f}")

# ============================================================================
# VISUALIZATION SECTION
# ============================================================================
st.markdown("## 🎯 3D Tensor Visualization")

if viz_library == 'Plotly (Interactive)':
    # Create subplots based on visualization option
    if visualization_option == 'Both':
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=('Permeability Tensor (κ)', 'Porous Resistivity Tensor (R)'),
            horizontal_spacing=0.1
        )
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'surface'}]],
            subplot_titles=(f'{visualization_option} Tensor',)
        )
    
    # Get matplotlib colormap
    matplotlib_cmap = cm.get_cmap(cmap_name)
    
    # Helper function to create Plotly surface
    def create_plotly_surface(tensor_matrix, title, colorbar_title, is_resistivity=False):
        if is_resistivity:
            # For resistivity, use sqrt of diagonal as scaling
            diag_vals = np.diag(tensor_matrix)
            a, b, c = np.sqrt(diag_vals)
            scaling_factors = (a, b, c)
        else:
            # For permeability, use sqrt of (kappa / beta) as scaling
            # This ensures consistent shape representation
            a = np.sqrt(k11 / Bx) if Bx > 0 else 1.0
            b = np.sqrt(k22 / By) if By > 0 else 1.0
            c = np.sqrt(k33 / Bz) if Bz > 0 else 1.0
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Create a proper Plotly colorscale by sampling the matplotlib colormap
        n_colors = 256
        sample_points = np.linspace(0, 1, n_colors)
        colors_rgba = matplotlib_cmap(sample_points)  # Shape: (n_colors, 4)
        colors_plotly = [
            [t, f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"] 
            for t, (r, g, b, a_val) in zip(sample_points, colors_rgba)
        ]
        
        # Create surface trace
        surface = go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=magnitude,
            colorscale=colors_plotly,
            colorbar=dict(title=colorbar_title, titleside='right') if show_colorbar else None,
            opacity=transparency,
            contours=dict(
                x=dict(show=show_contour, color='gray', width=1),
                y=dict(show=show_contour, color='gray', width=1),
                z=dict(show=show_contour, color='gray', width=1)
            ),
            lighting=dict(
                ambient=0.4,
                diffuse=0.9 if lighting_effects else 0.5,
                specular=0.3 if lighting_effects else 0.1,
                roughness=0.5
            ),
            name=title
        )
        
        return surface, X, Y, Z
    
    # Add traces based on visualization option
    if visualization_option == 'Both':
        # Permeability surface
        surf_perm, X_perm, Y_perm, Z_perm = create_plotly_surface(
            permeability_matrix, 'Permeability', 'Permeability (m²)', False
        )
        fig.add_trace(surf_perm, row=1, col=1)
        
        # Resistivity surface
        surf_res, X_res, Y_res, Z_res = create_plotly_surface(
            porous_resistivity_matrix, 'Resistivity', 'Resistivity (m⁻²)', True
        )
        fig.add_trace(surf_res, row=1, col=2)
        
        # Update scenes
        max_range_perm = max(np.ptp(X_perm), np.ptp(Y_perm), np.ptp(Z_perm))
        max_range_res = max(np.ptp(X_res), np.ptp(Y_res), np.ptp(Z_res))
        
        scene1 = dict(
            xaxis_title='X Direction',
            yaxis_title='Y Direction',
            zaxis_title='Z Direction',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube'
        )
        
        scene2 = dict(
            xaxis_title='X Direction',
            yaxis_title='Y Direction',
            zaxis_title='Z Direction',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube'
        )
        
        fig.update_layout(
            scene=scene1,
            scene2=scene2,
            width=1200,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50),
            showlegend=False
        )
    else:
        # Single surface
        if visualization_option == 'Permeability':
            tensor_matrix = permeability_matrix
            colorbar_title = 'Permeability (m²)'
            is_resistivity = False
        else:
            tensor_matrix = porous_resistivity_matrix
            colorbar_title = 'Resistivity (m⁻²)'
            is_resistivity = True
        
        surf, X, Y, Z = create_plotly_surface(
            tensor_matrix, visualization_option, colorbar_title, is_resistivity
        )
        fig.add_trace(surf)
        
        # Update layout
        scene = dict(
            xaxis_title='X Direction',
            yaxis_title='Y Direction',
            zaxis_title='Z Direction',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube'
        )
        
        fig.update_layout(
            scene=scene,
            width=800,
            height=600,
            margin=dict(l=50, r=50, b=50, t=50),
            showlegend=False,
            title=f"{visualization_option} Tensor Visualization"
        )
    
    # Add coordinate axes if requested
    if show_axes:
        # Determine axis length from data
        all_X = X_perm if visualization_option == 'Both' else X
        all_Y = Y_perm if visualization_option == 'Both' else Y
        all_Z = Z_perm if visualization_option == 'Both' else Z
        axis_length = max(np.max(np.abs(all_X)), np.max(np.abs(all_Y)), np.max(np.abs(all_Z))) * 1.2
        
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
    
    st.plotly_chart(fig, use_container_width=True)

else:  # Matplotlib visualization
    # Create figure based on visualization option
    if visualization_option == 'Both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
        titles = ['Permeability Tensor (κ)', 'Porous Resistivity Tensor (R)']
        tensors = [permeability_matrix, porous_resistivity_matrix]
        colorbar_titles = ['Permeability (m²)', 'Resistivity (m⁻²)']
        is_resistivity_list = [False, True]
    else:
        fig, ax = plt.subplots(figsize=(12, 9), subplot_kw={'projection': '3d'})
        axes = [ax]
        titles = [f'{visualization_option} Tensor']
        
        if visualization_option == 'Permeability':
            tensors = [permeability_matrix]
            colorbar_titles = ['Permeability (m²)']
            is_resistivity_list = [False]
        else:
            tensors = [porous_resistivity_matrix]
            colorbar_titles = ['Resistivity (m⁻²)']
            is_resistivity_list = [True]
    
    cmap = cm.get_cmap(cmap_name)
    
    for idx, (ax, tensor_matrix, title, colorbar_title, is_resistivity) in enumerate(zip(
        axes, tensors, titles, colorbar_titles, is_resistivity_list)):
        
        if is_resistivity:
            diag_vals = np.diag(tensor_matrix)
            a, b, c = np.sqrt(diag_vals)
            scaling_factors = (a, b, c)
        else:
            a = np.sqrt(k11 / Bx) if Bx > 0 else 1.0
            b = np.sqrt(k22 / By) if By > 0 else 1.0
            c = np.sqrt(k33 / Bz) if Bz > 0 else 1.0
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Normalize for coloring
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        
        # Create surface
        if show_wireframe:
            ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)
        
        surf = ax.plot_surface(X, Y, Z, facecolors=cmap(norm(magnitude)),
                              rstride=1, cstride=1, alpha=transparency,
                              linewidth=0, antialiased=True, shade=True)
        
        # Add coordinate axes
        if show_axes:
            axis_length = max(a, b, c) * 1.2
            ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
            ax.text(axis_length, 0, 0, 'X', fontsize=12, color='red')
            ax.text(0, axis_length, 0, 'Y', fontsize=12, color='green')
            ax.text(0, 0, axis_length, 'Z', fontsize=12, color='blue')
        
        # Set labels and title
        ax.set_xlabel('X Direction', fontsize=11)
        ax.set_ylabel('Y Direction', fontsize=11)
        ax.set_zlabel('Z Direction', fontsize=11)
        ax.set_title(title, fontsize=14)
        
        # Equal aspect ratio
        max_range = max(np.ptp(X), np.ptp(Y), np.ptp(Z))
        mid_x = (X.max() + X.min()) / 2
        mid_y = (Y.max() + Y.min()) / 2
        mid_z = (Z.max() + Z.min()) / 2
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # Add colorbar
        if show_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label(colorbar_title, fontsize=11, rotation=270, labelpad=15)
        
        # Style
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    
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
        mime="text/csv"
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
        mime="text/csv"
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
        mime="application/json"
    )

# ============================================================================
# INTERPRETATION GUIDE
# ============================================================================
with st.expander("🔍 Interpretation Guide", expanded=False):
    st.markdown("""
    ### How to Interpret the Visualizations
    
    #### 1. **Tensor Ellipsoid Shape**
    - **Spherical Shape**: Material is isotropic (same properties in all directions)
    - **Prolate Ellipsoid** (cigar-shaped): Material has one dominant flow direction
    - **Oblate Ellipsoid** (disc-shaped): Material has two similar flow directions, different from third
    - **Triaxial Ellipsoid**: All three principal directions have different properties
    
    #### 2. **Color Mapping**
    - Colors represent the magnitude of the tensor property in each direction
    - Warmer colors (red/yellow) typically indicate higher values
    - Cooler colors (blue/purple) indicate lower values
    - The color scale helps identify directional variations
    
    #### 3. **PDMS-Specific Insights**
    Based on the paper's default values:
    - **Y-direction** has highest permeability (1.13×10⁻⁹ m²)
    - **X and Z directions** have similar but lower permeability
    - This creates a **moderately anisotropic** ellipsoid
    - The resistivity tensor shows inverse relationships
    
    #### 4. **Practical Implications for PDMS Sponges**
    - Higher permeability directions allow easier fluid flow
    - Lower permeability directions create more resistance
    - Anisotropy affects:
        - Fluid transport efficiency
        - Adsorption contact time
        - Filter design optimization
        - Directional performance in water treatment
    
    #### 5. **Key Metrics to Monitor**
    - **Anisotropy Ratio**: Values > 3 indicate strong directional dependence
    - **Tensor Condition Number**: Ratio of largest to smallest eigenvalue
    - **Principal Direction Alignment**: How the tensor aligns with material axes
    
    #### 6. **Design Recommendations**
    - For uniform filtration: Aim for isotropic properties (spherical tensor)
    - For directional flow control: Design anisotropic materials
    - Optimize porosity distribution to achieve desired tensor properties
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
<p><strong>Orthotropic Tensor Analyzer</strong> • Based on: <em>"Microstructural contributions of PDMS sponges to their water treatment efficacy"</em></p>
<p>Tensor visualization using power law: $\\kappa_{ii} = \\beta_i \\times (\\phi_i)^{m_i}$ • PDMS defaults from experimental measurements</p>
</div>
""", unsafe_allow_html=True)
