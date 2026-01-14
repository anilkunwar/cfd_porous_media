import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.cm import ScalarMappable
import pandas as pd
from scipy.spatial import ConvexHull

# Page Configuration
st.set_page_config(
    page_title="Orthotropic Tensor Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MATHEMATICAL BACKGROUND SECTION
# ============================================================================
with st.sidebar.expander("📚 Mathematical Foundations", expanded=False):
    st.markdown("""
    ### Tensor Definitions
    
    **1. Porosity Tensor (Directional):**
    \[
    \phi_i = \text{Porosity normal to plane } jk
    \]
    
    **2. Permeability Power Law:**
    \[
    \kappa_{ii} = \beta_i \times (\phi_i)^{m_i}
    \]
    where:
    - \(\kappa_{ii}\): Principal permeability component (m²)
    - \(\beta_i\): Empirical constant (m²)
    - \(\phi_i\): Directional porosity (0-1)
    - \(m_i\): Empirical exponent
    
    **3. Permeability Tensor:**
    \[
    \boldsymbol{\kappa} = 
    \begin{bmatrix}
    \kappa_{xx} & 0 & 0 \\\\
    0 & \kappa_{yy} & 0 \\\\
    0 & 0 & \kappa_{zz}
    \end{bmatrix}
    \]
    
    **4. Porous Resistivity Tensor:**
    \[
    \mathbf{R} = \boldsymbol{\kappa}^{-1} = \mu \boldsymbol{\lambda}
    \]
    where \(\boldsymbol{\lambda}\) is the resistivity tensor in the momentum equation.
    
    **5. Ellipsoid Representation:**
    \[
    \frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1
    \]
    where \(a, b, c \propto \sqrt{\kappa_{ii}}\) for permeability visualization.
    """)

# ============================================================================
# DEFAULT VALUES FOR POROUS PDMS MATERIAL (from the paper)
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
    permeability_matrix = np.maximum(np.array([[k11, 0, 0],
                                               [0, k22, 0],
                                               [0, 0, k33]]), 1e-20)
    
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
    # Normalize by scaling factors to get directional components
    x_norm = x / scaling_factors[0]
    y_norm = y / scaling_factors[1]
    z_norm = z / scaling_factors[2]
    
    # Compute quadratic form: v^T * K * v
    magnitude = (tensor_matrix[0,0] * x_norm**2 + 
                 tensor_matrix[1,1] * y_norm**2 + 
                 tensor_matrix[2,2] * z_norm**2)
    
    return magnitude

# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================
st.title("🧭 Orthotropic Permeability & Resistivity Tensor Visualizer")
st.markdown("""
Visualize the directional dependence of permeability and porous resistivity tensors for anisotropic materials like porous PDMS.
""")

# ============================================================================
# SIDEBAR CONTROLS
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
    # Custom inputs
    with st.sidebar.expander("Porosity Values (φ)", expanded=True):
        phi_x = st.number_input("φ_x (YZ plane)", 0.01, 0.99, 0.2, 0.01,
                               help="Porosity normal to YZ plane")
        phi_y = st.number_input("φ_y (XZ plane)", 0.01, 0.99, 0.3, 0.01,
                               help="Porosity normal to XZ plane")
        phi_z = st.number_input("φ_z (XY plane)", 0.01, 0.99, 0.4, 0.01,
                               help="Porosity normal to XY plane")
    
    with st.sidebar.expander("Power Law Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**β (m²)**")
            Bx = st.number_input("β_x", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
            By = st.number_input("β_y", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
            Bz = st.number_input("β_z", 1.0e-11, 1.0e-5, 1.0e-9, 1.0e-10, format="%.2e")
        with col2:
            st.markdown("**Exponent m**")
            mx = st.number_input("m_x", 0.1, 5.0, 2.0, 0.1)
            my = st.number_input("m_y", 0.1, 5.0, 2.0, 0.1)
            mz = st.number_input("m_z", 0.1, 5.0, 2.0, 0.1)

# Visualization options
st.sidebar.header("🎨 Visualization Options")

visualization_option = st.sidebar.radio(
    "Tensor to Visualize:",
    ('Permeability', 'Porous Resistivity', 'Both'),
    index=0
)

# Enhanced colormap selection with categories
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
    
    "Miscellaneous": ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow',
                      'turbo', 'nipy_spectral', 'gist_ncar', 'gist_yarg',
                      'gist_gray', 'gray', 'binary', 'jet']
}

selected_category = st.sidebar.selectbox("Colormap Category:", list(cmap_categories.keys()))
cmap_name = st.sidebar.selectbox(f"Select {selected_category} colormap:", cmap_categories[selected_category])

# Additional visualization parameters
with st.sidebar.expander("Advanced Settings"):
    resolution = st.slider("Mesh Resolution", 50, 300, 100, 50)
    show_axes = st.checkbox("Show Coordinate Axes", True)
    show_colorbar = st.checkbox("Show Colorbar", True)
    transparency = st.slider("Surface Transparency", 0.0, 1.0, 0.8, 0.1)

# ============================================================================
# COMPUTE TENSORS
# ============================================================================
permeability_matrix, (k11, k22, k33) = compute_permeability_matrix(
    phi_x, phi_y, phi_z, Bx, By, Bz, mx, my, mz
)

# Calculate porous resistivity (inverse of permeability)
try:
    porous_resistivity_matrix = np.linalg.inv(permeability_matrix)
except np.linalg.LinAlgError:
    st.error("Singular matrix encountered. Please adjust parameters.")
    porous_resistivity_matrix = np.zeros((3, 3))

# ============================================================================
# MAIN DISPLAY AREA
# ============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Computed Tensors")
    
    # Display matrices in expandable sections
    with st.expander("Permeability Tensor (κ)", expanded=True):
        st.latex(r"\kappa = \begin{bmatrix}" +
                f"{k11:.2e} & 0 & 0 \\\\ 0 & {k22:.2e} & 0 \\\\ 0 & 0 & {k33:.2e}" +
                r"\end{bmatrix} \text{m}^2")
        
        # Bar chart for directional comparison
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        directions = ['X', 'Y', 'Z']
        values = [k11, k22, k33]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax_bar.bar(directions, values, color=colors)
        ax_bar.set_ylabel('Permeability (m²)')
        ax_bar.set_yscale('log')
        ax_bar.set_title('Directional Permeability Comparison')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1e}', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    
    with st.expander("Porous Resistivity Tensor (R)", expanded=False):
        st.latex(r"R = \kappa^{-1} = \begin{bmatrix}" +
                f"{porous_resistivity_matrix[0,0]:.2e} & 0 & 0 \\\\ " +
                f"0 & {porous_resistivity_matrix[1,1]:.2e} & 0 \\\\ " +
                f"0 & 0 & {porous_resistivity_matrix[2,2]:.2e}" +
                r"\end{bmatrix} \text{m}^{-2}")
    
    # Anisotropy metrics
    with st.expander("Anisotropy Analysis", expanded=False):
        anisotropy_ratio = max(k11, k22, k33) / min(k11, k22, k33)
        st.metric("Anisotropy Ratio", f"{anisotropy_ratio:.2f}")
        
        if anisotropy_ratio < 1.1:
            st.info("Material is nearly isotropic")
        elif anisotropy_ratio < 3:
            st.info("Moderate anisotropy")
        else:
            st.warning("Strong anisotropy present")
    
    # Download buttons
    st.subheader("💾 Export Data")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        if st.button("Download κ as CSV", use_container_width=True):
            df_k = pd.DataFrame(permeability_matrix, 
                               index=['X', 'Y', 'Z'], 
                               columns=['X', 'Y', 'Z'])
            csv = df_k.to_csv()
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="permeability_tensor.csv",
                mime="text/csv",
                key="download_k"
            )
    
    with col_d2:
        if st.button("Download R as CSV", use_container_width=True):
            df_r = pd.DataFrame(porous_resistivity_matrix,
                               index=['X', 'Y', 'Z'],
                               columns=['X', 'Y', 'Z'])
            csv = df_r.to_csv()
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="resistivity_tensor.csv",
                mime="text/csv",
                key="download_r"
            )

# ============================================================================
# 3D VISUALIZATION
# ============================================================================
with col2:
    st.subheader("🎯 3D Tensor Visualization")
    
    # Create figure based on visualization option
    if visualization_option == 'Both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': '3d'})
        axes = [ax]
    
    cmap = cm.get_cmap(cmap_name)
    
    for idx, tensor_type in enumerate(['Permeability', 'Porous Resistivity'] if visualization_option == 'Both' else [visualization_option]):
        ax = axes[idx] if visualization_option == 'Both' else axes[0]
        
        if tensor_type == 'Permeability':
            tensor_matrix = permeability_matrix
            title = 'Permeability Tensor (κ)'
            zlabel = 'Permeability (m²)'
            # Ellipsoid scaling
            scaling_factors = (np.sqrt(k11/Bx), np.sqrt(k22/By), np.sqrt(k33/Bz))
            a, b, c = scaling_factors
        else:
            tensor_matrix = porous_resistivity_matrix
            title = 'Porous Resistivity Tensor (R)'
            zlabel = 'Resistivity (m⁻²)'
            # Different scaling for resistivity
            a, b, c = np.sqrt(np.diag(tensor_matrix))
            scaling_factors = (a, b, c)
        
        # Generate ellipsoid points
        X, Y, Z = compute_ellipsoid_points(a, b, c, n_points=resolution)
        
        # Compute tensor magnitude at each point
        magnitude = compute_tensor_magnitude(X, Y, Z, tensor_matrix, scaling_factors)
        
        # Normalize for coloring
        norm = plt.Normalize(magnitude.min(), magnitude.max())
        colors = cmap(norm(magnitude.flatten())).reshape(X.shape[0], X.shape[1], 4)
        colors[:, :, 3] = transparency  # Set alpha
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, facecolors=colors,
                              rstride=2, cstride=2, linewidth=0.1,
                              antialiased=True, shade=True)
        
        # Add coordinate axes
        if show_axes:
            axis_length = max(a, b, c) * 1.2
            ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1, linewidth=2)
            ax.text(axis_length, 0, 0, 'X', fontsize=12, color='red', ha='center')
            ax.text(0, axis_length, 0, 'Y', fontsize=12, color='green', ha='center')
            ax.text(0, 0, axis_length, 'Z', fontsize=12, color='blue', ha='center')
        
        # Set labels and title
        ax.set_xlabel('X Direction', fontsize=11, labelpad=10)
        ax.set_ylabel('Y Direction', fontsize=11, labelpad=10)
        ax.set_zlabel('Z Direction', fontsize=11, labelpad=10)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Set aspect ratio
        ax.set_box_aspect([np.ptp(X), np.ptp(Y), np.ptp(Z)])
        
        # Add colorbar
        if show_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
            cbar.set_label(zlabel, fontsize=11, rotation=270, labelpad=15)
        
        # Add grid and remove panes
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================================
# ADDITIONAL INFORMATION
# ============================================================================
with st.expander("📖 Interpretation Guide", expanded=False):
    st.markdown("""
    ### How to Interpret the Visualization:
    
    1. **Ellipsoid Shape:**
       - **Spherical**: Isotropic material (same properties in all directions)
       - **Elongated**: Anisotropic material (different properties along axes)
       - **Flattened**: Material has preferential flow directions
    
    2. **Color Mapping:**
       - Colors represent the magnitude of the tensor property
       - Warmer colors (red/yellow) typically indicate higher values
       - Cooler colors (blue/purple) indicate lower values
    
    3. **PDMS Material Characteristics:**
       - From the paper: Y-direction has highest permeability (1.13×10⁻⁹ m²)
       - X and Z directions have similar but lower permeability
       - This creates a moderately anisotropic ellipsoid
    
    4. **Practical Implications:**
       - Higher permeability = easier fluid flow in that direction
       - Higher resistivity = greater resistance to flow
       - Anisotropy affects filter design and flow optimization
    """)

# Footer
st.markdown("---")
st.caption("""
*Based on: "Microstructural contributions of PDMS sponges to their water treatment efficacy" • 
Tensor visualization using power law: κᵢᵢ = βᵢ × (φᵢ)^mᵢ*
""")
