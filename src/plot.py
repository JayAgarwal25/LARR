import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import os

# ================= CONFIGURATION =================
DATA_DIR = "CEP_Data"
NC_LIST = [2, 3]

# Visualization Settings
OMEGA_MAX_PLOT = 500.0     # Max energy to show on X-axis (eV)
OMEGA_RES = 1000          # Resolution for energy axis
VMIN, VMAX = 1e-11, 1e-8 # Adjusted Log scale limits to match image range better

# --- MATPLOTLIB STYLE CONFIGURATION ---
# IMPORTANT: Requires a LaTeX installation on your system
try:
    mpl.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
except Exception as e:
    print("Warning: Could not configure LaTeX rendering. Ensure TeX Live/MikTeX is installed.")
    print(f"Error details: {e}")
    # Fallback to standard fonts if LaTeX fails
    mpl.rcParams.update({"font.family": "serif"})
# =================================================

def plot_styled_heatmap(nc):
    filename = os.path.join(DATA_DIR, f"Heatmap_nc{nc}.csv")
    if not os.path.exists(filename):
        print(f"File missing: {filename}")
        return

    print(f"Processing nc={nc} for styled plot...")
    df = pd.read_csv(filename)

    # 1. Prepare Data Axes
    unique_phis = np.sort(df['phi_deg'].unique())
    
    w_min_global = df['omega_eV'].min()
    w_max_global = max(df['omega_eV'].max(), OMEGA_MAX_PLOT) 
    master_omega = np.linspace(w_min_global, w_max_global, OMEGA_RES)

    # Matrix shape: (Number of Phis, Number of Omega points)
    data_matrix = np.zeros((len(unique_phis), len(master_omega)))

    # 2. Fill Matrix (Interpolate onto fixed omega grid)
    grouped = df.groupby('phi_deg')
    for i, phi in enumerate(unique_phis):
        group = grouped.get_group(phi)
        w_current = group['omega_eV'].values
        cdp_current = group['CDP'].values
        
        sort_idx = np.argsort(w_current)
        w_current = w_current[sort_idx]
        cdp_current = cdp_current[sort_idx]

        # Use a tiny epsilon for fill_value so log scale doesn't break on zeros
        data_matrix[i, :] = np.interp(master_omega, w_current, cdp_current, left=1e-30, right=1e-30)

    # 3. Plotting setup
    plt.figure(figsize=(10, 6))
    
    # --- AXIS SWAP ---
    # We want Omega on X, Phi on Y to match the reference image style.
    # X = Omega, Y = Phi. 
    # data_matrix is shape (N_phi, N_omega).
    X_grid, Y_grid = np.meshgrid(master_omega, unique_phis)
    
    # Plotting data_matrix directly aligns correctly with this meshgrid
    pcm = plt.pcolormesh(
        X_grid, Y_grid, data_matrix, 
        norm=colors.LogNorm(vmin=VMIN, vmax=VMAX), # 'viridis' matches the image's blue-green-yellow better than jet
        shading='auto' 
    )

    # 4. Styling Labels and Annotations (LaTeX)
    cbar = plt.colorbar(pcm)
    # You can set a title for the colorbar if desired, similar to the main title in the image
    # cbar.set_label(r"$d^3 E_{\mathbf{K}}(\mathbf{p})/d\omega_{\mathbf{K}} d^2\Omega_{\mathbf{K}}$ (a. u.)", rotation=270, labelpad=25)
    
    # Main Title using LaTeX
    plt.title(r"\textbf{CDP Spatial Distribution} ($n_c=" + str(nc) + r"$)", pad=15)
    
    # Axis Labels using LaTeX
    # Note: I'm using your quantities (eV and degrees) but styling them like the image.
    plt.xlabel(r"Electron Energy $\omega_x$ (eV)")
    plt.ylabel(r"CEP $\phi$ (degrees)")
    
    # Set limits
    plt.xlim(0, OMEGA_MAX_PLOT)
    plt.ylim(unique_phis.min(), unique_phis.max())

    # 5. Add the yellow text annotation in the bottom right corner
    # Calculate position based on axis limits (e.g., 80% of X, 15% of Y from bottom)
    x_pos = OMEGA_MAX_PLOT * 0.8
    y_pos = unique_phis.min() + (unique_phis.max() - unique_phis.min()) * 0.15
    
    plt.text(x_pos, y_pos, r"$\varphi_p = \pi$", color='yellow', fontsize=22, fontweight='bold')

    # Save and show
    out_file = f"Heatmap_Styled_nc{nc}.png"
    # bbox_inches='tight' helps ensure labels aren't cut off
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved styled plot: {out_file}")
    # plt.show() # Commented out for batch running

if __name__ == "__main__":
    # Check for data directory first
    if not os.path.exists(DATA_DIR):
        print(f"[Error] Data directory '{DATA_DIR}' not found.")
        print("Please run the computation script first.")
        exit()
        
    for nc in NC_LIST:
        plot_styled_heatmap(nc)
