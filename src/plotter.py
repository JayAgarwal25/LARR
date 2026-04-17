import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def plot_results(results_dir="results", output_type="DP", save_fig=True):
    """
    Plot CSV results from DP or CDP calculations
    
    Parameters:
    -----------
    results_dir : str
        Directory containing CSV files
    output_type : str
        "DP" or "CDP" - type of calculation to plot
    save_fig : bool
        Whether to save the figure
    """
    
    # Find all CSV files matching the pattern
    pattern = os.path.join(results_dir, f"{output_type}_nc*_phi*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    # Parse filenames to extract nc and phi values
    data_dict = {}
    for fname in csv_files:
        basename = os.path.basename(fname)
        # Extract nc and phi from filename like "DP_nc2_phi-90.csv"
        try:
            parts = basename.replace('.csv', '').split('_')
            nc = int(parts[1].replace('nc', ''))
            phi = int(parts[2].replace('phi', ''))
            
            # Load data
            data = np.loadtxt(fname, delimiter=',', skiprows=1)
            omega_x = data[:, 0]
            values = data[:, 1]
            
            if nc not in data_dict:
                data_dict[nc] = {}
            data_dict[nc][phi] = (omega_x, values)
        except Exception as e:
            print(f"Warning: Could not parse {basename}: {e}")
            continue
    
    if not data_dict:
        print("No valid data found!")
        return
    
    # Determine layout
    nc_list = sorted(data_dict.keys())
    all_phis = set()
    for nc in nc_list:
        all_phis.update(data_dict[nc].keys())
    phi_list = sorted(all_phis)
    
    n_rows = len(nc_list)
    n_cols = len(phi_list)
    
    print(f"\nCreating plot grid: {n_rows} rows (nc) × {n_cols} cols (phi)")
    print(f"nc values: {nc_list}")
    print(f"phi values: {phi_list}")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), 
                             squeeze=False, sharex=True, sharey=True)
    
    # Plot each configuration
    for row, nc in enumerate(nc_list):
        for col, phi in enumerate(phi_list):
            ax = axes[row, col]
            
            if phi in data_dict[nc]:
                omega_x, values = data_dict[nc][phi]
                
                # Plot with markers and lines
                ax.plot(omega_x, np.maximum(values, 1e-50), 'o-', 
                       markersize=4, linewidth=1.5, label=f'nc={nc}, φ={phi}°')
                
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3, which='both')
                ax.set_title(f'nc={nc}, φ={phi}°', fontsize=11, fontweight='bold')
                
                # Labels
                if row == n_rows - 1:
                    ax.set_xlabel('ω_x (eV)', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'{output_type} (a.u.)', fontsize=10)
                
                # Set reasonable limits
                ax.set_xlim(0, max(omega_x) * 1.05)
                ax.set_ylim(1e-20, max(values) * 10)
            else:
                # Empty subplot if data missing
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
    
    # Overall title
    title = f'Laser-Assisted Radiative Recombination: {output_type} Spectra'
    if 'TEST' in csv_files[0]:
        title += ' (TEST RUN)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    if save_fig:
        figdir = os.path.join(results_dir, 'figures')
        os.makedirs(figdir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figpath = os.path.join(figdir, f'{output_type}_spectra_{timestamp}.png')
        fig.savefig(figpath, dpi=200, bbox_inches='tight')
        print(f"\n[saved] {figpath}")
    
    plt.show()
    
    return fig, axes


def plot_single_csv(csv_file, save_fig=True, show_plot=True):
    """
    Plot a single CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    save_fig : bool
        Whether to save the figure
    show_plot : bool
        Whether to display the plot
    """
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        return None
    
    print(f"Loading: {csv_file}")
    
    # Load data
    try:
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        omega_x = data[:, 0]
        values = data[:, 1]
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Parse filename for title
    basename = os.path.basename(csv_file)
    title = basename.replace('.csv', '').replace('_', ' ')
    
    # Try to extract nc and phi
    try:
        if 'nc' in basename and 'phi' in basename:
            parts = basename.replace('.csv', '').split('_')
            nc = parts[1].replace('nc', '')
            phi = parts[2].replace('phi', '')
            title = f"nc={nc}, φ={phi}°"
    except:
        pass
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    ax.plot(omega_x, np.maximum(values, 1e-50), 'o-', 
           markersize=5, linewidth=2, color='steelblue')
    
    ax.set_yscale('log')
    ax.set_xlabel('ω_x (eV)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add statistics text box
    stats_text = f"Points: {len(omega_x)}\n"
    stats_text += f"ω_x range: [{omega_x.min():.1f}, {omega_x.max():.1f}] eV\n"
    stats_text += f"Max value: {values.max():.2e} a.u.\n"
    stats_text += f"Min value: {values[values>0].min():.2e} a.u."
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if save_fig:
        output_dir = os.path.dirname(csv_file)
        if not output_dir:
            output_dir = '.'
        figdir = os.path.join(output_dir, 'figures')
        os.makedirs(figdir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figname = basename.replace('.csv', f'_{timestamp}.png')
        figpath = os.path.join(figdir, figname)
        
        fig.savefig(figpath, dpi=200, bbox_inches='tight')
        print(f"[saved] {figpath}")
    
    if show_plot:
        plt.show()
    
    return fig, ax


def compare_DP_CDP(results_dir="results", nc=2, phi=0, save_fig=True):
    """
    Compare DP (quantum) vs CDP (semiclassical) for same configuration
    
    Parameters:
    -----------
    results_dir : str
        Directory containing CSV files
    nc : int
        Cycle number
    phi : int
        Phase in degrees
    save_fig : bool
        Whether to save the figure
    """
    
    # Load DP data
    dp_file = os.path.join(results_dir, f"DP_nc{nc}_phi{phi}.csv")
    cdp_file = os.path.join(results_dir, f"CDP_nc{nc}_phi{phi}.csv")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot DP if available
    if os.path.exists(dp_file):
        data = np.loadtxt(dp_file, delimiter=',', skiprows=1)
        omega_x = data[:, 0]
        dp_vals = data[:, 1]
        
        ax1.plot(omega_x, np.maximum(dp_vals, 1e-50), 'b-', linewidth=2, label='DP (Quantum)')
        ax1.set_yscale('log')
        ax1.set_ylabel('DP (a.u.)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Quantum (Eq. 6): nc={nc}, φ={phi}°', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, f'DP file not found:\n{os.path.basename(dp_file)}', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=11)
    
    # Plot CDP if available
    if os.path.exists(cdp_file):
        data = np.loadtxt(cdp_file, delimiter=',', skiprows=1)
        omega_x = data[:, 0]
        cdp_vals = data[:, 1]
        
        ax2.plot(omega_x, np.maximum(cdp_vals, 1e-50), 'r-', linewidth=2, label='CDP (Semiclassical)')
        ax2.set_yscale('log')
        ax2.set_xlabel('ω_x (eV)', fontsize=12)
        ax2.set_ylabel('CDP (a.u.)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Semiclassical (Eq. 13): nc={nc}, φ={phi}°', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=10)
    else:
        ax2.text(0.5, 0.5, f'CDP file not found:\n{os.path.basename(cdp_file)}', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=11)
    
    fig.suptitle('Comparison: Quantum vs Semiclassical', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    if save_fig:
        figdir = os.path.join(results_dir, 'figures')
        os.makedirs(figdir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figpath = os.path.join(figdir, f'comparison_nc{nc}_phi{phi}_{timestamp}.png')
        fig.savefig(figpath, dpi=200, bbox_inches='tight')
        print(f"\n[saved] {figpath}")
    
    plt.show()
    
    return fig


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot DP/CDP results from CSV files')
    parser.add_argument('--file', type=str, help='Plot single CSV file')
    parser.add_argument('--dir', type=str, default='results', help='Results directory')
    parser.add_argument('--type', type=str, default='DP', choices=['DP', 'CDP'], 
                       help='Type of calculation to plot')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare DP vs CDP for specific configuration')
    parser.add_argument('--nc', type=int, default=2, help='Cycle number for comparison')
    parser.add_argument('--phi', type=int, default=0, help='Phase (degrees) for comparison')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save figures')
    
    args = parser.parse_args()
    
    if args.file:
        print(f"Plotting single file: {args.file}")
        plot_single_csv(args.file, save_fig=not args.no_save)
    elif args.compare:
        print(f"Comparing DP vs CDP for nc={args.nc}, phi={args.phi}")
        compare_DP_CDP(args.dir, args.nc, args.phi, save_fig=not args.no_save)
    else:
        print(f"Plotting {args.type} results from {args.dir}")
        plot_results(args.dir, args.type, save_fig=not args.no_save)
