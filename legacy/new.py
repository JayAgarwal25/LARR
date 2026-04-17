import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ==========================================
# 1. Physics Constants & Parameters (Atomic Units)
# ==========================================
au_energy_to_eV = 27.211386
au_time_to_as = 24.188843265
au_intensity_unit = 3.50944e16 # W/cm^2

# Laser Parameters
intensity_Wcm2 = 6e14
E0 = np.sqrt(intensity_Wcm2 / au_intensity_unit) # Field amplitude in a.u.
omega_eV = 1.5498 
omega = omega_eV / au_energy_to_eV
period = 2 * np.pi / omega

# Atom & Electron Parameters
Z = 1.0 # Hydrogen
Ip = 0.5 * Z**2 # Ionization potential (0.5 a.u. = 13.6 eV)
E_electron_eV = 90.0
q_initial = np.sqrt(2 * E_electron_eV / au_energy_to_eV)

# Simulation Grid
dt = period / 1000.0 # Fine time resolution for accurate integration

# ==========================================
# 2. Core Functions
# ==========================================

def get_pulse_field(t, nc, phi_deg):
    """
    Calculates Electric Field E(t) for a sin^2 envelope pulse.
    """
    tau = nc * period
    phi = np.radians(phi_deg)
    
    # Field is zero outside [0, tau]
    mask = (t >= 0) & (t <= tau)
    
    envelope = np.sin(np.pi * t / tau)**2
    carrier = np.cos(omega * t + phi)
    
    E_t = np.zeros_like(t)
    E_t[mask] = E0 * envelope[mask] * carrier[mask]
    return E_t

def hydrogenic_matrix_element_squared(omega_x, q_inst):
    """
    Approximation of the squared matrix element |M|^2 for Hydrogen 1s.
    Uses the Stobbe formula scaling for photoionization/recombination.
    Scaling relation: Rate ~ omega^3 * |M|^2
    Stobbe formula gives sigma_PI ~ (1/omega^3) * Coulomb_Factor
    """
    # Prevent division by zero
    q_inst = np.maximum(q_inst, 1e-5)
    omega_x = np.maximum(omega_x, 1e-5)

    # Sommerfeld parameter nu = Z / q
    nu = Z / q_inst
    
    # Coulomb Factor F(nu)
    # To avoid numerical overflow in exp(2*pi*nu) for slow electrons, use stable forms
    # term = exp(-4 * nu * arccot(nu)) / (1 - exp(-2*pi*nu))
    # arccot(x) = arctan(1/x)
    
    arctan_term = np.arctan(1.0 / nu)
    numerator = np.exp(-4 * nu * arctan_term)
    denominator = 1.0 - np.exp(-2 * np.pi * nu)
    
    coulomb_factor = numerator / denominator
    
    # The semiclassical weight scales roughly as 1/omega (from dipole squared * density of states)
    # We use the proportionality: |M|^2 ~ coulomb_factor / omega^4
    # Because Equation 6 has omega^3 * |M|^2, the net prefactor is coulomb_factor / omega
    
    return coulomb_factor / (omega_x**4)

def simulate_cdp(nc, phi_deg):
    """
    Simulates the CDP(tau) using the Semiclassical Model (Eq. 13).
    Returns (bins, probability_density)
    """
    tau = nc * period
    t = np.arange(0, tau, dt)
    
    # 1. Calculate Fields
    E_t = get_pulse_field(t, nc, phi_deg)
    
    # 2. Vector Potential (k_L)
    # k_L(t) = - integral(E) dt (in a.u.)
    # Using cumulative trapezoidal integration
    A_t = -np.cumsum(E_t) * dt
    kL_t = A_t 
    
    # 3. Trajectory & Instantaneous Energy
    # q(t) = q_initial + k_L(t) (1D geometry along z)
    q_t = q_initial + kL_t
    
    # omega_x(t) = q(t)^2 / 2 + Ip (Eq. 10)
    omega_x_t = 0.5 * q_t**2 + Ip
    
    # 4. Semiclassical Probability Weighting
    # Eq 13 sums |M|^2 / |d_omega/dt|. 
    # In a histogram approach, we simply accumulate |M|^2 * omega^3 * dt
    # The 1/|Jacobian| is naturally handled by the binning process (more time spent in a bin = higher prob).
    
    M_squared = hydrogenic_matrix_element_squared(omega_x_t, np.abs(q_t))
    
    # Eq 13 Pre-factor: omega^3 / (2 pi^3) * |M|^2
    # We drop constants for qualitative matching, but keep omega scaling
    weights = (omega_x_t**3) * M_squared * dt
    
    return omega_x_t * au_energy_to_eV, weights

# ==========================================
# 3. Run Simulation & Plot (Matching Fig 1)
# ==========================================

phases = [-90, 90, 0]
cycles = [2, 3]

fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharey='row')
fig.suptitle(f'Semiclassical CDP Simulation (Recreating Figure 1)\n$I_L=6\\times10^{{14}}$ W/cm$^2$, $E_e=90$ eV', fontsize=14)

for col_idx, nc in enumerate(cycles):
    for row_idx, phi in enumerate(phases):
        ax = axes[row_idx, col_idx]
        
        # Run Simulation
        energies, weights = simulate_cdp(nc, phi)
        
        # Create Histogram (approximating the continuous distribution)
        # High bin count to capture the sharp caustic peaks
        bins = np.linspace(0, 500, 501) 
        hist, bin_edges = np.histogram(energies, bins=bins, weights=weights)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot
        # Use log scale for Y as in the paper
        ax.plot(bin_centers, hist + 1e-20, color='blue', lw=1.5, label=f'Semiclassical ($n_c={nc}, \phi={phi}^\circ$)')
        
        ax.set_yscale('log')
        ax.set_ylim(1e-14, 1e-5) # Matching paper scale roughly
        ax.set_xlim(0, 450)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Annotations
        ax.text(0.95, 0.9, f'$\phi={phi}^\circ$', transform=ax.transAxes, ha='right', fontsize=12, weight='bold')
        if row_idx == 0:
            ax.set_title(f'$n_c = {nc}$', fontsize=12, weight='bold')
        if row_idx == 2:
            ax.set_xlabel('$\omega_X$ (eV)', fontsize=11)
        if col_idx == 0:
            ax.set_ylabel('DDP($\\tau$) (a.u.)', fontsize=11)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
