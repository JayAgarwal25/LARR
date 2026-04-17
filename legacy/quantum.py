import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import gamma, j0
from numpy.polynomial.legendre import leggauss
import mpmath as mp
from multiprocessing import Pool, cpu_count

# Hypergeometric wrapper
def hyp1f1_complex(a, b, z):
    return complex(mp.hyper([a], [b], z))

# Constants
hartree_ev = 27.211386245988
c_au = 137.035999084
eps0 = 8.8541878128e-12
c_si = 299792458.0
ea_to_si = 5.14220652e11

# Problem parameters
I_L_Wcm2 = 6e14
omega_eV = 1.5498
Z = 1
eps_q_eV = 90.0
phi_list_deg = [-90.0, 0.0, 90.0]
nc_list = [2, 3]

omega_au = omega_eV / hartree_ev
eps_q_au = eps_q_eV / hartree_ev
q0 = np.sqrt(2.0 * eps_q_au)
I_0 = -0.5 * Z**2  # Ground state energy

I_SI = I_L_Wcm2 * 1e4
E0_SI = np.sqrt(2.0 * I_SI / (c_si * eps0))
E0_au = E0_SI / ea_to_si

print(f"[info] E0 = {E0_au:.3e} a.u., q0 = {q0:.3f} a.u., ω = {omega_au:.4e} a.u.")

# Pulse generator
def make_pulse(nc, phi_deg, n_t=4000):
    tau = nc * (2*np.pi/omega_au)
    t = np.linspace(0, tau, n_t)
    phi = np.deg2rad(phi_deg)
    env = np.sin(np.pi * t / tau)**2
    E_t = env * E0_au * np.cos(omega_au * t + phi)
    A_t = -c_au * np.cumsum(E_t) * (t[1]-t[0])
    A_t = A_t - np.interp(t, [0, tau], [A_t[0], A_t[-1]])
    kL_t = A_t / c_au
    q_t = q0 + kL_t
    omega_t = 0.5*q_t**2 + 0.5*Z**2
    return t, E_t, kL_t, q_t, omega_t, tau

# Matrix element M_q(t) - same as your CDP code
def compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                        N_r=150, R_max=80.0, N_mu=120):
    mu_nodes, mu_weights = leggauss(N_mu)
    r_grid = np.linspace(1e-6, R_max, N_r)

    norm_u0 = (Z**3/np.pi)**0.5
    u0_r = norm_u0 * np.exp(-Z * r_grid)

    v = Z/q0
    pref = np.exp(np.pi*v/2) * gamma(1.0 - 1j*v) / (2.0*np.pi)**1.5
    q_eff = q0 + kL_tr
    kx = omega_x_au/c_au

    integrand_r = np.zeros_like(r_grid, dtype=complex)
    for ir, r in enumerate(r_grid):
        s = kx*r*np.sqrt(1.0 - mu_nodes**2)
        j0_vals = j0(s)
        z_arg = 1j*q0*r*(mu_nodes-1.0)
        oneF1 = np.array([hyp1f1_complex(-1j*v, 1.0, zz) for zz in z_arg], dtype=complex)
        u_q_mu = pref*oneF1
        exp_factor = np.exp(1j*q_eff*r*mu_nodes)
        integrand_mu = mu_nodes*j0_vals*exp_factor*u_q_mu
        mu_integral = np.dot(integrand_mu, mu_weights)
        integrand_r[ir] = 2*np.pi*r**3*u0_r[ir]*mu_integral
    return np.trapz(integrand_r, r_grid)

# DP calculation (Equation 6) - KEY DIFFERENCE FROM CDP
def DP_single(omega_x_eV, t_grid, E_t, kL_t, q_t,
              Nr, Rmax, Nmu):
    """
    Calculate DP(τ) according to Eq. 6:
    DP(τ) = (ω_x³/2π³c³) |∫₀^τ dt' M(t') exp[i S(t')]|²
    
    where S(t') = ∫₀^t' dt'' [q²(t'')/2 + I₀ - ω_x]
    """
    omega_x_au = omega_x_eV / hartree_ev
    dt = t_grid[1] - t_grid[0]
    
    # Calculate the phase integral S(t) = ∫[q²(t)/2 + I₀ - ω_x] dt
    # Using cumulative integration
    integrand = 0.5 * q_t**2 + I_0 - omega_x_au
    S_t = np.cumsum(integrand) * dt
    
    # Compute M(t') for each time point
    M_vals = np.zeros(len(t_grid), dtype=complex)
    for i, t_val in enumerate(t_grid):
        kL_val = kL_t[i]
        M_vals[i] = compute_M_q_numeric(t_val, omega_x_au, kL_val, q0, Z,
                                        N_r=Nr, R_max=Rmax, N_mu=Nmu)
    
    # Compute the coherent sum: ∫ M(t') exp[i S(t')] dt'
    integrand_t = M_vals * np.exp(1j * S_t)
    integral = np.trapz(integrand_t, t_grid)
    
    # Final probability with prefactor
    prefactor = (omega_x_au**3) / (2 * np.pi**3 * c_au**3)
    DP = prefactor * np.abs(integral)**2
    
    return DP

# Parallel DP spectrum
def compute_DP_parallel(nc, phi_deg, omega_x_eV_grid,
                        Nr=150, Rmax=80.0, Nmu=120, t_nt=2500):
    t_grid, E_t, kL_t, q_t, omega_t, tau = make_pulse(nc, phi_deg, n_t=t_nt)
    
    args = [(om, t_grid, E_t, kL_t, q_t, Nr, Rmax, Nmu) 
            for om in omega_x_eV_grid]
    
    with Pool(processes=cpu_count()) as pool:
        vals = pool.starmap(DP_single, args)
    
    return np.array(vals)

# Main
if __name__ == "__main__":
    # Pick grid of ω_x
    t_test, _, _, _, omega_test, _ = make_pulse(2, 0.0, n_t=2000)
    wmin, wmax = np.min(omega_test)*hartree_ev, np.max(omega_test)*hartree_ev
    omega_x_eV_grid = np.linspace(max(1.0, wmin-5.0), wmax+5.0, 150)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    
    for row, nc in enumerate(nc_list):
        for col, phi in enumerate(phi_list_deg):
            print(f"[run] nc={nc}, phi={phi}")
            
            # Compute DP spectrum (Equation 6 - quantum mechanical)
            DP_vals = compute_DP_parallel(nc, phi, omega_x_eV_grid,
                                         Nr=120, Rmax=80.0, Nmu=100, t_nt=2000)
            
            # Save results to CSV
            outdir = "results"
            os.makedirs(outdir, exist_ok=True)
            fname = os.path.join(outdir, f"DP_nc{nc}_phi{int(phi)}.csv")
            out_data = np.column_stack((omega_x_eV_grid, DP_vals))
            np.savetxt(fname, out_data, delimiter=',', 
                      header='omega_x_eV,DP', comments='')
            print(f"[saved] {fname}")
            
            ax = axes[row, col]
            ax.plot(omega_x_eV_grid, np.maximum(DP_vals, 1e-50))
            ax.set_yscale("log")
            ax.set_xlim(0, 500)
            ax.set_ylim(1e-18, 1e-6)
            ax.set_title(f"nc={nc}, φ={phi}°")
            ax.set_xlabel("ω_x (eV)")
            if col == 0:
                ax.set_ylabel("DP (a.u.)")

    plt.suptitle("Laser-Assisted Radiative Recombination: DP spectra (Quantum)", 
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figpath = os.path.join(figdir, f'DP_spectra_{timestamp}.png')
    fig.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"[saved] figure -> {figpath}")
