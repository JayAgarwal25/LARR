import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import gamma, j0
from numpy.polynomial.legendre import leggauss
import mpmath as mp
from multiprocessing import Pool, cpu_count
from numpy import trapezoid
import time

# Hypergeometric wrapper with caching
_hyp_cache = {}
def hyp1f1_complex(a, b, z):
    key = (complex(a), complex(b), complex(z))
    if key not in _hyp_cache:
        _hyp_cache[key] = complex(mp.hyper([a], [b], z))
    return _hyp_cache[key]

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
I_0 = -0.5 * Z**2

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

# OPTIMIZED Matrix element - vectorized where possible
def compute_M_q_numeric_fast(tr, omega_x_au, kL_tr, q0, Z,
                             N_r=80, R_max=60.0, N_mu=60):
    """Optimized version with timing diagnostics"""
    t0 = time.time()
    
    mu_nodes, mu_weights = leggauss(N_mu)
    r_grid = np.linspace(1e-6, R_max, N_r)

    norm_u0 = (Z**3/np.pi)**0.5
    u0_r = norm_u0 * np.exp(-Z * r_grid)

    v = Z/q0
    pref = np.exp(np.pi*v/2) * gamma(1.0 - 1j*v) / (2.0*np.pi)**1.5
    q_eff = q0 + kL_tr
    kx = omega_x_au/c_au

    integrand_r = np.zeros_like(r_grid, dtype=complex)
    
    # Pre-compute mu-independent quantities
    mu_sq_root = np.sqrt(1.0 - mu_nodes**2)
    
    for ir, r in enumerate(r_grid):
        # Vectorized over mu
        s = kx * r * mu_sq_root
        j0_vals = j0(s)
        
        # This is still the bottleneck - hypergeometric calls
        z_args = 1j*q0*r*(mu_nodes-1.0)
        oneF1 = np.array([hyp1f1_complex(-1j*v, 1.0, z) for z in z_args], dtype=complex)
        
        u_q_mu = pref * oneF1
        exp_factor = np.exp(1j*q_eff*r*mu_nodes)
        integrand_mu = mu_nodes * j0_vals * exp_factor * u_q_mu
        mu_integral = np.dot(integrand_mu, mu_weights)
        integrand_r[ir] = 2*np.pi*r**3*u0_r[ir]*mu_integral
        
    result = trapezoid(integrand_r, r_grid)
    
    elapsed = time.time() - t0
    return result, elapsed

# DP calculation with detailed timing
def DP_single_timed(omega_x_eV, t_grid, E_t, kL_t, q_t,
                    Nr, Rmax, Nmu):
    """DP calculation with timing information"""
    t_start = time.time()
    
    omega_x_au = omega_x_eV / hartree_ev
    dt = t_grid[1] - t_grid[0]
    
    # Phase integral
    integrand = 0.5 * q_t**2 + I_0 - omega_x_au
    S_t = np.cumsum(integrand) * dt
    
    # Compute M(t') for each time point
    M_vals = np.zeros(len(t_grid), dtype=complex)
    n_t = len(t_grid)
    
    print(f"\n  ω_x={omega_x_eV:.1f} eV: Starting {n_t} time points...", flush=True)
    total_M_time = 0.0
    
    for i, t_val in enumerate(t_grid):
        if i % 20 == 0 and i > 0:
            avg_time = total_M_time / i
            est_remain = avg_time * (n_t - i)
            print(f"    [{i}/{n_t}] Avg: {avg_time:.2f}s/point, Est. remain: {est_remain/60:.1f} min", flush=True)
            
        kL_val = kL_t[i]
        M_vals[i], elapsed = compute_M_q_numeric_fast(t_val, omega_x_au, kL_val, q0, Z,
                                                      N_r=Nr, R_max=Rmax, N_mu=Nmu)
        total_M_time += elapsed
    
    # Coherent sum
    integrand_t = M_vals * np.exp(1j * S_t)
    integral = trapezoid(integrand_t, t_grid)
    
    # Final probability
    prefactor = (omega_x_au**3) / (2 * np.pi**3 * c_au**3)
    DP = prefactor * np.abs(integral)**2
    
    total_time = time.time() - t_start
    print(f"  ω_x={omega_x_eV:.1f} eV: DONE in {total_time/60:.1f} min (avg {total_M_time/n_t:.2f}s per M)", flush=True)
    
    return DP

# Wrapper for parallel processing
def DP_single_wrapper(args):
    return DP_single_timed(*args)

# Serial version for better progress tracking
def compute_DP_serial(nc, phi_deg, omega_x_eV_grid,
                      Nr=60, Rmax=50.0, Nmu=40, t_nt=100):
    """Serial computation with progress tracking"""
    print(f"\n{'='*60}")
    print(f"Starting DP calculation: nc={nc}, phi={phi_deg}°")
    print(f"Parameters: Nr={Nr}, Rmax={Rmax}, Nmu={Nmu}, t_nt={t_nt}")
    print(f"Total ω_x points: {len(omega_x_eV_grid)}")
    print(f"{'='*60}\n")
    
    t_grid, E_t, kL_t, q_t, omega_t, tau = make_pulse(nc, phi_deg, n_t=t_nt)
    
    DP_vals = []
    for idx, omega_x in enumerate(omega_x_eV_grid):
        print(f"\n[{idx+1}/{len(omega_x_eV_grid)}] Processing ω_x = {omega_x:.1f} eV", flush=True)
        val = DP_single_timed(omega_x, t_grid, E_t, kL_t, q_t, Nr, Rmax, Nmu)
        DP_vals.append(val)
    
    return np.array(DP_vals)

# Main
if __name__ == "__main__":
    # Pick grid of ω_x - IMPORTANT: Must cover full classical range!
    print("\n[info] Computing classical energy range...")
    t_test, _, _, q_test, omega_test, _ = make_pulse(2, 0.0, n_t=2000)
    
    # Classical photon energy: ω_x(t) = q²(t)/2 + I_0
    # where q(t) = q0 + kL(t)
    wmin_classical = np.min(omega_test) * hartree_ev
    wmax_classical = np.max(omega_test) * hartree_ev
    
    print(f"[info] Classical ω_x range: [{wmin_classical:.1f}, {wmax_classical:.1f}] eV")
    print(f"[info] This is where emission is classically allowed")
    
    # Extend slightly beyond classical range
    omega_x_eV_grid = np.linspace(max(1.0, wmin_classical - 10.0), 
                                   wmax_classical + 10.0, 100)
    
    print(f"[info] Using ω_x grid: [{omega_x_eV_grid[0]:.1f}, {omega_x_eV_grid[-1]:.1f}] eV")
    print(f"[info] Grid points: {len(omega_x_eV_grid)}")
    
    print(f"\n{'#'*60}")
    print("WARNING: DP calculation is VERY expensive!")
    print(f"Testing with {len(omega_x_eV_grid)} ω_x points")
    print("Estimated time per ω_x: 2-10 minutes")
    print(f"Total estimated time: {len(omega_x_eV_grid)*5/60:.1f} hours per configuration")
    print(f"{'#'*60}\n")
    
    # Test with just ONE configuration first
    nc_list_test = [2]
    phi_list_test = [0.0]
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    for nc in nc_list_test:
        for phi in phi_list_test:
            print(f"\n{'='*60}")
            print(f"RUNNING: nc={nc}, phi={phi}°")
            print(f"{'='*60}")
            
            # Use VERY reduced parameters for speed
            DP_vals = compute_DP_serial(nc, phi, omega_x_eV_grid,
                                       Nr=40, Rmax=40.0, Nmu=30, t_nt=50)
            
            # Save results
            outdir = "results"
            os.makedirs(outdir, exist_ok=True)
            fname = os.path.join(outdir, f"DP_nc{nc}_phi{int(phi)}_TEST.csv")
            out_data = np.column_stack((omega_x_eV_grid, DP_vals))
            np.savetxt(fname, out_data, delimiter=',', 
                      header='omega_x_eV,DP', comments='')
            print(f"\n[saved] {fname}")
            
            # Plot
            if isinstance(axes, np.ndarray):
                ax = axes[0]
            else:
                ax = axes
            ax.plot(omega_x_eV_grid, np.maximum(DP_vals, 1e-50), 'o-')
            ax.set_yscale("log")
            ax.set_xlabel("ω_x (eV)")
            ax.set_ylabel("DP (a.u.)")
            ax.set_title(f"DP spectrum: nc={nc}, φ={phi}° (TEST)")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save figure
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figpath = os.path.join(figdir, f'DP_spectrum_TEST_{timestamp}.png')
    fig.savefig(figpath, dpi=150, bbox_inches='tight')
    print(f"\n[saved] figure -> {figpath}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("If this worked, you can increase omega_x_eV_grid and t_nt")
    print("="*60)
