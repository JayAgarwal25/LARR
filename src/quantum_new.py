import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import gamma, j0
from numpy.polynomial.legendre import leggauss
import mpmath as mp
from multiprocessing import Pool, cpu_count
from flint import ctx, acb
ctx.prec = 53  # Standard Double Precision
# ============================================================
# CONFIGURATION
# ============================================================
mp.dps = 30
FAST_MODE = False           # quick test vs full run
RUN_SINGLE = True          # plot one (nc,phi)
REUSE_EXISTING = False      # skip recomputation if CSV exists

single_nc = 2
single_phi = 90.0

if FAST_MODE:
    Nr, Nmu, t_nt = 80,80, 2000
    omega_points = 192
    R_MAX_SIM = 80.0
else:
    Nr, Nmu, t_nt = 150, 120, 3000
    omega_points = 640
    R_MAX_SIM = 150.0

# ============================================================
# CONSTANTS AND PARAMETERS
# ============================================================
hartree_ev = 27.211386245988
c_au = 137.035999084
eps0 = 8.8541878128e-12
c_si = 299792458.0
ea_to_si = 5.14220652e11

I_L_Wcm2 = 6e14
omega_eV = 1.5498
Z = 1
eps_q_eV = 90.0
phi_list_deg = [-90.0, 0.0, 90.0]
nc_list = [2, 3]

omega_au = omega_eV / hartree_ev
eps_q_au = eps_q_eV / hartree_ev
q0 = np.sqrt(2.0 * eps_q_au)

I_SI = I_L_Wcm2 * 1e4
E0_SI = np.sqrt(2.0 * I_SI / (c_si * eps0))
E0_au = E0_SI / ea_to_si

print(f"[info] E0 = {E0_au:.3e} a.u., q0 = {q0:.3f} a.u., ω = {omega_au:.4e} a.u.")

# ============================================================
# BASIC BUILDING BLOCKS
# ============================================================

def hyp1f1_complex(a, b, z):
    #return complex(mp.hyper([a], [b], z))
    # Convert inputs to Flint High-Precision types
    #    (acb handles standard python complex/float/int automatically)
    val_a = acb(a)
    val_b = acb(b)
    val_z = acb(z)

    #    Use the syntax z.hypgeom_1f1(a, b)
    result = val_z.hypgeom_1f1(val_a, val_b)
    
    return complex(result)

def make_pulse(nc, phi_deg, n_t=4000):
    tau = nc * (2*np.pi/omega_au)
    t = np.linspace(0, tau, n_t,endpoint=False)
    phi = np.deg2rad(phi_deg)
    env = np.sin(np.pi * t / tau)**2
    E_t = env * E0_au * np.cos(omega_au * t + phi)
    dt = t[1] - t[0]
    A_t = -c_au * np.cumsum(E_t) * dt
    #A_t -= np.interp(t, [0, tau], [A_t[0], A_t[-1]])

    kL_t = A_t / c_au
    q_t = q0 + kL_t
    omega_t = 0.5*q_t**2 + 0.5*Z**2
    return t, E_t, kL_t, q_t, omega_t, tau

# ============================================================
# MATRIX ELEMENT WITH CACHE
# ============================================================
_M_cache = {}

def compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                        N_r=150, R_max=150.0, N_mu=120):
    
    key = (round(kL_tr,8), round(omega_x_au,12), round(q0,12), Z, int(N_r), int(N_mu), float(R_max))
    if key in _M_cache:
        return _M_cache[key]

    
    #key = (round(kL_tr,3), round(omega_x_au,3))


    mu_nodes, mu_weights = leggauss(N_mu)
    r_grid = np.linspace(1e-6, R_max, N_r)

    norm_u0 = (Z**3/np.pi)**0.5
    u0_r = norm_u0 * np.exp(-Z * r_grid)
    q_eff = q0 + kL_tr
    v = Z/q_eff
    pref = np.exp(np.pi*v/2) * gamma(1.0 - 1j*v) / (2.0*np.pi)**1.5
   
    kx = omega_x_au/c_au

    integrand_r = np.zeros_like(r_grid, dtype=complex)
    for ir, r in enumerate(r_grid):
        s = kx*r*np.sqrt(1.0 - mu_nodes**2)
        j0_vals = j0(s)
        z_arg = 1j*q_eff*r*(mu_nodes-1.0)
        oneF1 = np.array([hyp1f1_complex(-1j*v, 1.0, zz) for zz in z_arg], dtype=complex)
        u_q_mu = pref*oneF1
        exp_factor = np.exp(1j*q_eff*r*mu_nodes)
        integrand_mu = mu_nodes*j0_vals*exp_factor*u_q_mu
        mu_integral = np.dot(integrand_mu, mu_weights)
        integrand_r[ir] = 2*np.pi*r**3*u0_r[ir]*mu_integral

    val = np.trapezoid(integrand_r, r_grid)
    _M_cache[key] = val
    return val

# ============================================================
# CDP / DP CALCULATIONS
# ============================================================

def find_time_roots(omega_t, t_grid, omega_target_au):
    f = omega_t - omega_target_au
    idx = np.where(np.sign(f[:-1])*np.sign(f[1:]) <= 0)[0]
    roots = []
    for i in idx:
        f1, f2 = f[i], f[i+1]
        t1, t2 = t_grid[i], t_grid[i+1]
        tr = t1 - f1*(t2-t1)/(f2-f1) if not np.isclose(f1,f2) else 0.5*(t1+t2)
        roots.append(tr)
    return roots

def CDP_single(omega_x_eV, t_grid, E_t, kL_t, omega_t, Nr, Rmax, Nmu):
    omega_x_au = omega_x_eV/hartree_ev
    roots = find_time_roots(omega_t, t_grid, omega_x_au)
    if not roots: return 0.0
    total = 0.0
    for tr in roots:
        kL_tr = np.interp(tr, t_grid, kL_t)
        M_tr = compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                                   N_r=Nr, R_max=Rmax, N_mu=Nmu)
        E_val = np.interp(tr, t_grid, E_t)
        qeff = q0 + kL_tr
        wdot = -qeff*E_val
        if np.isclose(wdot, 0.0): 
            wdot = 1e-10
        pref = (omega_x_au**3)/(2*np.pi*c_au**3)
        total += pref*(np.abs(M_tr)**2)/abs(wdot)
    return total

def compute_CDP_parallel(nc, phi_deg, omega_x_eV_grid, Nr, Rmax, Nmu, t_nt):
    t_grid,E_t,kL_t,q_t,omega_t,tau = make_pulse(nc,phi_deg,n_t=t_nt)
    args = [(om,t_grid,E_t,kL_t,omega_t,Nr,Rmax,Nmu) for om in omega_x_eV_grid]

    MAX_PROCS = min(cpu_count(), 64)   # use 1 process per physical core
    CHUNKSIZE = 2       

    with Pool(processes=MAX_PROCS) as pool:
        vals = pool.starmap(CDP_single, args, chunksize=CHUNKSIZE)

    return np.array(vals)

def DP_single(omega_x_eV, t_grid, kL_t, omega_t, Nr, R_max, N_mu):
    omega_x_au = omega_x_eV / hartree_ev
    Nt = len(t_grid)
    S = np.zeros(Nt)
    for i in range(1, Nt):
        S[i] = S[i-1] + 0.5*(omega_t[i]+omega_t[i-1])*(t_grid[i]-t_grid[i-1])
    M_t = np.zeros(Nt, dtype=complex)
    for i, tval in enumerate(t_grid):
        M_t[i] = compute_M_q_numeric(tval, omega_x_au, kL_t[i], q0, Z,
                                     N_r=Nr, R_max=R_max, N_mu=N_mu)
    phase = omega_x_au * t_grid - S
    A = np.trapezoid(M_t * np.exp(1j*phase), t_grid)
    pref = (omega_x_au**3)/(2*np.pi*c_au**3)
    return pref * np.abs(A)**2

def DP_single_eq6_literal(omega_x_eV, t_grid, kL_t, Nr, R_max, N_mu):
    """
    Implements Eq.(6) from the paper *exactly*:
    DP = omega_x^3 / ((2*pi)^2 * c^3) * | integral_0^tau dt' [ exp( i*(omega_x + I0)*t' - i/2 * integral_0^{t'} [q + kL(t'')]^2 dt'' ) * M_q(t') ] |^2
    """
    omega_x_au = omega_x_eV / hartree_ev
    Nt = len(t_grid)
    dt_arr = np.diff(t_grid)
    # Ionization potential I0 in a.u. (paper's notation) -- mapping choice: I0 = 0.5 * Z^2
    I0_au = 0.5 * (Z**2)

    # Precompute inner cumulative integral J(t') = \int_0^{t'} [q + kL(s)]^2 ds
    # where q is q0 (asymptotic momentum)
    J = np.zeros(Nt, dtype=float)
    for i in range(1, Nt):
        # integrand at i and i-1
        integrand_i = (q0 + kL_t[i])**2
        integrand_im1 = (q0 + kL_t[i-1])**2
        J[i] = J[i-1] + 0.5 * (integrand_i + integrand_im1) * (t_grid[i] - t_grid[i-1])

    # Now build integrand I(t') = exp( i*(omega_x + I0)*t' - i/2 * J(t') ) * M_q(t')
    integrand = np.zeros(Nt, dtype=complex)
    for i, tval in enumerate(t_grid):
        kL_tr = kL_t[i]
        # compute M_q at this time (use same signature as before)
        M_tr = compute_M_q_numeric(tval, omega_x_au, kL_tr, q0, Z,
                                   N_r=Nr, R_max=R_max, N_mu=N_mu)
        phase_factor = np.exp(1j * ( (omega_x_au + I0_au) * tval ) - 0.5j * J[i])
        integrand[i] = phase_factor * M_tr

    # Integral A = \int_0^tau integrand(t') dt'  (use trapezoid)
    A = np.trapezoid(integrand, t_grid)

    # Prefactor exactly as in paper: omega_x^3 / (2*pi)^2 / c^3
    pref = (omega_x_au**3) / ((2.0 * np.pi)**2 * c_au**3)
    DP_val = pref * (np.abs(A)**2)
    return DP_val


def compute_DP_parallel(nc, phi_deg, omega_x_eV_grid, Nr, R_max, N_mu, t_nt):
    # Prepare pulse and precompute arrays
    t_grid, E_t, kL_t, q_t, omega_t, tau = make_pulse(nc, phi_deg, n_t=t_nt)
    
    # Prepare argument list for all ωₓ values
    args_dp = [(om, t_grid, kL_t, Nr, R_max, N_mu) for om in omega_x_eV_grid]
    
    # ⚙️ Parallel pool config (optimized for your server)
    MAX_PROCS = min(cpu_count(), 64)   # use 1 process per physical core
    CHUNKSIZE = 2                      # each process handles 2 ωₓ points at a time
    
    print(f"[info] Using {MAX_PROCS} parallel workers for DP computation")

    # Run parallel computation
    with Pool(processes=MAX_PROCS) as pool:
        vals = pool.starmap(DP_single_eq6_literal, args_dp, chunksize=CHUNKSIZE)
    
    return np.array(vals)


# ============================================================
# MAIN EXECUTION (QUICK SANITY CHECK)
# ============================================================
if __name__ == "__main__":
    import time, os

    nc, phi = single_nc, single_phi
    t_grid, E_t, kL_t, q_t, omega_t, tau = make_pulse(nc, phi, n_t=t_nt)

    print(f"[info] Running nc={nc}, φ={phi}°, {omega_points} ω-points")

    wmin, wmax = np.min(omega_t)*hartree_ev, np.max(omega_t)*hartree_ev
    omega_x_eV_grid = np.linspace(max(1.0, wmin - 5.0), wmax + 5.0, omega_points)

    print("[run] Computing CDP spectrum (parallel) ...")
    t0 = time.time()
    CDP_vals = compute_CDP_parallel(nc, phi, omega_x_eV_grid, Nr, R_MAX_SIM , Nmu, t_nt)
    print(f"[done] CDP took {(time.time()-t0)/60:.2f} min")

    #print("[run] Computing DP spectrum (parallel Eq.6) ...")
    #t0 = time.time()
    #DP_vals = compute_DP_parallel(nc, phi, omega_x_eV_grid, Nr, R_MAX_SIM, Nmu, t_nt)
    #print(f"[done] DP took {(time.time()-t0)/60:.2f} min")

    # ---- export ----
    #os.makedirs("results_fast", exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #out_csv = f"results_fast/CDP_DP_nc{single_nc}_phi{int(single_phi)}_{timestamp}.csv"
    #np.savetxt(out_csv,
    #           np.column_stack((omega_x_eV_grid, CDP_vals, DP_vals)),
    #           delimiter=",", header="omega_x_eV,CDP,DP", comments='')
    #print(f"[saved CSV] {out_csv}")

    # ---- plot ----
    plt.figure(figsize=(7,5))
    plt.plot(omega_x_eV_grid, np.maximum(CDP_vals, 1e-50), label='CDP (Eq.13)',linewidth=3.0, color='blue')
    #plt.plot(omega_x_eV_grid, np.maximum(DP_vals, 1e-50),  label='DP (Eq.6 literal)', color='red')
    plt.yscale('log'); plt.xlim(0, 500); plt.ylim(1e-18, 1e-3)
    plt.xlabel("ωₓ (eV)"); plt.ylabel("Probability (a.u.)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"CDP_{timestamp}.png", dpi=180)
    print(f"CDP_{timestamp}.png\n")

    
