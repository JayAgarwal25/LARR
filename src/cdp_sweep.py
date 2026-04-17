import os
import gc
import time
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import gamma, j0
from numpy.polynomial.legendre import leggauss
import mpmath as mp
from multiprocessing import Pool, cpu_count, RawArray

from flint import ctx, acb
ctx.prec = 53  # Standard Double Precision

# ============================================================
# SHARED MEMORY GLOBALS (Must be at module level)
# ============================================================
shared_t_grid = None
shared_E_t = None
shared_kL_t = None
shared_q_t = None
shared_omega_t = None

def init_worker(t_shape, t_raw, E_raw, kL_raw, q_raw, om_raw):
    """
    Initializes the worker process. 
    Maps the raw shared memory to numpy arrays so workers can read data 
    without allocating new RAM.
    """
    global shared_t_grid, shared_E_t, shared_kL_t, shared_q_t, shared_omega_t
    
    # Reconstruct numpy arrays from shared memory buffers
    shared_t_grid = np.frombuffer(t_raw, dtype=np.float64).reshape(t_shape)
    if E_raw:  shared_E_t = np.frombuffer(E_raw, dtype=np.float64).reshape(t_shape)
    if kL_raw: shared_kL_t = np.frombuffer(kL_raw, dtype=np.float64).reshape(t_shape)
    if q_raw:  shared_q_t = np.frombuffer(q_raw, dtype=np.float64).reshape(t_shape)
    if om_raw: shared_omega_t = np.frombuffer(om_raw, dtype=np.float64).reshape(t_shape)


# ============================================================
# CONFIGURATION
# ============================================================
mp.dps = 30
FAST_MODE = False           # quick test vs full run
RUN_SINGLE = True          # plot one (nc,phi)
REUSE_EXISTING = False      # skip recomputation if CSV exists

single_nc = 2
single_phi = -90.0

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
#_M_cache = {}

def compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                        N_r=150, R_max=150.0, N_mu=120):
    
    #key = (round(kL_tr,8), round(omega_x_au,12), round(q0,12), Z, int(N_r), int(N_mu), float(R_max))
    #if key in _M_cache:
    #    return _M_cache[key]

    
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
    #_M_cache[key] = val
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
    I0_au = -0.5 * (Z**2)

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
# OPTIMIZED WORKER FUNCTIONS (READS SHARED MEMORY)
# ============================================================

def CDP_single_optimized(args):
    """Calculates CDP for a single frequency using Shared Memory"""
    omega_x_eV, Nr, Rmax, Nmu = args
    
    # Access shared arrays directly
    t_grid = shared_t_grid
    omega_t = shared_omega_t
    kL_t = shared_kL_t
    E_t = shared_E_t
    
    omega_x_au = omega_x_eV / hartree_ev
    roots = find_time_roots(omega_t, t_grid, omega_x_au)
    
    if not roots: 
        return (omega_x_eV, 0.0)

    total = 0.0
    for tr in roots:
        kL_tr = np.interp(tr, t_grid, kL_t)
        M_tr = compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                                   N_r=Nr, R_max=Rmax, N_mu=Nmu)
        E_val = np.interp(tr, t_grid, E_t)
        qeff = q0 + kL_tr
        wdot = -qeff * E_val
        if np.isclose(wdot, 0.0): wdot = 1e-10
        pref = (omega_x_au**3)/(2*np.pi*c_au**3)
        total += pref*(np.abs(M_tr)**2)/abs(wdot)
    
    # Clear cache to prevent RAM explosion over long runs
    #_M_cache.clear()
    
    return (omega_x_eV, total)

def DP_single_optimized(args):
    """Calculates DP for a single frequency using Shared Memory"""
    omega_x_eV, Nr, R_max, N_mu = args
    
    # Access shared arrays directly
    t_grid = shared_t_grid
    kL_t = shared_kL_t
    
    # --- Existing Logic ---
    omega_x_au = omega_x_eV / hartree_ev
    Nt = len(t_grid)
    I0_au = -0.5 * (Z**2)

    # J calculation
    J = np.zeros(Nt, dtype=float)
    # Vectorized J calculation for speed
    integrand = (q0 + kL_t)**2
    J[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t_grid))

    integrand_arr = np.zeros(Nt, dtype=complex)
    for i, tval in enumerate(t_grid):
        kL_tr = kL_t[i]
        M_tr = compute_M_q_numeric(tval, omega_x_au, kL_tr, q0, Z,
                                   N_r=Nr, R_max=R_max, N_mu=N_mu)
        #changed sign
        phase_factor = np.exp(1j * ((omega_x_au + I0_au) * tval) - 0.5j * J[i])
        integrand_arr[i] = phase_factor * M_tr

    A = np.trapezoid(integrand_arr, t_grid)
    pref = (omega_x_au**3) / ((2.0 * np.pi)**2 * c_au**3)
    DP_val = pref * (np.abs(A)**2)

    gc.collect()
    
    # Clear cache to prevent RAM explosion
    #_M_cache.clear()
    
    return (omega_x_eV, DP_val)

def run_parallel_computation_safe(mode_name, task_list, t_grid, E_t, kL_t, q_t, omega_t, out_csv):
    """
    Generic runner for both CDP and DP.
    Handles Shared Memory allocation and Incremental Saving.
    """
    # 1. Prepare Raw Arrays (Shared Memory)
    # We create raw c-type arrays that processes can share
    t_raw = RawArray('d', t_grid.flatten())
    E_raw = RawArray('d', E_t.flatten()) if E_t is not None else None
    kL_raw = RawArray('d', kL_t.flatten()) if kL_t is not None else None
    q_raw = RawArray('d', q_t.flatten()) if q_t is not None else None
    om_raw = RawArray('d', omega_t.flatten()) if omega_t is not None else None
    
    # 2. Config Pool
    MAX_PROCS = min(cpu_count(), 64)
    init_args = (t_grid.shape, t_raw, E_raw, kL_raw, q_raw, om_raw)
    
    print(f"[run] Starting {mode_name} with {MAX_PROCS} workers. Saving to {out_csv}")

    # 3. Select Function
    if mode_name == "CDP":
        func = CDP_single_optimized
    else:
        func = DP_single_optimized

    results_map = {} # To store results for plotting later
        
    with open(out_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["omega_x_eV", "Value"]) 
        
        # FORCE FLUSH INITIAL HEADER
        f.flush()
        os.fsync(f.fileno())
        
        with Pool(processes=MAX_PROCS, initializer=init_worker, initargs=init_args) as pool:
            total = len(task_list)
            count = 0
            
            # Using chunksize=1 is good, but we also need to clear internal memory
            for res in pool.imap_unordered(func, task_list, chunksize=1):
                om_val, val = res
                
                writer.writerow([om_val, val])
                
                # --- CRITICAL FIX: FORCE WRITE TO DISK ---
                f.flush()
                os.fsync(f.fileno())
                # -----------------------------------------
                
                results_map[om_val] = val
                count += 1
                
                if count % 5 == 0 or count == total:
                    sys.stdout.write(f"\r   Progress: {count}/{total} points calculated")
                    sys.stdout.flush()
                    
    print(f"\n[done] {mode_name} finished.")
    
    # Return sorted results matching the original grid order (for plotting)
    # task_list[i][0] is the omega value
    sorted_vals = [results_map[task[0]] for task in task_list]
    return np.array(sorted_vals)



if __name__ == "__main__":
    
    # --- Sweep Settings ---
    nc_list = [2, 3]
    phi_list = list(range(-90, 91, 1))  # -90 to +90 step 1
    
    out_dir = "CEP_Data"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Starting CDP Sweep. Output: {out_dir}")

    # Use max available cores
    n_cores = min(cpu_count(), 64)

    # --- Loop over nc ---
    for nc in nc_list:
        
        # Create the Combined CSV for this nc
        combined_csv = os.path.join(out_dir, f"Heatmap_nc{nc}.csv")
        
        with open(combined_csv, "w") as f:
            f.write("phi_deg,omega_eV,CDP\n")

        print(f"\n[Set Start] nc={nc} -> {combined_csv}")

        # --- Loop over phi ---
        for phi in phi_list:
            
            # 1. Prepare Pulse (Local Arrays)
            t_grid, E_t, kL_t, q_t, omega_t, tau = make_pulse(nc, phi, n_t=t_nt)
            
            wmin = np.min(omega_t) * hartree_ev
            wmax = np.max(omega_t) * hartree_ev
            omega_x_eV_grid = np.linspace(max(1.0, wmin - 5.0), wmax + 5.0, omega_points)

            t_raw = RawArray('d', t_grid.flatten())
            E_raw = RawArray('d', E_t.flatten()) if E_t is not None else None
            kL_raw = RawArray('d', kL_t.flatten()) if kL_t is not None else None
            q_raw = RawArray('d', q_t.flatten()) if q_t is not None else None
            om_raw = RawArray('d', omega_t.flatten()) if omega_t is not None else None
    
            
            init_args = (t_grid.shape, t_raw, E_raw, kL_raw, q_raw, om_raw)

            args_list = [(om, Nr, R_MAX_SIM, Nmu) for om in omega_x_eV_grid]

            with Pool(processes=n_cores, initializer=init_worker, initargs=init_args) as pool:
                results = pool.map(CDP_single_optimized, args_list)
            
            CDP_vals = np.array([r[1] for r in results])

            # Save "Per-Case" Data 

            per_case_path = os.path.join(out_dir, f"nc{nc}_phi{phi}.dat")
            np.savetxt(
                per_case_path, 
                np.column_stack((omega_x_eV_grid, CDP_vals)),
                fmt="%.10e",
                header="omega_eV CDP"
            )

            #Append to Combined CSV
            with open(combined_csv, "a") as f:
                for om, val in zip(omega_x_eV_grid, CDP_vals):
                    f.write(f"{phi},{om:.6f},{val:.6e}\n")

            print(f"   -> nc={nc}, phi={phi}° completed.")

    print("\n[Done] All sweeps finished.")
