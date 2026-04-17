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
phi_list_deg = [-90.0, 0.0] #need to revert
nc_list = [3]

omega_au = omega_eV / hartree_ev
eps_q_au = eps_q_eV / hartree_ev
q0 = np.sqrt(2.0 * eps_q_au)

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

# Matrix element M_q(t)

def compute_M_q_numeric(tr, omega_x_au, kL_tr, q0, Z,
                        N_r=20, R_max=80.0, N_mu=20):
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
    return np.trapezoid(integrand_r, r_grid)

# Root finding

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

# CDP calculation for one ω_x

def CDP_single(omega_x_eV, t_grid, E_t, kL_t, omega_t,
               Nr, Rmax, Nmu):
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
        if np.isclose(wdot,0.0): continue
        pref = (omega_x_au**3)/(2*np.pi*c_au**3)
        total += pref*(np.abs(M_tr)**2)/abs(wdot)
    return total


# Parallel CDP spectrum

def compute_CDP_parallel(nc, phi_deg, omega_x_eV_grid,
                         Nr=150, Rmax=80.0, Nmu=120, t_nt=2500):
    t_grid,E_t,kL_t,q_t,omega_t,tau = make_pulse(nc,phi_deg,n_t=t_nt)
    args = [(om,t_grid,E_t,kL_t,omega_t,Nr,Rmax,Nmu) for om in omega_x_eV_grid]
    with Pool(processes=cpu_count()) as pool:
        vals = pool.starmap(CDP_single,args)
    return np.array(vals)


# Main

if __name__=="__main__":
    # pick grid of ω_x
    t_test,_,_,_,omega_test,_ = make_pulse(2,0.0,n_t=150)
    wmin, wmax = np.min(omega_test)*hartree_ev, np.max(omega_test)*hartree_ev
    omega_x_eV_grid = np.linspace(max(1.0,wmin-5.0),wmax+5.0,150)

    fig,axes = plt.subplots(2,3,figsize=(15,8),sharey=True)
    for row,nc in enumerate(nc_list):
        for col,phi in enumerate(phi_list_deg):
            print(f"[run] nc={nc}, phi={phi}")
            CDP_vals = compute_CDP_parallel(nc,phi,omega_x_eV_grid,
                                            Nr=120,Rmax=80.0,Nmu=120,t_nt=150)
            # Save results to CSV
            outdir = "results"
            os.makedirs(outdir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = os.path.join(outdir, f"CDP_nc{nc}_phi{int(phi)}_{timestamp}.csv")
            out_data = np.column_stack((omega_x_eV_grid, CDP_vals))
            np.savetxt(fname, out_data, delimiter=',', header='omega_x_eV,CDP', comments='')
            print(f"[saved] {fname}")
            ax = axes[row,col]
            ax.plot(omega_x_eV_grid, np.maximum(CDP_vals,1e-50))
            ax.set_yscale("log")
            ax.set_xlim(0,500)
            ax.set_ylim(1e-18,1e-3)
            ax.set_title(f"nc={nc}, φ={phi}°")
            ax.set_xlabel("ω_x (eV)")
            if col==0: ax.set_ylabel("CDP (a.u.)")

    plt.suptitle("Laser-Assisted Radiative Recombination: CDP spectra",fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.96])
    # Save the combined figure instead of showing it
    figdir = os.path.join(outdir, 'figures')
    os.makedirs(figdir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figpath = os.path.join(figdir, f'CDP_spectra_{timestamp}.png')
    fig.savefig(figpath, dpi=200, bbox_inches='tight')
    print(f"[saved] figure -> {figpath}")
