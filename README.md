# LARR — Laser-Assisted Radiative Recombination

Computational simulation of Laser-Assisted Radiative Recombination (LARR) of free electrons with hydrogenic ions under intense few-cycle laser pulses. Implements both a full quantum-mechanical model (Coulomb–Volkov approximation) and a semiclassical two-step model, with detailed carrier-envelope phase (CEP) parameter sweeps.

**BITS Pilani — Department of Physics**
Supervised by Dr. Amol R. Holkundkar

## Physics

In LARR, a free electron recombines with a positive ion and emits a photon. In the presence of an intense laser field, the electron's quiver motion broadens the emitted photon spectrum. For few-cycle pulses, the carrier-envelope phase (CEP) strongly controls the spectral cutoff positions and width.

### Quantum-Mechanical Model (Double Differential Probability — DP)

The transition amplitude uses the **Coulomb–Volkov approximation**, where the dressed continuum state is:

$$\Psi_q^+(\mathbf{r},t) = \exp\!\left[i[q+k_L(t)]\cdot\mathbf{r} - \frac{i}{2}\int_0^t [q+k_L(t')]^2 dt'\right] u_q^+(\mathbf{r})$$

The spatial part $u_q^+(\mathbf{r})$ involves the confluent hypergeometric function $_1F_1$, evaluated at high precision using [python-flint](https://github.com/fredrik-johansson/python-flint).

### Semiclassical Model (Classical Differential Probability — CDP)

Treats electron propagation classically. At recombination time $t_r$, energy conservation gives $\omega_X = \frac{1}{2}[q+k_L(t_r)]^2 + |I_0|$. The CDP sums incoherently over all contributing times — capturing spectral cutoffs and envelope without quantum interference fringes.

### Laser Pulse

Sin²-envelope, linearly polarized:

$$E(t) = E_0 \sin^2\!\left(\frac{\pi t}{\tau}\right)\cos(\omega t + \varphi), \quad 0 \le t \le \tau$$

Parameters: $I_L = 6\times10^{14}$ W/cm², $\omega = 1.55$ eV (Ti:Sapphire), $n_c \in \{2,3\}$, $\varphi \in [-90°, +90°]$, $\varepsilon_q = 90$ eV.

## Repository Structure

```
src/
  quantum_new.py              # main QM simulation (DP + CDP, multiprocessing)
  quantum_optimized.py        # optimized version using shared memory across workers
  cdp_sweep.py                # full CEP sweep φ = −90° to +90° in 1° steps
  semi_classical_with_matrix.py  # semiclassical model cross-check
  plotter.py                  # CLI tool to plot DP/CDP results from CSV output
  plot.py                     # CEP heatmap plots from sweep data

legacy/
  quantum.py … quantum4.py    # earlier development versions
  quantum_test.py, new.py

scripts/
  test.py, test2.py, check.py # quick sanity checks
```

## Dependencies

```
numpy scipy matplotlib mpmath python-flint pandas
```

```bash
pip install numpy scipy matplotlib mpmath python-flint pandas
```

> LaTeX is required for publication-quality plots in `plot.py` (TeX Live or MiKTeX). Falls back to standard fonts if unavailable.

## Usage

### Single (nc, phi) run

Edit the config block at the top of [src/quantum_new.py](src/quantum_new.py):
```python
RUN_SINGLE = True
single_nc  = 2
single_phi = 90.0
FAST_MODE  = False   # True for a quick test
```
```bash
python3 src/quantum_new.py
```

### Full CEP sweep (CDP only)
```bash
python3 src/cdp_sweep.py
```

### Plot results
```bash
python3 src/plotter.py --type DP                       # quantum DP spectrum
python3 src/plotter.py --compare --nc 2 --phi 90       # overlay DP vs CDP
python3 src/plotter.py --file results/CDP_nc2_phi90.csv
```

### CEP heatmaps
```bash
python3 src/plot.py   # reads CEP_Data/Heatmap_nc{2,3}.csv
```

## Output

Results are written to `results/` (full run) or `results_fast/` (fast mode) as CSV files named `DP_nc{N}_phi{P}.csv` and `CDP_nc{N}_phi{P}.csv`.

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `I_L` | 6×10¹⁴ W/cm² | Laser peak intensity |
| `ω` | 1.5498 eV | Photon energy (Ti:Sapphire) |
| `ε_q` | 90 eV | Final electron energy |
| `Z` | 1 | Nuclear charge (hydrogen) |
| `Nr × Nμ` | 150 × 120 | Spatial integration grid (full mode) |
| `t_nt` | 3000 | Time grid points |
| `φ` sweep | −90° to +90°, 1° steps | CEP parameter scan |

## References

1. S. Bivona et al., *Optics Express* **14**, 3715 (2006)
2. S. Bivona et al., *Laser Physics Letters* **4**, 44 (2007)
3. I. I. Fabrikant & H. B. Ambalampitiya, *Phys. Rev. A* **101**, 063418 (2020)
4. D. Kanti et al., *Phys. Rev. A* **103**, 043102 (2021)
