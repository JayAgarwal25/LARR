# LARR - Laser-Assisted Radiative Recombination

Simulation of LARR of free electrons with hydrogenic ions under few-cycle laser pulses. Implements both a quantum-mechanical model (Coulomb-Volkov approximation) and a semiclassical model, with CEP parameter sweeps.

BITS Pilani, Department of Physics. Supervised by Dr. Amol R. Holkundkar.

## Physics

A free electron recombines with a positive ion and emits a photon. In the presence of a laser field, the electron's quiver motion broadens the emitted photon spectrum. For few-cycle pulses, the carrier-envelope phase (CEP) strongly affects the spectral cutoff positions.

Two models are implemented:
- **DP** (Double differential Probability): full quantum-mechanical calculation using the Coulomb-Volkov approximation, involving the confluent hypergeometric function evaluated via [python-flint](https://github.com/fredrik-johansson/python-flint).
- **CDP** (Classical Differential Probability): semiclassical model, sums incoherently over recombination times. Faster, captures cutoffs and envelope but not interference fringes.

Laser pulse: sin² envelope, `I_L = 6e14 W/cm²`, `ω = 1.55 eV`, `nc = 2 or 3` cycles, CEP `φ` swept from -90° to +90°.

## Structure

```
src/          main simulation code
legacy/       earlier versions
scripts/      test and sanity check scripts
```

## Dependencies

```bash
pip install numpy scipy matplotlib mpmath python-flint pandas
```

LaTeX is needed for the styled plots in `plot.py`, but it falls back gracefully.

## Usage

**Single run** - edit the config at the top of `src/quantum_new.py`:
```python
RUN_SINGLE = True
single_nc  = 2
single_phi = 90.0
```
```bash
python3 src/quantum_new.py
```

**CEP sweep:**
```bash
python3 src/cdp_sweep.py
```

**Plotting:**
```bash
python3 src/plotter.py --type DP
python3 src/plotter.py --compare --nc 2 --phi 90
python3 src/plot.py        # heatmaps
```

## References

1. S. Bivona et al., *Optics Express* 14, 3715 (2006)
2. S. Bivona et al., *Laser Physics Letters* 4, 44 (2007)
3. I. I. Fabrikant & H. B. Ambalampitiya, *Phys. Rev. A* 101, 063418 (2020)
4. D. Kanti et al., *Phys. Rev. A* 103, 043102 (2021)
