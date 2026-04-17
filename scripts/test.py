import sys
import numpy as np
import scipy
from scipy.special import hyp1f1

print("="*40)
print(f"Python Version: {sys.version.split()[0]}")
print(f"SciPy Version:  {scipy.__version__}")
print("="*40)

# Define complex test inputs similar to your physics parameters
# a = -i*v (complex)
# b = 1.0 (real)
# z = complex argument
a = -1j * 0.5
b = 1.0
z = 2.0 + 3.0j

print(f"\n[Test] Attempting hyp1f1({a}, {b}, {z})...")

try:
    # 1. Scalar Test
    result = hyp1f1(a, b, z)
    print(f"[Success] Scalar result: {result}")
    
    # 2. Vectorized Test (What your code actually needs)
    z_array = np.array([z, z*2, z*3], dtype=complex)
    res_array = hyp1f1(a, b, z_array)
    print(f"[Success] Vectorized result: {res_array}")
    
    print("\n✅ VERDICT: Your SciPy SUPPORTS complex inputs!")
    print("You should switch to SciPy for 100x speedup.")

except TypeError as e:
    print(f"\n[Failed] TypeError: {e}")
    print("❌ VERDICT: Your SciPy is too old or does not support complex types.")
    print("You must stick to the mpmath wrapper solution.")

except Exception as e:
    print(f"\n[Failed] Error: {e}")
