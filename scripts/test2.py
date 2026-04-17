import time
import numpy as np
import mpmath as mp
from flint import ctx, acb

# ==========================================
# CONFIGURATION
# ==========================================
ctx.prec = 53
mp.dps = 15

# Test parameters
v = 10.5
a_val = -1j * v      # a = -10.5j
b_val = 1.0          # b = 1.0
z_val = 2.0 + 3.0j   # z = 2 + 3j

print(f"Testing 1F1(a={a_val}, b={b_val}, z={z_val})")
print("-" * 50)

# ==========================================
# 1. ACCURACY CHECK
# ==========================================

# --- Run mpmath ---
# Notation: hyp1f1(a, b, z)
t0 = time.time()
res_mp = complex(mp.hyp1f1(a_val, b_val, z_val))
print(f"mpmath result:  {res_mp:.15e}")

# --- Run flint ---
t0 = time.time()
a_flint = acb(a_val)
b_flint = acb(b_val)
z_flint = acb(z_val)

# CORRECT SYNTAX: z.function(a, b)
res_flint_obj = z_flint.hypgeom_1f1(a_flint, b_flint)

res_flint = complex(res_flint_obj)
print(f"flint result:   {res_flint:.15e}")

# --- Compare ---
diff = abs(res_mp - res_flint)
print(f"Difference:     {diff:.5e}")

if diff < 1e-10:
    print("✅ ACCURACY CHECK PASSED: Results match.")
else:
    print("❌ ACCURACY CHECK FAILED: Results diverge.")

print("-" * 50)

# ==========================================
# 2. SPEED STRESS TEST
# ==========================================
N_LOOPS = 10000
print(f"Running Speed Test ({N_LOOPS} iterations)...")

# --- Timing mpmath ---
start = time.time()
for _ in range(N_LOOPS):
    _ = complex(mp.hyp1f1(a_val, b_val, z_val))
end = time.time()
time_mp = end - start
print(f"mpmath time: {time_mp:.4f} sec")

# --- Timing flint ---
# Pre-convert constants
a_f = acb(a_val)
b_f = acb(b_val)
z_f = acb(z_val)

start = time.time()
for _ in range(N_LOOPS):
    # CORRECT SYNTAX: z.function(a, b)
    val = z_f.hypgeom_1f1(a_f, b_f)
    _ = complex(val)
end = time.time()
time_flint = end - start
print(f"flint time:  {time_flint:.4f} sec")

# --- Verdict ---
speedup = time_mp / time_flint
print("-" * 50)
print(f"🚀 Speedup Factor: {speedup:.2f}x FASTER")
print("-" * 50)
