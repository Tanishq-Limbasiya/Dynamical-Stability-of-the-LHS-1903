import rebound
import numpy as np
import matplotlib.pyplot as plt

M_STAR = 0.538
ME     = 3e-6
STAR_R = 0.002506

PLANET_B = dict(m=3.28*ME, a=0.02656, e=0.015, omega=np.radians(216), r=5.89e-5)
PLANET_C = dict(m=4.55*ME, a=0.05387, e=0.089, omega=np.radians(288), r=8.72e-5)
PLANET_D = dict(m=5.96*ME, a=0.08604, e=0.112, omega=np.radians(233), r=1.07e-4)
PLANET_E = dict(m=5.79*ME, a=0.15135, e=0.014, omega=np.radians(263), r=7.38e-5)

sim = rebound.Simulation()
sim.integrator = "whfast"
sim.dt         = 0.0003
sim.units      = ('yr', 'AU', 'Msun')

sim.add(m=M_STAR, r=STAR_R)
sim.add(m=PLANET_B['m'], a=PLANET_B['a'], e=PLANET_B['e'], omega=PLANET_B['omega'], r=PLANET_B['r'])
sim.add(m=PLANET_C['m'], a=PLANET_C['a'], e=PLANET_C['e'], omega=PLANET_C['omega'], r=PLANET_C['r'])
sim.add(m=PLANET_D['m'], a=PLANET_D['a'], e=PLANET_D['e'], omega=PLANET_D['omega'], r=PLANET_D['r'])
sim.add(m=PLANET_E['m'], a=PLANET_E['a'], e=PLANET_E['e'], omega=PLANET_E['omega'], r=PLANET_E['r'])
sim.move_to_com()

T_YEARS = 10000
N_STEPS = 10000

times       = np.linspace(0, T_YEARS, N_STEPS)
omega_d     = []
omega_e     = []
delta_omega = []
lambda_d    = []
lambda_e    = []
phi_73      = []

print(f"Integrating {T_YEARS} years ({N_STEPS} snapshots)...")

for t in times:
    sim.integrate(t, exact_finish_time=1)

    star = sim.particles[0]
    pd   = sim.particles[3]
    pe   = sim.particles[4]

    orb_d = pd.orbit(primary=star)
    orb_e = pe.orbit(primary=star)

    od = np.degrees(orb_d.omega) % 360.0
    oe = np.degrees(orb_e.omega) % 360.0
    omega_d.append(od)
    omega_e.append(oe)

    dw = od - oe
    if dw >  180: dw -= 360
    if dw < -180: dw += 360
    delta_omega.append(dw)

    ld = np.degrees(orb_d.Omega + orb_d.omega + orb_d.M) % 360.0
    le = np.degrees(orb_e.Omega + orb_e.omega + orb_e.M) % 360.0
    lambda_d.append(ld)
    lambda_e.append(le)

    phi = (7*le - 3*ld - 4*oe) % 360.0
    phi_73.append(phi)

print("Done. Plotting...")

times       = np.array(times)
omega_d     = np.array(omega_d)
omega_e     = np.array(omega_e)
delta_omega = np.array(delta_omega)
phi_73      = np.array(phi_73)

dw_range  = delta_omega.max() - delta_omega.min()
phi_range = phi_73.max()      - phi_73.min()

if dw_range < 180:
    apsidal_mode = f"LIBRATING  (range = {dw_range:.1f}°) → Apsidal resonance"
else:
    apsidal_mode = f"CIRCULATING (range = {dw_range:.1f}°) → No apsidal lock"

if phi_range < 180:
    resonance_mode = f"LIBRATING  (range = {phi_range:.1f}°) → In 7:3 MMR"
else:
    resonance_mode = f"CIRCULATING (range = {phi_range:.1f}°) → Not in 7:3 MMR"

print(f"\n{'─'*55}")
print(f"  Δω (d-e):     {apsidal_mode}")
print(f"  Φ (7:3 MMR):  {resonance_mode}")
print(f"  Mean Δω:      {delta_omega.mean():.2f}°")
print(f"  Std  Δω:      {delta_omega.std():.2f}°")
print(f"  Mean Φ:       {phi_73.mean():.2f}°")
print(f"  Std  Φ:       {phi_73.std():.2f}°")
print(f"{'─'*55}\n")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    "LHS 1903 — Apsidal Resonance Analysis (Planets d & e)\n"
    "WHFast | dt = 0.0003 yr | T = 10,000 yr | Wilson et al. 2026",
    fontsize=13
)

ax = axes[0]
ax.plot(times, omega_d, color="#2ecc71", lw=0.6, alpha=0.8, label="$\\omega_d$")
ax.plot(times, omega_e, color="#e74c3c", lw=0.6, alpha=0.8, label="$\\omega_e$")
ax.set_ylabel("$\\omega$ (deg)", fontsize=11)
ax.set_ylim(0, 360)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.88, "Arguments of Periapsis",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.8))

ax = axes[1]
ax.plot(times, delta_omega, color="#f39c12", lw=0.6, alpha=0.9)
ax.axhline(0,    color="black", ls="--", lw=0.8, alpha=0.4)
ax.axhline(180,  color="gray",  ls=":",  lw=0.6, alpha=0.4)
ax.axhline(-180, color="gray",  ls=":",  lw=0.6, alpha=0.4)
ax.set_ylabel("$\\Delta\\omega$ (deg)", fontsize=11)
ax.set_ylim(-185, 185)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.88,
        f"Apsidal difference: {apsidal_mode}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.8))

ax = axes[2]
ax.plot(times, phi_73, color="#3498db", lw=0.6, alpha=0.9)
ax.axhline(180, color="black", ls="--", lw=0.8, alpha=0.4)
ax.set_ylabel("$\\Phi_{7:3}$ (deg)", fontsize=11)
ax.set_xlabel("Time (years)", fontsize=11)
ax.set_ylim(0, 360)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.88,
        f"7:3 MMR angle: {resonance_mode}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(fc="white", ec="none", alpha=0.8))

plt.tight_layout()
plt.savefig("apsidal_resonance.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → apsidal_resonance.png")
