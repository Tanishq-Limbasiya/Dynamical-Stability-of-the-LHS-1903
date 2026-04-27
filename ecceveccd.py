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

times = np.linspace(0, T_YEARS, N_STEPS)
ecc_d = []
ecc_e = []

print(f"Integrating {T_YEARS} years ({N_STEPS} snapshots)...")

for t in times:
    sim.integrate(t, exact_finish_time=1)
    orb_d = sim.particles[3].orbit(primary=sim.particles[0])
    orb_e = sim.particles[4].orbit(primary=sim.particles[0])
    ecc_d.append(orb_d.e)
    ecc_e.append(orb_e.e)

ecc_d = np.array(ecc_d)
ecc_e = np.array(ecc_e)

print("Done. Plotting...")

fig, ax = plt.subplots(figsize=(8, 7))

scatter = ax.scatter(
    ecc_d, ecc_e,
    c=times,
    cmap="plasma",
    s=2,
    alpha=0.6,
    linewidths=0
)

ax.axvline(0.112, color="black", ls="--", lw=1.0, alpha=0.7,
           label="Nominal $e_d$ = 0.112")
ax.axhline(0.014, color="gray",  ls="--", lw=1.0, alpha=0.7,
           label="Nominal $e_e$ = 0.014")

ax.plot(ecc_d[0],  ecc_e[0],  "o", color="lime",  markersize=8,
        zorder=5, label="t = 0 yr")
ax.plot(ecc_d[-1], ecc_e[-1], "s", color="red",   markersize=8,
        zorder=5, label="t = 10,000 yr")

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Time (years)", fontsize=11)
cbar.set_ticks([0, 2500, 5000, 7500, 10000])

ax.set_xlabel("Eccentricity of Planet d  ($e_d$)", fontsize=12)
ax.set_ylabel("Eccentricity of Planet e  ($e_e$)", fontsize=12)
ax.set_title(
    "LHS 1903 — Phase Space: $e_d$ vs $e_e$\n"
    "WHFast | dt = 0.0003 yr | T = 10,000 yr | One point per year | Wilson et al. 2026",
    fontsize=12
)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("phase_space_ecc_d_vs_ecc_e.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → phase_space_ecc_d_vs_ecc_e.png")
