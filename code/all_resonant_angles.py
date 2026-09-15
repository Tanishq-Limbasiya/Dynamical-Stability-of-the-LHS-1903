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
times   = np.linspace(0, T_YEARS, N_STEPS)

phi1 = []
phi2 = []
phi3 = []
phi4 = []
phi5 = []

print("Integrating 10000 years...")

for t in times:
    sim.integrate(t, exact_finish_time=1)

    star  = sim.particles[0]
    orb_d = sim.particles[3].orbit(primary=star)
    orb_e = sim.particles[4].orbit(primary=star)

    ld = np.degrees(orb_d.Omega + orb_d.omega + orb_d.M) % 360.0
    le = np.degrees(orb_e.Omega + orb_e.omega + orb_e.M) % 360.0
    vd = np.degrees(orb_d.Omega + orb_d.omega) % 360.0
    ve = np.degrees(orb_e.Omega + orb_e.omega) % 360.0

    base = 7*le - 3*ld

    phi1.append((base - 4*ve        ) % 360.0)
    phi2.append((base - 4*vd        ) % 360.0)
    phi3.append((base - 3*ve - vd   ) % 360.0)
    phi4.append((base - 2*ve - 2*vd ) % 360.0)
    phi5.append((base - ve  - 3*vd  ) % 360.0)

print("Done.")

phi1 = np.array(phi1)
phi2 = np.array(phi2)
phi3 = np.array(phi3)
phi4 = np.array(phi4)
phi5 = np.array(phi5)

names = ["Phi1 = 7le-3ld-4ve",
         "Phi2 = 7le-3ld-4vd",
         "Phi3 = 7le-3ld-3ve-vd",
         "Phi4 = 7le-3ld-2ve-2vd",
         "Phi5 = 7le-3ld-ve-3vd"]

phis = [phi1, phi2, phi3, phi4, phi5]

print("--------------------------------------------------")
print("Resonance Angle Analysis -- 7:3 MMR (d:e)")
print("--------------------------------------------------")
for name, phi in zip(names, phis):
    rng  = phi.max() - phi.min()
    if rng < 180:
        mode = "LIBRATING -> IN RESONANCE"
    else:
        mode = "circulating"
    print(name + "  range=" + str(round(rng, 1)) + " deg  " + mode)
print("--------------------------------------------------")

labels = [
    "$\\Phi_1 = 7\\lambda_e - 3\\lambda_d - 4\\varpi_e$",
    "$\\Phi_2 = 7\\lambda_e - 3\\lambda_d - 4\\varpi_d$",
    "$\\Phi_3 = 7\\lambda_e - 3\\lambda_d - 3\\varpi_e - \\varpi_d$",
    "$\\Phi_4 = 7\\lambda_e - 3\\lambda_d - 2\\varpi_e - 2\\varpi_d$",
    "$\\Phi_5 = 7\\lambda_e - 3\\lambda_d - \\varpi_e - 3\\varpi_d$",
]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig.suptitle(
    "LHS 1903 -- All 7:3 MMR Resonance Angles (Planets d and e)\n"
    "WHFast | dt = 0.0003 yr | T = 10,000 yr | Wilson et al. 2026",
    fontsize=13
)

for i in range(5):
    ax    = axes[i]
    phi   = phis[i]
    rng   = phi.max() - phi.min()
    if rng < 180:
        mode = "LIBRATING"
    else:
        mode = "circulating"

    ax.plot(times, phi, color=colors[i], lw=0.5, alpha=0.8)
    ax.axhline(180, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.set_ylabel("Angle (deg)", fontsize=9)
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.85,
            labels[i] + "   range = " + str(round(rng, 1)) + " deg  ->  " + mode,
            transform=ax.transAxes, fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8))

axes[-1].set_xlabel("Time (years)", fontsize=11)

plt.tight_layout()
plt.savefig("resonance_angles_all.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved -> resonance_angles_all.png")
