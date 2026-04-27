import rebound
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

T_ORBIT = 29.32 / 365.25
N_STEPS = 1000
times   = np.linspace(0, T_ORBIT, N_STEPS)

x = {"b": [], "c": [], "d": [], "e": []}
y = {"b": [], "c": [], "d": [], "e": []}

for t in times:
    sim.integrate(t, exact_finish_time=1)
    for i, name in enumerate(["b", "c", "d", "e"], start=1):
        x[name].append(sim.particles[i].x)
        y[name].append(sim.particles[i].y)

fig, ax = plt.subplots(figsize=(8, 8))

colors = {
    "b": "#f39c12",
    "c": "#3498db",
    "d": "#2ecc71",
    "e": "#e74c3c",
}

labels = {
    "b": "Planet b  |  a = 0.02656 AU  |  m = 3.28 $M_\\oplus$",
    "c": "Planet c  |  a = 0.05387 AU  |  m = 4.55 $M_\\oplus$",
    "d": "Planet d  |  a = 0.08604 AU  |  m = 5.96 $M_\\oplus$",
    "e": "Planet e  |  a = 0.15135 AU  |  m = 5.79 $M_\\oplus$",
}

for name in ["b", "c", "d", "e"]:
    ax.plot(x[name], y[name], color=colors[name], lw=1.5, alpha=0.9)
    ax.scatter(x[name][-1], y[name][-1], color=colors[name], s=60, zorder=5)

ax.scatter(0, 0, color="yellow", s=250, zorder=6, label="LHS 1903  |  0.538 $M_\\odot$")

patches = [mpatches.Patch(color=colors[n], label=labels[n]) for n in ["b", "c", "d", "e"]]
patches.insert(0, mpatches.Patch(color="yellow", label="LHS 1903  |  0.538 $M_\\odot$"))
ax.legend(handles=patches, fontsize=8.5, loc="upper right")

ax.set_xlabel("x (AU)", fontsize=12)
ax.set_ylabel("y (AU)", fontsize=12)
ax.set_title(
    "LHS 1903 — Orbital Architecture\n"
    "One orbit of Planet e (29.32 days) | Wilson et al. 2026",
    fontsize=12
)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("orbit_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → orbit_plot.png")
