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
ecc   = {"b": [], "c": [], "d": [], "e": []}

print(f"Integrating {T_YEARS} years ({N_STEPS} snapshots)...")

for t in times:
    sim.integrate(t, exact_finish_time=1)
    for i, name in enumerate(["b", "c", "d", "e"], start=1):
        orb = sim.particles[i].orbit(primary=sim.particles[0])
        ecc[name].append(orb.e)

print("Done. Plotting...")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle(
    "LHS 1903 — Eccentricity Evolution (Nominal System)\n"
    "WHFast | dt = 0.0003 yr | T = 10,000 yr | Wilson et al. 2026",
    fontsize=13
)

planets  = ["b",      "c",      "d",      "e"    ]
colors   = ["#f39c12","#3498db","#2ecc71","#e74c3c"]
nominals = [0.015,    0.089,    0.112,    0.014  ]
masses   = [3.28,     4.55,     5.96,     5.79   ]

for i, (name, color, e_nom, mass) in enumerate(
        zip(planets, colors, nominals, masses)):

    ax    = axes[i]
    e_arr = np.array(ecc[name])

    ax.plot(times, e_arr, color=color, lw=0.6, alpha=0.9)
    ax.axhline(e_nom, color="black", ls="--", lw=0.8, alpha=0.6,
               label=f"Nominal e = {e_nom}")
    ax.fill_between(times, e_arr.min(), e_arr.max(),
                    alpha=0.15, color=color)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Eccentricity", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.02, 0.85,
        f"Planet {name}  |  m = {mass} $M_\\oplus$  |  $e_{{nom}}$ = {e_nom}",
        transform=ax.transAxes,
        fontsize=9, color=color,
        bbox=dict(fc="white", ec="none", alpha=0.8)
    )

    print(f"  Planet {name}: "
          f"min={e_arr.min():.5f}  "
          f"max={e_arr.max():.5f}  "
          f"mean={e_arr.mean():.5f}  "
          f"std={e_arr.std():.5f}")

axes[-1].set_xlabel("Time (years)", fontsize=11)

plt.tight_layout()
plt.savefig("eccentricity_evolution.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved → eccentricity_evolution.png")
