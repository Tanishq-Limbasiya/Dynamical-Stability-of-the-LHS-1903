import rebound
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os

M_STAR = 0.538
ME     = 3e-6
STAR_R = 0.002506

PLANET_B = dict(m=3.28*ME, a=0.02656, e=0.015, omega=np.radians(216), r=5.89e-5)
PLANET_C = dict(m=4.55*ME, a=0.05387, e=0.089, omega=np.radians(288), r=8.72e-5)
PLANET_D = dict(m=5.96*ME, a=0.08604, e=0.112, omega=np.radians(233), r=1.07e-4)
PLANET_E = dict(m=5.79*ME, a=0.15135, e=0.014, omega=np.radians(263), r=7.38e-5)

N_GRID     = 50
T_INT      = 80000

ECC_D_MIN, ECC_D_MAX = 0.0, 0.50
ECC_E_MIN, ECC_E_MAX = 0.0, 0.30

def run_sim(params):
    ecc_e, ecc_d = params

    sim = rebound.Simulation()
    sim.integrator = "whfast"
    sim.dt         = 0.0003
    sim.units      = ('yr', 'AU', 'Msun')

    sim.add(m=M_STAR, r=STAR_R)
    sim.add(m=PLANET_B['m'], a=PLANET_B['a'], e=PLANET_B['e'],
            omega=PLANET_B['omega'], r=PLANET_B['r'])
    sim.add(m=PLANET_C['m'], a=PLANET_C['a'], e=PLANET_C['e'],
            omega=PLANET_C['omega'], r=PLANET_C['r'])
    sim.add(m=PLANET_D['m'], a=PLANET_D['a'], e=ecc_d,
            omega=PLANET_D['omega'], r=PLANET_D['r'])
    sim.add(m=PLANET_E['m'], a=PLANET_E['a'], e=ecc_e,
            omega=PLANET_E['omega'], r=PLANET_E['r'])

    sim.move_to_com()
    sim.collision         = "direct"
    sim.exit_max_distance = 2.0

    N_CHUNKS = 200
    dt_chunk = T_INT / N_CHUNKS

    for i in range(N_CHUNKS):
        try:
            sim.integrate(sim.t + dt_chunk)
        except:
            return (params, sim.t)

    return (params, T_INT)

if __name__ == "__main__":
    ecc_e_vals = np.linspace(ECC_E_MIN, ECC_E_MAX, N_GRID)
    ecc_d_vals = np.linspace(ECC_D_MIN, ECC_D_MAX, N_GRID)

    work_list = [(ee, ed) for ee in ecc_e_vals for ed in ecc_d_vals]
    total     = len(work_list)

    n_workers = max(1, os.cpu_count() - 1)
    print("LHS 1903 -- Survival Time Grid: Ecc of e vs Ecc of d")
    print("Grid: " + str(N_GRID) + "x" + str(N_GRID) + " = " + str(total) + " sims")
    print("T = " + str(T_INT) + " yr | WHFast | dt = 0.0003 yr")
    print("Workers: " + str(n_workers))
    print("-" * 50)

    results_dict = {}
    completed    = 0

    with Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(run_sim, work_list):
            params, survival_time = result
            results_dict[params]  = survival_time
            completed += 1

            pct    = (completed / total) * 100
            filled = int(30 * completed / total)
            bar    = "#" * filled + "-" * (30 - filled)
            print("\r  [" + bar + "] " + str(completed) + "/" + str(total) +
                  "  (" + str(round(pct, 1)) + "%)", end="", flush=True)

    print("")
    print("-" * 50)
    print("All simulations complete!")

    survival_vals = [results_dict[(ee, ed)]
                     for ee in ecc_e_vals for ed in ecc_d_vals]
    grid          = np.array(survival_vals).reshape(N_GRID, N_GRID)

    n_stable = np.sum(grid >= T_INT)
    print("Stable:   " + str(n_stable) + "/" + str(total))
    print("Unstable: " + str(total - n_stable) + "/" + str(total))

    np.save('survival_ecc_e_vs_ecc_d.npy', grid)
    print("Grid saved -- survival_ecc_e_vs_ecc_d.npy")

    fig, ax = plt.subplots(figsize=(10, 7))

    img = ax.imshow(
        np.log10(grid + 1),
        extent=[ECC_D_MIN, ECC_D_MAX, ECC_E_MIN, ECC_E_MAX],
        origin='lower',
        cmap='RdYlGn',
        vmin=0,
        vmax=np.log10(T_INT),
        aspect='auto'
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('Survival Time (log scale, years)', fontsize=11)
    tick_years = [100, 500, 1000, 5000, 10000, 40000, 80000]
    tick_pos   = [np.log10(t) for t in tick_years]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([str(t) for t in tick_years])

    ax.axhline(0.014, color='black', ls='--', lw=1.5,
               label='Nominal $e_e$ = 0.014')
    ax.axvline(0.112, color='blue',  ls='--', lw=1.5,
               label='Nominal $e_d$ = 0.112')
    ax.plot(0.112, 0.014, '*', color='black',
            markersize=14, zorder=5, label='Nominal system')

    ax.set_xlabel('Eccentricity of Planet d  ($e_d$)', fontsize=12)
    ax.set_ylabel('Eccentricity of Planet e  ($e_e$)', fontsize=12)
    ax.set_title(
        'LHS 1903 -- Survival Time Map: Ecc of e vs Ecc of d\n'
        'WHFast | dt=0.0003 yr | T=80,000 yr | 50x50 | Wilson et al. 2026',
        fontsize=11
    )
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig('survival_ecc_e_vs_ecc_d.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved -- survival_ecc_e_vs_ecc_d.png")
