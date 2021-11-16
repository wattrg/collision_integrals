from collision_integrals import CollisionIntegral, ColIntCurveFitCollection
import matplotlib.pyplot as plt
import numpy as np
from data.hcb_ci_data import hcb_ci_data
from data.wright_ci_data import wright_ci_data

def ci_comparison_n2():
    """
    Compare the collision integrals as computed by Gupta Yos and Wright et. al.
    """

    temps = np.linspace(300, 3000, 250)
    ci = CollisionIntegral()

    gupta_yos_n2_11 = ci.construct_ci(ci_type="gupta_yos", curve_fit_type="pi_Omega",
                                   coeffs=[0.0, -0.0112, -0.1182, 4.8464])
    wright_et_al_n2_11 = ci.construct_ci(ci_type="curve_fit", curve_fit_type="Omega",
                                   temps=wright_ci_data["N2:N2"]["Omega_11"]["temps"],
                                   cis=wright_ci_data["N2:N2"]["Omega_11"]["cis"])
    gupta_yos_n2_22 = ci.construct_ci(ci_type="gupta_yos", curve_fit_type="pi_Omega",
                                   coeffs=[0.0, -0.0203, 0.0683, 4.09])
    wright_et_al_n2_22 = ci.construct_ci(ci_type="curve_fit", curve_fit_type="Omega",
                                   temps=wright_ci_data["N2:N2"]["Omega_22"]["temps"],
                                   cis=wright_ci_data["N2:N2"]["Omega_22"]["cis"])
    laricchiuta_n2_n2_11 = ci.construct_ci(ci_type="laricchiuta", l=1, s=1, alpha=1.71, N=8)
    laricchiuta_n2_n2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=1.71, N=8)

    wright_n2_n2_eval = wright_et_al_n2_11.eval(temps)
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("$N_2 - N_2 $ collision integrals")
    ax[0].plot(temps, laricchiuta_n2_n2_11.eval(temps), 'k', label="Laricchiuta")
    ax[0].plot(temps, gupta_yos_n2_11.eval(temps), 'k:', label="Gupta et al (Eilmer)")
    ax[0].plot(temps, wright_n2_n2_eval, 'k--', label="Wright et al")
    ax[0].fill_between(temps,
                       wright_n2_n2_eval - wright_n2_n2_eval*0.1,
                       wright_n2_n2_eval + wright_n2_n2_eval*0.1, alpha=0.5,
                       label="Wright et al uncertainty")
    ax[0].set_ylabel("$\\Omega^{(1,1)}$, $\\AA^2$")
    ax[0].legend()
    ax[0].set_ylim(bottom=0)
    ax[0].grid()
    wright_eval = wright_et_al_n2_22.eval(temps)
    ax[1].plot(temps, laricchiuta_n2_n2_22.eval(temps), 'k', label="Laricchiuta")
    ax[1].plot(temps, gupta_yos_n2_22.eval(temps), 'k:', label="Gupta et al (Eilmer)")
    ax[1].plot(temps, wright_eval, 'k--', label="Wright et al")
    ax[1].fill_between(temps, 0.9*wright_eval, 1.1*wright_eval, label="Wright et al uncertainty", alpha=0.5)
    ax[1].set_ylabel("$\\Omega^{(2,2)}$, $\\AA^2$")
    ax[1].set_xlabel("Temperature, K")
    ax[1].set_ylim(bottom=0)
    ax[1].grid()
    fig.savefig("./figs/N2_N2_comparison.png")

def ci_comparison_co2():
    temps = np.linspace(300, 3000, 250)
    ci = CollisionIntegral()

    # wright collision integrals curve fitted
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="Omega")

    # Laricchiuta collision integrals for co2:n2
    laricchiuta_co2_n2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2,
                                            alphas=[2.507, 1.71], Ns=[16, 10])
    laricchiuta_co2_n2_11 = ci.construct_ci(ci_type="laricchiuta", l=1, s=1,
                                            alphas=[2.507, 1.71], Ns=[16, 10])

    laricchiuta_co2_co2_11 = ci.construct_ci(ci_type="laricchiuta", l=1, s=1, alpha=2.507, N=16)
    laricchiuta_co2_co2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=2.507, N=16)

    laricchiuta_omega_co2_co2_11 = laricchiuta_co2_co2_11.eval(temps)
    laricchiuta_omega_co2_n2_11 = laricchiuta_co2_n2_11.eval(temps)
    laricchiuta_omega_co2_co2_22 = laricchiuta_co2_co2_22.eval(temps)
    laricchiuta_omega_co2_n2_22 = laricchiuta_co2_n2_22.eval(temps)

    wright_omega_co2_co2_11 = cfs.get_col_ints(pair="CO2:CO2", ci_type="Omega_11").eval(temps)
    wright_omega_co2_co2_22 = cfs.get_col_ints(pair="CO2:CO2", ci_type="Omega_22").eval(temps)
    wright_omega_co2_n2_11 = cfs.get_col_ints(pair="CO2:N2", ci_type="Omega_11").eval(temps)
    wright_omega_co2_n2_22 = cfs.get_col_ints(pair="CO2:N2", ci_type="Omega_22").eval(temps)

    fig_co2_co2, ax_co2_co2 = plt.subplots(2, 1, sharex=True)
    fig_co2_co2.suptitle("$CO_2$ - $CO_2$ Collision Integrals")
    ax_co2_co2[0].plot(temps, laricchiuta_omega_co2_co2_11, 'k', label="Laricchiuta")
    ax_co2_co2[0].plot(temps, wright_omega_co2_co2_11, 'k--', label="Wright et al (Eilmer)")
    ax_co2_co2[0].fill_between(temps, wright_omega_co2_co2_11*0.8, wright_omega_co2_co2_11*1.2,
                              alpha = 0.5, label="Wright et al. uncertainty")
    ax_co2_co2[0].set_ylabel("$\\Omega^{1,1}_{CO_2, CO_2}$, $\\AA^2$")
    ax_co2_co2[0].legend()
    ax_co2_co2[0].set_ylim(bottom=0)
    ax_co2_co2[0].grid()

    ax_co2_co2[1].plot(temps, laricchiuta_omega_co2_co2_22, 'k', label="Laricchiuta")
    ax_co2_co2[1].plot(temps, wright_omega_co2_co2_22, 'k--', label="Wright et al (Eilmer)")
    ax_co2_co2[1].fill_between(temps, wright_omega_co2_co2_22*0.8, wright_omega_co2_co2_22*1.2,
                              alpha = 0.5, label="Wright et al. uncertainty")
    ax_co2_co2[1].set_xlabel("Temperature [K]")
    ax_co2_co2[1].set_ylabel("$\\Omega^{2,2}_{CO_2, CO_2}$, $\\AA^2$")
    ax_co2_co2[1].set_ylim(bottom=0)
    ax_co2_co2[1].grid()

    fig_co2_n2, ax_co2_n2 = plt.subplots(2, 1, sharex=True)
    fig_co2_n2.suptitle("$CO_2$ - $N_2$ Collision Integrals")
    ax_co2_n2[0].plot(temps, laricchiuta_omega_co2_n2_11, 'k', label="Laricchiuta")
    ax_co2_n2[0].plot(temps, wright_omega_co2_n2_11, 'k--', label="Wright et al")
    ax_co2_n2[0].fill_between(temps, wright_omega_co2_n2_11*0.8, wright_omega_co2_n2_11*1.2,
                              alpha = 0.5, label="Wright et al. uncertainty")
    ax_co2_n2[0].set_ylabel("$\\Omega^{1,1}_{CO_2, N_2}$, $\\AA^2$")
    ax_co2_n2[0].legend()
    ax_co2_n2[0].set_ylim(bottom=0)
    ax_co2_n2[0].grid()

    ax_co2_n2[1].plot(temps, laricchiuta_omega_co2_n2_22, 'k', label="Laricchiuta")
    ax_co2_n2[1].plot(temps, wright_omega_co2_n2_22, 'k--', label="Wright et al")
    ax_co2_n2[1].fill_between(temps, wright_omega_co2_n2_22*0.8, wright_omega_co2_n2_22*1.2,
                              alpha = 0.5, label="Wright et al. uncertainty")
    ax_co2_n2[1].set_xlabel("Temperature [K]")
    ax_co2_n2[1].set_ylabel("$\\Omega^{2,2}_{CO_2, N_2}$, $\\AA^2$")
    ax_co2_n2[1].set_ylim(bottom=0)
    ax_co2_n2[1].grid()

    fig_co2_co2.savefig("./figs/CO2_CO2_comparison.png")
    fig_co2_n2.savefig("./figs/CO2_N2_comparison.png")

def plot_curve_fit_data():
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="Omega")
    co2_co2_ci = cfs.get_col_ints(pair="CO2:CO2", ci_type="Omega_11")
    wright_co2_co2_temps = co2_co2_ci._temps
    wright_co2_co2_cis = co2_co2_ci._cis

    temps = np.linspace(300, 20000)
    curve_fit_ci = co2_co2_ci.eval(temps)

    fig, ax = plt.subplots()
    ax.plot(wright_co2_co2_temps, wright_co2_co2_cis, label="Wright et al")
    ax.plot(temps, curve_fit_ci, label="Curve fit")
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("$\\Omega^{(1,1)}_{CO_2, CO_2}$")

def numeric_deflection_integrand(impact, rel_vel):
    ci = CollisionIntegral()
    numeric_ci = ci.construct_ci(ci_type="numerical",
                                 sigma=3.7, epsilon=244,
                                 l=1, s=1,
                                 potential="lennard_jones",
                                 mu=0.02)
    r_m = numeric_ci._calc_r_m(impact, rel_vel)
    radii = np.linspace(r_m, 100, 100)
    integrand = numeric_ci._deflection_integrand(radii, impact, rel_vel)
    fig, ax = plt.subplots()
    ax.plot(radii, integrand)
    ax.set_xlabel("radius, $\AA$")
    ax.set_ylabel("deflection integrand")

def numeric_deflection_angle(ci, impacts, rel_vel, n=100):
    deflection = np.zeros_like(impacts)
    for i, impact in enumerate(impacts):
        deflection[i] = ci._deflection_angle(impact, rel_vel)
    return deflection

def numeric_deflection_angles(rel_vels):
    ci = CollisionIntegral()
    numeric_ci = ci.construct_ci(ci_type="numerical",
                                 sigma=3.7, epsilon=244,
                                 l=1, s=1,
                                 potential="lennard_jones",
                                 mu=0.02)

    fig, ax = plt.subplots()
    impacts = np.linspace(0, 20, 1000)
    for i, rel_vel in enumerate(rel_vels):
        deflection_angles = numeric_deflection_angle(numeric_ci, impacts, rel_vel, 100)
        ax.plot(impacts, 1-np.cos(deflection_angles), label=f"g = {rel_vel}")
    ax.legend()
    ax.set_xlabel("impact parameter $\AA$")
    ax.set_ylabel("$1-\\cos\\Theta$")

def plot_numeric_cross_section(l=1):
    ci = CollisionIntegral()
    numeric_ci = ci.construct_ci(ci_type="numerical",
                                 potential="lennard_jones",
                                 sigma=3.7, epsilon=244, mu=0.02,
                                 l=l, s=l)
    rel_vels = np.linspace(10, 500)
    numeric_ci._collision_cross_section(rel_vels)

if __name__ == "__main__":
    # plot_curve_fit_data()
    ci_comparison_co2()
    ci_comparison_n2()

    # numeric_deflection_integrand(17, 10)
    # numeric_deflection_angles([10, 100, 500])
    #plot_numeric_cross_section()
    plt.show()
