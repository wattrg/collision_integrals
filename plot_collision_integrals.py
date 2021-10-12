from collision_integrals import CollisionIntegral, ColIntCurveFitCollection
import matplotlib.pyplot as plt
import numpy as np
from hcb_ci_data import hcb_ci_data
from wright_ci_data import wright_ci_data

def ci_comparison():
    temps = np.linspace(300, 30000, 250)
    ci = CollisionIntegral()
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
    hcb_co2_co2_11 = ci.construct_ci(ci_type="hcb",
                                     sigma=3.763, epsilon=244, mu=0.04401*0.5,
                                     l=1, s=1)
    hcb_co2_co2_22 = ci.construct_ci(ci_type="hcb",
                                     sigma=3.763, epsilon=244, mu=0.04401*0.5,
                                     l=2, s=2)
    hcb_co2_n2_11 = ci.construct_ci(ci_type="hcb",
                                    sigma=3.692, epsilon=154.26, mu=0.01712,
                                    l=1, s=1)
    hcb_co2_n2_22 = ci.construct_ci(ci_type="hcb",
                                    sigma=3.692, epsilon=154.26, mu=0.01712,
                                    l=2, s=2)
    laricchiuta_co2_co2_11 = ci.construct_ci(ci_type="laricchiuta",
                                          l=1, s=1, beta=10,
                                          sigma=3.763, epsilon=244)
    laricchiuta_co2_n2_11 = ci.construct_ci(ci_type="laricchiuta",
                                          l=1, s=1, beta=10,
                                            sigma=3.692, epsilon=154.26)
    laricchiuta_co2_co2_22 = ci.construct_ci(ci_type="laricchiuta",
                                          l=2, s=2, beta=10,
                                          sigma=3.763, epsilon=244)
    laricchiuta_co2_n2_22 = ci.construct_ci(ci_type="laricchiuta",
                                          l=2, s=2, beta=10,
                                            sigma=3.692, epsilon=154.26)

    laricchiuta_omega_co2_co2_11 = laricchiuta_co2_co2_11.eval(temps)
    laricchiuta_omega_co2_n2_11 = laricchiuta_co2_n2_11.eval(temps)
    laricchiuta_omega_co2_co2_22 = laricchiuta_co2_co2_11.eval(temps)
    laricchiuta_omega_co2_n2_22 = laricchiuta_co2_n2_11.eval(temps)
    hcb_omega_co2_co2_11 = hcb_co2_co2_11.eval(temps)
    hcb_omega_co2_co2_22 = hcb_co2_co2_22.eval(temps)
    hcb_omega_co2_n2_11 = hcb_co2_n2_11.eval(temps)
    hcb_omega_co2_n2_22 = hcb_co2_n2_22.eval(temps)
    wright_omega_co2_co2_11 = cfs.get_col_ints(pair="CO2:CO2", ci_type="pi_Omega_11").eval(temps)
    wright_omega_co2_co2_22 = cfs.get_col_ints(pair="CO2:CO2", ci_type="pi_Omega_22").eval(temps)
    wright_omega_co2_n2_11 = cfs.get_col_ints(pair="CO2:N2", ci_type="pi_Omega_11").eval(temps)
    wright_omega_co2_n2_22 = cfs.get_col_ints(pair="CO2:N2", ci_type="pi_Omega_22").eval(temps)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(temps, hcb_omega_co2_co2_11, label="HCB")
    ax[0, 0].plot(temps, wright_omega_co2_co2_11, label="Wright et al")
    ax[0, 0].plot(temps, laricchiuta_omega_co2_co2_11, label="Laricchiuta")
    ax[0, 0].set_xlabel("Temperature [K]")
    ax[0, 0].set_ylabel("$\\Omega^{1,1}_{CO_2, CO_2}$")
    ax[0, 0].legend()
    ax[0 ,1].plot(temps, hcb_omega_co2_co2_22, label="HDB")
    ax[0 ,1].plot(temps, wright_omega_co2_co2_22, label="Wright et al")
    ax[0, 1].plot(temps, laricchiuta_omega_co2_co2_22, label="Laricchiuta")
    ax[0 ,1].set_xlabel("Temperature [K]")
    ax[0 ,1].set_ylabel("$\\Omega^{2,2}_{CO_2, CO_2}$")
    ax[0 ,1].legend()
    ax[1 ,0].plot(temps, hcb_omega_co2_n2_11, label="HCB")
    ax[1 ,0].plot(temps, wright_omega_co2_n2_11, label="Wright et al")
    ax[1, 0].plot(temps, laricchiuta_omega_co2_n2_11, label="Laricchiuta")
    ax[1 ,0].set_xlabel("Temperature [K]")
    ax[1 ,0].set_ylabel("$\\Omega^{1,1}_{CO_2, N_2}$")
    ax[1 ,0].legend()
    ax[1 ,1].plot(temps, hcb_omega_co2_n2_22, label="HCB")
    ax[1 ,1].plot(temps, wright_omega_co2_n2_22, label="Wright et al")
    ax[1, 1].plot(temps, laricchiuta_omega_co2_n2_22, label="Laricchiuta")
    ax[1 ,1].set_xlabel("Temperature [K]")
    ax[1 ,1].set_ylabel("$\\Omega^{2,2}_{CO_2, N_2}$")
    ax[1 ,1].legend()

def plot_curve_fit_data():
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
    co2_co2_ci = cfs.get_col_ints(pair="CO2:CO2", ci_type="pi_Omega_11")
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
    # ci_comparison()

    # numeric_deflection_integrand(17, 10)
    # numeric_deflection_angles([10, 100, 500])
    plot_numeric_cross_section()
    plt.show()
