from eilmer.gas import GasModel, GasState
from transprop.collision_integrals import collision_integral, ColIntCurveFitCollection
import matplotlib.pyplot as plt
import numpy as np
from data.wright_ci_data import wright_ci_data


def ci_comparison_n2():
    """
    Compare the collision integrals as computed by Gupta Yos and Wright et. al.
    """

    temps = np.linspace(300, 3000, 250)
    gas_state = {"temp": temps}

    gupta_yos_n2_11 = collision_integral("gupta_yos", order=(1,1), curve_fit_type="pi_Omega",
                                         coeffs=[0.0, -0.0112, -0.1182, 4.8464],
                                         species=("N2", "N2"))
    wright_et_al_n2_11 = collision_integral("wright", order=(1,1), species=("N2", "N2"))
    gupta_yos_n2_22 = collision_integral("gupta_yos", order=(2,2), curve_fit_type="pi_Omega",
                                         coeffs=[0.0, -0.0203, 0.0683, 4.09],
                                         species=("N2", "N2"))
    wright_et_al_n2_22 = collision_integral("wright", order=(2,2), species=("N2", "N2"))
    laricchiuta_n2_n2_11 = collision_integral("laricchiuta", order=(1,1), alpha=1.71, N=8,
                                              species=("N2", "N2"))
    laricchiuta_n2_n2_22 = collision_integral("laricchiuta", order=(2,2), alpha=1.71, N=8,
                                              species=("N2", "N2"))

    wright_n2_n2_eval = wright_et_al_n2_11.eval(gas_state)
    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("$N_2 - N_2 $ collision integrals")
    ax[0].plot(temps, laricchiuta_n2_n2_11.eval(gas_state), 'k', label="Laricchiuta")
    ax[0].plot(temps, gupta_yos_n2_11.eval(gas_state), 'k:', label="Gupta et al (Eilmer)")
    ax[0].plot(temps, wright_n2_n2_eval, 'k--', label="Wright et al")
    ax[0].fill_between(temps,
                       wright_n2_n2_eval - wright_n2_n2_eval*0.1,
                       wright_n2_n2_eval + wright_n2_n2_eval*0.1, alpha=0.5,
                       label="Wright et al uncertainty")
    ax[0].set_ylabel("$\\Omega^{(1,1)}$, $\\AA^2$")
    ax[0].legend()
    ax[0].set_ylim(bottom=0)
    ax[0].grid()
    wright_eval = wright_et_al_n2_22.eval(gas_state)
    ax[1].plot(temps, laricchiuta_n2_n2_22.eval(gas_state), 'k', label="Laricchiuta")
    ax[1].plot(temps, gupta_yos_n2_22.eval(gas_state), 'k:', label="Gupta et al (Eilmer)")
    ax[1].plot(temps, wright_eval, 'k--', label="Wright et al")
    ax[1].fill_between(temps, 0.9*wright_eval, 1.1*wright_eval, label="Wright et al uncertainty", alpha=0.5)
    ax[1].set_ylabel("$\\Omega^{(2,2)}$, $\\AA^2$")
    ax[1].set_xlabel("Temperature, K")
    ax[1].set_ylim(bottom=0)
    ax[1].grid()
    fig.savefig("./figs/N2_N2_comparison.svg")

def ci_comparison_co2():
    temps = np.linspace(300, 3000, 250)
    gas_state = {"temp": temps}

    # Laricchiuta collision integrals for co2:n2
    laricchiuta_co2_n2_22 = collision_integral("laricchiuta", order=(2,2),
                                               alphas=[2.507, 1.71], Ns=[16, 10],
                                               species=("CO2", "N2"))
    laricchiuta_co2_n2_11 = collision_integral("laricchiuta", order=(1,1),
                                               alphas=[2.507, 1.71], Ns=[16, 10],
                                               species=("CO2", "N2"))

    laricchiuta_co2_co2_11 = collision_integral("laricchiuta", order=(1,1), alpha=2.507, N=16,
                                                species=("CO2", "CO2"))
    laricchiuta_co2_co2_22 = collision_integral("laricchiuta", order=(2,2), alpha=2.507, N=16,
                                                species=("CO2", "CO2"))

    laricchiuta_omega_co2_co2_11 = laricchiuta_co2_co2_11.eval(gas_state)
    laricchiuta_omega_co2_n2_11 = laricchiuta_co2_n2_11.eval(gas_state)
    laricchiuta_omega_co2_co2_22 = laricchiuta_co2_co2_22.eval(gas_state)
    laricchiuta_omega_co2_n2_22 = laricchiuta_co2_n2_22.eval(gas_state)

    wright_omega_co2_co2_11 = collision_integral("wright", species=("CO2", "CO2"), order=(1,1)).eval(gas_state)
    wright_omega_co2_co2_22 = collision_integral("wright", species=("CO2", "CO2"), order=(2,2)).eval(gas_state)
    wright_omega_co2_n2_11 = collision_integral("wright", species=("CO2", "N2"), order=(1,1)).eval(gas_state)
    wright_omega_co2_n2_22 = collision_integral("wright", species=("CO2", "N2"), order=(2,2)).eval(gas_state)

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

    fig_co2_co2.savefig("./figs/CO2_CO2_comparison.svg")
    fig_co2_n2.savefig("./figs/CO2_N2_comparison.svg")

def ne_from_ep(gas_state):
        kB = 1.38066e-23
        temp = gas_state["temp"]
        pressure = gas_state["ep"]
        gas_state["ne"] = pressure / (kB * temp)

def ep_from_ne(gas_state):
    kB = 1.38066e-23
    ne = gas_state["ne"]
    temp = gas_state["temp"]
    gas_state["ep"] = ne * kB * temp

def plot_N2p_N2p_interaction():
    # construct collision integrals
    gupta_n2p_n2p_11 = collision_integral("gupta_yos", charge=(1,1), order=(1,1),
                                          curve_fit_type="Omega",
                                          coeffs=[0.1251, -3.5134, 31.2277, -80.6515],
                                          species=("N2+", "N2+"))
    gupta_n2p_n2p_22 = collision_integral("gupta_yos", charge=(1,1), order=(2,2),
                                          curve_fit_type="pi_Omega",
                                          coeffs=[0.1251, -3.5135, 31.2281, -80.1163],
                                          species=("N2+", "N2+"))
    mason_n2p_n2p_11 = collision_integral("mason", order=(1,1), charge=(1, 1),
                                          species=("N2+", "N2+"))
    mason_n2p_n2p_22 = collision_integral("mason", order=(2,2), charge=(2, 2),
                                          species=("N2+", "N2+"))

    # setup the gas state
    temps = np.linspace(300, 15000, 100)
    #gas_state = {"temp": temps, "ne": 1e20}
    gas_state = {"temp": temps, "ep": 101325*0.01}
    #ep_from_ne(gas_state)
    ne_from_ep(gas_state)

    # plot results
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(temps, gupta_n2p_n2p_11.eval(gas_state), 'k',
               label="Gupta et al")
    ax[0].plot(temps, mason_n2p_n2p_11.eval(gas_state), 'k--', label="Mason")
    ax[0].set_ylabel("$\\Omega^{(1,1)}$")
    ax[0].legend()
    ax[1].plot(temps, gupta_n2p_n2p_22.eval(gas_state), 'k',
               label="Gupta et al")
    ax[1].plot(temps, mason_n2p_n2p_22.eval(gas_state), 'k--', label="Mason")
    ax[1].set_ylabel("$\\Omega^{(2,2)}$")


def plot_curve_fit_data():
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="Omega")
    co2_co2_ci = collision_integral("wright", species=("CO2", "CO2"), order=(1,1))
    wright_co2_co2_temps = co2_co2_ci._temps
    wright_co2_co2_cis = co2_co2_ci._cis

    temps = np.linspace(300, 20000)
    gas_state = {"temp": temps}
    curve_fit_ci = co2_co2_ci.eval(gas_state)

    fig, ax = plt.subplots()
    ax.plot(wright_co2_co2_temps, wright_co2_co2_cis, label="Wright et al")
    ax.plot(temps, curve_fit_ci, label="Curve fit")
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("$\\Omega^{(1,1)}_{CO_2, CO_2}$")


if __name__ == "__main__":
    plot_curve_fit_data()
    ci_comparison_co2()
    ci_comparison_n2()
    plot_N2p_N2p_interaction()

    plt.show()
