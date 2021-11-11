import numpy as np
from uncertainties import unumpy as unp
import uncertainties as unct
from matplotlib import pyplot as plt
from data.wright_ci_data import wright_ci_data
from collision_integrals import CollisionIntegral, ColIntCurveFitCollection

NA = 6.022e23
kB = 1.38e-23
def viscosity(T, ci, molar_mass):
    return 2.6693e-5 * unp.sqrt(molar_mass * T) / ci

def co2_n2_mixture_viscosity(T, cis):
    """
    cis should be: [ci_CO2_CO2, ci_N2_N2, ci_N2_CO2]
    """
    M_N2 = 28
    M_CO2 = 44
    delta_2_CO2_CO2 = 16/5 * unp.sqrt((2 * M_N2 * M_CO2)/(np.pi*NA*kB*T*(M_N2+M_CO2))) * cis[0] * np.pi
    delta_2_N2_N2 = 16/5 * unp.sqrt((2 * M_N2 * M_CO2)/(np.pi*NA*kB*T*(M_N2+M_CO2))) * cis[1] * np.pi
    delta_2_N2_CO2 = 16/5 * unp.sqrt((2 * M_N2 * M_CO2)/(np.pi*NA*kB*T*(M_N2+M_CO2))) * cis[2] * np.pi

    mu = M_CO2 * 0.5 / (0.5 * delta_2_CO2_CO2 + 0.5 * delta_2_N2_CO2)
    mu += M_N2 * 0.5 / (0.5 * delta_2_N2_CO2 + 0.5 * delta_2_N2_N2)
    return mu

def plot_co2_n2_viscosity():
    ci = CollisionIntegral()
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="Omega")
    temps = np.linspace(300, 3000, 100)

    laricchiuta_co2_co2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=2.507, N=16).eval(temps)
    wright_omega_co2_co2_22 = cfs.get_col_ints(pair="CO2:CO2", ci_type="Omega_22").eval(temps)
    wright_co2_co2_unct = unp.uarray(wright_omega_co2_co2_22, 0.2*wright_omega_co2_co2_22)

    laricchiuta_n2_n2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=1.71, N=8).eval(temps)
    wright_et_al_n2_22 = ci.construct_ci(ci_type="curve_fit", curve_fit_type="Omega",
                                   temps=wright_ci_data["N2:N2"]["Omega_22"]["temps"],
                                   cis=wright_ci_data["N2:N2"]["Omega_22"]["cis"]).eval(temps)
    wright_n2_n2_unct = unp.uarray(wright_et_al_n2_22, 0.1*wright_et_al_n2_22)

    laricchiuta_co2_n2_22 = ci.construct_ci(ci_type="laricchiuta", l=2, s=2,
                                            alphas=(2.507, 1.71), Ns=(16, 10)).eval(temps)
    wright_omega_co2_n2_22 = cfs.get_col_ints(pair="CO2:N2", ci_type="Omega_22").eval(temps)
    wright_co2_n2_unct = unp.uarray(wright_omega_co2_n2_22, 0.2*wright_omega_co2_n2_22)

    wright_viscosity = co2_n2_mixture_viscosity(temps, (wright_co2_co2_unct,
                                                wright_n2_n2_unct, wright_co2_n2_unct))
    laricchiuta_viscosity = co2_n2_mixture_viscosity(temps, (laricchiuta_co2_co2_22,
                                                     laricchiuta_co2_n2_22,
                                                     laricchiuta_n2_n2_22))
    fig, ax = plt.subplots()
    ax.plot(temps, laricchiuta_viscosity, 'k', label="Laricchiuta")
    ax.plot(temps, unp.nominal_values(wright_viscosity), 'k--', label="Wright et al")
    ax.legend()
    ax.fill_between(temps, unp.nominal_values(wright_viscosity) - unp.std_devs(wright_viscosity),
                    unp.nominal_values(wright_viscosity) +unp.std_devs(wright_viscosity),
                    alpha=0.5)

def plot_n2_viscosity():
    ci = CollisionIntegral()
    gupta_yos = ci.construct_ci(ci_type="gupta_yos", curve_fit_type="pi_Omega",
                                    coeffs=[0.0, -0.0203, 0.0683, 4.09])

    wright_et_al = ci.construct_ci(ci_type="curve_fit", curve_fit_type="Omega",
                                   temps=wright_ci_data["N2:N2"]["Omega_22"]["temps"],
                                   cis=wright_ci_data["N2:N2"]["Omega_22"]["cis"])
    laricchiuta = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=1.71, N=8)

    temps = np.linspace(300, 3000, 100)
    gupta_eval = viscosity(temps, gupta_yos.eval(temps), 28)
    laricchiuta_eval = viscosity(temps, laricchiuta.eval(temps), 28)
    wright_eval = wright_et_al.eval(temps)
    wright_eval_unct = unp.uarray(wright_eval, 0.1*wright_eval)
    wright_eval_unct = viscosity(temps, wright_eval_unct, 28)

    fig, ax = plt.subplots()
    ax.plot(temps, laricchiuta_eval, 'k', label="laricchiuta")
    ax.plot(temps, gupta_eval, 'k:', label="Gupta, Yos, Thompson (Eilmer)")
    ax.plot(temps, unp.nominal_values(wright_eval_unct), 'k--', label="Wright et al")
    ax.fill_between(temps,
                    unp.nominal_values(wright_eval_unct)-unp.std_devs(wright_eval_unct),
                    unp.nominal_values(wright_eval_unct)+unp.std_devs(wright_eval_unct),
                    alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Viscosity [g/cm/s]")
    ax.set_title("$N_2$ viscosity")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.savefig("../04_talks/collision_integrals/figs/N2_viscosity.png")

def plot_co2_viscosity():
    ci = CollisionIntegral()
    temps = np.linspace(300, 3000, 100)
    # wright collision integrals curve fitted
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="Omega")

    laricchiuta = ci.construct_ci(ci_type="laricchiuta", l=2, s=2, alpha=2.507, N=16)

    wright_eval = cfs.get_col_ints(pair="CO2:CO2", ci_type="Omega_22").eval(temps)

    laricchiuta_eval = viscosity(temps, laricchiuta.eval(temps), 44)
    wright_eval_unct = unp.uarray(wright_eval, 0.2*wright_eval)
    wright_eval_unct = viscosity(temps, wright_eval_unct, 44)

    fig, ax = plt.subplots()
    ax.plot(temps, laricchiuta_eval, 'k', label="laricchiuta")
    ax.plot(temps, unp.nominal_values(wright_eval_unct), 'k--', label="Wright et al")
    ax.fill_between(temps,
                    unp.nominal_values(wright_eval_unct)-unp.std_devs(wright_eval_unct),
                    unp.nominal_values(wright_eval_unct)+unp.std_devs(wright_eval_unct),
                    alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.legend()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Visocisity [g/cm/s]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title("$CO_2$ viscosity")
    ax.grid()
    plt.savefig("../04_talks/collision_integrals/figs/CO2_viscosity.png")

if __name__ == "__main__":
    plot_n2_viscosity()
    plot_co2_viscosity()
    plot_co2_n2_viscosity()
    plt.show()
