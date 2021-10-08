from collision_integrals import CollisionIntegral, ColIntCurveFitCollection
import matplotlib.pyplot as plt
import numpy as np
from hcb_ci_data import hcb_ci_data
from wright_ci_data import wright_ci_data

def plot_dimensionless_ci():
    ci = CollisionIntegral()
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
    hcb_co2_co2_11 = ci.construct_ci(ci_type="dimensionless",
                                     sigma=3.763, epsilon=244, mu=0.04401*0.5,
                                     l=1, s=1)
    hcb_co2_co2_22 = ci.construct_ci(ci_type="dimensionless",
                                     sigma=3.763, epsilon=244, mu=0.04401*0.5,
                                     l=2, s=2)
    hcb_co2_n2_11 = ci.construct_ci(ci_type="dimensionless",
                                    sigma=3.692, epsilon=154.26, mu=0.01712,
                                    l=1, s=1)
    hcb_co2_n2_22 = ci.construct_ci(ci_type="dimensionless",
                                    sigma=3.692, epsilon=154.26, mu=0.01712,
                                    l=2, s=2)
    temps = np.linspace(300, 20000, 150)
    hcb_omega_co2_co2_11 = hcb_co2_co2_11.eval(temps)
    hcb_omega_co2_co2_22 = hcb_co2_co2_22.eval(temps)
    hcb_omega_co2_n2_11 = hcb_co2_n2_11.eval(temps)
    hcb_omega_co2_n2_22 = hcb_co2_n2_22.eval(temps)
    wright_omega_11 = cfs.get_cis(pair="CO2:CO2", ci_type="pi_Omega_11").eval(temps)
    wright_omega_22 = cfs.get_cis(pair="CO2:CO2", ci_type="pi_Omega_22").eval(temps)
    wright_omega_co2_n2_11 = cfs.get_cis(pair="CO2:N2", ci_type="pi_Omega_11").eval(temps)
    wright_omega_co2_n2_22 = cfs.get_cis(pair="CO2:N2", ci_type="pi_Omega_22").eval(temps)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(temps, hcb_omega_co2_co2_11, label="dimensionless")
    ax[0, 0].plot(temps, wright_omega_11, label="Wright et al")
    ax[0, 0].set_xlabel("Temperature [K]")
    ax[0, 0].set_ylabel("$\\Omega^{1,1}_{CO_2, CO_2}$")
    ax[0, 0].legend()
    ax[0 ,1].plot(temps, hcb_omega_co2_co2_22, label="dimensionless")
    ax[0 ,1].plot(temps, wright_omega_22, label="Wright et al")
    ax[0 ,1].set_xlabel("Temperature [K]")
    ax[0 ,1].set_ylabel("$\\Omega^{2,2}_{CO_2, CO_2}$")
    ax[0 ,1].legend()
    ax[1 ,0].plot(temps, hcb_omega_co2_n2_11, label="dimensionless")
    ax[1 ,0].plot(temps, wright_omega_co2_n2_11, label="Wright et al")
    ax[1 ,0].set_xlabel("Temperature [K]")
    ax[1 ,0].set_ylabel("$\\Omega^{1,1}_{CO_2, N_2}$")
    ax[1 ,0].legend()
    ax[1 ,1].plot(temps, hcb_omega_co2_n2_22, label="dimensionless")
    ax[1 ,1].plot(temps, wright_omega_co2_n2_22, label="Wright et al")
    ax[1 ,1].set_xlabel("Temperature [K]")
    ax[1 ,1].set_ylabel("$\\Omega^{2,2}_{CO_2, N_2}$")
    ax[1 ,1].legend()

def plot_curve_fit_data():
    cfs = ColIntCurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
    co2_co2_ci = cfs.get_cis(pair="CO2:CO2", ci_type="pi_Omega_11")
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


if __name__ == "__main__":
    plot_curve_fit_data()
    plot_dimensionless_ci()
    plt.show()
