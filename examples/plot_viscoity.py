import numpy as np
from uncertainties import unumpy as unp
import uncertainties as unct
from matplotlib import pyplot as plt
from data.wright_ci_data import wright_ci_data
from collision_integrals import collision_integral, ColIntCurveFitCollection
from transport_properties import TwoTempTransProp
from examples.gas_models.N2 import nitrogen
from examples.gas_models.CO2 import carbon_dioxide
from ci_models import ci_models_laricchiuta, ci_models_wright


def plot_n2_viscosity():
    wright = TwoTempTransProp(nitrogen, ci_models_wright)
    gupta = TwoTempTransProp(nitrogen)
    laricchiuta = TwoTempTransProp(nitrogen, ci_models_laricchiuta)

    temps = np.linspace(300, 3000, 100)
    gas_state = {"molef": {"N2": 1.0}}

    wright_mus, gupta_mus, laricchiuta_mus = [], [], []
    for temp in temps:
        gas_state["temp"] = temp
        wright_mus.append(wright.viscosity(gas_state))
        gupta_mus.append(gupta.viscosity(gas_state))
        laricchiuta_mus.append(laricchiuta.viscosity(gas_state))

    fig, ax = plt.subplots()
    l, = ax.plot(temps, laricchiuta_mus, 'k')
    g, = ax.plot(temps, gupta_mus, 'k:')
    w, = ax.plot(temps, unp.nominal_values(wright_mus), 'k--')
    w_fill = ax.fill_between(temps,
                                unp.nominal_values(wright_mus)-unp.std_devs(wright_mus),
                                unp.nominal_values(wright_mus)+unp.std_devs(wright_mus),
                                alpha=0.5)
    ax.set_ylim(bottom=0)
    ax.legend([l, g, (w, w_fill)], ["Laricchiuta", "Gupta, Yos, Thomson (Eilmer)", "Wright"])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Viscosity [g/cm/s]")
    ax.set_title("$N_2$ viscosity")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.savefig("./figs/N2_viscosity.png")

def plot_co2_viscosity():
    temps = np.linspace(300, 3000, 100)
    gas_state = {"molef": {"CO2": 1.0}}
    wright = TwoTempTransProp(carbon_dioxide, {("CO2", "CO2"): ("wright", {"eval_acc": True})})
    laricchiuta = TwoTempTransProp(carbon_dioxide, {("CO2", "CO2"): ("laricchiuta", {})})

    mu_l = []
    mu_w = []
    for temp in temps:
        gas_state["temp"] = temp
        mu_l.append(laricchiuta.viscosity(gas_state))
        mu_w.append(wright.viscosity(gas_state))

    fig, ax = plt.subplots()
    l, = ax.plot(temps, mu_l, 'k')
    w, = ax.plot(temps, unp.nominal_values(mu_w), 'k--')
    w_fill = ax.fill_between(temps,
                    unp.nominal_values(mu_w)-unp.std_devs(mu_w),
                    unp.nominal_values(mu_w)+unp.std_devs(mu_w),
                    alpha=0.5)
    ax.set_ylim(bottom=0)
    plt.legend([l, (w, w_fill)], ["Laricchiuta", "Wright"])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Visocisity [g/cm/s]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title("$CO_2$ viscosity")
    ax.grid()
    plt.savefig("./figs/CO2_viscosity.png")

if __name__ == "__main__":
    plot_n2_viscosity()
    plot_co2_viscosity()
    plt.show()
