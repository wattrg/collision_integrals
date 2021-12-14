import numpy as np
from uncertainties import unumpy as unp
import uncertainties as unct
from matplotlib import pyplot as plt
from transport_properties import TwoTempTransProp
from examples.gas_models.N2 import nitrogen
from examples.gas_models.CO2 import carbon_dioxide
from examples.gas_models.nitrogen_oxygen import nitrogen_oxygen
from ci_models import ci_models_laricchiuta, ci_models_wright, ci_models_laricchiuta_non_polar
from examples.gas_models.air_5_species import air_5_species
from eilmer.gas import GasModel, GasState


def plot_n2_viscosity():
    wright = TwoTempTransProp(nitrogen, {("N2", "N2"): ("wright", {"eval_acc": True})})
    gupta = TwoTempTransProp(nitrogen)
    laricchiuta = TwoTempTransProp(nitrogen,
                                   {("N2", "N2"): ("laricchiuta", {"param_priority": "polarisability"})})

    temps = np.linspace(300, 30000, 100)
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
    ax.set_ylabel("Viscosity [kg/m/s]")
    ax.set_title("$N_2$ viscosity")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.grid()
    plt.savefig("./figs/N2_viscosity.svg")

def plot_co2_viscosity():
    temps = np.linspace(300, 30000, 100)
    gas_state = {"molef": {"CO2": 1.0}}
    wright = TwoTempTransProp(carbon_dioxide, {("CO2", "CO2"): ("wright", {"eval_acc": True})})
    laricchiuta = TwoTempTransProp(carbon_dioxide, {("CO2", "CO2"): ("laricchiuta", {"param_priority": "polarisability"})})

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
    plt.legend([l, (w, w_fill)], ["Laricchiuta", "Wright (Eilmer)"])
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Visocisity [kg/m/s]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax.set_title("$CO_2$ viscosity")
    ax.grid()
    plt.savefig("./figs/CO2_viscosity.svg")

def plot_5_species_air_mixture():
    gupta_trans_prop = TwoTempTransProp(air_5_species)
    wright_trans_prop = TwoTempTransProp(air_5_species, ci_models_wright)
    laricchiuta_trans_prop = TwoTempTransProp(air_5_species, ci_models_laricchiuta)
    laricchiuta_trans_prop_np = TwoTempTransProp(air_5_species, ci_models_laricchiuta_non_polar)

    gmodel = GasModel("examples/two_temp_gas_5_species.lua")
    gs = GasState(gmodel)
    gs.p = 1e5
    #gs.massf = {"N2": 0.8, "O2": 0.1, "O": 0.05, "N": 0.05}
    gs.massf = {"N2": 0.8, "O2": 0.2}
    molef = gs.molef_as_dict
    gas_state = {"molef": molef}
    temps = np.linspace(300, 30000, 100)
    eilmer_mus = []
    gupta_mus = []
    wright_mus = []
    laricchiuta_mus = []
    laricchiuta_np_mus = []
    for temp in temps:
        # update gas state
        gs.T = temp
        gs.T_modes = [temp]
        gs.update_thermo_from_pT()
        gas_state["temp"] = temp

        # update viscosity
        gs.update_trans_coeffs()
        eilmer_mus.append(gs.mu)
        gupta_mus.append(gupta_trans_prop.viscosity(gas_state))
        wright_mus.append(wright_trans_prop.viscosity(gas_state))
        laricchiuta_mus.append(laricchiuta_trans_prop.viscosity(gas_state))
        laricchiuta_np_mus.append(laricchiuta_trans_prop_np.viscosity(gas_state))

    fig, ax = plt.subplots()
    line_e, = ax.plot(temps, eilmer_mus, 'k--')
    line_w, = ax.plot(temps, unp.nominal_values(wright_mus), 'k')
    fill_w = ax.fill_between(temps,
                     unp.nominal_values(wright_mus)+unp.std_devs(wright_mus),
                     unp.nominal_values(wright_mus)-unp.std_devs(wright_mus),
                     alpha=0.5)
    line_l, = ax.plot(temps, laricchiuta_mus, 'k:')
    line_lnp, = ax.plot(temps, laricchiuta_np_mus, 'k-.')
    ax.legend([line_e, (line_w, fill_w), line_l, line_lnp],
              ["Gupta-Yos/Eilmer", "Wright", r"Laricchiuta ($\alpha$, $N$)", r"Laricchiuta ($\sigma$, $\epsilon$ from Eilmer)"])
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"Viscosity $[kg/(m \cdot s)]$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("./figs/air_viscosity.svg")

def plot_co2_n2_viscosity():
    wright_ci_models = {
        ("CO2", "CO2"): ("wright", {"eval_acc": True}),
        ("N2", "N2"): ("wright", {"eval_acc": True}),
        ("CO2", "N2"): ("wright", {"eval_acc": True})
    }
    laricchiuta_ci_models = {
        ("CO2", "CO2"): ("laricchiuta", {"param_priority": "polarisability"}),
        ("N2", "N2"):   ("laricchiuta", {"param_priority": "polarisability"}),
        ("CO2", "N2"):  ("laricchiuta", {"param_priority": "polarisability"})
    }
    trans_prop_laricchiuta = TwoTempTransProp(nitrogen_oxygen, laricchiuta_ci_models)
    trans_prop_wright = TwoTempTransProp(nitrogen_oxygen, wright_ci_models)

    temps = np.linspace(300, 30000, 100)
    gas_state = {"molef": {"N2": 0.05, "CO2": 0.95}}

    mu_l = []
    mu_w = []
    for temp in temps:
        gas_state["temp"] = temp
        mu_l.append(trans_prop_laricchiuta.viscosity(gas_state))
        mu_w.append(trans_prop_wright.viscosity(gas_state))

    fig, ax = plt.subplots()
    line_l, = ax.plot(temps, mu_l, 'k--')
    line_w, = ax.plot(temps, unp.nominal_values(mu_w), 'k')
    fill_w = ax.fill_between(temps,
                     unp.nominal_values(mu_w)+unp.std_devs(mu_w),
                     unp.nominal_values(mu_w)-unp.std_devs(mu_w),
                     alpha=0.5)
    ax.legend([line_l, (line_w, fill_w)],
              ["Laricchiuta", "Wright"])
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"Viscosity $[kg/(m \cdot s)]$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("./figs/co2_n2_viscosity.svg")

if __name__ == "__main__":
    plot_n2_viscosity()
    plot_co2_viscosity()
    plot_5_species_air_mixture()
    plot_co2_n2_viscosity()
    plt.show()
