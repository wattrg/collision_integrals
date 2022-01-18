import numpy as np
from uncertainties import unumpy as unp
import uncertainties as unct
from matplotlib import pyplot as plt
from transprop.transport_properties import TwoTempTransProp
from gas_models.N2 import nitrogen
from gas_models.CO2 import carbon_dioxide
from gas_models.nitrogen_oxygen import nitrogen_oxygen
from ci_models import ci_models_laricchiuta, ci_models_wright, ci_models_laricchiuta_non_polar
from gas_models.air_5_species import air_5_species
from eilmer.gas import GasModel, GasState
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True

def plot_5_species_air_mixture():
    gupta_trans_prop = TwoTempTransProp(air_5_species)
    wright_trans_prop = TwoTempTransProp(air_5_species, ci_models_wright)
    laricchiuta_trans_prop = TwoTempTransProp(air_5_species, ci_models_laricchiuta)
    laricchiuta_trans_prop_np = TwoTempTransProp(air_5_species, ci_models_laricchiuta_non_polar)

    gmodel = GasModel("two_temp_gas_5_species.lua")
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
        eilmer_mus.append(gs.k)
        gupta_mus.append(gupta_trans_prop.thermal_conductivity(gas_state)[0])
        wright_mus.append(wright_trans_prop.thermal_conductivity(gas_state)[0])
        laricchiuta_mus.append(laricchiuta_trans_prop.thermal_conductivity(gas_state)[0])
        laricchiuta_np_mus.append(laricchiuta_trans_prop_np.thermal_conductivity(gas_state)[0])

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
    ax.set_ylabel(r"Thermal conductivity")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("./figs/air_thermal_conductivity.svg")

if __name__ == "__main__":
    plot_5_species_air_mixture()
    plt.show()
