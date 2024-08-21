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
    temps = np.linspace(500, 10000, 100)
    eilmer_ks, eilmer_k_modes = [], []
    gupta_ks, gupta_k_modes = [], []
    wright_ks, wright_k_modes = [], []
    laricchiuta_ks, laricchiuta_k_modes = [], []
    laricchiuta_np_ks, laricchiuta_np_k_modes = [], []
    for temp in temps:
        # update gas state
        gs.T = temp
        gs.T_modes = [temp]
        gs.update_thermo_from_pT()
        gas_state["temp"] = temp

        # update thermal conductivity
        gs.update_trans_coeffs()
        eilmer_ks.append(gs.k)
        eilmer_k_modes.append(gs.k_modes[0])

        k, k_modes = gupta_trans_prop.thermal_conductivity(gas_state)
        gupta_ks.append(k)
        gupta_k_modes.append(k_modes)

        k, k_modes = wright_trans_prop.thermal_conductivity(gas_state)
        wright_ks.append(k)
        wright_k_modes.append(k_modes)

        k, k_modes = laricchiuta_trans_prop.thermal_conductivity(gas_state)
        laricchiuta_ks.append(k)
        laricchiuta_k_modes.append(k_modes)

        k, k_modes = laricchiuta_trans_prop_np.thermal_conductivity(gas_state)
        laricchiuta_np_ks.append(k)
        laricchiuta_np_k_modes.append(k_modes)

    fig, ax = plt.subplots()
    line_e, = ax.plot(temps, eilmer_ks, 'k--')
    line_w, = ax.plot(temps, unp.nominal_values(wright_ks), 'k')
    fill_w = ax.fill_between(temps,
                     unp.nominal_values(wright_ks)+unp.std_devs(wright_ks),
                     unp.nominal_values(wright_ks)-unp.std_devs(wright_ks),
                     alpha=0.5)
    line_l, = ax.plot(temps, laricchiuta_ks, 'k:')
    line_lnp, = ax.plot(temps, laricchiuta_np_ks, 'k-.')
    ax.legend([line_e, (line_w, fill_w), line_l, line_lnp],
              ["Gupta-Yos/Eilmer", "Wright", r"Laricchiuta ($\alpha$, $N$)", r"Laricchiuta ($\sigma$, $\epsilon$ from Eilmer)"])
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"Thermal conductivity [$J/(m.s.K)$]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("./figs/air_thermal_conductivity.svg")

    fig_mode, ax_mode = plt.subplots()
    line_e, = ax_mode.plot(temps, eilmer_k_modes, 'k--')
    line_w, = ax_mode.plot(temps, unp.nominal_values(wright_k_modes), 'k')
    fill_w = ax_mode.fill_between(temps,
                     unp.nominal_values(wright_k_modes)+unp.std_devs(wright_k_modes),
                     unp.nominal_values(wright_k_modes)-unp.std_devs(wright_k_modes),
                     alpha=0.5)
    line_l, = ax_mode.plot(temps, laricchiuta_k_modes, 'k:')
    line_lnp, = ax_mode.plot(temps, laricchiuta_np_k_modes, 'k-.')
    ax_mode.legend([line_e, (line_w, fill_w), line_l, line_lnp],
              ["Gupta-Yos/Eilmer", "Wright", r"Laricchiuta ($\alpha$, $N$)", r"Laricchiuta ($\sigma$, $\epsilon$ from Eilmer)"])
    ax_mode.set_ylim(bottom=0)
    ax_mode.grid()
    ax_mode.set_xlabel("Temperature [K]")
    ax_mode.set_ylabel(r"Model thermal conductivity [$J/(m.s.K)$]")
    ax_mode.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.savefig("./figs/air_modal_thermal_conductivity.svg")

if __name__ == "__main__":
    plot_5_species_air_mixture()
    plt.show()
