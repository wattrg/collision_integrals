import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
import uncertainties.unumpy as unp
from eilmer.gas import GasModel, GasState
from transport_properties import TwoTempTransProp
from examples.gas_models.air_7_species import air_7_species
import examples.plasma_trans_props.ci_models as ci_models

# setup Eilmer gas model
gmodel = GasModel("examples/plasma_trans_props/two-temp-7sp-air.lua")
gas_state = GasState(gmodel)
gas_state.p = 1e5
gas_state.massf = {"N2": 0.69, "O2": 0.09, "O": 0.1, "N": 0.1, "NO+": 0.01, "e-": 0.01}

# set up TwoTempTransProp
wright_trans_prop = TwoTempTransProp(air_7_species, ci_models.wright)
laricchiuta_trans_prop = TwoTempTransProp(air_7_species, ci_models.laricchiuta)
gs = {"molef": gas_state.molef_as_dict}

temps = np.linspace(300, 30000, 100)
mu_e = np.zeros(len(temps))
mu_w = np.zeros(len(temps), dtype=object)
mu_l = np.zeros(len(temps))
for i, temp in enumerate(temps):
    gas_state.T = temp
    gas_state.T_modes = [temp]
    gas_state.update_thermo_from_pT()
    gas_state.update_trans_coeffs()
    mu_e[i] = gas_state.mu

    # compute properties for Mason collision integrals
    gs["temp"] = temp
    gs["T_modes"] = [temp]
    gs["ep"] = gas_state.rho * gas_state.massf_as_dict["e-"] * gas_state.T_modes[0] * 8.314 / air_7_species["e-"]["M"]
    kB = 1.38066e-23
    gs["ne"] = gs["ep"] / (kB * gs["T_modes"][0])
    mu_w[i] = wright_trans_prop.viscosity(gs)
    mu_l[i] = laricchiuta_trans_prop.viscosity(gs)

fig, ax = plt.subplots()
line_w, = ax.plot(temps, unp.nominal_values(mu_w), "k")
fill_w = ax.fill_between(temps,
                         unp.nominal_values(mu_w)+unp.std_devs(mu_w),
                         unp.nominal_values(mu_w)-unp.std_devs(mu_w),
                         alpha=0.5)
line_e, = ax.plot(temps, mu_e, "k--")
line_l, = ax.plot(temps, mu_l, "k-.")
ax.legend([(line_w, fill_w), line_e, line_l], ["Wright et. al.", "Eilmer", "Laricchiuta et. al."])
ax.set_ylim(bottom=0)
plt.show()
