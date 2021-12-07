# Transport property calculations
#
# Computation of transport properties from collision integrals
#
# Author: Robert Watt

from collision_integrals import collision_integral
from data.gupta_yos_data import gupta_yos_data
import numpy as np
from abc import ABC, abstractmethod

AVOGADRO_NUMBER = 6.022e23

class TransProp(ABC):
    """ Base class for calculation of transport properties """

    @abstractmethod
    def viscosity(self, gas_state):
        raise NotImplementedError()

    @abstractmethod
    def thermal_conductivity(self, gas_state):
        raise NotImplementedError()


class TwoTempTransProp(TransProp):
    """ Calculates transport properties of two temperature gasses """

    def __init__(self, gas_model, ci_models=None):
        """
        Initialise a two temperature transport properties

        mixutre:
            Database of gas model

        ci_models (Dict):
            The collision integral models to use. If no collision integral
            model is specified for a given pair, a default one will be chosen
        """

        # if ci_models weren't supplied, set them to be an empty dictionary
        if not ci_models:
            ci_models = {}

        self._species_names = []
        self._particle_mass = {}
        for species_name, species_data in gas_model.items():
            self._species_names.append(species_name)
            self._particle_mass[species_name] = species_data["M"]*1000/AVOGADRO_NUMBER
        self._gas_model = gas_model
        self._cis_11 = {}
        self._cis_22 = {}
        self._mu = {}
        for name_i in self._species_names:
            for name_j in self._species_names[self._species_names.index(name_i):]:
                pair = name_i, name_j
                # construct the collision integrals
                model = self._choose_col_int_model(ci_models, pair)
                params_11 = self._get_col_int_parameters(model,
                                                         pair, (1,1))
                params_22 = self._get_col_int_parameters(model,
                                                         pair, (2,2))
                self._cis_11[pair] = collision_integral(model, **params_11)
                self._cis_11[pair[::-1]] = self._cis_11[pair]
                self._cis_22[pair] = collision_integral(model, **params_22)
                self._cis_22[pair[::-1]] = self._cis_22[pair]

                # calculate reduced mass
                mu_a = gas_model[name_i]["M"]
                mu_b = gas_model[name_j]["M"]
                self._mu[pair] = mu_a * mu_b / (mu_a + mu_b) * 1000 # kg -> g
                self._mu[pair[::-1]] = self._mu[pair]

    def _get_col_int_parameters(self, ci_model, pair, order):
        params = {"order": order, "species": pair}
        if ci_model == "gupta_yos":
            if pair not in gupta_yos_data:
                pair = pair[::-1]
            params["coeffs"] = gupta_yos_data[pair][order]
        elif ci_model == "laricchiuta":
            sigma_a = self._gas_model[pair[0]]["sigma"]
            sigma_b = self._gas_model[pair[1]]["sigma"]
            params["sigma"] = 0.5 * (sigma_a + sigma_b)
            epsilon_a = self._gas_model[pair[0]]["epsilon"]
            epsilon_b = self._gas_model[pair[1]]["epsilon"]
            params["epsilon"] = np.sqrt(epsilon_a * epsilon_b)
        elif ci_model == "mason":
            pass
        return params

    def _choose_col_int_model(self, ci_models, pair):
        # First, check if the user has asked for a particular model
        model = ci_models.get(pair, None)
        if model is not None:
            return model
        # maybe they've specified the pair in the opposite order
        model = ci_models.get(pair[::-1], None)
        if model is not None:
            return model

        # No user specified model, so choose the default one to apply
        # If Gupta-Yos is available, use that.
        if pair in gupta_yos_data or pair[::-1] in gupta_yos_data:
            return "gupta_yos"

        # Gupta-Yos wasn't available. If both species are charged, use mason
        if ("+" in pair[0] or "-" in pair[0]) and ("+" in pair[1] or "-" in pair[1]):
            return "mason"

        # The species weren't charged, use laricchiuta
        return "laricchiuta"

    def _compute_delta(self, gas_state, order):
        # storage for the deltas
        deltas = {}

        # decide which order collision integrals to use
        if order == (1, 1):
            ci = self._cis_11
            factor = 8./3.
        elif order == (2, 2):
            ci = self._cis_22
            factor = 16./5.
        else:
            raise NotImplementedError(f"deltas for order = {order} not implemented")

        # calculate all the deltas
        for name_i in self._species_names:
            # choose temperature to use to evaluate collision integral
            if name_i == "e-":
                T_ci = gas_state["T_modes"][0]
            else:
                T_ci = gas_state["temp"]
            for name_j in self._species_names[self._species_names.index(name_i):]:
                pair = (name_i, name_j)
                # compute delta for this pair
                tmp = factor*1.546e-20*np.sqrt(2.0*self._mu[pair]/(np.pi*1.987*T_ci))
                deltas[pair] = tmp * np.pi * ci[pair].eval(gas_state)
                # the pair written in the opposite order is the same
                deltas[pair[::-1]] = deltas[pair]
        return deltas

    def viscosity(self, gas_state):
        assert np.isclose(sum(gas_state["molef"].values()), 1), "mole fractions don't sum to one"
        delta_22 = self._compute_delta(gas_state, (2, 2))
        sumA = 0.0
        for name_i in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                denom += gas_state["molef"][name_j]*delta_22[name_i, name_j]
            if name_i == "e-": continue
            sumA += self._particle_mass[name_i] * gas_state["molef"][name_i] / denom
        # add term if electron present.
        if "e-" in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                denom += gas_state["molef"][name_j]*delta_22["e-", name_j]
            sumA += self._particle_mass["e-"]*gas_state["molef"]["e-"]/denom
        return sumA * 1e-1 # g/(cm*s) -> kg/(m*s)

    def thermal_conductivity(gas_state):
        raise Exception("thermal conductivity not implemented for two temperature gas")


if __name__ == "__main__":
    from tests.ci_models import ci_models_wright, ci_models_laricchiuta
    from tests.air_gas_model import air as air_gas_model
    from eilmer.gas import GasModel, GasState
    import matplotlib.pyplot as plt

    gupta_trans_prop = TwoTempTransProp(air_gas_model)
    wright_trans_prop = TwoTempTransProp(air_gas_model, ci_models_wright)
    laricchiuta_trans_prop = TwoTempTransProp(air_gas_model, ci_models_laricchiuta)

    gmodel = GasModel("tests/two_temp_gas_5_species.lua")
    gs = GasState(gmodel)
    gs.p = 1e5
    gs.massf = {"N2": 0.8, "O2": 0.2}
    molef = gs.molef_as_dict
    gas_state = {"molef": molef}
    temps = np.linspace(300, 3000, 100)
    eilmer_mus = []
    gupta_mus = []
    wright_mus = []
    laricchiuta_mus = []
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

    plt.plot(temps, eilmer_mus, 'k--', label="Eilmer/Eilmer")
    plt.plot(temps, wright_mus, label="Wright")
    plt.plot(temps, laricchiuta_mus, label="Laricchiuta")
    plt.legend()
    plt.ylim(bottom=0)
    plt.show()
