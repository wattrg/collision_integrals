# Transport property calculations
#
# Computation of transport properties from collision integrals
#
# Author: Robert Watt

from collision_integrals import collision_integral
from data.gupta_yos_data import gupta_yos_data
import numpy as np
from abc import ABC, abstractmethod
from slpp import slpp as lua

AVOGADRO_NUMBER = 6.022e23

class TransProp(ABC):
    """ Base class for calculation of transport properties """

    @abstractmethod
    def viscosity(self, gas_state):
        pass

    @abstractmethod
    def thermal_conductivity(self, gas_state):
        pass


class TwoTempTransProp(TransProp):
    """ Calculates transport properties of two temperature gasses """

    def __init__(self, gas_model, ci_models):
        """
        Initialise a two temperature transport properties

        mixutre:
            Database of gas model

        ci_models (Dict):
            The collision integral models to use. If no collision integral
            model is specified for a given pair, defaults are available
        """

        self._species_names = []
        self._particle_mass = {}
        for species_name, species_data in gas_model:
            self._species_names.append(species_name)
            self._particle_mass = species_data["M"]*1000/AVOGADRO_NUMBER
        self._gas_model = gas_model
        self._cis_11 = {}
        self._cis_22 = {}
        self._mu = {}
        for name_i in self._species_names:
            for name_j in self._species_names:
                pair = (name_i, name_j)
                # construct the collision integrals
                model = self._choose_col_int_model(ci_models, pair)
                params_11 = self._get_col_int_parameters(model, gas_model,
                                                         pair, (1,1))
                params_22 = self._get_col_int_parameters(model, gas_model,
                                                         pair, (2,2))
                self._cis_11[pair] = collision_integral(model, **params_11)
                self._cis_22[pair] = collision_integral(mode, **params_22)

                # calculate reduced mass
                mu_a = gas_model[name_i]["M"]
                mu_b = gas_model[name_j]["M"]
                self._mu[pair] = mu_a * mu_b / (mu_a + mu_b)


    def _get_col_int_parameters(self, ci_model, gas_model, pair, order):
        params = {"order": order, "species": pair}
        if ci_model == "gupta_yos":
            params["coeffs"] = gupta_yos_data[pair, ls]
        elif ci_model == "laricchiuta":
            sigma_a = gas_model[pair[0]]["sigma"]
            sigma_b = gas_model[pair[1]]["sigma"]
            params["sigma"] = 0.5 * (sigma_a + simga_b)
            epsilon_a = gas_model[pair[0]]["epsilon"]
            epsilon_b = gas_model[pair[1]]["epsilon"]
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
        model = ci_modelts.get(pair[::-1], None)
        if model is not None:
            return model

        # No user specified model, so choose the default one to apply
        # check if gupta-yos is available
        if pair in gupta_yos_data or pair[::-1] in gupta_yos_data:
            return "gupta_yos"

        # If both species are charged, use mason
        if ("+" in pair[0] or "-" in pair[0]) and ("+" in pair[1] or "-" in pair[1]):
            return "mason"

        # finally, use laricchiuta
        return "laricchiuta"

    def _compute_delta(self, gas_state, order):
        # storage for the deltas
        deltas = {}

        # point to the collision integrals to use
        if order == (1, 1):
            ci = self._cis_11
        elif order == (2, 2):
            ci = self._cis_22

        # calculate all the deltas
        for name_i in self._species_names:
            for name_j in self._species_names:
                pair = (name_i, name_j)
                # choose temperature to use to evaluate collision integral at
                if name_i == "e-":
                    T_ci = gas_state["T_modes"][0]
                else:
                    T_ci = gas_state["T"]
                # compute delta for this pair
                tmp = (8/3)*1.546e-20*np.sqrt(2*self._mu[pair]/np.pi*8.314*T_ci)
                deltas[pair] = tmp * np.pi * ci[pair].eval(gas_state)
                # the pair written in opposite order is the same
                deltas[pair[::-1]] = deltas[pair]
        return deltas

    def viscosity(self, gas_state):
        delta_22 = self._compute_delta(gas_state, (2, 2))
        sumA = 0.0
        for name_i in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                denom += gas_state["molef"][name_j]*delta_22[name_i, name_j]
            if name_i == "e-": continue
            sumA += self._particle_mass[name_i] * gas_state["molef"][name_i]/denom
        # add term if electron present.
        if "e-" in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                denom += gas_state["molef"][name_j]*delta_22["e-", name_j]
            sumA += self._particle_mass["e-"]*gas_state["molef"]["e-"]/denom
        return sumA * (1e-3/1e-2)

    def thermal_conductivity(gas_state):
        raise Exception("thermal conductivity not implemented for two temperature gas")


def read_lua_gas_model(fname):
    with open(fname) as lua_file:
        lua_data = lua_file.read()
        print(lua_data)
        python_gas_model = lua.decode(lua_data)
        print(lua_data)
    return python_gas_model

if __name__ == "__main__":
    gas_model = read_lua_gas_model("tests/two-temp-mix.lua")
    print(type(gas_model))
