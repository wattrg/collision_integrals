# Transport property calculations
#
# Computation of transport properties from collision integrals
#
# Author: Robert Watt

from .collision_integrals import collision_integral
from transprop.data.gupta_yos_data import gupta_yos_data
from transprop.data.very_viscous_gupta_yos import very_viscous_gupta_yos_data
import numpy as np
import sympy as sp
from abc import ABC, abstractmethod

AVOGADRO_NUMBER = 6.022e23

class TransProp(ABC):
    """ Base class for calculation of transport properties """

    @abstractmethod
    def _viscosity(self, gas_state):
        raise NotImplementedError()

    @abstractmethod
    def _thermal_conductivity(self, gas_state):
        raise NotImplementedError()

    def viscosity(self, gas_state):
        self._choose_math_package(gas_state)
        return self._viscosity(gas_state)

    def thermal_conductivity(self, gas_state):
        self._choose_math_package(gas_state)
        return self._thermal_conductivity(gas_state)

    def _choose_math_package(self, gas_state):
        for gas_var in gas_state.values():
            if isinstance(gas_var, sp.Expr):
                self._math = sp
                break
            else:
                self._math = np


class TwoTempTransProp(TransProp):
    """ Calculates transport properties of two temperature gasses """

    def __init__(self, gas_model, ci_models=None):
        """
        Initialise a two temperature transport properties

        mixutre:
            Database of gas model

        (optional) ci_models (Dict):
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
        self._alpha = {}
        for name_i in self._species_names:
            for name_j in self._species_names[self._species_names.index(name_i):]:
                pair = name_i, name_j
                # construct the collision integrals
                model, user_params = self._choose_col_int_model(ci_models, pair)
                params_11 = self._get_col_int_parameters(model,
                                                         pair, (1,1), user_params)
                params_22 = self._get_col_int_parameters(model,
                                                         pair, (2,2), user_params)
                self._cis_11[pair] = collision_integral(model, **params_11)
                self._cis_11[pair[::-1]] = self._cis_11[pair]
                self._cis_22[pair] = collision_integral(model, **params_22)
                self._cis_22[pair[::-1]] = self._cis_22[pair]

                # calculate reduced mass
                mu_a = gas_model[name_i]["M"]
                mu_b = gas_model[name_j]["M"]
                self._mu[pair] = mu_a * mu_b / (mu_a + mu_b) * 1000 # kg -> g
                self._mu[pair[::-1]] = self._mu[pair]

                # compute alpha
                M_ratio = np.array([gas_model[name_i]["M"]/gas_model[name_j]["M"],
                                    gas_model[name_j]["M"]/gas_model[name_i]["M"]])
                denom = (1 + M_ratio)**2
                numer = (1 - M_ratio) * (0.45 - 2.54*M_ratio)
                alphas = 1 + numer/denom
                self._alpha[name_i, name_j] = alphas[0]
                self._alpha[name_j, name_i] = alphas[1]

    def _get_col_int_parameters(self, ci_model, pair, order, user_params):
        params = {"order": order, "species": pair}
        if ci_model == "gupta_yos":
            if user_params.get("very-viscous"):
                gy_data = very_viscous_gupta_yos_data
            else:
                gy_data = gupta_yos_data
            if pair not in gy_data:
                pair = pair[::-1]
            params["coeffs"] = gy_data[pair][order]
        elif ci_model == "laricchiuta":
            sigma_a = self._gas_model[pair[0]].get("sigma", None)
            sigma_b = self._gas_model[pair[1]].get("sigma", None)
            if sigma_a and sigma_b:
                params["sigma"] = 0.5 * (sigma_a + sigma_b)
            epsilon_a = self._gas_model[pair[0]].get("epsilon", None)
            epsilon_b = self._gas_model[pair[1]].get("epsilon", None)
            if epsilon_a and epsilon_b:
                params["epsilon"] = np.sqrt(epsilon_a * epsilon_b)
            if "polarisability" in self._gas_model[pair[0]] and "polarisability" in self._gas_model[pair[1]]:
                params["alphas"] = [self._gas_model[pair[0]]["polarisability"],
                                    self._gas_model[pair[1]]["polarisability"]]
            if "N" in self._gas_model[pair[0]] and "N" in self._gas_model[pair[1]]:
                params["Ns"] = [self._gas_model[pair[0]]["N"],
                                self._gas_model[pair[1]]["N"]]
        elif ci_model == "mason":
            pass
        params.update(user_params)
        return params

    def _choose_col_int_model(self, ci_models, pair):
        # First, check if the user has asked for a particular model
        model, params = ci_models.get(pair, (None, {}))
        if model is not None:
            return model, params
        # maybe they've specified the pair in the opposite order
        model, params = ci_models.get(pair[::-1], (None, {}))
        if model is not None:
            return model, params

        # No user specified model, so choose the default one to apply
        # If Gupta-Yos is available, use that.
        if pair in gupta_yos_data or pair[::-1] in gupta_yos_data:
            return "gupta_yos", {}

        # Gupta-Yos wasn't available. If both species are charged, use mason
        if ("+" in pair[0] or "-" in pair[0]) and ("+" in pair[1] or "-" in pair[1]):
            return "mason", {}

        # The species weren't charged, use laricchiuta
        return "laricchiuta", {}

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
            gs = gas_state.copy()
            gs["temp"] = T_ci
            for name_j in self._species_names[self._species_names.index(name_i):]:
                pair = (name_i, name_j)
                # compute delta for this pair
                tmp = factor*1.546e-20*self._math.sqrt(2.0*self._mu[pair]/(self._math.pi*1.987*T_ci))
                deltas[pair] = tmp * self._math.pi * ci[pair].eval(gs)
                # the pair written in the opposite order is the same
                deltas[pair[::-1]] = deltas[pair]
        return deltas

    def _viscosity(self, gas_state):
        """
        Compute the viscosity of a mixture in `gas_state`
        """
        # assert np.isclose(sum(gas_state["molef"].values()), 1), "mole fractions don't sum to one"
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

    def _thermal_conductivity(self, gas_state):
        """
        Compute the thermal conductivity of a mixture in `gas_state`
        """
        # assert np.isclose(sum(gas_state["molef"].values()), 1), "mole fractions don't sum to one"
        delta_22 = self._compute_delta(gas_state, (2, 2))
        delta_11 = self._compute_delta(gas_state, (1, 1))

        # 1. k_tr
        sumA = 0.0
        for name_i in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                if name_j != "e-":
                    denom += self._alpha[name_i, name_j]*gas_state["molef"][name_j]*delta_22[name_i, name_j]
                else:
                    denom += 3.54*self._alpha[name_i, name_j]*gas_state["molef"][name_j]*delta_22[name_i, name_j]
            if name_i == "e-": continue
            sumA += gas_state["molef"][name_i]/denom
        kB_erg = 1.38066e-16
        k_tr = 2.3901e-8*(15.0/4.0)*kB_erg*sumA
        k_tr *= 4.184/1e-2

        # 2. k_rot
        # assuming fully excited, eq (75) in Gnoffo
        k_rot = 0.0
        for name_i in self._species_names:
            if self._gas_model[name_i]["type"] == "molecule":
                denom = 0.0
                for name_j in self._species_names:
                    denom += gas_state["molef"][name_j]*delta_11[name_i, name_j]
                k_rot += gas_state["molef"][name_i]/denom
        k_rot *= 2.3901e-8*kB_erg
        k_rot *= 4.184/1e-2
        k = k_tr + k_rot

        # 3. k_vib = k_rot
        k_vib = k_rot

        # 4. k_e
        k_E = 0.0
        if "e-" in self._species_names:
            denom = 0.0
            for name_j in self._species_names:
                denom += 1.45*gas_state["molef"][name_j]*delta_22["e-", name_j]
            k_E = gas_state["molef"]["e-"]/denom
            k_E *= 2.3901e-8 * (15.0/4)*kB_erg
            k_E *= 4.184/1e-2
        return k, k_vib + k_E
