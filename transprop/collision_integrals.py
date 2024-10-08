# Collision integral calculation
#
# Provides a number of models for calculating collision integrals
# which can then be used to compute some transport properties
# of high temperature gasses
#
# Author: Robert Watt

from scipy import optimize, interpolate
from scipy.special import factorial
from scipy import integrate
import numpy as np
import sympy as sp
from uncertainties import ufloat
import matplotlib.pyplot as plt

from transprop.data.wright_ci_data import wright_ci_data
from transprop.data.Laricchiuta import laricchiuta_coeffs
from transprop.data.gupta_yos_data import gupta_yos_data
from transprop.data.very_viscous_gupta_yos import very_viscous_gupta_yos_data
from abc import ABC, abstractmethod



class ColIntModel(ABC):
    """ Base class for collision integral models """
    def __init__(self, **kwargs):
        # get the order of the collision integral
        try:
            order = kwargs["order"]
            self._l, self._s = order[0], order[1]
        except KeyError:
            raise Exception("The order of the collision integral must be supplied")

        # get the species involved in the collision
        try:
            self._species = kwargs["species"]
        except KeyError:
            raise Exception("Colliding species not specified")

        # get the charge of the species in the collision
        self._charge = kwargs.get("charge", [None, None])
        if self._charge == [None, None]:
            for i in range(2):
                if "+" in self._species[i]:
                    self._charge[i] = 1
                elif "-" in self._species[i]:
                    self._charge[i] = -1
                else:
                    self._charge[i] = 0

        self._model = kwargs["model"]

        self._acc = kwargs.get("acc", None)
        self._eval_acc = kwargs.get("eval_acc", False)
        if not self._acc and self._eval_acc:
            self._eval_acc = False

    def get_charge(self):
        """ Return the charge of the species """
        return tuple(self._charge)

    def get_order(self):
        """ Return the order of the collision integral """
        return self._l, self._s

    def get_species(self):
        """ Return the species in the collision integral """
        return self._species

    def get_model(self):
        """ Return the model of the collision integral """
        return self._model

    @abstractmethod
    def _eval(self, gas_state):
        raise NotImplementedError()

    def eval(self, gas_state):
        """
        Evaluate the collision integral

        Parameters:
            gas_state (Dict): The gas state to evaluate the collision integral at

        Returns:
            ci (float): The evaluated collision integral
        """
        self._choose_math_package(gas_state)
        return self._eval(gas_state)


    def _choose_math_package(self, gas_state):
        for gas_var in gas_state.values():
            if isinstance(gas_var, sp.Expr):
                self._math = sp
                break
            else:
                self._math = np


def omega_curve_fit(temp, a_ii, b_ii, c_ii, d_ii, m=np):
    """
    The curve fit of collision integrals proposed by Gupta, Yos, Thomson (1990)

    Parameters:
       temp (float): The temperature to evaluate the CI at
       a_ii (float): Curve fit parameter A_ii
       b_ii (float): Curve fit parameter B_ii
       c_ii (float): Curve fit parameter C_ii
       d_ii (float): Curve fit parameter D_ii

    Returns:
       pi_omega_ii (float): The estimated value of pi_omega_ii
                            using the parameters provided
       """

    log_T = m.log(temp)
    return m.exp(d_ii) * temp**(a_ii*log_T**2 + b_ii*log_T + c_ii)

def psi(s):
    """ Compute Psi(s) """
    if s == 1:
        return 0.0
    else:
        tmp = 0
        for n in range(1, s):
            tmp += 1/n
        return tmp


def gamma_function(s, m=np):
    """ Compute the gamma function for integer values of s """
    if s <= 2:
        return 1.0
    else:
        return m.prod(self._math.arange(2, s))


class MasonColInt(ColIntModel):
    """
    Model for collision integrals between charged particles.
    The model assumes a shielded coulomb potential, and evaluates a curve fit

    Ref: "Transport Coefficients of Ionized Gases" by Mason, Munn, Smith
    Curve fit from Wright et al 2007
    """

    # electron charge and Bolzmann's constant are in these units
    # to match the values given by Mason
    _e = 4.803e-10 # electron charge (esu)
    _k_B = 1.3806e-16 # Boltzmann's constant (erg/K)

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = "mason"
        super().__init__(**kwargs)
        assert self._l == self._s, "l must equal s"

        # repulsive potential
        if self._charge[0] * self._charge[1] > 0:
            if self._l == 1:
                self._Cn = 0.138
                self._cn = 0.0106
                self._Dn = 0.765
            elif self._l == 2:
                self._Cn = 0.157
                self._cn = 0.0274
                self._Dn = 1.235

        # attractive potential
        elif self._charge[0] * self._charge[1] < 0:
            if self._l == 1:
                self._Cn = -0.476
                self._cn = 0.0313
                self._Dn = 0.784
            elif self._l == 2:
                self._Cn = -0.146
                self._cn = 0.0377
                self._Dn = 1.262
        else:
            raise ValueError("Mason Collision integral not valid for neutral particles")

    def _debye_length(self, gas_state):
        temp = gas_state["temp"]
        ne = gas_state["ne"]
        return self._math.sqrt(self._k_B * temp / (4 * np.pi * ne * self._e**2))

    def _temp_star(self, gas_state):
        temp = gas_state["temp"]
        return self._debye_length(gas_state) * self._k_B * temp / self._e**2

    def _eval(self, gas_state):
        temp_star = self._temp_star(gas_state)
        debye = self._debye_length(gas_state)
        log_term = self._math.log(self._Dn * temp_star * (1 - self._Cn*np.exp(-self._cn * temp_star))+1)
        return 5e15 * (debye / temp_star)**2 * log_term


class GhoruiColInt(ColIntModel):
    """
    Charged - charged collision integrals, from Ghorui and Das 2013
    """

    _gamma = 0.5772
    _beta = 1.0
    _epsilon = 1.0
    _epsilon_0 = 8.854e-12
    _e = 1.602e-19
    _k_B = 1.38e-23

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = "ghorui"
        super().__init__(**kwargs)

        if self._l == 1:
            self._l_term = 0.5
        elif self._l == 2:
            self._l_term = 1.0
        else:
            raise ValueError(f"l={self._l} not supported")

        self._m_star = mass[0] * mass[1] / (mass[0] + mass[1])

    def _compute_screening_distance(self, gas_state):
        """ Calculate the screening distance """
        pre_factor = self._e**2 / self._epsilon_0 / self._k_B / gas_state["Te"]
        tmp = gas_state["ne"]
        for species in gas_state.items():
            tmp += species["charge"]**2 * species["num_den"]
        return pre_factor * tmp

    def _eval(self, gas_state):
        self._temp_star = (mass[0] * temp[1] + mass[1] * temp[0]) / (mass[0] + mass[1])
        delta = self._beta * (1/4/self._math.pi/self._epsilon
                              * charge[0]*charge[1] * self._e**2 / self._k_B / self._temp_star)
        self._b_0 = delta / 2 / self._beta
        pre_factor = self._math.sqrt(2 * np.pi * self._k_B * self._temp_star / self._m_star)
        pre_factor *= self._beta**2 * self._b_0**2 * gamma_function(self._s)
        log_Lambda = self._math.log(2 * self._compute_screening_distance / self._beta / self._b_0)
        return pre_factor * (log_Lambda - self._l_term - 2 * self._gamma + psi(self._s))


class ColIntLaricchiuta(ColIntModel):
    """
    Dimensionless collision integrals from Laricchiuta et al:
    Classical transport collision integrals for a Lennard-Jones like
    phenomenological model potential 2007

    Good for neutral/neutral or neutral/ion collisions
    """

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            kwargs["model"] = "laricchiuta"
        super().__init__(**kwargs)

        # set the zeta values and collision type
        if self._charge[0] != 0 or self._charge[1] != 0:
            self._zeta = [0.7564, 0.064605]
            self._col_type = "ion-neutral"
        if self._charge[0] == 0 and self._charge[1] == 0:
            self._zeta = [0.8002, 0.049256]
            self._col_type = "neutral-neutral"
        else:
            raise Exception("Invalid collision type for Laricchiuta integrals")

        # physical parameters
        param_priority = kwargs.get("param_priority", "LJ")
        if param_priority == "LJ":
            if not self._get_sigma_epsilon(kwargs):
                if not self._get_alphas(kwargs):
                    if not self._get_alpha(kwargs):
                        raise KeyError("No parameters to Laricchiuta model provided")
        elif param_priority == "polarisability":
            if not self._get_alphas(kwargs):
                if not self._get_alpha(kwargs):
                    if not self._get_sigma_epsilon(kwargs):
                        raise KeyError("No parameters to Laricchiuta model provided")
        else:
            raise KeyError("Unknown parameter priority")

        # compute x_0
        self._x_0 = self._zeta[0] * self._beta ** self._zeta[1]

        # get the values of c
        self._cs = laricchiuta_coeffs[f"omega_{self._l}{self._s}"]

        # compute coefficients
        self._coeffs = []
        for i in range(7):
            coeff = 0.0
            for j, c in enumerate(self._cs[i + 1]):
                coeff += c * self._beta**j
            self._coeffs.append(coeff)

    def _get_sigma_epsilon(self, kwargs):
        try:
            self._sigma = kwargs["sigma"]
            self._epsilon = kwargs["epsilon"]
            self._beta = kwargs.get("beta", 8)
            return True
        except KeyError:
            return False

    def _get_alphas(self, kwargs):
        try:
            self._alphas = kwargs["alphas"]
            self._Ns = kwargs["Ns"]
            self._compute_parameters()
            return True
        except KeyError:
            return False

    def _get_alpha(self, kwargs):
        try:
            alpha = kwargs["alpha"]
            N = kwargs["N"]
            self._Ns = np.array([N, N])
            self._alphas = np.array([alpha, alpha])
            self._compute_parameters()
            return True
        except KeyError:
            return False

    def get_col_type(self):
        """ Return type of collision """
        return self._col_type

    def _compute_parameters(self):
        """ Compute epsilon_0, r_e, and beta """

        alphas = np.array(self._alphas)
        Ns = np.array(self._Ns)
        softness = np.cbrt(alphas)
        self._beta = 6 + 5 / ( np.sum(softness) )
        col_type = self.get_col_type()

        if col_type == "neutral-neutral":
            self._sigma = 1.767 * np.sum(softness) / ( np.prod(alphas)**0.095 )
            c_d = 15.7 * np.prod(alphas) / np.sum( np.sqrt(alphas/Ns ))
            # compute epsilon in eV, and convert to K
            self._epsilon = 0.72 * c_d / self._sigma**6 * 11604.525

        elif col_type == "ion-netrual":
            raise NotImplementedError("calculating parameters for ion-neutral "
                                      "collisions not implemented yet")

    def get_coeffs(self):
        """ Return a_i(beta) """
        return self._coeffs

    def _eval(self, gas_state):
        temp = gas_state["temp"]
        temp_star = temp / self._epsilon
        if self._math.any(temp_star < 2e-4) or np.any(temp_star > 1e3):
            raise ValueError("Temperature outside of valid range")
        a1, a2, a3, a4, a5, a6, a7 = self.get_coeffs()
        x = self._math.log(temp_star)
        ln_omega = (a1 + a2*x) * \
                    self._math.exp((x-a3)/a4) / (np.exp((x-a3)/a4) + np.exp((a3-x)/a4)) +\
                    a5 * self._math.exp((x-a6)/a7) / (np.exp((x-a6)/a7) + np.exp((a6-x)/a7))
        return self._math.exp(ln_omega) * (self._x_0 * self._sigma)**2


class ColIntTable(ColIntModel):
    """ Tabulated collision integral """
    _model = "table"

    def __init__(self, **kwargs):
        self._temps = kwargs["temps"]
        self._cis = kwargs["cis"]
        self._kind = kwargs.get("kind", "linear")
        self._acc = kwargs.get("acc", None)
        self._eval_acc = kwargs.get("eval_acc", False)
        self._interp = interpolate.interp1d(self._temps, self._cis, kind=self._kind)

    def _eval(self, gas_state):
        col_int = self._interp(gas_state["temp"])
        if self._eval_acc:
            if not self._acc:
                raise ValueError("Accuracy not specified for tabular collision integral")
            col_int = ufloat(col_int, self._acc * col_int)
        return col_int


class ColIntCurveFitModel(ColIntModel):
    """ Base class for curve fitted collision integrals"""

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            if hasattr(self, "_model"):
                kwargs["model"] = self._model
            else:
                kwargs["model"] = "curve_fit"
        super().__init__(**kwargs)

        if "cis" in kwargs:
            self._temps = kwargs["temps"]
            self._cis = kwargs["cis"]
            self._evaluate_coeffs()
        elif "coeffs" in kwargs:
            self._coeffs = kwargs["coeffs"]
        else:
            raise ValueError("Curve fit model must be provide tabulated "
                             "collision integrals or coefficients")

    @abstractmethod
    def _curve_fit_form(self, temp, *args):
        pass

    def _evaluate_coeffs(self):
        self._math = np
        self._coeffs, _ = optimize.curve_fit(self._curve_fit_form,
                                             self._temps,
                                             self._cis,
                                             self._guess)

    def _check_interp_bounds(self, temp):
        if hasattr(self, "_temps"):
            if self._math.any(temp < min(self._temps)):
                raise ValueError(f"Temperature ({temp}K) below curve fit range "
                                 f" {min(self._temps)}-{max(self._temps)} K")
            elif self._math.any(temp > max(self._temps)):
                raise ValueError(f"Temperature ({temp}K) above curve fit range "
                                 f"{min(self._temps)}-{max(self._temps)} K")

    def _eval(self, gas_state):
        temp = gas_state["temp"]
        self._check_interp_bounds(temp)
        return self._curve_fit_form(temp, *self._coeffs)

    def __repr__(self):
        return f"{self._coeffs}"


class ColIntGYCurveFitPiOmega(ColIntCurveFitModel):
    """ Curve fit of pi * Omega """
    _guess = [-0.01, 0.3, -2.5, 11]
    _model = "curve_fit_pi_omega"

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d, m=self._math) / self._math.pi


class ColIntGYCurveFitOmega(ColIntCurveFitModel):
    """ Curve fitted collision integral """
    _guess = [-0.01, 0.3, -2.5, 11]
    _model = "curve_fit_omega"

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d, m=self._math)


class ColIntWrightGYCurveFit(ColIntGYCurveFitOmega):
    """
    Collision integrals from Wright et. al.
    """

    def _eval(self, gas_state):
        col_int = super()._eval(gas_state)
        if self._eval_acc:
            col_int = ufloat(col_int, self._acc*col_int)
        return col_int

class ColIntGuptaYos(ColIntGYCurveFitPiOmega):
    """
    Collision integral from Gupta Yos
    """

    def __init__(self, **kwargs):
        if "coeffs" not in kwargs:
            species = kwargs["species"]
            if kwargs["very-viscous"] == True:
                gy_data = very_viscous_gupta_yos_data
                print("Using very viscous gupta-yos data")
            else:
                gy_data = gupta_yos_data
            if species in gy_data:
                kwargs["coeffs"] = gy_data[species]
            elif species[::-1] in gy_data:
                kwargs["coeffs"] = gy_data[species[::-1]]
            else:
                raise KeyError(f"Couldn't find collision {species} in gupta yos data")
        super().__init__(**kwargs)

    def _eval(self, gas_state):
        col_int = super()._eval(gas_state)
        if self._charge[0] * self._charge[1] != 0:
            # charged collision, so need to correct for electron pressure
            temp = gas_state["temp"]
            pe = gas_state["ep"]/101325
            col_int *= self._math.log(2.09e-2 * (temp/1000/pe**0.25)**4
                                + 1.52*(temp/1000/pe**0.25)**(8/3))
            col_int /= self._math.log(2.09e-2 * (temp/1000)**4 + 1.52*(temp/1000)**(8/3))
        return col_int


class ColIntCurveFitCollection:
    """
    Collection of Curve fit collision integrals
    """

    def __init__(self, **kwargs):
        self._data = kwargs["ci_table"]
        self._curve_fit_type = kwargs["curve_fit_type"]
        self._ci_coeffs = {}
        for pair, pair_ci in self._data.items():
            self._ci_coeffs[pair] = {}
            species = pair

            # check if both species are charged
            charge = [0, 0]
            for i in range(2):
                if "+" in pair[i]:
                    charge[i] = 1
                if "-" in pair[i]:
                    charge[i] = -1

            for order in [(1,1), (2,2)]:
                self._ci_coeffs[pair][order] = _col_int_curve_fit(
                    curve_fit_type=self._curve_fit_type,
                    order=order,
                    temps=pair_ci[order]["temps"],
                    cis=pair_ci[order]["cis"],
                    charge=charge,
                    species=species
                )

    def get_col_ints(self, pair=None, ci_type=None):
        """ Return the collision integrals """
        if pair:
            if ci_type:
                return self._ci_coeffs[pair][ci_type]
            return self._ci_coeffs[pair]
        return self._ci_coeffs

def _col_int_wright_table(**kwargs):
    """ Create a collision integral from Wright et al """
    try:
        species = kwargs["species"]
        order = kwargs["order"]
    except KeyError:
        raise KeyError("You must supply the species and order for Wright collision integrals")

    if species in wright_ci_data:
        data = wright_ci_data[species][order]
    elif species[::-1] in wright_ci_data:
        data = wright_ci_data[species[::-1]][order]
    else:
        raise KeyError(f"Couldn't find collision {species} in wright data base")
    kwargs["temps"] = data["temps"]
    kwargs["cis"] = data["cis"]
    kwargs["acc"] = data["acc"]
    return ColIntTable(**kwargs)


def _col_int_wright_gy_curve_fit(**kwargs):
    """
    Create a collision integral from Wright et al, curve fitted to the form
    given by Gupta-Yos
    """
    try:
        species = kwargs["species"]
        order = kwargs["order"]
    except KeyError:
        raise KeyError("You must supply the species and order for Wright collision integrals")

    if species in wright_ci_data:
        data = wright_ci_data[species][order]
    elif species[::-1] in wright_ci_data:
        data = wright_ci_data[species[::-1]][order]
    else:
        raise KeyError(f"Couldn't find collision {species} in wright data base")
    kwargs["temps"] = data["temps"]
    kwargs["cis"] = data["cis"]
    kwargs["acc"] = data["acc"]
    kwargs["curve_fit_type"] = "Omega"
    return ColIntWrightGYCurveFit(**kwargs)

def _col_int_curve_fit(**kwargs):
    """
    Factory for curve fitted collision integrals
    """
    CURVE_FIT_TYPES = {
        "Omega": ColIntGYCurveFitOmega,
        "pi_Omega": ColIntGYCurveFitPiOmega,
    }

    curve_fit_type = kwargs["curve_fit_type"]
    if curve_fit_type not in CURVE_FIT_TYPES:
        raise ValueError(curve_fit_type)
    return CURVE_FIT_TYPES[curve_fit_type](**kwargs)

def collision_integral(ci_type, **kwargs):
    """
    Handles the high level creation of collision integrals

    Parameters:
        ci_type (string): The type of collision integral

    Optional parameters:
        Any parameters required to construct the desired collision integral

    Returns:
        ci: A concrete collision integral instsance
    """
    CI_TYPES = {
        "laricchiuta": ColIntLaricchiuta,
        "gupta_yos": ColIntGuptaYos,
        "wright_curve_fit": _col_int_wright_gy_curve_fit,
        "wright_table": _col_int_wright_table,
        "table": ColIntTable,
        "curve_fit": _col_int_curve_fit,
        "mason": MasonColInt,
    }

    return CI_TYPES[ci_type](**kwargs)


if __name__ == "__main__":
    gas_state = {"ne": 1e20, "temp": 1000}
    ci = collision_integral(
        "mason",
        order=(1, 1),
        charge=(-1, -1),
        species=("NA", "NA")
    )
    print("collision integral = ", ci.eval(gas_state))
    print("temp star = ", ci._temp_star(gas_state))
    print("debye length = ", ci._debye_length(gas_state))
    print("temp star square * ci = ", ci.eval(gas_state) * ci._temp_star(gas_state)**2)

    ci = collision_integral(
        "gupta_yos",
        order=(1,1),
        charge=(0,0),
        species=("N2", "N2"),
        coeffs=[0.0, -0.0112, -0.1182, 4.8464]
    )
    gas_state = {"temp": 1000}
    print(ci.eval(gas_state))

    ci = collision_integral(
        "wright_curve_fit",
        order=(1,1),
        species=("N2", "N2"),
        eval_acc=True
    )
    ci_table = collision_integral(
        "wright_table",
        order=(1,1),
        species=("N2", "N2"),
        eval_acc=True
    )
    print(ci.eval(gas_state))
    print(ci_table.eval(gas_state))
