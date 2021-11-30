# Collision integral calculation
#
# Provides a number of models for calculating collision integrals
# which can then be used to compute some transport properties
# of high temperature gasses
#
# Author: Robert Watt

from eilmer.gas import GasModel, GasState
from scipy import optimize, interpolate
from scipy.special import factorial
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from data.hcb_ci_data import hcb_ci_data
from data.wright_ci_data import wright_ci_data
from data.Laricchiuta import laricchiuta_coeffs
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

    def get_charge(self):
        """ Return the charge of the species """
        return tuple(self._charge)

    def get_order(self):
        """ Return the order of the collision integral """
        return self._l, self._s

    def get_species(self):
        """ Return the species in the collision integral """
        return self._species

    @abstractmethod
    def eval(self, gas_state):
        """
        Evaluate the collision integral

        Parameters:
            gas_state (Dict): The gas state to evaluate the collision integral at

        Returns:
            ci (float): The evaluated collision integral
        """
        pass


def omega_curve_fit(temp, a_ii, b_ii, c_ii, d_ii):
    """
       The functional form to fit the data to. Note that a division by
       pi is present to account for Gupta Yos using pi * Omega,
       whereas usually Omega itself is reported

       Parameters:
       temp (float): The temperature to evaluate the CI at
       a_ii (float): Curve fit parameter A_ii in Gupta et al. (1989)
       b_ii (float): Curve fit parameter B_ii in Gupta et al. (1989)
       c_ii (float): Curve fit parameter C_ii in Gupta et al. (1989)
       d_ii (float): Curve fit parameter D_ii in Gupta et al. (1989)

       Returns:
       pi_omega_ii (float): The estimated value of pi_omega_ii
                            using the parameters provided
       """

    log_T = np.log(temp)
    return np.exp(d_ii) * np.power(temp, a_ii * log_T**2 + b_ii * log_T + c_ii)


def lennard_jones(radius, sigma, epsilon):
    """ compute the Lennard Jones potential at some separation """
    return 4 * epsilon * ((sigma/radius)**12 - (sigma/radius)**6)


def psi(s):
    """ Compute Psi(s) """
    if s == 1:
        return 0.0
    else:
        tmp = 0
        for n in range(1, s):
            tmp += 1/n
        return tmp


def gamma_function(s):
    """ Compute the gamma function for integer values of s """
    if s <= 2:
        return 1.0
    else:
        return np.prod(np.arange(2, s))


class MasonColInt(ColIntModel):
    """
    Model for collision integrals between charged particles.
    The model assumes a shielded coulomb potential, and evaluates a curve fit

    Ref: "Transport Coefficients of Ionized Gases" by Mason, Munn, Smith
    Curve fit from Wright et al 2007
    """

    _e = 4.803e-10 # electron charge (esu)
    _k_B = 1.3806e-16 # Boltzmann's constant (erg/K)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self._l == self._s, "l must equal s"
        charge = kwargs["charge"]

        # repulsive potential
        if charge[0] * charge[1] > 0:
            if self._l == 1:
                self._Cn = 0.138
                self._cn = 0.0106
                self._Dn = 0.765
            elif self._l == 2:
                self._Cn = 0.157
                self._cn = 0.0274
                self._Dn = 1.235

        # attractive potential
        elif charge[0] * charge[1] < 0:
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
        return np.sqrt(self._k_B * temp / (4 * np.pi * ne * self._e**2))

    def _temp_star(self, gas_state):
        temp = gas_state["temp"]
        return self._debye_length(gas_state) * self._k_B * temp / self._e**2

    def eval(self, gas_state):
        temp_star = self._temp_star(gas_state)
        debye = self._debye_length(gas_state)
        log_term = np.log(self._Dn * temp_star * (1 - self._Cn*np.exp(-self._cn * temp_star))+1)
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

    def eval(self, gas_state):
        self._temp_star = (mass[0] * temp[1] + mass[1] * temp[0]) / (mass[0] + mass[1])
        delta = self._beta * (1/4/np.pi/self._epsilon
                              * charge[0]*charge[1] * self._e**2 / self._k_B / self._temp_star)
        self._b_0 = delta / 2 / self._beta
        pre_factor = np.sqrt(2 * np.pi * self._k_B * self._temp_star / self._m_star)
        pre_factor *= self._beta**2 * self._b_0**2 * gamma_function(self._s)
        log_Lambda = np.log(2 * self._compute_screening_distance / self._beta / self._b_0)
        return pre_factor * (log_Lambda - self._l_term - 2 * self._gamma + psi(self._s))


class NumericCollisionIntegral(ColIntModel):
    """ Collision integral performed numerically from scratch """

    POTENTIALS = {
        "lennard_jones": lennard_jones,
    }

    def __init__(self, order, **kwargs):
        super().__init__(order)
        if callable(kwargs["potential"]):
            self._potential_func = kwargs["potential"]
        else:
            self._potential_func = self.POTENTIALS[kwargs["potential"]]
        self._epsilon = kwargs["epsilon"]
        self._sigma = kwargs["sigma"]
        self._mu = kwargs["mu"]

    def _potential(self, radius):
        """ Evaluate the potential at the particular radius """
        return self._potential_func(radius, self._sigma, self._epsilon)

    def _r_m_func(self, radius, impact_param, rel_vel):
        gamma_2 = 0.5 * self._mu * rel_vel**2
        # tmp = -radius**2/impact_param
        # tmp *= np.sqrt(1 - self._potential(radius)/gamma_2 - (impact_param/radius)**2)
        return self._potential(radius)/gamma_2 + (impact_param/radius)**2 - 1

    def _calc_r_m(self, impact_param, rel_vel):
        """ Compute the value of r_m for computing the deflection angle """
        if hasattr(rel_vel, "__iter__"):
            r_ms = np.zeros_like(rel_vel)
            for i, g in enumerate(rel_vel):
                r_ms[i] = optimize.root_scalar(self._r_m_func,
                                               x0=1e-10, x1=1e-9,
                                               method="secant",
                                               args=(impact_param, g)).root
            return r_ms
        else:
            return optimize.root_scalar(self._r_m_func,
                                        x0=1e-10, x1=1e-9,
                                        method="secant",
                                        args=(impact_param, rel_vel)).root

    def _deflection_integrand(self, radius, impact_param, rel_vel):
        """ Integrand for computing the deflection angle """

        gamma_2 = 0.5 * self._mu * rel_vel**2
        tmp = 1 - self._potential(radius)/gamma_2 - (impact_param/radius)**2
        if hasattr(tmp, "__iter__"):
            tmp[tmp < 0.0] = 1e-300
        else:
            tmp = max(tmp, 1e-300)
        if np.any(tmp < 0):
            print(f"WARNING: Invalid value under square root\n"
                  f"         radius = {radius}\n"
                  f"         impact_param = {impact_param}\n"
                  f"         rel_vel = {rel_vel}\n"
                  f"         sqrt = {tmp}\n"
                  f"         r_m = {self._calc_r_m(impact_param, rel_vel)}")
        tmp = np.sqrt(tmp)
        return 1 / tmp / radius**2

    def _deflection_angle(self, impact_param, rel_vel):
        """
        Compute the deflection angle for a given relative velocity and
        impact parameter
        """
        r_m = self._calc_r_m(impact_param, rel_vel)
        if hasattr(rel_vel, "__iter__"):
            integral = np.zeros_like(rel_vel)
            for i, g in enumerate(rel_vel):
                integral[i],_ = integrate.quad(self._deflection_integrand,
                                               r_m + 1e-10, np.inf,
                                               limit=100,
                                               args=(impact_param, g))
        else:
            integral, _ = integrate.quad(self._deflection_integrand,
                                        r_m + 1e-10, np.inf,
                                        limit=100,
                                        args=(impact_param, rel_vel))
        return np.pi - 2 * impact_param * integral

    def _collision_cross_section_integrand(self, impact_parameter, rel_vel):
        """ Compute the integrand for the collision cross section """
        deflection_angle = self._deflection_angle(impact_parameter, rel_vel)
        return (1 - np.cos(deflection_angle)**self._l) * impact_parameter

    def _collision_cross_section(self, rel_vel):
        """Compute the collision cross section for a given relative velocity"""
        if hasattr(rel_vel, "__iter__"):
            integral = np.zeros_like(rel_vel)
            for i, g in enumerate(rel_vel):
                integral[i], _ = integrate.quad(self._collision_cross_section_integrand,
                                            0.0, np.inf,
                                            limit=100,
                                            args=(g))
        else:
            integral, _ = integrate.quad(self._collision_cross_section_integrand,
                                        0.0, np.inf,
                                        limit=100,
                                        args=(rel_vel))
        return 2 * np.pi * integral

    def eval(self, gas_state):
        pass


class ColIntLaricchiuta(ColIntModel):
    """
    Dimensionless collision integrals from Laricchiuta et al:
    Classical transport collision integrals for a Lennard-Jones like
    phenomenological model potential 2007

    Good for neutral/neutral or neutral/ion collisions
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # physical parameters
        if "sigma" in kwargs:
            self._sigma = kwargs["sigma"]
            self._epsilon = kwargs["epsilon"]
            self._beta = kwargs.get("beta", 8)
        elif "alphas" in kwargs:
            self._alphas = kwargs["alphas"]
            self._Ns = kwargs["Ns"]
            self._compute_parameters()
        elif "alpha" in kwargs:
            alpha = kwargs["alpha"]
            N = kwargs["N"]
            self._alphas = np.array([alpha, alpha])
            self._Ns = np.array([N, N])
            self._compute_parameters()
        else:
            raise ValueError("No sigma or alpha value provided")

        # set the zeta values
        if self._charge[0] != 0 or self._charge[1] != 0:
            self._zeta = [0.7564, 0.064605]
        if self._charge[0] == 0 and self._charge[1] == 0:
            self._zeta = [0.8002, 0.049256]

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

    def eval(self, gas_state):
        temp = gas_state["temp"]
        temp_star = temp / self._epsilon
        if np.any(temp_star < 2e-4) or np.any(temp_star > 1e3):
            raise ValueError("Temperature outside of valid range")
        a1, a2, a3, a4, a5, a6, a7 = self.get_coeffs()
        x = np.log(temp_star)
        ln_omega = (a1 + a2*x) * \
                    np.exp((x-a3)/a4) / (np.exp((x-a3)/a4) + np.exp((a3-x)/a4)) +\
                    a5 * np.exp((x-a6)/a7) / (np.exp((x-a6)/a7) + np.exp((a6-x)/a7))
        return np.exp(ln_omega) * (self._x_0 * self._sigma)**2


class ColIntCurveFitModel(ColIntModel):
    """ Base class for curve fitted collision integrals"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "cis" in kwargs:
            self._temps = kwargs["temps"]
            self._cis = kwargs["cis"]
            self._evaluate_coeffs()
        elif "coeffs" in kwargs:
            self._a, self._b, self._c, self._d = kwargs["coeffs"]
        else:
            raise ValueError("Curve fit model must have collision "
                             "integrals or coefficients")

    @abstractmethod
    def _curve_fit_form(self, temp, a, b, c, d):
        pass

    def _evaluate_coeffs(self):
        (a, b, c, d), _ = optimize.curve_fit(self._curve_fit_form,
                                             self._temps,
                                             self._cis,
                                             [-0.01, 0.3, -2.5, 11])
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def eval(self, gas_state):
        temp = gas_state["temp"]
        return self._curve_fit_form(temp, self._a, self._b, self._c, self._d)

    def __repr__(self):
        return f"[a={self._a}, b={self._b}, c={self._c}, d={self._d}]"


class ColIntCurveFitPiOmega(ColIntCurveFitModel):
    """ Curve fit of pi * Omega """

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d) / np.pi


class ColIntCurveFitOmega(ColIntCurveFitModel):
    """ Curve fitted collision integral """

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d)


class ColIntCurveFit:
    """
    Factory class for curve fitted collision integrals
    """
    CURVE_FIT_TYPES = {
        "Omega": ColIntCurveFitOmega,
        "pi_Omega": ColIntCurveFitPiOmega,
    }

    def __new__(cls, **kwargs):
        curve_fit_type = kwargs["curve_fit_type"]
        if curve_fit_type not in cls.CURVE_FIT_TYPES:
            raise ValueError(curve_fit_type)
        return cls.CURVE_FIT_TYPES[curve_fit_type](**kwargs)


class ColIntGuptaYos(ColIntCurveFitPiOmega):
    """
    Collision integral from Gupta Yos
    """

    def eval(self, gas_state):
        col_int = super().eval(gas_state)
        if self._charge[0] * self._charge[1] != 0:
            # charged collision, so need to correct for electron pressure
            temp = gas_state["temp"]
            pe = gas_state["ep"]/100000
            col_int *= np.log(2.09e-2 * (temp/1000/pe**0.25)**4
                                + 1.52*(temp/1000/pe**0.25)**(8/3))
            col_int /= np.log(2.09e-2 * (temp/1000)**4 + 1.52*(temp/1000)**(8/3))
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
            species = pair.split(":")

            # check if both species are charged
            charge = [0, 0]
            for i in range(2):
                if "+" in pair[i]:
                    charge[i] = 1
                if "-" in pair[i]:
                    charge[i] = -1

            for ii in ["11", "22"]:
                # extract the numeric order from the string representation
                l = int(ii[0])

                self._ci_coeffs[pair][f"Omega_{ii}"] = ColIntCurveFit(
                    curve_fit_type=self._curve_fit_type,
                    order=(l,l),
                    temps=pair_ci[f"Omega_{ii}"]["temps"],
                    cis=pair_ci[f"Omega_{ii}"]["cis"],
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


def collision_integral(ci_type, **kwargs):
    """
    Create a collision integral

    Parameters:
        ci_type (string): The type of collision integral

    Returns:
        ci: A concrete collision integral instsance
    """
    CI_TYPES = {
        "laricchiuta": ColIntLaricchiuta,
        "gupta_yos": ColIntGuptaYos,
        "curve_fit": ColIntCurveFit,
        #"numerical": NumericCollisionIntegral,
        "mason": MasonColInt,
    }

    return CI_TYPES[ci_type](**kwargs)


if __name__ == "__main__":
    ci = collision_integral(
        "mason",
        order=(1, 1),
        charge=(-1, -1),
        species=("NA", "NA")
    )
    gas_state = {"ne": 1e20, "temp": 1000}
    print("collision integral = ", ci.eval(gas_state))
    print("temp star = ", ci._temp_star(gas_state))
    print("debye length = ", ci._debye_length(gas_state))
    print("temp star square * ci = ", ci.eval(gas_state) * ci._temp_star(gas_state)**2)
