# Collision integral curve fitting program
#
# Automates curve fitting collision integrals to the form used by Eilmer4
# The form is given by Gupta, Yos, Thompson et. al. (1989) on page 20.
#
# Author: Robert Watt
# Versions: October 2021: first attempt

from scipy import optimize, interpolate
from scipy.special import factorial
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from hcb_ci_data import hcb_ci_data
from wright_ci_data import wright_ci_data
from abc import ABC, abstractmethod

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
    return np.exp(d_ii) * \
        np.power(temp, a_ii * log_T**2 + b_ii * log_T + c_ii)



class ColIntModel(ABC):
    """ Base class for collision integral models """

    @abstractmethod
    def eval(self, temp):
        pass

def lennard_jones(radius, sigma, epsilon):
    """ compute the Lennard Jones potential at some separation """
    return 4 * epsilon * ((sigma/radius)**12 - (sigma/radius)**6)


class NumericCollisionIntegral(ColIntModel):
    """ Collision integral performed numerically from scratch """

    POTENTIALS = {
        "lennard_jones": lennard_jones,
    }

    def __init__(self, **kwargs):
        if callable(kwargs["potential"]):
            self._potential_func = kwargs["potential"]
        else:
            self._potential_func = self.POTENTIALS[kwargs["potential"]]
        self._epsilon = kwargs["epsilon"]
        self._sigma = kwargs["sigma"]
        self._mu = kwargs["mu"]
        self._l = kwargs["l"]
        self._s = kwargs["s"]

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
        return optimize.root_scalar(self._r_m_func,
                                    bracket=[1e-10, 1e5],
                                    args=(impact_param, rel_vel))

    def _deflection_integrand(self, radius, impact_param, rel_vel):
        """ Integrand for computing the deflection angle """
        gamma_2 = 0.5 * self._mu * rel_vel**2
        tmp = np.sqrt(1 - self._potential(radius)/gamma_2 - (impact_param/radius)**2)
        return radius**2 / tmp

    def _deflection_angle(self, impact_param, rel_vel):
        """
        Compute the deflection angle for a given relative velocity and
        impact parameter
        """
        r_m = self._calc_r_m(impact_param, rel_vel).root
        integral, _ = integrate.quad(self._deflection_integrand,
                                     r_m, np.inf,
                                     limit=100,
                                     args=(impact_param, rel_vel))
        return np.pi - 2 * impact_param * integral

    def _collision_cross_section_integrand(self, impact_parameter, rel_vel):
        """ Compute the integrand for the collision cross section """
        deflection_angle = self._deflection_angle(impact_parameter, rel_vel)
        return (1 - np.cos(deflection_angle)**self._l) * impact_parameter

    def _collision_cross_section(self, rel_vel):
        """Compute the collision cross section for a given relative velocity"""
        integral, _ = integrate.quad(self._collision_cross_section_integrand,
                                     0.0, np.inf,
                                     limit=100,
                                     args=(rel_vel))
        return 2 * np.pi * integral

    def eval(self, temp):
        pass

class DimensionlessColInt(ColIntModel):
    """
    Collision integrals calculated using the non-dimensional collision integrals
    from Hirschfelder, Curtiss, and Bird (1964) appendix I-M (page 1126)
    """
    # Boltzmann's consant
    k_B = 1.3806e-3 # angstrom^2 kg s^-2 K^-1

    # Avagradro's number
    N_A = 6.022e23

    def __init__(self, **kwargs):
        # order of the collision integral
        self.l = kwargs["l"]
        self.s = kwargs["s"]

        # set L-J potential parmeters
        self._sigma = kwargs["sigma"]
        self._epsilon = kwargs["epsilon"]

        # reduced mass
        self._mu = kwargs["mu"]

        # get the non-dimensional collision integral data
        temp_star = hcb_ci_data[:, 0]
        if self.l == 1 and self.s == 1:
            omega_star = hcb_ci_data[:, 1]
        elif self.l == 2 and self.s == 2:
            omega_star = hcb_ci_data[:, 2]
        else:
            raise ValueError((l, s))

        # non-dimensionalise the collision integrals
        temps = temp_star * self._epsilon
        factor = 0.5 * factorial((self.s + 1)) * (1 - 0.5 * ((1 + (-1)**self.l)/(1 + self.l)))
        mass_factor = np.sqrt(2 * np.pi * self._mu / (self.k_B * temps))
        omega = factor * np.pi * self._sigma**2 * omega_star * mass_factor

        # interpolate the data
        self._omega_interp = interpolate.interp1d(temps, omega, kind="quadratic")

    def eval(self, temp):
        return self._omega_interp(temp)

    def get_sigma(self):
        return self._sigma

    def get_epsilon(self):
        return self._epsilon


class CICurveFitModel(ColIntModel):
    """ Base class for curve fitted collision integrals"""

    def __init__(self, temps, cis):
        self._temps = temps
        self._cis = cis
        self._evaluate_coeffs()

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

    def eval(self, temp):
        return self._curve_fit_form(temp, self._a, self._b, self._c, self._d)

    def __repr__(self):
        return f"[a={self._a}, b={self._b}, c={self._c}, d={self._d}]"


class CICurveFitPiOmega(CICurveFitModel):
    """ Curve fit of pi * Omega """

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d) / np.pi


class CICurveFitOmega(CICurveFitModel):
    """ Curve fitted collision integral """

    def _curve_fit_form(self, temp, a, b, c, d):
        return omega_curve_fit(temp, a, b, c, d)


class CICurveFit:
    """ Factory for collision integral curve fits """
    CURVE_FIT_TYPES = {
        "Omega": CICurveFitOmega,
        "pi_Omega": CICurveFitPiOmega,
    }

    def construct_ci(self, **kwargs):
        curve_fit_type = kwargs["curve_fit_type"]
        temps = kwargs["temps"]
        cis = kwargs["cis"]
        if curve_fit_type not in self.CURVE_FIT_TYPES:
            raise ValueError(curve_fit_type)
        return self.CURVE_FIT_TYPES[curve_fit_type](temps=temps, cis=cis)


class CollisionIntegral:
    """ Factory class for collision integrals """
    CI_TYPES = {
        "dimensionless": DimensionlessColInt,
        "curve_fit": CICurveFit,
        "numerical": NumericCollisionIntegral,
    }

    def construct_ci(self, **kwargs):
        ci_type = kwargs["ci_type"]
        return self.CI_TYPES[ci_type](**kwargs)


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
            for ii in ["11", "22"]:
                self._ci_coeffs[pair][f"pi_Omega_{ii}"] = CICurveFit().construct_ci(
                    curve_fit_type=self._curve_fit_type,
                    temps=pair_ci[f"pi_Omega_{ii}"]["temps"],
                    cis=pair_ci[f"pi_Omega_{ii}"]["cis"]
                )

    def get_cis(self, pair=None, ci_type=None):
        """ Return the collision integrals """
        if pair:
            if ci_type:
                return self._ci_coeffs[pair][ci_type]
            return self._ci_coeffs[pair]
        return self._ci_coeffs
