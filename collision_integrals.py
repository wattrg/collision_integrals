# Collision integral curve fitting program
#
# Automates curve fitting collision integrals to the form used by Eilmer4
# The form is given by Gupta, Yos, Thompson et. al. (1989) on page 20.
#
# Author: Robert Watt
# Versions: October 2021: first attempt

from scipy.optimize import curve_fit
from scipy import interpolate
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
                     d_ii should be provided to get pi * Omega, rather
                     than Omega

       Returns:
       pi_omega_ii (float): The estimated value of pi_omega_ii
                            using the parameters provided
       """

    log_T = np.log(temp)
    return np.exp(d_ii) * \
        np.power(temp, a_ii * log_T**2 + b_ii * log_T + c_ii)



class CollisionIntegralModel(ABC):

    @abstractmethod
    def eval(self, temp):
        pass


class HcbCollisionIntegral(CollisionIntegralModel):
    """
    Collision integrals calculated using the non-dimensional collision integrals
    from Hirschfelder, Curtiss, and Bird (1964) appendix I-M (page 1126)
    """

    def __init__(self, sigma, epsilon, l, s):
        self.l = l
        self.s = s

        # get the non-dimensional collision integral data
        temp_star = hcb_ci_data[:, 0]
        if l == 1 and s == 1:
            omega_star = hcb_ci_data[:, 1]
        elif l == 2 and s == 2:
            omega_star = hcb_ci_data[:, 2]
        else:
            raise ValueError((l, s))

        # set L-J potential parmeters
        self._sigma = sigma
        self._epsilon = epsilon

        # non-dimensionalise the collision integrals
        temps = temp_star * self._epsilon
        factor = 1 if l == 1 else 2
        omega = factor * np.pi * self._sigma**2 * omega_star

        # interpolate the data
        self._omega_interp = interpolate.interp1d(temps, omega)

    def eval(self, temp):
        return self._omega_interp(temp)

    def get_sigma(self):
        return self._sigma

    def get_epsilon(self):
        return self._epsilon

class CICurveFitModel(CollisionIntegralModel):
    """ Base class for curve fitted collision integrals"""

    def __init__(self, temps, cis):
        self._temps = temps
        self._cis = cis
        self._evaluate_coeffs()

    @abstractmethod
    def _curve_fit_form(self, temp, a, b, c, d):
        pass

    def _evaluate_coeffs(self):
        (a, b, c, d), _ = curve_fit(self._curve_fit_form,
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
    """Factory for collision integral curve fits"""
    CURVE_FIT_TYPES = {
        "Omega": CICurveFitOmega,
        "pi_Omega": CICurveFitPiOmega,
    }

    def get_curve_fit(self, **kwargs):
        curve_fit_type = kwargs["curve_fit_type"]
        temps = kwargs["temps"]
        cis = kwargs["cis"]
        if curve_fit_type not in self.CURVE_FIT_TYPES:
            raise ValueError(curve_fit_type)
        return self.CURVE_FIT_TYPES[curve_fit_type](temps=temps, cis=cis)


class CICurveFitCollection:
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
                self._ci_coeffs[pair][f"pi_Omega_{ii}"] = CICurveFit().get_curve_fit(
                    curve_fit_type=self._curve_fit_type,
                    temps=pair_ci[f"pi_Omega_{ii}"]["temps"],
                    cis=pair_ci[f"pi_Omega_{ii}"]["cis"]
                )

    def get_coeffs(self, pair=None):
        if pair:
            return self._ci_coeffs[pair]
        return self._ci_coeffs


class CollisionIntegral:
    """ Factory class for collision integrals """
    CI_TYPES = {
        "hcb": HcbCollisionIntegral,
        "curve_fit": CICurveFit
    }

    def __init__(self, **kwargs):
        ci_type = kwargs["ci_type"]
        self.collision_integral = self.CI_TYPES[ci_type](**kwargs)


if __name__ == "__main__":
    cf = CICurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
