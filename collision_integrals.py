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



class CollisionIntegralModel(ABC):
    """ Base class for collision integral models """

    @abstractmethod
    def eval(self, temp):
        pass

def lennard_jones(radius, sigma, epsilon):
    """ compute the Lennard Jones potential at some separation """
    pass


class NumericCollisionIntegral(CollisionIntegralModel):
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
        tmp = -(radius**2/impact_param)
        tmp *= np.sqrt(1 - self._potential(radius)/gamma_2 - (impact_param/radius)**2)
        return tmp

    def _calc_r_m(self, impact_param, rel_vel):
        """ Compute the value of r_m for computing the deflection angle """
        return optimize.root_scalar(self._r_m_func, args=(impact_param, rel_vel))


    def _deflection_integrand(self, rel_vel, impact_param):
        """ Integrand for computing the deflection angle """

    def _deflection_angle(self, rel_vel, impact_param):
        """
        Compute the deflection angle for a given relative velocity and
        impact parameter
        """
        r_m = self._calc_r_m(impact, param, rel_vel)
        integrate.quad()
        pass


class HcbCollisionIntegral(CollisionIntegralModel):
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
        print(temps)
        factor = 0.5 * factorial((self.s + 1)) * (1 - 0.5 * ((1 + (-1)**self.l)/(1 + self.l)))
        mass_factor = np.sqrt(self.k_B * temps / (2 * np.pi * self._mu))
        omega = factor * np.pi * self._sigma**2 * omega_star * mass_factor

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

    def get_curve_fit(self, **kwargs):
        curve_fit_type = kwargs["curve_fit_type"]
        temps = kwargs["temps"]
        cis = kwargs["cis"]
        if curve_fit_type not in self.CURVE_FIT_TYPES:
            raise ValueError(curve_fit_type)
        return self.CURVE_FIT_TYPES[curve_fit_type](temps=temps, cis=cis)


class CollisionIntegral:
    """ Factory class for collision integrals """
    CI_TYPES = {
        "hcb": HcbCollisionIntegral,
        "curve_fit": CICurveFit
    }

    def construct_ci(self, **kwargs):
        ci_type = kwargs["ci_type"]
        return self.CI_TYPES[ci_type](**kwargs)


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


if __name__ == "__main__":
    cf = CICurveFitCollection(ci_table=wright_ci_data, curve_fit_type="pi_Omega")
    ci = CollisionIntegral()
    hcb_co2_co2_11 = ci.construct_ci(ci_type="hcb",
                                     sigma=3.763, epsilon=244, mu=0.04401,
                                     l=1, s=1)
    hcb_co2_co2_22 = ci.construct_ci(ci_type="hcb",
                                     sigma=3.763, epsilon=244, mu=0.04401,
                                     l=2, s=2)
