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


def omega_ii_curve_fit(temp, a_ii, b_ii, c_ii, d_ii):
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


class CurveFitCoeffs:
    """
    Handles the curve fit parameters
    """
    def __init__(self, a, b, c, d, ci_type = "omega"):
        self._type = ci_type
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._d_pi_omega = self._d + np.log(np.pi)


    def __repr__(self):
        return f"[a = {self._a}, b = {self._b}, c = {self._c}, d = {self._d_pi_omega}]"

    def get_a(self):
        return self._a

    def get_b(self):
        return self._b

    def get_c(self):
        return self._c

    def get_d(self):
        return self._d_pi_omega


class CICurveFit:
    """
    Class to handle curve fitting collision integral data
    in the form given by Gupta, Yos, Thompson et al (1989)
    """

    def __init__(self, ci_table):
        self.data = ci_table
        self.ci_coeffs = {}
        for pair, pair_ci in ci_table.items():
            self.ci_coeffs[pair] = {}
            for ii in ["11", "22"]:
                (a, b, c, d), _ = curve_fit(omega_ii_curve_fit,
                                            pair_ci[f"pi_Omega_{ii}"]["temps"],
                                            pair_ci[f"pi_Omega_{ii}"]["cis"],
                                            [-0.01, 0.3, -2.5, 11])
                self.ci_coeffs[pair][f"pi_Omega_{ii}"] = CurveFitCoeffs(a, b, c, d)

    def get_coeffs(self):
        return self.ci_coeffs


    def plot_fit(self, pair, ci_type):
        """
        Plots the curve fit to compare to the actual data
        """
        temps = np.linspace(300, 20000)
        a = self.ci_coeffs[pair][f"pi_Omega_{ci_type}"].get_a()
        b = self.ci_coeffs[pair][f"pi_Omega_{ci_type}"].get_b()
        c = self.ci_coeffs[pair][f"pi_Omega_{ci_type}"].get_c()
        d = self.ci_coeffs[pair][f"pi_Omega_{ci_type}"].get_d()
        curve_fit = omega_ii_curve_fit(temps, a, b, c, d)

        fig, ax = plt.subplots()
        ax.scatter(self.data[pair][f"pi_Omega_{ci_type}"]["temps"],
                   self.data[pair][f"pi_Omega_{ci_type}"]["cis"]*np.pi,
                   label="Wright et al")
        ax.plot(temps, curve_fit, label="Curve fit")
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel(f"$\\pi \\Omega^{{({ci_type})}}_{{{pair}}}$, $\AA$")
        ax.legend()
        fig.show()


class CollisionIntegral:
    """
    Factory class for collision integrals
    """


class HCB_CI:
    """
    Collision integrals calculated using the non-dimensional collision integrals
    from Hirschfelder, Curtiss, and Bird (1964) appendix I-M (page 1126)
    """

    def __init__(self, sigma, epsilon):
        self.sigma = sigma
        self.epsilon = epsilon

        self._omega_11_star = interpolate.interp1d(hcb_ci_data[:,0], hcb_ci_data[:,1])
        self._omega_22_star = interpolate.interp1d(hcb_ci_data[:,0], hcb_ci_data[:,2])

    def omega_11(self, temp):
        temp_star = temp/self.epsilon
        return np.pi * self.sigma**2 * self._omega_11_star(temp_star)

    def omega_22(self, temp):
        temp_star = temp/self.epsilon
        return 2 * np.pi * self.sigma**2 * self._omega_22_star(temp_star)


if __name__ == "__main__":
    pass
