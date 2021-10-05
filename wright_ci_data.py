# Collision integral data for Air, Mars and Venus
# Sources:
#   Wright, Hwang and Schewenke (2007)
#   Recommended Collision Integrals for Transport Property Computations
#   Part 2: Mars and Venus Entries
#   AIAA Journal, 45(1) pp. 281--288
#
#   Wright, Bose, Palmer and Levin (2005)
#   Recommended Collision Integrals for Transport Property Computations
#   Part 1: Air species
#   AIAA Journal, 43(12), pp. 2558--2564

import numpy as np

ci_data = {
    "CO2:CO2": {
        "pi_Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([17.35, 14.39, 13.66, 12.12, 10.66, 9.47, 9.13, 8.86, 8.44, 8.12, 7.54, 7.15])
        },
        "pi_Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([20.35, 16.45, 15.50, 13.58, 11.92, 10.69, 10.32, 10.02, 9.57, 9.21, 8.59, 8.15])
        }
    },
    "CO2:CO": {
        "pi_Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([14.18, 12.36, 11.88, 10.77, 9.52, 8.26, 7.86, 7.54, 7.05, 6.68, 6.04, 5.60])
        },
        "pi_Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([16.24, 13.92, 13.32, 12.03, 10.74, 9.50, 9.08, 8.74, 8.21, 7.79, 7.07, 6.56])
        }
    },
    "CO2:O2": {
        "pi_Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([13.68, 11.84, 11.36, 10.27, 9.08, 7.97, 7.64, 7.38, 6.98, 6.69, 6.17, 5.82])
        },
        "pi_Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([15.74, 13.35, 12.76, 11.47, 10.23, 9.15, 8.80, 8.53, 8.10, 7.77, 7.19, 6.79])
        }
    }
}
