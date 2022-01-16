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

wright_ci_data = {
    "N2:N2": {
        "Omega_11": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([12.23, 10.60, 9.79, 8.60, 7.49, 6.87, 6.43, 6.06]),
            "acc": 0.1
        },
        "Omega_22": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([13.72, 11.80, 10.94, 9.82, 8.70, 8.08, 7.58, 7.32]),
            "acc": 0.1
        }
    },
    "N2:O2": {
        "Omega_11": {
            "temps": np.array([300, 1000, 2000, 4000, 5000, 10000, 15000]),
            "cis": np.array([10.16, 7.39, 6.42, 5.59, 5.35, 4.6, 4.2]),
            "acc": 0.2
        },
        "Omega_22": {
            "temps": np.array([300, 1000, 2000, 4000, 5000, 10000, 15000]),
            "cis": np.array([11.23, 8.36, 7.35, 6.47, 6.21, 5.42, 4.94]),
            "acc": 0.2
        }
    },
    "N2:O": {
        "Omega_11": {
            "temps": np.array([300, 1000, 2000, 4000, 5000, 10000, 15000]),
            "cis": np.array([8.07, 5.93, 5.17, 4.77, 4.31, 3.71, 3.38]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 1000, 2000, 4000, 5000, 10000, 15000]),
            "cis": np.array([8.99, 6.72, 5.91, 5.22, 5.01, 4.36, 3.95]),
            "acc": 0.2
        }
    },
    "N2:N": {
        "Omega_11": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([10.1, 8.57, 7.7, 6.65, 5.65, 5.05, 4.61, 4.25]),
            "acc": 0.1,
        },
        "Omega_22": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([11.21, 9.68, 8.81, 7.76, 6.73, 6.18, 5.74, 5.36]),
            "acc": 0.1,
        }
    },
    "N2:NO": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([11.8, 10.61, 10.24, 9.35, 8.12, 6.82, 6.43, 6.12, 5.66, 5.31, 4.71]),
            "acc": 0.25,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([13.44, 11.87, 11.44, 10.48, 9.32, 8.04, 7.61, 7.27, 6.74, 6.33, 5.62]),
            "acc": 0.25,
        }
    },
    "O2:O2": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([11.12, 9.88, 9.53, 8.69, 7.6, 6.52, 6.22, 5.99, 5.64, 5.39, 4.94]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([12.62, 11.06, 10.65, 9.72, 8.70, 7.7, 7.38, 7.12, 6.73, 6.42, 5.89]),
            "acc": 0.2
        }
    },
    "O2:N": {
        "Omega_11": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([7.56, 7.26, 6.55, 5.6, 4.75, 4.49, 4.28, 3.96, 3.72, 3.31]),
            "acc": 0.25,
        },
        "Omega_22": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([8.79, 8.47, 7.68, 6.63, 5.67, 5.38, 5.14, 4.78, 4.51, 4.04]),
            "acc": 0.25,
        }
    },
    "O2:O": {
        "Omega_11": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([9.1, 7.58, 6.74, 5.7, 4.78, 4.29, 3.96, 3.71]),
            "acc": 0.1,
        },
        "Omega_22": {
            "temps": np.array([300, 600, 1000, 2000, 4000, 6000, 8000, 10000]),
            "cis": np.array([10.13, 8.61, 7.78, 6.71, 5.67, 5.13, 4.78, 4.5]),
            "acc": 0.1,
        }
    },
    "O2:NO": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([11.39, 10.1, 9.75, 8.89, 7.74, 6.56, 6.23, 5.98, 5.59, 5.31, 4.82]),
            "acc": 0.25,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([12.93, 11.32, 10.9, 9.94, 8.89, 7.8, 7.45, 7.17, 6.73, 6.39, 5.8]),
            "acc": 0.25,
        }
    },
    "N:N": {
        "Omega_11": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([8.07, 7.03, 5.96, 5.15, 4.39, 4.14, 3.94, 3.61, 3.37, 2.92, 2.62]),
            "acc": 0.05,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([9.11, 7.94, 6.72, 5.82, 4.98, 4.7, 4.48, 4.14, 3.88, 3.43, 3.11]),
            "acc": 0.05,
        }
    },
    "N:O": {
        "Omega_11": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([8.32, 7.34, 6.22, 5.26, 4.45, 4.21, 4.01, 3.69, 3.43, 2.98, 2.66]),
            "acc": 0.05,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([9.08, 8.15, 7.09, 6.06, 5.14, 4.88, 4.67, 4.34, 4.07, 3.56, 3.21]),
            "acc": 0.05,
        },
    },
    "N:NO": {
        "Omega_11": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([8.21, 7.86, 6.99, 5.9, 4.91, 4.61, 4.37, 4.01, 3.73, 3.27]),
            "acc": 0.25,
        },
        "Omega_22": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([9.65, 9.26, 8.29, 7.07, 5.94, 5.6, 5.33, 4.91, 4.6, 4.06]),
            "acc": 0.25,
        },
    },
    "O:O": {
        "Omega_11": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([8.53, 7.28, 5.89, 4.84, 4.0, 3.76, 3.57, 3.27, 3.05, 2.65, 2.39]),
            "acc": 0.05,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([9.46, 8.22, 6.76, 5.58, 4.67, 4.41, 4.2, 3.88, 3.64, 3.21, 2.91]),
            "acc": 0.05,
        },
    },
    "O:NO": {
        "Omega_11": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([7.57, 7.27, 6.55, 5.62, 4.78, 4.52, 4.31, 4.0, 3.76, 3.35]),
            "acc": 0.25,
        },
        "Omega_22": {
            "temps": np.array([500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([8.79, 8.47, 7.66, 6.64, 5.69, 5.4, 5.17, 4.82, 4.55, 4.08]),
            "acc": 0.25
        },
    },
    "NO:NO": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([11.66, 10.33, 9.97, 9.09, 7.9, 6.6, 6.24, 5.96, 5.54, 5.23, 4.7]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000]),
            "cis": np.array([13.25, 11.58, 11.15, 10.16, 9.07, 7.91, 7.53, 7.21, 6.73, 6.36, 5.72]),
            "acc": 0.2,
        },
    },
    "CO2:CO2": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([17.35, 14.39, 13.66, 12.12, 10.66, 9.47, 9.13, 8.86, 8.44, 8.12, 7.54, 7.15]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([20.35, 16.45, 15.50, 13.58, 11.92, 10.69, 10.32, 10.02, 9.57, 9.21, 8.59, 8.15]),
            "acc": 0.2,
        }
    },
    "CO2:N2": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([14.17, 12.36, 11.88, 10.77, 9.52, 8.25, 7.86, 7.54, 7.05,6.68, 6.04, 5.60]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([16.24, 13.91, 13.32, 12.03, 10.74, 9.49, 9.08, 8.74, 8.20, 7.79, 7.06, 6.5620]),
            "acc": 0.2,
        }
    },
    "N2:CO2": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([14.17, 12.36, 11.88, 10.77, 9.52, 8.25, 7.86, 7.54, 7.05,6.68, 6.04, 5.60]),
            "acc": 0.2,
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([16.24, 13.91, 13.32, 12.03, 10.74, 9.49, 9.08, 8.74, 8.20, 7.79, 7.06, 6.5620]),
            "acc": 0.2,
        }
    },
    "CO2:CO": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([14.18, 12.36, 11.88, 10.77, 9.52, 8.26, 7.86, 7.54, 7.05, 6.68, 6.04, 5.60])
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([16.24, 13.92, 13.32, 12.03, 10.74, 9.50, 9.08, 8.74, 8.21, 7.79, 7.07, 6.56])
        }
    },
    "CO2:O2": {
        "Omega_11": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([13.68, 11.84, 11.36, 10.27, 9.08, 7.97, 7.64, 7.38, 6.98, 6.69, 6.17, 5.82])
        },
        "Omega_22": {
            "temps": np.array([300, 500, 600, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]),
            "cis": np.array([15.74, 13.35, 12.76, 11.47, 10.23, 9.15, 8.80, 8.53, 8.10, 7.77, 7.19, 6.79])
        }
    }
}
