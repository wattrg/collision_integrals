# Collision integral parameters for 11-species air.
# Source:
#    Gupta, Yos, Thompson and Lee (1990)
#    A Review of Reaction Rates and Thermodynamics and Transport Properties
#    for an 11-Species Air Model for Chemical and Thermal Nonequilibrium Calculations to 30 000 K

from math import log
viscous_factor = 100
very_viscous_gupta_yos_data = {
    ('N2', 'N2'): {
        (1,1): [log(viscous_factor), -0.0112, -0.1182, 4.8464],
        (2,2): [log(viscous_factor), -0.0203, 0.0683, 4.0900]
    },
    ("O2", "N2"): {
        (1,1): [log(viscous_factor), -0.0465, 0.5729, 1.6185],
        (2,2): [log(viscous_factor), -0.0558, 0.7590,  0.8955],
    },
    ("O2", "O2"): {
        (1,1): [log(viscous_factor), -0.0410,  0.4977, 1.8302],
        (2,2): [log(viscous_factor), -0.0485,  0.6475, 1.2607]
    },
    ("N", "N2"): {
        (1,1):  [log(viscous_factor), -0.0194,  0.0119, 4.1055],
        (2,2):  [log(viscous_factor), -0.0190,  0.0239, 4.1782],
    },
    ("N", "O2"): {
        (1,1):  [log(viscous_factor), -0.0179,  0.0152, 3.9996],
        (2,2):  [log(viscous_factor), -0.0203,  0.0703, 3.8818],
    },
    ("N", "N"): {
        (1,1):   [log(viscous_factor), -0.0033, -0.0572, 5.0452],
        (2,2):   [log(viscous_factor), -0.0118, -0.0960, 4.3252],
    },
    ("O", "N2"): {
        (1,1):  [log(viscous_factor), -0.0139, -0.0825, 4.5785],
        (2,2):  [log(viscous_factor), -0.0169, -0.0143, 4.4195],
    },
    ("O", "O2"): {
        (1,1):  [log(viscous_factor), -0.0226,  0.1300, 3.3363],
        (2,2):  [log(viscous_factor), -0.0247,  0.1783, 3.2517],
    },
    ("O", "N"): {
        (1,1):   [log(viscous_factor), 0.0048, -0.4195, 5.7774],
        (2,2):   [log(viscous_factor), 0.0065, -0.4467, 6.0426],
    },
    ("O", "O"): {
        (1,1):   [log(viscous_factor), -0.0034, -0.0572, 4.9901],
       (2,2):   [log(viscous_factor), -0.0207,  0.0780, 3.5658],
    },
    ("NO", "N2"): {
        (1,1): [log(viscous_factor), -0.0291,  0.2324, 3.2082],
        (2,2): [log(viscous_factor), -0.0385,  0.4226, 2.4507],
    },
    ("NO", "O2"): {
        (1,1): [log(viscous_factor), -0.0438,  0.5352, 1.7252],
        (2,2): [log(viscous_factor), -0.0522,  0.7045, 1.0738],
    },
    ("NO", "N"): {
        (1,1):  [log(viscous_factor), -0.0185,  0.0118, 4.0590],
        (2,2):  [log(viscous_factor), -0.0196,  0.0478, 4.0321],
    },
    ("NO", "O"): {
        (1,1):  [log(viscous_factor), -0.0179,  0.0152, 3.9996],
        (2,2):  [log(viscous_factor), -0.0203,  0.0703, 3.8818],
    },
    ("NO", "NO"): {
        (1,1): [log(viscous_factor), -0.0364,  0.3825, 2.4718],
        (2,2): [log(viscous_factor), -0.0453,  0.5624, 1.7669],
    },
    ('NO+', 'N2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('NO+', 'O2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760]
    },
    ('NO+', 'N'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('NO+', 'O'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('NO+', 'NO'): {
        (1,1): [log(viscous_factor), -0.0047, -0.0551, 4.8737],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('NO+', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('e-', 'N2'): {
        (1,1): [0.1147 + log(viscous_factor), -2.8945,  24.5080, -67.3691],
        (2,2): [0.1147 + log(viscous_factor), -2.8945,  24.5080, -67.3691],
    },
    ('e-', 'O2'): {
        (1,1): [0.0241+log(viscous_factor), -0.3467,  1.3887, -0.0110],
        (2,2): [0.0241+log(viscous_factor), -0.3467,  1.3887, -0.0110],
    },
    ('e-', 'N'): {
        (1,1): [log(viscous_factor), 0.0000,  0.0000, 1.6094],
        (2,2): [log(viscous_factor), 0.0000,  0.0000, 1.6094],
    },
    ('e-', 'O'): {
        (1,1): [0.0164, -0.2431,  1.1231, -1.5561],
        (2,2): [0.0164, -0.2431,  1.1231, -1.5561],
    },
    ('e-', 'NO'): {
        (1,1): [-0.2202, 5.2265, -40.5659, 104.7126],
        (2,2): [-0.2202, 5.2265, -40.5659, 104.7126],
    },
    ('e-', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3061],
    },
    ('e-', 'e-'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3061],
    },
    ('N+', 'N2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N+', 'O2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N+', 'N'): {
        (1,1): [log(viscous_factor), -0.0033, -0.0572, 5.0452],
        (2,2): [log(viscous_factor), 0.0000, -0.4146, 6.9078],
    },
    ('N+', 'O'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N+', 'NO'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N+', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('N+', 'e-'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3601],
    },
    ('N+', 'N+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O+', 'N2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O+', 'O2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O+', 'N'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O+', 'O'): {
        (1,1): [log(viscous_factor), -0.0034, -0.0572, 4.9901],
        (2,2): [log(viscous_factor), 0.0000, -0.4235, 6.7787],
    },
    ('O+', 'NO'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O+', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O+', 'e-'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3601],
    },
    ('O+', 'N+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O+', 'O+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('N2+', 'N2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N2+', 'O2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N2+', 'N'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N2+', 'O'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N2+', 'NO'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('N2+', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('N2+', 'e-'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3601],
    },
    ('N2+', 'N+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('N2+', 'O+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('N2+', 'N2+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O2+', 'N2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O2+', 'O2'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O2+', 'N'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O2+', 'O'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O2+', 'NO'): {
        (1,1): [log(viscous_factor), 0.0000, -0.4000, 6.8543],
        (2,2): [log(viscous_factor), 0.0000, -0.4000, 6.7760],
    },
    ('O2+', 'NO+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O2+', 'e-'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3601],
    },
    ('O2+', 'N+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O2+', 'O+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O2+', 'N2+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
    ('O2+', 'O2+'): {
        (1,1): [log(viscous_factor), 0.0000, -2.0000, 23.8237],
        (2,2): [log(viscous_factor), 0.0000, -2.0000, 24.3602],
    },
}
