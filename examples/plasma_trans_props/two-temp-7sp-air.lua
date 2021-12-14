-- Auto-generated by prep-gas on: 14-Dec-2021 14:37:38

model = 'CompositeGas'
species = {'N2', 'O2', 'N', 'O', 'NO', 'NO+', 'e-', }

physical_model = 'two-temperature-gas'
db = {}
db['N2'] = {}
db['N2'].type = 'molecule'
db['N2'].molecule_type = 'linear'
db['N2'].theta_v = 3393.440
db['N2'].atomicConstituents = { N=2, }
db['N2'].charge = 0
db['N2'].M = 2.80134000e-02
db['N2'].Hf = 0.00000000e+00
db['N2'].M = 2.80134000e-02
db['N2'].sigma = 3.62100000
db['N2'].epsilon = 97.53000000
db['N2'].Lewis = 1.15200000
db['N2'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 3, 
  T_break_points = { 200.00, 1000.00, 6000.00, 20000.00, },
  T_blend_ranges = { 400.0, 1000.0, },
  segment0 = {
    2.210371497e+04,
   -3.818461820e+02,
    6.082738360e+00,
   -8.530914410e-03,
    1.384646189e-05,
   -9.625793620e-09,
    2.519705809e-12,
    7.108460860e+02,
   -1.076003744e+01,
  },
  segment1 = {
    5.877124060e+05,
   -2.239249073e+03,
    6.066949220e+00,
   -6.139685500e-04,
    1.491806679e-07,
   -1.923105485e-11,
    1.061954386e-15,
    1.283210415e+04,
   -1.586640027e+01,
  },
  segment2 = {
    8.310139160e+08,
   -6.420733540e+05,
    2.020264635e+02,
   -3.065092046e-02,
    2.486903333e-06,
   -9.705954110e-11,
    1.437538881e-15,
    4.938707040e+06,
   -1.672099740e+03,
  },
}
db['O2'] = {}
db['O2'].type = 'molecule'
db['O2'].molecule_type = 'linear'
db['O2'].theta_v = 2273.530
db['O2'].atomicConstituents = { O=2, }
db['O2'].charge = 0
db['O2'].M = 3.19988000e-02
db['O2'].Hf = 0.00000000e+00
db['O2'].M = 3.19988000e-02
db['O2'].sigma = 3.45800000
db['O2'].epsilon = 107.40000000
db['O2'].Lewis = 1.08600000
db['O2'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 3, 
  T_break_points = { 200.00, 1000.00, 6000.00, 20000.00, },
  T_blend_ranges = { 400.0, 1000.0, },
  segment0 = {
   -3.425563420e+04,
    4.847000970e+02,
    1.119010961e+00,
    4.293889240e-03,
   -6.836300520e-07,
   -2.023372700e-09,
    1.039040018e-12,
   -3.391454870e+03,
    1.849699470e+01,
  },
  segment1 = {
   -1.037939022e+06,
    2.344830282e+03,
    1.819732036e+00,
    1.267847582e-03,
   -2.188067988e-07,
    2.053719572e-11,
   -8.193467050e-16,
   -1.689010929e+04,
    1.738716506e+01,
  },
  segment2 = {
    4.975294300e+08,
   -2.866106874e+05,
    6.690352250e+01,
   -6.169959020e-03,
    3.016396027e-07,
   -7.421416600e-12,
    7.278175770e-17,
    2.293554027e+06,
   -5.530621610e+02,
  },
}
db['N'] = {}
db['N'].type = 'atom'
db['N'].atomicConstituents = { N=1, }
db['N'].charge = 0
db['N'].M = 1.40067000e-02
db['N'].Hf = 4.72680000e+05
db['N'].M = 1.40067000e-02
db['N'].sigma = 3.29800000
db['N'].epsilon = 71.40000000
db['N'].Lewis = 1.15200000
db['N'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 4, 
  T_break_points = { 200.00, 1000.00, 6000.00, 20000.00, 50000.00, },
  T_blend_ranges = { 400.0, 1000.0, 1000.0, },
  segment0 = {
    0.000000000e+00,
    0.000000000e+00,
    2.500000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    5.610610630e+04,
    4.194251390e+00,
  },
  segment1 = {
   -2.270732770e+05,
    8.140529440e+02,
    1.327051370e+00,
    8.627217310e-04,
   -3.357470890e-07,
    6.290106870e-11,
   -3.906745870e-15,
    5.109431410e+04,
    1.228237130e+01,
  },
  segment2 = {
   -2.047389940e+09,
    1.458428470e+06,
   -4.188338240e+02,
    6.259944070e-02,
   -4.965428220e-06,
    1.982470520e-10,
   -3.054701940e-15,
   -1.127277300e+07,
    3.584874170e+03,
  },
  segment3 = {
    5.742919020e+11,
   -1.290392940e+08,
    1.153814670e+04,
   -5.250785680e-01,
    1.292190900e-05,
   -1.639742310e-10,
    8.418785850e-16,
    1.152618360e+09,
   -1.116492320e+05,
  },
}
db['O'] = {}
db['O'].type = 'atom'
db['O'].atomicConstituents = { O=1, }
db['O'].charge = 0
db['O'].M = 1.59994000e-02
db['O'].Hf = 2.49175003e+05
db['O'].M = 1.59994000e-02
db['O'].sigma = 2.75000000
db['O'].epsilon = 80.00000000
db['O'].Lewis = 0.71200000
db['O'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 4, 
  T_break_points = { 200.00, 1000.00, 6000.00, 20000.00, 50000.00, },
  T_blend_ranges = { 400.0, 1000.0, 1000.0, },
  segment0 = {
    0.000000000e+00,
    0.000000000e+00,
    2.500000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    2.927275780e+04,
    5.204407390e+00,
  },
  segment1 = {
   -4.594930000e+05,
    1.360042990e+03,
    9.307798890e-01,
    9.081572960e-04,
   -2.817466540e-07,
    4.494516750e-11,
   -2.685644900e-15,
    2.061677860e+04,
    1.639289840e+01,
  },
  segment2 = {
   -1.361804880e+09,
    9.176836840e+05,
   -2.468849420e+02,
    3.485201650e-02,
   -2.617134300e-06,
    9.915441590e-11,
   -1.441920000e-15,
   -7.153289770e+06,
    2.140150020e+03,
  },
  segment3 = {
    4.381213090e+11,
   -8.983106230e+07,
    7.336265460e+03,
   -3.053691850e-01,
    6.903180540e-06,
   -8.092578220e-11,
    3.863224590e-16,
    8.105768910e+08,
   -7.164011440e+04,
  },
}
db['NO'] = {}
db['NO'].type = 'molecule'
db['NO'].molecule_type = 'linear'
db['NO'].theta_v = 2739.700
db['NO'].atomicConstituents = { N=1, O=1, }
db['NO'].charge = 0
db['NO'].M = 3.00061000e-02
db['NO'].Hf = 9.12713100e+04
db['NO'].M = 3.00061000e-02
db['NO'].sigma = 3.62100000
db['NO'].epsilon = 97.53000000
db['NO'].Lewis = 1.15200000
db['NO'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 3, 
  T_break_points = { 200.00, 1000.00, 6000.00, 20000.00, },
  T_blend_ranges = { 400.0, 1000.0, },
  segment0 = {
   -1.143916503e+04,
    1.536467592e+02,
    3.431468730e+00,
   -2.668592368e-03,
    8.481399120e-06,
   -7.685111050e-09,
    2.386797655e-12,
    9.098214410e+03,
    6.728725490e+00,
  },
  segment1 = {
    2.239018716e+05,
   -1.289651623e+03,
    5.433936030e+00,
   -3.656034900e-04,
    9.880966450e-08,
   -1.416076856e-11,
    9.380184620e-16,
    1.750317656e+04,
   -8.501669090e+00,
  },
  segment2 = {
   -9.575303540e+08,
    5.912434480e+05,
   -1.384566826e+02,
    1.694339403e-02,
   -1.007351096e-06,
    2.912584076e-11,
   -3.295109350e-16,
   -4.677501240e+06,
    1.242081216e+03,
  },
}
db['NO+'] = {}
db['NO+'].type = 'molecule'
db['NO+'].molecule_type = 'linear'
db['NO+'].theta_v = 2739.700
db['NO+'].atomicConstituents = { N=1, O=1, }
db['NO+'].charge = 1
db['NO+'].M = 3.00055514e-02
db['NO+'].Hf = 9.90809704e+05
db['NO+'].M = 3.00055514e-02
db['NO+'].sigma = 3.62100000
db['NO+'].epsilon = 97.53000000
db['NO+'].Lewis = 1.15200000
db['NO+'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 3, 
  T_break_points = { 298.15, 1000.00, 6000.00, 20000.00, },
  T_blend_ranges = { 400.0, 1000.0, },
  segment0 = {
    1.398106635e+03,
   -1.590446941e+02,
    5.122895400e+00,
   -6.394388620e-03,
    1.123918342e-05,
   -7.988581260e-09,
    2.107383677e-12,
    1.187495132e+05,
   -4.398433810e+00,
  },
  segment1 = {
    6.069876900e+05,
   -2.278395427e+03,
    6.080324670e+00,
   -6.066847580e-04,
    1.432002611e-07,
   -1.747990522e-11,
    8.935014060e-16,
    1.322709615e+05,
   -1.519880037e+01,
  },
  segment2 = {
    2.676400347e+09,
   -1.832948690e+06,
    5.099249390e+02,
   -7.113819280e-02,
    5.317659880e-06,
   -1.963208212e-10,
    2.805268230e-15,
    1.443308939e+07,
   -4.324044462e+03,
  },
}
db['e-'] = {}
db['e-'].type = 'electron'
db['e-'].atomicConstituents = { }
db['e-'].charge = -1
db['e-'].M = 5.48579903e-07
db['e-'].Hf = 0.00000000e+00
db['e-'].M = 5.48579903e-07
db['e-'].sigma = 3.62100000
db['e-'].epsilon = 97.53000000
db['e-'].Lewis = 1.15200000
db['e-'].thermoCoeffs = {
  origin = 'CEA',
  nsegments = 3, 
  T_break_points = { 298.15, 1000.00, 6000.00, 20000.00, },
  T_blend_ranges = { 400.0, 1000.0, },
  segment0 = {
    0.000000000e+00,
    0.000000000e+00,
    2.500000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
   -7.453750000e+02,
   -1.172081224e+01,
  },
  segment1 = {
    0.000000000e+00,
    0.000000000e+00,
    2.500000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
   -7.453750000e+02,
   -1.172081224e+01,
  },
  segment2 = {
    0.000000000e+00,
    0.000000000e+00,
    2.500000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
    0.000000000e+00,
   -7.453750000e+02,
   -1.172081224e+01,
  },
}
db.CIs = {}
db.CIs['N2:N2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0112, C= -0.1182, D=  4.8464},
   pi_Omega_22 = {A=  0.0000, B= -0.0203, C=  0.0683, D=  4.0900},
}
db.CIs['O2:N2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0465, C=  0.5729, D=  1.6185},
   pi_Omega_22 = {A=  0.0000, B= -0.0558, C=  0.7590, D=  0.8955},
}
db.CIs['O2:O2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0410, C=  0.4977, D=  1.8302},
   pi_Omega_22 = {A=  0.0000, B= -0.0485, C=  0.6475, D=  1.2607},
}
db.CIs['N:N2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0194, C=  0.0119, D=  4.1055},
   pi_Omega_22 = {A=  0.0000, B= -0.0190, C=  0.0239, D=  4.1782},
}
db.CIs['N:O2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0179, C=  0.0152, D=  3.9996},
   pi_Omega_22 = {A=  0.0000, B= -0.0203, C=  0.0703, D=  3.8818},
}
db.CIs['N:N'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0033, C= -0.0572, D=  5.0452},
   pi_Omega_22 = {A=  0.0000, B= -0.0118, C= -0.0960, D=  4.3252},
}
db.CIs['O:N2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0139, C= -0.0825, D=  4.5785},
   pi_Omega_22 = {A=  0.0000, B= -0.0169, C= -0.0143, D=  4.4195},
}
db.CIs['O:O2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0226, C=  0.1300, D=  3.3363},
   pi_Omega_22 = {A=  0.0000, B= -0.0247, C=  0.1783, D=  3.2517},
}
db.CIs['O:N'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0048, C= -0.4195, D=  5.7774},
   pi_Omega_22 = {A=  0.0000, B=  0.0065, C= -0.4467, D=  6.0426},
}
db.CIs['O:O'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0034, C= -0.0572, D=  4.9901},
   pi_Omega_22 = {A=  0.0000, B= -0.0207, C=  0.0780, D=  3.5658},
}
db.CIs['NO:N2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0291, C=  0.2324, D=  3.2082},
   pi_Omega_22 = {A=  0.0000, B= -0.0385, C=  0.4226, D=  2.4507},
}
db.CIs['NO:O2'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0438, C=  0.5352, D=  1.7252},
   pi_Omega_22 = {A=  0.0000, B= -0.0522, C=  0.7045, D=  1.0738},
}
db.CIs['NO:N'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0185, C=  0.0118, D=  4.0590},
   pi_Omega_22 = {A=  0.0000, B= -0.0196, C=  0.0478, D=  4.0321},
}
db.CIs['NO:O'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0179, C=  0.0152, D=  3.9996},
   pi_Omega_22 = {A=  0.0000, B= -0.0203, C=  0.0703, D=  3.8818},
}
db.CIs['NO:NO'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0364, C=  0.3825, D=  2.4718},
   pi_Omega_22 = {A=  0.0000, B= -0.0453, C=  0.5624, D=  1.7669},
}
db.CIs['NO+:N2'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.8543},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.7760},
}
db.CIs['NO+:O2'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.8543},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.7760},
}
db.CIs['NO+:N'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.8543},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.7760},
}
db.CIs['NO+:O'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.8543},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.7760},
}
db.CIs['NO+:NO'] = {
   pi_Omega_11 = {A=  0.0000, B= -0.0047, C= -0.0551, D=  4.8737},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -0.4000, D=  6.7760},
}
db.CIs['NO+:NO+'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  23.8237},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  24.3602},
}
db.CIs['e-:N2'] = {
   pi_Omega_11 = {A=  0.1147, B= -2.8945, C=  24.5080, D= -67.3691},
   pi_Omega_22 = {A=  0.1147, B= -2.8945, C=  24.5080, D= -67.3691},
}
db.CIs['e-:O2'] = {
   pi_Omega_11 = {A=  0.0241, B= -0.3467, C=  1.3887, D= -0.0110},
   pi_Omega_22 = {A=  0.0241, B= -0.3467, C=  1.3887, D= -0.0110},
}
db.CIs['e-:N'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C=  0.0000, D=  1.6094},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C=  0.0000, D=  1.6094},
}
db.CIs['e-:O'] = {
   pi_Omega_11 = {A=  0.0164, B= -0.2431, C=  1.1231, D= -1.5561},
   pi_Omega_22 = {A=  0.0164, B= -0.2431, C=  1.1231, D= -1.5561},
}
db.CIs['e-:NO'] = {
   pi_Omega_11 = {A= -0.2202, B=  5.2265, C= -40.5659, D=  104.7126},
   pi_Omega_22 = {A= -0.2202, B=  5.2265, C= -40.5659, D=  104.7126},
}
db.CIs['e-:NO+'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  23.8237},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  24.3061},
}
db.CIs['e-:e-'] = {
   pi_Omega_11 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  23.8237},
   pi_Omega_22 = {A=  0.0000, B=  0.0000, C= -2.0000, D=  24.3061},
}
