def lua_to_python(line1, line2, line3):
    _, pair_string, _ = line1.split("'")
    pair = pair_string.split(":")
    python_string = f"    {tuple(pair)}: {{\n"
    _, coeffs_11 = line2.split("{")
    coeffs_11,_ = coeffs_11.split("}")
    python_string += f"        (1,1): [{coeffs_11}],\n"
    _, coeffs_22 = line3.split("{")
    coeffs_22,_ = coeffs_22.split("}")
    python_string += f"        (2,2): [{coeffs_22}],\n"
    python_string += "    },\n"
    return python_string

def read_string(lua_string):
    string_line_breaks = lua_string.split("\n")
    python_string = ""
    for line_i in range(0,len(string_line_breaks),3):
        python_string += lua_to_python(*string_line_breaks[line_i:line_i+3])
    return python_string


if __name__== "__main__":
    #print(lua_to_python("#cis['NO+:N'] = {", "#  pi_Omega_11 = {0.0, 0.0, -0.4, 6.8},", "#  pi_Omega_22 = {0.0, 0.0, -0.4, 6.77}"))
    print(read_string("""#cis['NO+:N'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['NO+:O'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['NO+:NO'] = {
#  pi_Omega_11 = {0.0000, -0.0047, -0.0551, 4.8737},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['NO+:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['e-:N2'] = {
#  pi_Omega_11 = {0.1147, -2.8945,  24.5080, -67.3691},
#  pi_Omega_22 = {0.1147, -2.8945,  24.5080, -67.3691}
#cis['e-:O2'] = {
#  pi_Omega_11 = {0.0241, -0.3467,  1.3887, -0.0110},
#  pi_Omega_22 = {0.0241, -0.3467,  1.3887, -0.0110}
#cis['e-:N'] = {
#  pi_Omega_11 = {0.0000, 0.0000,  0.0000, 1.6094},
#  pi_Omega_22 = {0.0000, 0.0000,  0.0000, 1.6094}
#cis['e-:O'] = {
#  pi_Omega_11 = {0.0164, -0.2431,  1.1231, -1.5561},
#  pi_Omega_22 = {0.0164, -0.2431,  1.1231, -1.5561}
#cis['e-:NO'] = {
#  pi_Omega_11 = {A= -0.2202, 5.2265, -40.5659, 104.7126},
#  pi_Omega_22 = {A= -0.2202, 5.2265, -40.5659, 104.7126}
#cis['e-:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3061}
#cis['e-:e-'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3061}
#cis['N+:N2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N+:O2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N+:N'] = {
#  pi_Omega_11 = {0.0000, -0.0033, -0.0572, 5.0452},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4146, 6.9078}
#cis['N+:O'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N+:NO'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N+:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['N+:e-'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3601}
#cis['N+:N+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O+:N2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O+:O2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O+:N'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O+:O'] = {
#  pi_Omega_11 = {0.0000, -0.0034, -0.0572, 4.9901},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4235, 6.7787}
#cis['O+:NO'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O+:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O+:e-'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3601}
#cis['O+:N+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O+:O+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['N2+:N2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N2+:O2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N2+:N'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N2+:O'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N2+:NO'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['N2+:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['N2+:e-'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3601}
#cis['N2+:N+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['N2+:O+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['N2+:N2+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O2+:N2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O2+:O2'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O2+:N'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O2+:O'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O2+:NO'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -0.4000, 6.8543},
#  pi_Omega_22 = {0.0000, 0.0000, -0.4000, 6.7760}
#cis['O2+:NO+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O2+:e-'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3601}
#cis['O2+:N+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O2+:O+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O2+:N2+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}
#cis['O2+:O2+'] = {
#  pi_Omega_11 = {0.0000, 0.0000, -2.0000, 23.8237},
#  pi_Omega_22 = {0.0000, 0.0000, -2.0000, 24.3602}"""))


