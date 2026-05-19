from transprop.collision_integrals import (
    ColIntLaricchiuta,
    ColIntGYCurveFitPiOmega,
    ColIntTable,
)
import matplotlib.pyplot as plt
import numpy as np
from air_CO_He import database
from ci_data import ci_data

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.serif"] = "Computer Modern"


def plot_air_CO_He_col_ints():
    species = list(database.keys())
    temps = np.linspace(600, 10000, 250)
    gas_state = {"temp": temps}

    for a, species_a in enumerate(species):
        for b in range(a + 1):
            species_b = species[b]
            laricchiuta_11 = ColIntLaricchiuta(
                order=(1, 1),
                database=database,
                species=(species_a, species_b),
            )
            laricchiuta_22 = ColIntLaricchiuta(
                order=(2, 2),
                database=database,
                species=(species_a, species_b),
            )

            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 4))
            ax[0].plot(temps, laricchiuta_11.eval(gas_state), "k", label="Laricchiuta")
            ax[1].plot(temps, laricchiuta_22.eval(gas_state), "k", label="Laricchiuta")

            known_data = None
            acc_11 = None
            acc_22 = None
            if (species_a, species_b) in ci_data:
                known_data = ci_data[(species_a, species_b)]
            elif (species_b, species_a) in ci_data:
                known_data = ci_data[(species_b, species_a)]
            if known_data:
                if known_data["type"] == "table":
                    known_data_11 = known_data[(1, 1)]
                    known_data_22 = known_data[(2, 2)]
                    known_col_int_11 = ColIntTable(
                        temps=known_data_11["temps"],
                        cis=known_data_11["cis"],
                        kind="linear",
                    )
                    acc_11 = known_data_11["acc"]
                    known_col_int_22 = ColIntTable(
                        temps=known_data_22["temps"],
                        cis=known_data_22["cis"],
                        kind="linear",
                    )
                    acc_22 = known_data_22["acc"]
                    known_col_int_11_vals = known_col_int_11.eval(gas_state)
                    known_col_int_22_vals = known_col_int_22.eval(gas_state)
                elif known_data["type"] == "gupta_yos_curve_fit":
                    known_data_11 = known_data[(1, 1)]
                    known_data_22 = known_data[(2, 2)]
                    known_col_int_11 = ColIntGYCurveFitPiOmega(
                        coeffs=known_data_11,
                        order=(1, 1),
                        species=(species_a, species_b),
                    )
                    known_col_int_22 = ColIntGYCurveFitPiOmega(
                        coeffs=known_data_11,
                        order=(2, 2),
                        species=(species_a, species_b),
                    )
                    known_col_int_11_vals = known_col_int_11.eval(gas_state)
                    known_col_int_22_vals = known_col_int_22.eval(gas_state)

                ax[0].plot(temps, known_col_int_11_vals, "b", label="Published")
                ax[1].plot(temps, known_col_int_22_vals, "b", label="Published")

                if acc_11:
                    ax[0].fill_between(
                        temps,
                        (1 + acc_11) * known_col_int_11_vals,
                        (1 - acc_11) * known_col_int_11_vals,
                        alpha=0.5,
                    )
                if acc_22:
                    ax[1].fill_between(
                        temps,
                        (1 + acc_22) * known_col_int_22_vals,
                        (1 - acc_22) * known_col_int_22_vals,
                        alpha=0.5,
                    )

            ax[0].set_ylabel("$\\Omega^{(1,1)}$, $\\AA^2$")
            ax[0].set_ylim(bottom=0)
            ax[0].grid()
            ax[1].set_ylabel("$\\Omega^{(2,2)}$, $\\AA^2$")
            ax[1].set_xlabel("Temperature, K")
            ax[1].set_ylim(bottom=0)
            ax[1].grid()
            ax[1].legend()
            fig.tight_layout()
            fig.savefig(f"figs/{species_a}_{species_b}_comparison.png")
            plt.close()


if __name__ == "__main__":
    plot_air_CO_He_col_ints()
