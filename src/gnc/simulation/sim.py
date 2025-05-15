from colorama import init
import numpy as np
import os

init(autoreset=True)

CYBER = "\033[38;2;88;176;140m"
CYAN_VEC = "\033[38;2;120;220;232m"
PINK_VEC = "\033[38;2;255;97;136m"
RESET = "\033[0m"

ascii_logo = [
    f"{CYBER}            +....            {RESET}",
    f"{CYBER}           +++....           {RESET}",
    f"{CYBER}          +++++....          {RESET}",
    f"{CYBER}         +++++++....         {RESET}",
    f"{CYBER}       +++++#++++....        {RESET}",
    f"{CYBER}      +++++###++++....       {RESET}",
    f"{CYBER}     +++++#### ++++.....     {RESET}",
    f"{CYBER}    +++++####   ++++.....    {RESET}",
    f"{CYBER}   +++++####     ++++.....   {RESET}",
    f"{CYBER}  +++++####       ++++.....  {RESET}",
    f"{CYBER} +++++####.................. {RESET}",
    f"{CYBER}+++++####....................{RESET}",
    f"{CYBER} +++######################## {RESET}",
    f"{CYBER}  +########################  {RESET}"
]

def display_sim_dashboard(coe, P, TA0, X, kepler_iters):
    os.system('cls' if os.name == 'nt' else 'clear')  # clear the terminal
    position_label = "Apoapsis" if np.pi/2 < TA0 < 1.5 * np.pi else "Periapsis"

    state_fmt = lambda vec: np.array2string(
    vec,
    precision=4,
    formatter={'float_kind': lambda x: f"{x:.4f}"},
    max_line_width=999
    )

    info_lines = [
        f"{CYBER}       Semi-major axis{RESET}       (a)  | {coe[0]:.2f} km",
        f"{CYBER}       Eccentricity{RESET}          (e)  | {coe[1]:.4f}",
        f"{CYBER}       Inclination{RESET}           (i)  | {np.rad2deg(coe[2]):.2f}°",
        f"{CYBER}       RAAN{RESET}                  (Ω)  | {np.rad2deg(coe[3]):.2f}°",
        f"{CYBER}       Arg. of Periapsis{RESET}     (ω)  | {np.rad2deg(coe[4]):.2f}°",
        f"{CYBER}       True Anomaly{RESET}          (M₀) | {np.rad2deg(coe[5]):.2f}°",
        "",
        f"{CYBER}       Orbital Period{RESET}        (P)  | {P:.2f} s",
        f"{CYBER}       Convergence {RESET}               | {kepler_iters}",
        f"{CYBER}       Location along orbit{RESET}       | {position_label}",
        "",
        f"{CYAN_VEC}       Initial State Vector{RESET}       {state_fmt(X[:, 0])}",
        f"{PINK_VEC}       Final State Vector{RESET}         {state_fmt(X[:, -1])}",
    ]

    logo_height = len(ascii_logo)
    info_height = len(info_lines)
    top_padding = (logo_height - info_height) // 2
    info_lines = [''] * top_padding + info_lines
    info_lines += [''] * (logo_height - len(info_lines))

    print("\n")
    for l, r in zip(ascii_logo, info_lines):
        print(f"{l:<36} {r}")
    print("\n")

