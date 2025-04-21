import platform
import getpass
import socket
from colorama import init
import numpy as np
import os
import time



init(autoreset=True)

CYBER = "\033[38;2;88;176;140m"
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
    f"{CYBER}   +++++####     ++++....-   {RESET}",
    f"{CYBER}  +++++####       ++++.....  {RESET}",
    f"{CYBER} +++++####.................. {RESET}",
    f"{CYBER}+++++####....................{RESET}",
    f"{CYBER} +++######################## {RESET}",
    f"{CYBER}  +########################  {RESET}"
]

def display_sim_dashboard(coe, P, E0, TA0, kepler_iters):
    os.system('cls' if os.name == 'nt' else 'clear')  # clear the terminal

    info_lines = [
        f"{CYBER}       Semi-major axis{RESET}       (a)  | {coe[0]:.2f} km",
        f"{CYBER}       Eccentricity{RESET}          (e)  | {coe[1]:.4f}",
        f"{CYBER}       Inclination{RESET}           (i)  | {np.rad2deg(coe[2]):.2f}°",
        f"{CYBER}       RAAN{RESET}                  (Ω)  | {np.rad2deg(coe[3]):.2f}°",
        f"{CYBER}       Arg. of Periapsis{RESET}     (ω)  | {np.rad2deg(coe[4]):.2f}°",
        f"{CYBER}       True Anomaly{RESET}          (M₀) | {np.rad2deg(coe[5]):.2f}°",
        "",
        f"{CYBER}       Orbital Period{RESET}        (P)  | {P:.2f} s",
        f"{CYBER}       Eccentric Anomaly{RESET}          | {E0:.4f} rad",
        f"{CYBER}       Convergence {RESET}               | {kepler_iters}",
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
