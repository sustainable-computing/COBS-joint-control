import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import gspread
from oauth2client.service_account import ServiceAccountCredentials

current_model = "PPO"
occupancy = ("FixedBldgOcc", "CustomRoomOcc")[0]
blind = ("NoBlindCtrl", "SameBlindCtrl", "DiffBlindCtrl")[0]
light = ("NoAutoDim", "AutoDim")[0]
season = ("cooling", "heating")[0]

if __name__ == '__main__':
    scope = ['https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('../utils/secret_info.json', scope)
    client = gspread.authorize(creds).open("Result playground")

    sheets = {"PPO": client.worksheet("ppo"),
              "SAC": client.worksheet("sac"),
              "DUELING": client.worksheet("dqn")}

    data = sheets[current_model].get_all_values()

    i = 0
    for i, row in enumerate(data):
        if not row[0]:
            continue
        elif row[0] == f"{season} {occupancy} {blind} {light}":
            break

    assert i != len(data)

    xs, ys, zs, energy = ([], [], [], [])
    for j, reward_weight in enumerate(data[0][1:], start=1):
        x, y, z = reward_weight.split('-')
        xs.append(x)
        ys.append(y)
        zs.append(z)
        energy.append(data[i][j])

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)
    energy = np.array(energy, dtype=float)
    energy = (energy - energy.min()) / (energy.max() - energy.min())

    j = np.argmin(energy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = ax.scatter(xs, ys, zs, c=energy, cmap='coolwarm')
    ax.scatter(xs[j], ys[j], zs[j], marker='*', s=300)
    cbar = fig.colorbar(plot, ax=ax, shrink=0.5, aspect=5, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Small', 'Large'])
    ax.set_xlabel("Power Multiplier")
    ax.set_ylabel("Thermal Comfort Multiplier")
    ax.set_zlabel("Visual Comfort Multiplier")
    plt.show()
