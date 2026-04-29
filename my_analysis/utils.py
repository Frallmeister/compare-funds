from datetime import datetime, timedelta
from collections import Counter
from itertools import cycle
from pathlib import Path
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


COLORS = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
DARK_THEME = False
BASE_DIR = Path(__file__).resolve().parents[1]

NAME_MAP = {
    "avanza_global": "avanza_global",
    "avanza_zero": "avanza_zero",
    "cliens_smabolag_a": "cliens",
    "spiltan_aktiefond_investmentbolag": "spiltan",
    "swedbank_robur_tech_a": "swedbank",
    "nordea_alfa": "nordea",
    "coeli_frontiers_mkt": "coeli",
    "ct_lux_global_tech": "ct_lux",
    "dnb_teknologi_s": "dnb",
    "finserve": "finserve",
    "jpm_us_tech": "jpm_tech",
    "bgf_tech_a2": "bgf",
}

def load_csv_files():
    frames = []
    for file in BASE_DIR.glob("data/*.csv"):
        if file.stem not in NAME_MAP.keys():
            continue
        frame = pd.read_csv(file)
        frame = frame[["Datum", "Senaste"]]
        frame = frame.rename(columns={"Datum": "date", "Senaste": "nav"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["nav"] = frame["nav"].str.replace(".", "").str.replace(",", ".").astype(float)
        frame = frame.sort_values("date")
        frame["name"] = NAME_MAP[file.stem]
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return df

def set_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": "#111111",
        "axes.facecolor": "#111111",
        "text.color": "#f0f0f0",
        "axes.labelcolor": "#f0f0f0",
        "axes.edgecolor": "#f0f0f0",
        "xtick.color": "#f0f0f0",
        "ytick.color": "#f0f0f0",
        "legend.facecolor": "#111111",
        "legend.edgecolor": "#f0f0f0",
        "legend.labelcolor": "#f0f0f0",
        "grid.color": "#666666",
    })