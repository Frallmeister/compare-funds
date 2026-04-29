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
    "dnb_teknologi_s": "dnb_s",
    "dnb_teknologi_a": "dnb_a",
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


class FundValue:
    """Represents a single fund time series and simulates recurring monthly investments.

    The class stores NAV history, identifies valid monthly contribution dates, and
    computes portfolio development over a selected date interval.

    Parameters
    ==========
    data : pd.DataFrame
        Fund history with at least the columns "date" and "nav".

    Attributes
    ==========
    data : pd.DataFrame
        Internal dataset with derived columns such as
        "first_in_month", "units_", "invested", "value", and "profit".
    first_datetime : pd.Timestamp
        First available date in the input data.
    last_datetime : pd.Timestamp
        Last available date in the input data.

    Notes
    =====
    - Monthly contributions are added on the first available trading date in each
    month, constrained to day < 5.
    - Calling run mutates the internal data by appending simulation results.
    """

    def __init__(self, data: pd.DataFrame):
        data = data.copy().dropna()
        self.data = data
        self.first_datetime = data["date"].min()
        self.last_datetime = data["date"].max()
        self.data = self._first_day_in_month()

    def __getitem__(self, key):
        return self.data[key]

    def run(self, monthly_contrib: int, start: datetime = None, end: datetime = None):
        """
        Simulate periodic investing and compute invested capital, current value, and profit.

        Args:
            monthly_contrib (int): Amount invested each contribution month.
            start (datetime | None): Simulation start date. Defaults to the first available date.
            end (datetime | None): Simulation end date. Defaults to the last available date.

        Raises:
            ValueError: If start is not earlier than end.
            ValueError: If start is earlier than the first date in the dataset.

        Notes:
            - The first row in the selected range always gets an initial contribution.
            - Additional contributions are made on rows flagged as first_in_month.
            - This method mutates self.data by merging calculated columns:
            units_, invested, value, and profit.
        """
        start = start or self.first_datetime
        end = end or self.last_datetime
        if start >= end:
            raise ValueError("The start date must be earlier than the end date")
        if start < self.first_datetime:
            raise ValueError("The start date is earlier than the first date in the data")

        frame = self.data.query("@start <= date <= @end")
        units = []
        contribs = []
        for _, row in frame.iterrows():
            if not units:
                units.append(monthly_contrib / row["nav"])
                contribs.append(monthly_contrib)
            elif row["first_in_month"]:
                units_to_add = monthly_contrib / row["nav"]
                units.append(units[-1] + units_to_add)
                contribs.append(monthly_contrib)
            else:
                units.append(units[-1])
                contribs.append(0)
        new_data = pd.DataFrame(
            {"date": frame["date"], "units_": units, "invested": contribs}
        )
        new_data = new_data.assign(invested=new_data["invested"].cumsum())
        self.data = self.data.merge(new_data, on="date", how="outer")
        self.data["units_"] = self.data["units_"].fillna(0)
        self.data = self.data.assign(value=lambda x: x["units_"].mul(x["nav"]))
        self.data = self.data.assign(profit=lambda x: x["value"].sub(x["invested"]))

    def _first_day_in_month(self):
        first_days = self.data.groupby(self.data['date'].dt.to_period('M'))['date'].transform('min') == self.data["date"]
        first_days = first_days & self.data["date"].dt.day.lt(5)
        return self.data.assign(first_in_month=first_days)


def backtest_simulation(monthly_contribution: int, start_date: datetime|None = None) -> pd.DataFrame:
    if start_date is None:
        start_date = datetime(2017, 7, 25)
    end_date = datetime.now()
    dates = pd.date_range(start_date, end_date, freq="30D")

    data = load_csv_files()
    results = []
    for date in tqdm(dates):
        for name, frame in data.groupby("name"):
            frame = FundValue(frame)
            try:
                frame.run(monthly_contribution, start=date)
            except ValueError:
                continue
            value = int(frame["value"].iloc[-1].item())
            profit = int(frame["profit"].iloc[-1].item())
            invested = int(frame["invested"].iloc[-1].item())
            results.append([date, name, value, profit, invested])

    results = pd.DataFrame(results, columns=["date", "name", "value", "profit", "invested"])
    results = results.pivot_table(index="date", columns="name", values=["value", "profit", "invested"])
    return results