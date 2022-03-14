# %% [markdown]
# # Libraries and Data
# %%
from zipfile import ZipFile

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()
# %% [markdown]
# After downloading the dataset, we can load the train set from the zip file.
#
# However, for faster reloads, we stored it in a feather file.
#
# You can comment/uncomment the specific rows for each use case.

# %%
def first_load(save_feather=False):
    # Original data load
    zipfile = ZipFile("./data/avazu-ctr-prediction.zip")
    train = pd.read_csv(
        zipfile.open("train.gz"), compression="gzip", usecols=["click", "hour"]
    )

    # Save to feather, for faster data reloads
    if save_feather:
        train.to_feather("./data/train.feather")
    assert train.equals(pd.read_feather("./data/train.feather"))

    return train


train = first_load(save_feather=True)

# Load from feather
# train = pd.read_feather("./data/train.feather")
# %% [markdown]
# We validate data features. Because of the size of the data set,
# we rely on numerical operations for a faster process in exchange for some
# intelligibility.
# %%
assert all(train["click"].unique() == [0, 1]), "Invalid `click` values."
assert all(train["hour"] // 1e6 == 14), "Invalid year data"
assert all((train["hour"] - 14e6) // 1e4 == 10), "Invalid month data"
assert all(
    ((train["hour"] - 14e6 - 10e4) // 1e2).isin(range(1, 31 + 1))
), "Invalid day data"
assert all(((train["hour"] - 14e6 - 10e4) % 1e2).isin(range(24))), "Invalid hour data"
# %%
# Transform the datetimeformat to pandas datetime

train["dthour"] = pd.to_datetime(train["hour"], format="%y%m%d%H")
assert (
    train["hour"].astype(str).str[-2:].astype(int) == train["dthour"].dt.hour
).all(), "Hour transformation do not match"

train = train.set_index("dthour").drop(columns="hour")
# %%
# Validate transformation results

assert all(
    pd.Series(train.index).diff().iloc[1:].dt.total_seconds().unique() / 60**2
    == [
        0,
        1,
    ]
), f"Incorrect timestamp deltas"

assert train.index.is_monotonic, "Timestamp is not monotonic."

# %% [markdown]
# # Data aggregation
#
# - We assume that it is meaningful to use all the ads in a single
# group and to plot them all onto the same time series.
# - We assume that a row in the dataset stands for an 'impression' and,
# therefore, we can get hourly CTRs by dividing the number of clicks with
# the number of total impressions within that hour.
#
#
# %% [markdown]
# The number of records varies a lot by each hour.
# %%
display(train.groupby(pd.Grouper(freq="h")).size())
alt.Chart(
    train.groupby(pd.Grouper(freq="h")).size().rename("records").reset_index()
).mark_line().encode(x="dthour:T", y="records:Q").properties(
    title="Number of Records", width=600, height=150
)

# %%

hourly = pd.DataFrame()
hourly["clicks"] = train.resample("H")["click"].sum()
hourly["impressions"] = train.resample("H")["click"].count()

display(
    hourly[["clicks", "impressions"]]
    .describe()
    .loc[["mean", "std", "min", "max"], :]
    .style.format(precision=0, thousands=" ")
)

# %% [markdown]
# Average CTR is around 17% with some considerable deviation
# between ~10% and ~22%.
# %%
hourly["CTR"] = train.resample("H")["click"].mean()

mean = hourly["clicks"].sum() / hourly["impressions"].sum()
std = np.sqrt(
    (((hourly["clicks"] / hourly["impressions"] - mean) ** 2).sum() / hourly.size)
)

display(
    pd.DataFrame(
        {
            "CTR": {
                "mean": mean,
                "std": std,
                "min": hourly["CTR"].min(),
                "max": hourly["CTR"].max(),
            }
        }
    ).loc[["mean", "std", "min", "max"], :]
)

line = alt.Chart(hourly.reset_index()).mark_line().encode(x="dthour:T", y="CTR:Q")
points = (
    alt.Chart(hourly.reset_index())
    .mark_point()
    .encode(
        x="dthour:T",
        y=alt.Y("CTR:Q", scale=alt.Scale(zero=False)),
        tooltip=["dthour", "CTR"],
    )
)

(line + points).properties(title="CTR", width=600, height=150)

# %% [markdown]
# # Outlier detection
#
# As the data contains only a single weekend, we cannot tell too much about the weekly
# patterns.
#
#
# ## Assumptions
#
# - We do this for retrospective analysis, and therefore we can use a
# centered moving window.
# - We do not have a specific use case, so we can experiment with different
# window sizes. For in-day outliers we can set it to 6H, while for in-week
# or in-month outliers we can set it to 3D, 7D, etc.
#
#
# ## Calculation
#
# Because CTR is already an aggregate metric, we need to calculate the
# rolling metrics from the original `clicks` column.
# %%


def rolling_metrics(hourly, window):

    try:
        hourly.set_index("dthour", inplace=True)
    except KeyError:
        print("`dthour` is already an index")

    # CTR mean
    hourly[f"{window}-mean"] = (
        hourly.rolling(window, center=True)["clicks", "impressions"]
        .sum()
        .apply(lambda x: x["clicks"] / x["impressions"], axis=1)
    )

    # CTR std
    hourly["squared_error"] = (hourly["CTR"] - hourly[f"{window}-mean"]) ** 2
    hourly[f"{window}-squared_error"] = (
        hourly["squared_error"].rolling(window, center=True).sum()
    )
    hourly["hours_in_window"] = (
        hourly.rolling(window, center=True).apply(lambda x: x.size).iloc[:, 0]
    )
    hourly[f"{window}-std"] = np.sqrt(
        hourly[f"{window}-squared_error"] / hourly["hours_in_window"]
    )

    return hourly


# %%
def define_outliers(hourly, window):
    hourly["top"] = hourly[f"{window}-mean"] + hourly[f"{window}-std"] * 1.5
    hourly["bottom"] = hourly[f"{window}-mean"] - hourly[f"{window}-std"] * 1.5
    hourly["outlier"] = (hourly["CTR"] > hourly["top"]) | (
        hourly["CTR"] < hourly["bottom"]
    ).astype(bool)
    return hourly


# %%
def plot_outliers(hourly):
    try:
        hourly.reset_index("dthour", inplace=True)
    except KeyError:
        print("`dthour` is already a column")

    points = (
        alt.Chart(hourly)
        .mark_point()
        .encode(
            x="dthour:T",
            y=alt.Y("CTR:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("outlier:N"),
            tooltip=["dthour:T", "CTR", "outlier"],
        )
    )

    lines = alt.layer(
        alt.Chart(hourly)
        .mark_line(opacity=0.5, color="grey")
        .encode(x="dthour:T", y="CTR:Q"),
        alt.Chart(hourly)
        .mark_line(opacity=0.5, color="red")
        .encode(x="dthour:T", y=f"{window}-mean:Q"),
        alt.Chart(hourly)
        .mark_area(opacity=0.2)
        .encode(x="dthour:T", y="top:Q", y2="bottom:Q"),
    )

    (points + lines).properties(title="CTR Outliers", width=600, height=150).display()


# %%
window = "1D"

hourly = rolling_metrics(hourly, window)
hourly = define_outliers(hourly, window)

display(
    hourly.loc[
        :, ["CTR", f"{window}-mean", f"{window}-std", "top", "bottom", "outlier"]
    ].sample(10)
)

plot_outliers(hourly)

# %% [markdown]
# # Possible improvements
#
# - For a more programmatic identification, use an error/distance metric
# to measure the distance of the outliers from the rest of the samples
# - Examine the relationship between CTR and change of total impressions
# - Combine smaller and bigger windows
