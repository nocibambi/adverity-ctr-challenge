# %%
from xmlrpc.client import Boolean
from zipfile import ZipFile

import altair as alt
import numpy as np
import pandas as pd

# %%
zipfile = ZipFile("./data/avazu-ctr-prediction.zip")
train = pd.read_csv(
    zipfile.open("train.gz"), compression="gzip", usecols=["click", "hour"]
)
# %%
# For faster data reloads
# train.to_feather("./data/train.feather")
# assert train.equals(pd.read_feather("./data/train.feather"))
train = pd.read_feather("./data/train.feather")

# %%
train["dthour"] = pd.to_datetime(train["hour"], format="%y%m%d%H")
# %%
assert (train["hour"].astype(str).str[-2:].astype(int) == train["dthour"].dt.hour).all()
train = train.set_index("dthour").drop(columns="hour")
# %%
window = "5D"
hourly = pd.DataFrame()
hourly["CTR"] = train.resample("H")["click"].mean()
hourly["clicks"] = train.resample("H")["click"].sum()
hourly["impressions"] = train.resample("H")["click"].count()
# %%
hourly[f"{window}-mean"] = (
    hourly.rolling(window, center=True)["clicks", "impressions"]
    .sum()
    .apply(lambda x: x["clicks"] / x["impressions"], axis=1)
)

hourly["squared_error"] = (hourly["CTR"] - hourly[f"{window}-mean"]) ** 2
hourly[f"{window}-squared_error"] = (
    hourly["squared_error"].rolling(window, center=True).sum()
)

hourly["hours"] = hourly.rolling(window, center=True).apply(lambda x: x.size).iloc[:, 0]
hourly[f"{window}-std"] = np.sqrt(hourly[f"{window}-squared_error"] / hourly["hours"])
# %%
hourly["top"] = hourly[f"{window}-mean"] + hourly[f"{window}-std"] * 1.5
hourly["bottom"] = hourly[f"{window}-mean"] - hourly[f"{window}-std"] * 1.5
hourly["outlier"] = (hourly["CTR"] > hourly["top"]) | (
    hourly["CTR"] < hourly["bottom"]
).astype(bool)
# %%
print(hourly["outlier"].astype(int).describe())
hourly.loc[:, ["CTR", f"{window}-mean", f"{window}-std", "top", "bottom", "outlier"]]
# %%
try:
    hourly.reset_index("dthour", inplace=True)
except KeyError:
    print("`dthour` is already index")
# %%
points = (
    alt.Chart(hourly)
    .mark_point()
    .encode(
        x="dthour:T",
        y="CTR:Q",
        color=alt.Color("outlier:N"),
        tooltip=["dthour:T", "CTR", "outlier"],
        # tooltip=alt.Tooltip('dthour', timeUnit='hours')
    )
)

lines = alt.layer(
    alt.Chart(hourly).mark_line().encode(x="dthour:T", y="CTR:Q"),
    alt.Chart(hourly)
    .mark_line(opacity=0.5)
    .encode(
        x="dthour:T",
        y=f"{window}-mean:Q",
    ),
    alt.Chart(hourly)
    .mark_area(opacity=0.2)
    .encode(x="dthour:T", y="top:Q", y2="bottom:Q"),
)

(points + lines).properties(title="CTR Outliers", width=600, height=150)

# %%
