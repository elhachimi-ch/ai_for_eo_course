# %% [markdown]
# # Lab 4 — Forecasting reference evapotranspiration (ETo) with an LSTM
# **Deep Learning for Earth Observation** · Part V
#
# Task: from the last **30 days** of weather, forecast daily FAO-56 reference ETo for the **next 7 days** at Marrakech.
#
# Data: Open-Meteo historical-weather API (ERA5 / ERA5-Land reanalysis from Copernicus C3S), which also provides
# `et0_fao_evapotranspiration` (FAO-56 Penman–Monteith). No key needed.
#
# We compare the LSTM with two honest baselines: **persistence** and **day-of-year climatology**, using a **temporal split**.

# %%
!pip -q install requests pandas scikit-learn

# %%
import numpy as np, pandas as pd, requests, torch, torch.nn as nn, matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(0); np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
url = "https://archive-api.open-meteo.com/v1/archive"
q = dict(latitude=31.63, longitude=-8.01, start_date="2000-01-01", end_date="2024-12-31",
         daily="et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min,"
               "shortwave_radiation_sum,wind_speed_10m_max,precipitation_sum",
         timezone="UTC")
r = requests.get(url, params=q, timeout=120); r.raise_for_status()
df = pd.DataFrame(r.json()["daily"]); df["time"] = pd.to_datetime(df["time"]); df = df.set_index("time")
df = df.interpolate(limit=3).dropna()
doy = df.index.dayofyear.values
df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
print(df.shape); df.head()

# %%
df["et0_fao_evapotranspiration"].plot(figsize=(12, 3), lw=0.5, title="Daily ETo at Marrakech (mm/day)"); plt.show()

# %% [markdown]
# ## Windows and temporal split
# train 2000–2018 · validation 2019–2021 · test 2022–2024. Normalisation statistics come from the training period only.

# %%
TARGET = "et0_fao_evapotranspiration"
FEATS = [TARGET, "temperature_2m_max", "temperature_2m_min", "shortwave_radiation_sum",
         "wind_speed_10m_max", "precipitation_sum", "doy_sin", "doy_cos"]
LOOKBACK, HORIZON = 30, 7

train_mask = df.index < "2019-01-01"
mu, sd = df.loc[train_mask, FEATS].mean(), df.loc[train_mask, FEATS].std()
A = ((df[FEATS] - mu) / sd).values.astype("float32")
T = df[TARGET].values.astype("float32")

def windows(a, target, idx_start, idx_end, lookback=LOOKBACK, horizon=HORIZON):
    X, Y, t0 = [], [], []
    for t in range(max(lookback, idx_start), min(len(a) - horizon, idx_end)):
        X.append(a[t - lookback:t]); Y.append(target[t:t + horizon]); t0.append(t)
    return np.stack(X), np.stack(Y), np.array(t0)

i_val = np.searchsorted(df.index, pd.Timestamp("2019-01-01"))
i_test = np.searchsorted(df.index, pd.Timestamp("2022-01-01"))
Xtr, Ytr, _ = windows(A, T, 0, i_val)
Xva, Yva, _ = windows(A, T, i_val, i_test)
Xte, Yte, t_te = windows(A, T, i_test, len(A))
print(Xtr.shape, Xva.shape, Xte.shape)

# %%
class EToLSTM(nn.Module):
    def __init__(self, n_in, hidden=64, horizon=HORIZON):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, horizon)
    def forward(self, x):                 # x: (B, 30, n_in)
        out, (h, c) = self.lstm(x)
        return self.head(out[:, -1])      # (B, 7) in mm/day

model = EToLSTM(len(FEATS)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
tr_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)), batch_size=128, shuffle=True)

@torch.no_grad()
def predict(X):
    model.eval(); return model(torch.tensor(X).to(device)).cpu().numpy()

best, best_state, patience = 1e9, None, 0
for ep in range(60):
    model.train()
    for xb, yb in tr_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)      # guard against exploding gradients
        opt.step()
    va = np.sqrt(((predict(Xva) - Yva) ** 2).mean())
    if va < best - 1e-4:
        best, best_state, patience = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        patience += 1
    if ep % 5 == 0: print(f"epoch {ep:02d}  val RMSE {va:.3f} mm/day")
    if patience >= 8: print("early stop at", ep); break
model.load_state_dict(best_state)

# %% [markdown]
# ## Evaluation vs baselines (RMSE per lead day)

# %%
P = predict(Xte)
persist = np.repeat(T[t_te - 1][:, None], HORIZON, axis=1)                         # tomorrow = today
clim_table = df.loc[train_mask, TARGET].groupby(df.index[train_mask].dayofyear).mean()
clim = np.stack([clim_table.reindex(df.index[t_te + h].dayofyear).values for h in range(HORIZON)], 1)

def rmse(a, b): return np.sqrt(((a - b) ** 2).mean(0))
res = pd.DataFrame({"LSTM": rmse(P, Yte), "persistence": rmse(persist, Yte), "climatology": rmse(clim, Yte)},
                   index=[f"day+{h+1}" for h in range(HORIZON)])
print(res.round(3))
res.plot(marker="o", title="Test RMSE (mm/day) by lead time"); plt.show()

# %%
k = slice(0, 365)
plt.figure(figsize=(12, 3))
plt.plot(df.index[t_te[k]], Yte[k, 0], label="observed (ERA5-based)", lw=1)
plt.plot(df.index[t_te[k]], P[k, 0], label="LSTM day+1", lw=1)
plt.legend(); plt.title("2022 test year"); plt.show()

# %% [markdown]
# ## Exercises
# 1. Drop the ETo history from the inputs (keep only meteorology). How much does skill change?
# 2. Replace `nn.LSTM` with `nn.GRU`. Same accuracy with fewer parameters?
# 3. Train on several Moroccan sites at once (e.g. Marrakech, Meknès, Agadir, Oujda) with a site embedding. Does it generalise to a held-out site?
# 4. Physics check: compute Hargreaves–Samani ETo from Tmax/Tmin and compare its error with the LSTM.
