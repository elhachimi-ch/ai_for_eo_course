# %% [markdown]
# # Lab 3 — Reinforcement learning: irrigation scheduling and satellite tasking
# **AI for Earth Observation** · Part IV
#
# Goals
# 1. Write an irrigation season as a **Markov decision process** using the FAO-56 soil water balance.
# 2. Solve it exactly with **value iteration** (we know the model).
# 3. Learn it from experience with **Q-learning** (we only interact with a simulator) and compare.
# 4. Solve the simplest RL problem — a **bandit** — for choosing satellite acquisition windows: ε-greedy vs UCB1 vs Thompson sampling.
#
# NumPy only. Runtime: **CPU**, about 2 minutes.
#
# ⚠️ The simulator is a toy: one crop, weekly steps, three rain amounts. Real RL for agriculture lives or dies by the quality of its simulator.

# %%
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# %% [markdown]
# ## 1 · The environment: a 20-week season
# * **State**: week $t$ and root-zone **depletion** $D$ (mm of water missing below field capacity), in 10-mm bins.
# * **Action**: irrigate 0, 20 or 40 mm.
# * **Dynamics** (FAO-56): crop water use $ET_c = K_c\,ET_o$; when depletion exceeds the readily available water RAW the crop is stressed, $K_s = (TAW-D)/(TAW-RAW)$; then $D' = D + K_s ET_c - I - \text{rain}$.
# * **Reward**: crop response $w_t K_s$ (stress hurts most around flowering) minus the cost of water.

# %%
WEEKS, TAW, RAW, DSTEP = 20, 120.0, 60.0, 10.0          # weeks, total / readily available water (mm), bin width
N_D = int(TAW / DSTEP) + 1                               # depletion states 0, 10, …, 120 mm
IRR = np.array([0.0, 20.0, 40.0])                        # actions (mm)
t = np.arange(WEEKS)
ETO = 7 * (4.0 + 2.5 * np.sin(np.pi * (t + 2) / 24))    # reference ET, mm/week: spring → summer (Marrakech-like)
KC = np.interp(t, [0, 4, 9, 15, 19], [0.4, 0.7, 1.15, 1.15, 0.6])       # crop coefficient
ETC = KC * ETO
YIELD_W = np.interp(t, [0, 6, 9, 14, 19], [0.5, 1.0, 2.0, 1.5, 0.5])    # sensitivity to stress (flowering ≈ weeks 9–14)
WATER_COST = 0.02                                        # reward units per mm of irrigation
RAIN_MM = np.array([0.0, 10.0, 30.0])
p_wet = np.interp(t, [0, 8, 19], [0.45, 0.15, 0.05])    # rain becomes rare into summer
RAIN_P = np.stack([1 - p_wet, 0.7 * p_wet, 0.3 * p_wet], axis=1)
b = lambda D: int(round(D / DSTEP))                      # depletion (mm) → state index


def step(week, D, a, rng):
    rain = rng.choice(RAIN_MM, p=RAIN_P[week])
    ks = 1.0 if D <= RAW else max(0.0, (TAW - D) / (TAW - RAW))         # FAO-56 water-stress coefficient
    D2 = DSTEP * np.round(np.clip(D + ks * ETC[week] - IRR[a] - rain, 0, TAW) / DSTEP)
    reward = YIELD_W[week] * ks - WATER_COST * IRR[a]
    return D2, reward


fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(t, ETC, label="ETc (mm/week)"); ax[0].plot(t, RAIN_P[:, 1:] @ RAIN_MM[1:], label="expected rain (mm/week)")
ax[0].legend(); ax[0].set_xlabel("week")
ax[1].plot(t, YIELD_W, color="#B5502E"); ax[1].set_title("yield sensitivity to stress"); ax[1].set_xlabel("week")
plt.show()

# %% [markdown]
# ## 2 · Value iteration: the optimal policy when the model is known
# For a finite horizon, value iteration is **backward induction**: $V_T=0$, then for $t = T-1, \dots, 0$
# $$Q_t(s,a)=\sum_{s'}P_t(s'\mid s,a)\big[R_t(s,a,s')+V_{t+1}(s')\big],\qquad V_t(s)=\max_a Q_t(s,a).$$

# %%
def build_model():
    """Transition probabilities P[t, s, a, s'] and expected rewards R[t, s, a] from the simulator's equations."""
    P = np.zeros((WEEKS, N_D, 3, N_D)); R = np.zeros((WEEKS, N_D, 3))
    for w in range(WEEKS):
        for s in range(N_D):
            D = s * DSTEP
            ks = 1.0 if D <= RAW else max(0.0, (TAW - D) / (TAW - RAW))
            for a in range(3):
                R[w, s, a] = YIELD_W[w] * ks - WATER_COST * IRR[a]
                for rain, p in zip(RAIN_MM, RAIN_P[w]):
                    D2 = DSTEP * np.round(np.clip(D + ks * ETC[w] - IRR[a] - rain, 0, TAW) / DSTEP)
                    P[w, s, a, b(D2)] += p
    return P, R


def value_iteration(P, R, horizon):
    V = np.zeros((horizon + 1, N_D)); policy = np.zeros((horizon, N_D), dtype=int)
    for w in reversed(range(horizon)):
        Q = R[w] + P[w] @ V[w + 1]                 # (N_D, 3)
        V[w], policy[w] = Q.max(1), Q.argmax(1)
    return V, policy


P, R = build_model()
V, policy = value_iteration(P, R, horizon=WEEKS)
plt.figure(figsize=(9, 3.6))
plt.imshow(IRR[policy].T, origin="lower", aspect="auto", cmap="Blues", extent=[-0.5, WEEKS - 0.5, -5, TAW + 5])
plt.colorbar(label="irrigation (mm)"); plt.axhline(RAW, color="#B5502E", ls="--", label="RAW (stress starts)")
plt.xlabel("week"); plt.ylabel("depletion D (mm)"); plt.title("Optimal policy (value iteration)"); plt.legend(loc="upper left"); plt.show()

# %% [markdown]
# ## 3 · Compare with rules of thumb over 2 000 simulated seasons

# %%
D0 = 20.0                                          # start of season: soil close to field capacity


def run(policy_fn, n=2000, seed=1):
    rng = np.random.default_rng(seed); ret = water = stress = 0.0
    for _ in range(n):
        D = D0
        for w in range(WEEKS):
            a = policy_fn(w, D)
            ks = 1.0 if D <= RAW else max(0.0, (TAW - D) / (TAW - RAW))
            D, r = step(w, D, a, rng)
            ret += r; water += IRR[a]; stress += 1 - ks
    return ret / n, water / n, stress / (n * WEEKS)


policies = {
    "optimal (value iteration)": lambda w, D: policy[w, b(D)],
    "FAO rule: 40 mm when D > RAW": lambda w, D: 2 if D > RAW else 0,
    "calendar: 20 mm every week": lambda w, D: 1,
    "no irrigation": lambda w, D: 0,
}
for name, fn in policies.items():
    r, wtr, st = run(fn)
    print(f"{name:32s} return {r:6.2f} · water {wtr:5.0f} mm · mean stress {st:.1%}")

# %% [markdown]
# ## 4 · Q-learning: the same policy from experience alone
# Q-learning never sees `P` or `R`. It plays seasons in the simulator and nudges its estimates towards the observed reward plus the value of the next state:
# $$Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\max_{a'}Q(s',a')-Q(s,a)\big].$$

# %%
Q = np.zeros((WEEKS, N_D, 3))
EPISODES, returns = 50_000, []
for ep in range(EPISODES):
    eps = max(0.05, 1.0 - ep / (0.6 * EPISODES))           # explore a lot first, then exploit
    alpha = max(0.02, 0.5 * (1 - ep / EPISODES))
    D, G = D0, 0.0
    for w in range(WEEKS):
        s = b(D)
        a = rng.integers(3) if rng.random() < eps else int(Q[w, s].argmax())
        D2, r = step(w, D, a, rng)
        target = r + (Q[w + 1, b(D2)].max() if w < WEEKS - 1 else 0.0)
        Q[w, s, a] += alpha * (target - Q[w, s, a])
        D, G = D2, G + r
    returns.append(G)

q_policy = Q.argmax(2)
visited = np.abs(Q).sum(2) > 0
agree = (q_policy == policy)[visited].mean()
print(f"agreement with the optimal policy on visited states: {agree:.1%}")
r, wtr, st = run(lambda w, D: q_policy[w, b(D)])
print(f"Q-learning policy: return {r:.2f} · water {wtr:.0f} mm · mean stress {st:.1%}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].plot(np.convolve(returns, np.ones(500) / 500, mode="valid")); ax[0].set_title("return per season (moving average)"); ax[0].set_xlabel("episode")
im = ax[1].imshow(np.where(visited, IRR[q_policy], np.nan).T, origin="lower", aspect="auto", cmap="Blues", extent=[-0.5, WEEKS - 0.5, -5, TAW + 5])
ax[1].axhline(RAW, color="#B5502E", ls="--"); ax[1].set_title("Q-learning policy (blank = never visited)"); ax[1].set_xlabel("week")
plt.colorbar(im, ax=ax[1], label="irrigation (mm)"); plt.show()

# %% [markdown]
# ## 5 · Bandits: which acquisition window gives a cloud-free image?
# You can request one acquisition per day in one of five windows. Each window has an unknown probability of a clear image. Every request is both **exploration** (learning about a window) and **exploitation** (getting images). Regret = clear images lost compared with always choosing the best window.

# %%
P_CLEAR = np.array([0.35, 0.50, 0.62, 0.55, 0.70])        # unknown to the agent
T, RUNS = 500, 300


def play(strategy, seed):
    rng = np.random.default_rng(seed); n = np.zeros(5); s = np.zeros(5); regret = np.zeros(T)
    for t_ in range(T):
        if strategy == "ε-greedy (0.1)":
            a = rng.integers(5) if (rng.random() < 0.1 or n.min() == 0) else int(np.argmax(s / np.maximum(n, 1)))
        elif strategy == "UCB1":
            a = int(np.argmin(n)) if n.min() == 0 else int(np.argmax(s / n + np.sqrt(2 * np.log(t_ + 1) / n)))
        else:                                                # Thompson sampling with Beta(1, 1) priors
            a = int(np.argmax(rng.beta(1 + s, 1 + n - s)))
        clear = rng.random() < P_CLEAR[a]
        n[a] += 1; s[a] += clear
        regret[t_] = P_CLEAR.max() - P_CLEAR[a]
    return regret.cumsum()


plt.figure(figsize=(8, 4))
for strat in ["ε-greedy (0.1)", "UCB1", "Thompson"]:
    plt.plot(np.mean([play(strat, k) for k in range(RUNS)], axis=0), label=strat)
plt.xlabel("day"); plt.ylabel("cumulative regret (clear images lost)"); plt.legend(); plt.title("Exploration strategies"); plt.show()

# %% [markdown]
# ε-greedy keeps exploring at a fixed rate forever, so its regret grows linearly. UCB1 and Thompson sampling explore **where they are still uncertain** and their regret grows only logarithmically.
#
# ## 6 · Why tables stop working
# Our state was two small integers. A real irrigation agent would see a **weather forecast**, soil-moisture maps from SAR and a Sentinel-2 NDVI time series; a tasking agent would see cloud forecasts as images. No table can index that state — the Q-table becomes a **neural network** (DQN), or the policy itself does (PPO). That is the bridge to Part V.
#
# ## Exercises
# 1. **Expensive water.** Set `WATER_COST = 0.05`. When does the optimal policy accept stress, and in which weeks?
# 2. **Drier climate.** Halve `p_wet`. How much more water does the optimal policy use?
# 3. **No exploration.** Run Q-learning with `eps = 0` from the start. Why does it get stuck?
# 4. **Coarser state.** Use 20-mm bins (`DSTEP = 20`). What do you lose?
# 5. **Non-stationary clouds.** Make `P_CLEAR` drift during the 500 days. Which bandit strategy copes, and how would you modify the others?
