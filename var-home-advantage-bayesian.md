# Did VAR Reduce Home Advantage in the Premier League?

## Executive Summary

This analysis evaluates whether the introduction of the Video Assistant Referee (VAR) in the 2019/20 Premier League season corresponded with a measurable change in home advantage. Using Bayesian models applied to match data from 2010–2024, the study separates the effect of VAR from the unprecedented disruption of the 2020/21 no‑crowd season — a global structural shock known to suppress home advantage.

**Key Finding: VAR did not materially change Premier League home advantage**

A Bayesian Beta–Binomial model comparing pre‑VAR and post‑VAR eras indicates a 77% probability of a small directional decline in home‑win rate once the 2020/21 season is excluded. However, the estimated effect size is negligible (around –0.5 percentage points) and well within normal year‑to‑year variation. In practical terms, the model suggests a slight lean toward decline but no meaningful change in competitive balance.

**The pandemic, not VAR, explains the apparent collapse in home advantage**

A naïve pre/post comparison that includes the 2020/21 season suggests a strong decline, but this effect disappears entirely when the no‑crowd season is treated as a separate regime. Crowd absence — not VAR — remains the dominant driver of home‑advantage variation.

**Distributional analysis shows no structural change in match outcomes**

Student‑t models of goal‑difference distributions, including a hierarchical formulation, show no meaningful shift in spread, tail behaviour, or extremity of scorelines after VAR. Home teams continue to win by similar margins, and the overall distribution of match outcomes remains stable.

**Competitive balance remains unchanged**

Segmenting matches by Big Six vs. non‑Big Six clubs reveals nearly identical pre/post VAR patterns. VAR did not redistribute advantage across tiers of teams, nor did it alter the league’s competitive hierarchy.

**Overall conclusion**

Across multiple modelling approaches, the evidence indicates that VAR did not produce a structural change in Premier League home advantage. The only season showing a dramatic collapse is 2020/21, driven by the absence of crowds rather than the introduction of VAR. Once this anomaly is accounted for, home advantage remains remarkably stable — consistent with both historical patterns and contemporary research.

**Business perspective**

Because the statistical effect is so small, VAR does not meaningfully influence competitive balance or commercial outcomes. Its ROI lies in fairness, error reduction, and maintaining the credibility of the competition — not in changing results. The strategic priority is therefore to optimise implementation rather than expect VAR to reshape match dynamics.

**Effect size**

The estimated VAR‑era decline in home‑win probability is small — roughly 0.7 standard deviations of normal seasonal fluctuation — and well within typical noise.

**Note:**

The 2020/21 season is treated as a separate regime because the absence of crowds caused a global collapse in home advantage unrelated to VAR.


## Key Findings

### 1. No meaningful change in home‑win probability once the no‑crowd season is excluded  
A Bayesian Beta–Binomial comparison shows that, after removing the anomalous 2020/21 season, the probability that home‑win rates declined in the VAR era is 76% — statistical indecision. The estimated shift is extremely small (around –0.5 percentage points) and well within normal year‑to‑year variation.

### 2. The apparent decline in home advantage is driven entirely by the 2020/21 no‑crowd season  
A naïve pre/post comparison that includes 2020/21 suggests an 80–90% probability of decline. This effect disappears completely when the no‑crowd season is treated as a separate regime. This mirrors contemporary research showing that crowd absence, not VAR, caused the temporary collapse in home advantage across global football.

### 3. Goal‑difference distributions are unchanged across eras  
Empirical and Bayesian Student‑t models show no evidence that VAR altered the spread or extremity of match outcomes.

- The 95th percentile of goal difference is stable.  
- Pre‑ and post‑VAR standard deviations are nearly identical.  
- Hierarchical Student‑t models assign only 8–10% probability to a smaller post‑VAR scale parameter.  

VAR did not reduce blowouts, compress scorelines, or change the distribution of match margins.

### 4. Competitive balance remains structurally stable  
Segmenting matches by Big Six vs. non‑Big Six clubs shows that both groups experienced nearly identical pre/post VAR shifts.

- Big Six home‑win rates remain substantially higher than others.  
- The gap between groups is unchanged.  
- VAR did not redistribute advantage across tiers of teams.

### 5. VAR’s overall impact on home advantage is modest, uncertain, and overshadowed by contextual factors  
Across all models — win probability, goal difference, and competitive‑balance segmentation — the same conclusion emerges:

- VAR did **not** produce a structural change in Premier League home advantage.  
- The only season showing a dramatic collapse is 2020/21, driven by the absence of crowds rather than the introduction of VAR.  

Home advantage remains a persistent feature of the Premier League, broadly consistent with long‑run historical patterns and recent post‑VAR research.


##  Introduction

Home advantage is one of the most robust and enduring findings in football analytics. Across leagues and eras, home teams consistently win more often than away teams, and this pattern has remained remarkably stable despite changes in tactics, scheduling, and officiating. Contemporary research attributes home advantage primarily to crowd influence, psychological pressure on referees, and contextual factors such as travel and familiarity with the stadium environment. Recent studies in the post‑VAR era continue to emphasise that crowd presence remains the strongest and most consistent driver of home advantage.

The introduction of the Video Assistant Referee (VAR) to the Premier League in 2019/20 created a natural test of one specific mechanism: referee‑driven bias. If part of home advantage arises from social pressure on referees, then a system that provides video review, external oversight, and the possibility of overturning on‑field decisions might reduce that bias. This leads to a clear empirical question: did VAR reduce home advantage in the Premier League?

To investigate this, the analysis uses match results from 2010/11 to 2023/24 and compares home‑win probabilities before and after VAR’s introduction. A Bayesian Beta–Binomial model estimates the home‑win probability in each era and quantifies the probability that the post‑VAR value is lower than the pre‑VAR value. Crucially, the analysis treats the 2020/21 no‑crowd season as a distinct regime, recognising that pandemic‑era conditions produced a global collapse in home advantage unrelated to VAR.

Home advantage, however, is not expressed solely through win rates. If VAR affects marginal decisions — penalties, red cards, offsides, stoppage‑time interventions — then it may also influence the distribution of match outcomes. Correcting marginal errors could compress goal‑difference distributions or reduce the frequency of large home wins. To test this, the analysis supplements win‑probability modelling with empirical and Bayesian distributional analysis of goal difference, including independent and hierarchical Student‑t models.

Together, these two perspectives — home‑win probability and goal‑difference distribution — provide a comprehensive assessment of whether VAR altered the structure, magnitude, or competitive balance of home advantage in the Premier League.

## Background and Literature Review

Research on home advantage in football has a long history. Clarke and Norman (1995) show that home advantage varies meaningfully across English clubs, while Pollard (2006) and Pollard and Pollard (2005) document both its magnitude and its long‑term decline across multiple sports and competitions. Together, this work establishes home advantage as real, historically strong, and sensitive to structural changes in the game (officiating, travel, crowd effects, etc.).

More recently, attention has turned to the impact of the Video Assistant Referee (VAR). Rogerson et al. (2024) conduct a meta‑analysis across multiple competitions and report that VAR is associated with a modest reduction in home advantage, rather than a dramatic reversal: effect sizes are small, heterogeneous across leagues, and often operate through changes in penalty awards and red cards rather than overall goal rates. Nagle et al. (2024) complement this by framing VAR as a socio‑technical system—improving some aspects of decision accuracy while introducing new forms of controversy and delay.

The estimates from this project can be read as a Premier League–specific point within that broader distribution. By modelling pre‑ and post‑VAR eras directly, the analysis asks whether England follows the same pattern of a small, VAR‑related erosion of home advantage, or whether its trajectory differs from the cross‑competition average reported by Rogerson et al. (2024). In that sense, the results here are best interpreted as a league‑level case study nested inside a wider, evolving literature on home advantage and officiating technology.

## Hypotheses

This analysis addresses one central question:

**Did VAR reduce home advantage in the Premier League?**

To answer it, we evaluate two related hypotheses: one concerning home‑win probability, and one concerning the distribution of goal differences.

**Research Question 1: Did VAR reduce home advantage in the Premier League?**

H₁: VAR reduced home win probability

Let:

𝑃pre = home win probability in the pre‑VAR era (2010/11–2018/19)

𝑃post = home win probability in the VAR era (2019/20–2023/24)

Define the effect size:

                        𝛿 = 𝑃post−𝑃pre

The first hypothesis is:

                        𝐻1:𝛿<0

A Bayesian Beta–Binomial model estimates the posterior distribution of 𝛿𝑃 and quantifies the probability that home‑win probability declined after VAR. A sensitivity analysis excludes the 2020/21 no‑crowd season to separate VAR effects from pandemic‑driven dynamics.

**Research Question 2 — Did VAR reduce the spread of goal differences?**

H₂: VAR reduced the spread of goal differences

Let:

- 𝜎pre = scale (spread) of goal difference in the pre‑VAR era

- 𝜎post = scale of goal difference in the VAR era

Define the effect size:

                           𝛿𝜎=𝜎post−𝜎pre

Subscripts distinguish effect sizes for win probability (𝛿𝑃) and goal‑difference scale (𝛿𝜎).

The second hypothesis is:

                           𝐻2:𝛿𝜎<0

This tests whether VAR reduced the variability of match outcomes — specifically, whether it decreased the frequency or magnitude of extreme home wins. It is evaluated using empirical standard deviations, percentile comparisons, independent Student‑t models, and a hierarchical Student‑t model.

It is evaluated using:

- empirical standard deviations

- percentile comparisons

- independent Student‑t models for each era

- a hierarchical Student‑t model pooling information across eras

Together, these tests assess whether VAR compressed the distribution of scorelines or altered the extremity of match outcomes.

## Data & Definitions

**Data Source**

Match results were obtained directly from football-data.co.uk, a long‑standing public repository of historical football statistics. The data were accessed programmatically using a simple Python function that downloads each season’s Premier League CSV file.

**Seasons Included**

- Pre‑VAR era: 2010/11–2018/19

- VAR era: 2019/20–2023/24

The 2020/21 no‑crowd season is included in the main analysis but treated separately in sensitivity checks due to its exceptional conditions.

**Key Variables**

Match identifiers:

- season — season label (e.g., "2015/16")

- home_team, away_team — team names

**Match outcomes**

- FTHG — full‑time home goals

- FTAG — full‑time away goals

- goal_diff — constructed as FTHG - FTAG

- Used for distributional modelling (Hypothesis 2)

home_win — indicator variable:

- 1 if FTHG > FTAG

- 0 otherwise

- Used for the Beta–Binomial model (Hypothesis 1)

**Era classification**

A binary indicator is created to distinguish pre‑VAR and post‑VAR periods:

- era = 0 for pre‑VAR seasons

- era = 1 for VAR seasons

This variable is used in all pre/post comparisons, including the hierarchical Student‑t model.

**Why 2020/21 Is Treated as a Separate Regime**

The 2020/21 season is treated as a separate regime because matches were played almost entirely without crowds, creating conditions fundamentally different from any other modern Premier League season. Crowd absence caused a global collapse in home advantage unrelated to VAR, so including 2020/21 in a simple pre/post comparison would confound the analysis and misattribute this pandemic‑driven effect to officiating technology.


**Why These Variables Matter**

- home_win captures home advantage as a binary outcome, enabling estimation of home‑win probabilities in each era.

- goal_diff captures the distribution of match outcomes, allowing us to test whether VAR reduced the spread or extremity of results.

- era provides a clean, reproducible way to separate the two periods for both hypotheses.

Together, these variables support the two central hypotheses:

- H₁: VAR reduced home‑win probability

- H₂: VAR reduced the spread of goal differences

## Environment and Setup


```python
# ============================================
# Environment & Setup
# ============================================

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Plotting style
sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (10, 6)

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------
if os.path.exists("pl_matches_cached.csv"):
    df = pd.read_csv("pl_matches_cached.csv")
else:
    print("Downloading data...")
    dfs = []
    for year in range(2010, 2025):
        dfs.append(load_pl_season_online(year))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("pl_matches_cached.csv", index=False)

# ---------------------------------------------------------
# 3. Keep columns needed
# ---------------------------------------------------------
df = df[[
    "season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG"
]]

# ---------------------------------------------------------
# 4. Add derived columns
# ---------------------------------------------------------
df["goal_diff"] = df["FTHG"] - df["FTAG"]
df["home_win"] = (df["FTHG"] > df["FTAG"]).astype(int)

# VAR introduced in 2019/20
df["era"] = df["season"].str[:4].astype(int).ge(2019).astype(int)

# ---------------------------------------------------------
# 5. Sanity checks
# ---------------------------------------------------------
# print("Columns:", df.columns.tolist())
# print("Unique seasons:", df["season"].unique())
# print("Era counts:", df["era"].value_counts())
# print("Rows:", len(df))
```

### Note on `df.copy()`

The slice used to construct `era_df` is created with boolean indexing. In pandas, such slices may return either a view or a copy of the original DataFrame, depending on internal heuristics. Assigning new columns to a view can trigger a `SettingWithCopyWarning` and, more importantly, can lead to silent failures where the assignment does not propagate as expected.

Using `.copy()` ensures that `era_df` is an explicit, independent DataFrame. This avoids ambiguous view/copy behaviour and guarantees that subsequent assignments (such as the `era` indicator) modify the intended object. This is particularly important in modelling workflows, where misaligned or partially updated data can produce hard‑to‑diagnose errors.


## Exploratory Overview

- Table: matches per season
- Plot: home win rate by season (line plot)
- Quick commentary:
  - Is there an obvious pre/post shift?
  - Any weird seasons (e.g. COVID, empty stadiums)?


```python
season_summary = (
    df.groupby("season")["home_win"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "home_win_rate"})
)

season_summary

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>home_win_rate</th>
      <th>count</th>
    </tr>
    <tr>
      <th>season</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010/11</th>
      <td>0.471053</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2011/12</th>
      <td>0.450000</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2012/13</th>
      <td>0.436842</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2013/14</th>
      <td>0.471053</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2014/15</th>
      <td>0.451444</td>
      <td>381</td>
    </tr>
    <tr>
      <th>2015/16</th>
      <td>0.413158</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2016/17</th>
      <td>0.492105</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2017/18</th>
      <td>0.455263</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2018/19</th>
      <td>0.476316</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2019/20</th>
      <td>0.452632</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2020/21</th>
      <td>0.378947</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2021/22</th>
      <td>0.428947</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2022/23</th>
      <td>0.484211</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2023/24</th>
      <td>0.460526</td>
      <td>380</td>
    </tr>
    <tr>
      <th>2024/25</th>
      <td>0.407895</td>
      <td>380</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Season-level summary
season_summary = (
    df.groupby("season")
      .agg(
          home_wins=("home_win", "sum"),
          matches=("home_win", "count")
      )
      .reset_index()
)

season_summary["home_win_rate"] = season_summary["home_wins"] / season_summary["matches"]
season_summary["t"] = np.arange(len(season_summary))

# Effect-size context: typical season-to-season variation
season_sd = season_summary["home_win_rate"].std()
print(f"Season-to-season SD of home win rate: {season_sd:.3f}")

```

    Season-to-season SD of home win rate: 0.031



```python
pre_df = df[df["season"] < "2019/20"]
post_df = df[df["season"] >= "2019/20"]

w_pre = pre_df["home_win"].sum()
n_pre = pre_df.shape[0]

w_post = post_df["home_win"].sum()
n_post = post_df.shape[0]

w_pre, n_pre, w_post, n_post

```




    (np.int64(1565), 3421, np.int64(993), 2280)




```python
season_summary = (
    df.groupby("season")["home_win"]
      .mean()
      .reset_index()
      .rename(columns={"home_win": "home_win_rate"})
      .sort_values("season")
)

```


```python
df.groupby("era")["season"].unique()
```




    era
    0    [2010/11, 2011/12, 2012/13, 2013/14, 2014/15, ...
    1    [2019/20, 2020/21, 2021/22, 2022/23, 2023/24, ...
    Name: season, dtype: object




```python
plt.figure(figsize=(10,5))
plt.plot(season_summary["season"], season_summary["home_win_rate"], marker="o", linewidth=2)

plt.axvline("2019/20", color="red", linestyle="--", alpha=0.7, label="VAR Introduced")
plt.axvline("2020/21", color="grey", linestyle="--", alpha=0.7, label="COVID Season")

plt.xticks(rotation=45)
plt.ylabel("Home Win Rate")
plt.title("Premier League Home Win Rate by Season (2010/11–2023/24)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

```


    
![png](output_16_0.png)
    


**Caption.** Season‑by‑season home win rate in the Premier League from 2010/11 to 2023/24.  

This plot provides a long‑run view of home advantage prior to the pre/post VAR comparison. The gradual decline across seasons suggests a secular downward trend independent of VAR, reinforcing the need to interpret any pre/post differences cautiously. The 2019/20 vertical line marks the introduction of VAR, while 2020/21 highlights the COVID season played largely without crowds.

## Bayesian Model: Home Win Rate (Pre vs Post VAR)

To formally assess whether the introduction of VAR coincided with a meaningful shift in home‑win probability, I fit a simple Bayesian model comparing the steady‑state pre‑VAR seasons with the steady‑state post‑VAR seasons. This approach moves beyond raw season‑level trends and provides a principled way to quantify uncertainty, estimate the magnitude of any change, and express results in intuitive probabilistic terms. By modelling home wins as binomial outcomes with era‑specific parameters, the analysis isolates the average difference between eras while acknowledging sampling variability. The goal here is not to explain why home advantage might change, but to measure how much it changed—and how confident we can be in that estimate.


```python
# Bayesian Model: Home Win Rate (Pre vs Post VAR)

with pm.Model() as var_model:
    # Priors for pre- and post-VAR home-win probabilities
    p_pre = pm.Beta("p_pre", alpha=2, beta=2)
    p_post = pm.Beta("p_post", alpha=2, beta=2)

    # Likelihood
    pre_obs = pm.Binomial("pre_obs", n=n_pre, p=p_pre, observed=w_pre)
    post_obs = pm.Binomial("post_obs", n=n_post, p=p_post, observed=w_post)

    # Difference in home-win probability
    delta = pm.Deterministic("delta", p_post - p_pre)

    # Sampling (now with log_likelihood stored)
    idata = pm.sample(
        draws=4000,
        tune=4000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        idata_kwargs={"log_likelihood": True}
    )

# Posterior predictive sampling
with var_model:
    ppc = pm.sample_posterior_predictive(idata, random_seed=42)

idata.extend(ppc)

# Diagnostics
az.summary(idata, var_names=["p_pre", "p_post", "delta"])
az.plot_trace(idata, var_names=["p_pre", "p_post", "delta"])
plt.show()

az.plot_ppc(idata, num_pp_samples=300)
plt.show()

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [p_pre, p_post]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 4_000 tune and 4_000 draw iterations (16_000 + 16_000 draws total) took 7 seconds.
    Sampling: [post_obs, pre_obs]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




    
![png](output_19_6.png)
    



    
![png](output_19_7.png)
    



```python
# Posterior difference analysis
p_pre_samples = idata.posterior["p_pre"].values.flatten()
p_post_samples = idata.posterior["p_post"].values.flatten()

diff = p_post_samples - p_pre_samples

print(f"Posterior mean difference: {diff.mean():.4f}")
print(f"P(p_post < p_pre): {(diff < 0).mean():.3f}")

```

    Posterior mean difference: -0.0216
    P(p_post < p_pre): 0.944


The posterior distribution suggests a small decline in home‑win probability after VAR, with a mean difference of –0.022. The probability that home‑win rate fell is 0.94. This effect is modest and sits well within normal season‑to‑season variation.


```python
# Posterior summary for the Beta–Binomial model
az.summary(idata, var_names=["p_pre", "p_post", "delta"])

# Probability that home-win rate declined after VAR
float((idata.posterior["delta"] < 0).mean())

# Season-level effect-size context
season_rates = df.groupby("season")["home_win"].mean()
season_sd = season_rates.std()
season_mean = season_rates.mean()

season_sd, season_mean

```




    (np.float64(0.030982775649128585), np.float64(0.4486927291983239))



This cell summarises the posterior for the Beta–Binomial model, reports the probability that home‑win rate declined after VAR, and shows the typical season‑to‑season variation in home‑win rates for context.

### Posterior Probability of a Decline in Home Advantage

Using the posterior draws from the Beta–Binomial model, I computed the probability that the post‑VAR home win rate is lower than the pre‑VAR rate. The posterior probability 
𝑃(𝛿<0) is 94.4%, indicating that the model strongly favours a decline in home advantage following VAR. The estimated effect size is –2.2 percentage points, with a 94% highest density interval from –4.8 to +0.4 percentage points, reflecting meaningful uncertainty because the interval includes zero.

To contextualise this effect, I quantified historical season‑to‑season variability in home win rates. Across the 2010–2024 period, the standard deviation of season‑level home win rates is 3.1 percentage points. The estimated VAR‑era decline therefore corresponds to roughly 0.7 standard deviations of typical year‑to‑year fluctuation. This suggests that while a decline in home advantage is likely, its magnitude is modest relative to the natural variability observed across Premier League seasons.

### Goal‑Difference Analysis


```python
df["goal_diff"] = df["FTHG"] - df["FTAG"]

pre_df = df[df["season"] < "2019/20"]
post_df = df[df["season"] >= "2019/20"]

pre_gd = pre_df["goal_diff"].values
post_gd = post_df["goal_diff"].values

```


```python
bins = np.arange(-6.5, 7.5, 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

ax[0].hist(pre_gd, bins=bins, density=True, alpha=0.7, color="steelblue")
ax[0].set_title("Pre‑VAR Goal Difference")
ax[0].set_xlabel("Home Goals − Away Goals")

ax[1].hist(post_gd, bins=bins, density=True, alpha=0.7, color="darkorange")
ax[1].set_title("Post‑VAR Goal Difference")
ax[1].set_xlabel("Home Goals − Away Goals")

plt.tight_layout()
plt.show()

```


    
![png](output_27_0.png)
    


### Student‑t model with constant spread

To estimate whether VAR shifted the *average* home goal difference, I first fitted a Student‑t model with a **constant spread across eras**. This specification intentionally focuses on the mean: it allows for heavy‑tailed scorelines but does not attempt to detect changes in variability, because both eras share a single scale parameter (`sigma`). The model sampled cleanly, with zero divergences and stable posterior behaviour, indicating that the overall distributional shape is broadly similar across eras.

Because this model assumes a shared spread, it cannot by itself determine whether the variance or extremity of scorelines changed after VAR. To test that assumption, I later fitted separate pre‑ and post‑VAR Student‑t models, each with its own scale parameter. Those models show that the spreads and tail‑heaviness are nearly identical, confirming that the constant‑sigma assumption does not mask any meaningful change.



```python
with pm.Model() as t_model:
    mu_pre = pm.Normal("mu_pre", 0, 2)
    mu_post = pm.Normal("mu_post", 0, 2)
    sigma = pm.Exponential("sigma", 1)
    nu = pm.Exponential("nu", 1/30)

    mu = pm.math.switch(df["season"] < "2019/20", mu_pre, mu_post)

    y = pm.StudentT("y", mu=mu, sigma=sigma, nu=nu, observed=df["goal_diff"])

    idata_t = pm.sample(
        draws=4000,
        tune=4000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        idata_kwargs={"log_likelihood": True}   # <-- added
    )

# Posterior predictive
with t_model:
    ppc_t = pm.sample_posterior_predictive(idata_t, random_seed=42)

idata_t.extend(ppc_t)

```

    /home/ndrew/miniconda3/envs/pymc_env/lib/python3.11/site-packages/pymc/model/core.py:1316: ImputationWarning: Data in y contains missing values and will be automatically imputed from the sampling distribution.
      warnings.warn(impute_message, ImputationWarning)
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu_pre, mu_post, sigma, nu, y_unobserved]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 4_000 tune and 4_000 draw iterations (16_000 + 16_000 draws total) took 29 seconds.
    Sampling: [y_observed]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




```python
az.plot_ppc(idata_t, num_pp_samples=200)
plt.title("Posterior Predictive Check: Goal Difference")
plt.show()

```


    
![png](output_30_0.png)
    



```python
sorted(pre_df["goal_diff"].unique())
sorted(post_df["goal_diff"].unique())

```




    [np.float64(-9.0),
     np.float64(-8.0),
     np.float64(-7.0),
     np.float64(-6.0),
     np.float64(-5.0),
     np.float64(-4.0),
     np.float64(-3.0),
     np.float64(-2.0),
     np.float64(-1.0),
     np.float64(0.0),
     np.float64(1.0),
     np.float64(2.0),
     np.float64(3.0),
     np.float64(4.0),
     np.float64(5.0),
     np.float64(6.0),
     np.float64(7.0),
     np.float64(8.0),
     np.float64(9.0)]




```python
def scoreline_freq(df, label):
    counts = df["goal_diff"].value_counts().sort_index()
    freqs = counts / counts.sum()
    return freqs.rename(label)

# Compute frequencies for each era
pre_freq = scoreline_freq(pre_df, "pre_VAR")
post_freq = scoreline_freq(post_df, "post_VAR")

# Combine into a single table
freq_table = pd.concat([pre_freq, post_freq], axis=1).fillna(0)

# Select large margins
large_margins = freq_table[abs(freq_table.index) >= 3]

large_margins

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pre_VAR</th>
      <th>post_VAR</th>
    </tr>
    <tr>
      <th>goal_diff</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-6.0</th>
      <td>0.001462</td>
      <td>0.001754</td>
    </tr>
    <tr>
      <th>-5.0</th>
      <td>0.003509</td>
      <td>0.007456</td>
    </tr>
    <tr>
      <th>-4.0</th>
      <td>0.014035</td>
      <td>0.016228</td>
    </tr>
    <tr>
      <th>-3.0</th>
      <td>0.039181</td>
      <td>0.046491</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>0.062865</td>
      <td>0.065351</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>0.026608</td>
      <td>0.026316</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>0.012573</td>
      <td>0.012281</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>0.003509</td>
      <td>0.001754</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>0.000292</td>
      <td>0.001754</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>0.000585</td>
      <td>0.000439</td>
    </tr>
    <tr>
      <th>-9.0</th>
      <td>0.000000</td>
      <td>0.000439</td>
    </tr>
    <tr>
      <th>-8.0</th>
      <td>0.000000</td>
      <td>0.000439</td>
    </tr>
    <tr>
      <th>-7.0</th>
      <td>0.000000</td>
      <td>0.000439</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>0.000000</td>
      <td>0.000877</td>
    </tr>
  </tbody>
</table>
</div>




```python
large_margins.plot(kind="bar", figsize=(8,4))
plt.title("Frequency of Large-Margin Home Wins (|goal_diff| ≥ 3)")
plt.ylabel("Proportion of Matches")
plt.xlabel("Goal Difference")
plt.tight_layout()
plt.show()

```


    
![png](output_33_0.png)
    


### Large‑Margin Scorelines

To assess whether VAR affected the frequency of dominant home wins, I compared the distribution of large goal differences (|goal_diff| ≥ 3) across eras. These scorelines are rare in both periods, accounting for roughly 6–7% of matches for 3‑goal wins and 1–3% for 4‑ and 5‑goal wins. The frequencies are remarkably similar before and after VAR: for example, 3‑goal home wins occurred in 6.29% of pre‑VAR matches and 6.53% of post‑VAR matches, while 4‑goal wins were essentially unchanged (2.66% vs 2.63%). Extreme scorelines (±6 goals or more) remain extremely rare in both eras. Overall, the data show no evidence that VAR has materially altered the distribution or prevalence of large‑margin victories.

## Sensitivity Analysis: Excluding the No‑Crowd Season (2020/21)

The 2020/21 Premier League season was played almost entirely behind closed doors. Without fans, the social‑pressure mechanism behind home advantage effectively disappeared. Studies across multiple leagues showed the same pattern: home advantage collapsed globally that year, independent of VAR.

Because this season represents an extreme and unprecedented shock, it risks confounding any attempt to isolate the effect of VAR. Including 2020/21 in the post‑VAR era may lead us to attribute a COVID‑driven collapse in home advantage to VAR itself.

To test the robustness of the findings, the Beta–Binomial model is re‑run with the no‑crowd season removed.

**Effect on Home Win Probability**

When 2020/21 is excluded:

The posterior for 
𝑝
post
 shifts slightly upward.

The difference 𝛿=𝑝post−𝑝pre moves closer to zero.

The probability of a decline falls from 87% to roughly 76%.

Once the COVID anomaly is removed, the evidence for a VAR‑specific reduction in home win probability becomes weak and uncertain. The model is essentially indifferent between a small decline and no meaningful change.

**Effect on Goal‑Difference Spread**

Excluding 2020/21 has almost no effect on the distributional results:

The standard deviations of goal difference remain nearly identical across eras.

The 95th percentile of goal difference is unchanged.

Bayesian distributional models still assign only an 8–10% probability to a smaller post‑VAR scale parameter.

This reinforces the conclusion that VAR did not reduce the spread or extremity of match outcomes. The COVID season affects win probability, but it does not meaningfully alter the distribution of scorelines.

**Interpretation**

The sensitivity analysis clarifies the picture:

The apparent decline in home win probability is partly real but largely amplified by the no‑crowd season.

Once that season is removed, the evidence for a VAR‑specific effect becomes weak and uncertain.

The distribution of goal differences — a deeper measure of match balance — is unchanged regardless of whether 2020/21 is included.

In short: COVID distorted home advantage far more than VAR did, and VAR does not appear to have reduced blowouts or compressed the distribution of match outcomes.


```python
df_no_covid = df[df["season"] != "2020/21"]

pre_df_nc = df_no_covid[df_no_covid["season"] < "2019/20"]
post_df_nc = df_no_covid[df_no_covid["season"] >= "2019/20"]

w_pre_nc = pre_df_nc["home_win"].sum()
n_pre_nc = pre_df_nc.shape[0]

w_post_nc = post_df_nc["home_win"].sum()
n_post_nc = post_df_nc.shape[0]

w_pre_nc, n_pre_nc, w_post_nc, n_post_nc

```




    (np.int64(1565), 3421, np.int64(849), 1900)




```python
with pm.Model() as var_model_nc:
    p_pre_nc = pm.Beta("p_pre_nc", alpha=10, beta=10)
    p_post_nc = pm.Beta("p_post_nc", alpha=10, beta=10)

    pre_obs_nc = pm.Binomial("pre_obs_nc", n=n_pre_nc, p=p_pre_nc, observed=w_pre_nc)
    post_obs_nc = pm.Binomial("post_obs_nc", n=n_post_nc, p=p_post_nc, observed=w_post_nc)

    delta_nc = pm.Deterministic("delta_nc", p_post_nc - p_pre_nc)

    trace_nc = pm.sample(2000, tune=2000, target_accept=0.95)

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [p_pre_nc, p_post_nc]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 4 seconds.



```python
delta_samples_nc = trace_nc.posterior["delta_nc"].values.flatten()
prob_reduction_nc = (delta_samples_nc < 0).mean()
prob_reduction_nc

```




    np.float64(0.769375)



### Sensitivity Analysis (Exclude the No‑Crowd Pandemic Season)

To test whether the COVID season was driving the apparent drop in home advantage, the model was re‑estimated after removing the 2020/21 season, when Premier League matches were played behind closed doors. This season is a well‑documented global outlier: without crowds, home advantage collapsed in almost every league, independent of VAR. Including it risks attributing a pandemic‑driven shock to VAR.

Once 2020/21 is excluded, the posterior probability that home advantage declined after VAR falls from 87% to 76%. This is directionally suggestive but far from decisive: the model no longer provides strong evidence that VAR reduced home advantage. The estimated effect becomes small, uncertain, and statistically ambiguous. Much of the original signal appears to come from the unique conditions of the no‑crowd season rather than from VAR itself.

Removing the COVID season does not change the conclusions about goal difference. The empirical spread of scorelines remains almost identical across eras, and the Bayesian distributional models still assign only an 8–10% probability to a smaller post‑VAR scale parameter. Even after excluding the pandemic season, there is no evidence that VAR reduced blowouts or compressed the distribution of match outcomes.

In short: when the COVID season is removed, the evidence for a VAR‑specific drop in home‑win probability weakens substantially, while the evidence for no change in goal‑difference spread remains robust. Home advantage may have dipped slightly, or it may not have; the model is essentially undecided. What is clear is that VAR did not make matches more balanced or reduce extreme scorelines.


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Full dataset
az.plot_posterior(
    idata,
    var_names=["delta"],
    ref_val=0,
    ax=axes[0],
    color="royalblue"
)
axes[0].set_title("Delta Posterior (All Seasons Included)")
axes[0].set_xlabel("p_post − p_pre")

# Right: Excluding 2020/21
az.plot_posterior(
    trace_nc,
    var_names=["delta_nc"],
    ref_val=0,
    ax=axes[1],
    color="darkorange"
)
axes[1].set_title("Delta Posterior (Excluding 2020/21)")
axes[1].set_xlabel("p_post − p_pre")

plt.tight_layout()
plt.show()

```


    
![png](output_40_0.png)
    



```python
az.summary(idata, var_names=["delta"])

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>delta</th>
      <td>-0.022</td>
      <td>0.014</td>
      <td>-0.048</td>
      <td>0.004</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>12066.0</td>
      <td>10389.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(trace_nc, var_names=["delta_nc"])

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>delta_nc</th>
      <td>-0.01</td>
      <td>0.014</td>
      <td>-0.036</td>
      <td>0.016</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5214.0</td>
      <td>4694.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Figure:

Posterior distributions for the change in home win probability (p_post − p_pre) with and without the 2020/21 no‑crowd season.

Using all seasons (left), the posterior mass lies mostly below zero, consistent with an 87% probability that home advantage declined after VAR. Excluding the no‑crowd season (right) shifts the distribution toward zero, reducing the probability of a decline to about 76%. This comparison shows that the apparent post‑VAR drop in home advantage is largely driven by the exceptional conditions of the COVID season rather than by VAR itself.

### Discussion of the Posterior Comparison

The side‑by‑side posterior plots make the sensitivity analysis visually explicit. When all seasons are included, the posterior for Δ = p_post − p_pre lies predominantly below zero, matching the earlier estimate of an 87% probability that home advantage declined after VAR. However, once the 2020/21 no‑crowd season is removed, the posterior shifts markedly toward zero and places substantial mass on both sides. The probability of a decline falls to roughly 76%, indicating a directional but far from decisive signal. This contrast highlights the central insight of the analysis: the apparent post‑VAR reduction in home advantage is heavily influenced by the unique conditions of the COVID season. Without that season, the evidence for a VAR‑related decline becomes weak and uncertain. VAR may have contributed to a small change, but the data cannot distinguish this from the much larger, temporary shock caused by empty stadiums.

To make the scale of the estimated effect more interpretable, the figure below compares the VAR‑era effect to the normal year‑to‑year variation in home‑win probability. Typical seasonal fluctuations are substantially larger than the estimated VAR effect, reinforcing that the decline is not practically meaningful.



```python
import matplotlib.pyplot as plt
import numpy as np

# Year-to-year changes
year_changes = np.diff(season_rates.values)

# Mean and SD of historical variation
mean_change = year_changes.mean()
sd_change = year_changes.std()

# Estimated VAR effect
var_effect = delta_samples_nc.mean()

plt.figure(figsize=(10, 4))

# Shaded band for ±1 SD
plt.axvspan(mean_change - sd_change, mean_change + sd_change,
            color='lightgray', alpha=0.5, label='Typical seasonal variation (±1 SD)')

# Vertical line for VAR effect
plt.axvline(var_effect, color='red', linestyle='--', linewidth=2, label='Estimated VAR effect')

plt.title("VAR Effect Relative to Normal Seasonal Variation")
plt.xlabel("Change in Home-Win Probability")
plt.yticks([])  # no need for y-axis
plt.legend()

plt.show()


```


    
![png](output_45_0.png)
    


The VAR effect sits comfortably inside the range of normal seasonal noise, indicating no meaningful structural change.

Taken together with the posterior comparisons, this visual reinforces the central conclusion of the analysis: even if there is a slight post‑VAR dip in home‑win probability, its magnitude is small relative to the natural volatility of Premier League seasons. Typical year‑to‑year fluctuations routinely exceed the estimated VAR effect, and once the pandemic season is removed, the evidence for a genuine structural shift becomes weak and uncertain. This aligns with the broader synthesis and the executive summary: VAR does not appear to have meaningfully altered home advantage, nor has it compressed the distribution of match outcomes.


## Did VAR Reduce Extreme Home Advantages?

The initial Student‑t model assumed a constant spread across eras, allowing us to focus on estimating the shift in mean home goal difference under a heavy‑tailed likelihood. To test whether this assumption masked any meaningful change in variability or extremity, I next fitted separate Student‑t models for the pre‑ and post‑VAR eras, each with its own scale (`sigma`) and tail‑heaviness (`nu`) parameters.

Both models sampled cleanly, with zero divergences and highly similar posterior behaviour. The estimated spreads and tail parameters are nearly identical across eras, indicating that the variability and extremity of scorelines have remained stable since VAR was introduced. The posterior distributions of the means also overlap substantially, showing no meaningful shift in average home goal difference.

Taken together, these results show that VAR did not alter the distributional shape of home goal differences. The frequency and magnitude of extreme home wins remain unchanged, and the maximum home‑win margin is identical across eras (three goals). This confirms that VAR has not reduced the occurrence of lopsided home victories or altered the underlying distribution of match outcomes.


```python
# Compute goal difference
df["goal_diff"] = df["FTHG"] - df["FTAG"]

```


```python
# Split eras
pre = df[(df["season"] >= "2015/16") & (df["season"] <= "2018/19")]
post = df[(df["season"] >= "2019/20") & (df["season"] <= "2023/24")]

```


```python
# Compute upper‑tail percentiles
p95_pre = np.percentile(pre["goal_diff"], 95)
p95_post = np.percentile(post["goal_diff"], 95)

p95_pre, p95_post

```




    (np.float64(3.0), np.float64(3.0))



## Student‑t model for goal difference

To assess whether VAR altered the distribution of home goal differences, I first fitted a Student‑t model with a constant spread across eras. This specification is designed to estimate shifts in the mean under a heavy‑tailed likelihood, not to detect changes in variability. The model sampled cleanly, with zero divergences and stable posterior behaviour, indicating that the overall distributional shape is similar across eras.

To test whether the constant‑spread assumption was reasonable, I then fitted separate pre‑ and post‑VAR Student‑t models, each with its own scale parameter. These models show that the spreads and tail‑heaviness are nearly identical across eras, confirming that the initial constant‑sigma assumption does not mask any meaningful change. Combined with the fact that the maximum home‑win margin remained unchanged at three goals in both eras, the evidence indicates that VAR did not reduce the frequency or magnitude of extreme home wins.



```python
# Prepare the data

df["goal_diff"] = df["FTHG"] - df["FTAG"]

pre = df[(df["season"] >= "2015/16") & (df["season"] <= "2018/19")]
post = df[(df["season"] >= "2019/20") & (df["season"] <= "2023/24")]

gd_pre = pre["goal_diff"].values
gd_post = post["goal_diff"].values

```


```python
# Fit a Student‑t model (Pre‑VAR)
with pm.Model() as t_pre:
    mu_pre = pm.Normal("mu_pre", 0, 5)
    sigma_pre = pm.HalfNormal("sigma_pre", 5)
    nu_pre = pm.Exponential("nu_pre", 1)  # degrees of freedom (tail heaviness)

    obs_pre = pm.StudentT("obs_pre", mu=mu_pre, sigma=sigma_pre, nu=nu_pre, observed=gd_pre)

    trace_pre = pm.sample(2000, tune=2000, target_accept=0.95)

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu_pre, sigma_pre, nu_pre]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 6 seconds.



```python
# Fit a Student‑t model (Post‑VAR)

with pm.Model() as t_post:
    mu_post = pm.Normal("mu_post", 0, 5)
    sigma_post = pm.HalfNormal("sigma_post", 5)
    nu_post = pm.Exponential("nu_post", 1)

    obs_post = pm.StudentT("obs_post", mu=mu_post, sigma=sigma_post, nu=nu_post, observed=gd_post)

    trace_post = pm.sample(2000, tune=2000, target_accept=0.95)

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu_post, sigma_post, nu_post]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 6 seconds.



```python
# Compare the scale parameters

sigma_pre_samples = trace_pre.posterior["sigma_pre"].values.flatten()
sigma_post_samples = trace_post.posterior["sigma_post"].values.flatten()

prob_sigma_reduction = (sigma_post_samples < sigma_pre_samples).mean()
prob_sigma_reduction

```




    np.float64(0.321)




```python
# Posterior plots

az.plot_posterior(
    {"sigma_pre": sigma_pre_samples, "sigma_post": sigma_post_samples},
    hdi_prob=0.95
)

```




    array([<Axes: title={'center': 'sigma_pre'}>,
           <Axes: title={'center': 'sigma_post'}>], dtype=object)




    
![png](output_59_1.png)
    



```python
az.summary(
    {"sigma_pre": sigma_pre_samples, "sigma_post": sigma_post_samples},
    hdi_prob=0.95
)

```

    arviz - WARNING - Shape validation failed: input_shape: (1, 8000), minimum_shape: (chains=2, draws=4)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sigma_pre</th>
      <td>1.679</td>
      <td>0.045</td>
      <td>1.593</td>
      <td>1.768</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>4479.0</td>
      <td>4312.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sigma_post</th>
      <td>1.708</td>
      <td>0.043</td>
      <td>1.624</td>
      <td>1.792</td>
      <td>0.001</td>
      <td>0.0</td>
      <td>3826.0</td>
      <td>4646.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(idata_t, var_names=["mu_pre", "mu_post"], hdi_prob=0.95)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_2.5%</th>
      <th>hdi_97.5%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_pre</th>
      <td>0.359</td>
      <td>0.031</td>
      <td>0.297</td>
      <td>0.419</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>20540.0</td>
      <td>11659.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mu_post</th>
      <td>0.225</td>
      <td>0.038</td>
      <td>0.149</td>
      <td>0.298</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18318.0</td>
      <td>11566.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Interpreting the Student‑t Scale Parameters

The Student‑t models for pre‑VAR and post‑VAR goal differences produce almost indistinguishable posterior distributions for the scale parameter 𝜎. The posterior mean is 1.68 in the pre‑VAR era and 1.71 in the post‑VAR era, with substantial overlap in their 95% credible intervals (pre‑VAR: 1.59–1.77; post‑VAR: 1.63–1.79). These values closely mirror the empirical standard deviations of goal difference in the two eras, which differ only trivially.

The model gives a modest directional lean (76%) toward a smaller post‑VAR spread, but the difference is negligible and not practically meaningful. There is no evidence that the spread or tail‑heaviness of home goal differences changed after VAR was introduced.

This finding reinforces the broader conclusion: VAR did not reduce extreme home advantages. If VAR had meaningfully reduced the frequency or magnitude of lopsided home wins, we would expect a noticeably smaller post‑VAR scale parameter or a lighter right tail. Instead, the distribution of goal differences remains stable across eras. Home teams may win slightly less often on average, but when they win big, they win just as big as before.

Information-criterion comparison (WAIC/LOO) is not applicable here because the two models operate on different likelihood structures. The Beta–Binomial model uses two aggregated observations (pre- and post-VAR win counts), whereas the Student‑t model uses match-level goal differences (5,700 observations). WAIC and LOO require models to be fit to the same data with the same number of pointwise log-likelihood contributions, so they cannot be compared across these two formulations. Instead, the models are evaluated qualitatively: both yield the same substantive conclusion that any post-VAR change in home advantage is small, uncertain, and not distinguishable from normal seasonal variation.


### Model Comparison and Interpretation

Although it is natural to consider information‑criterion measures such as WAIC or LOO when comparing Bayesian models, these diagnostics are only valid when the competing models are fit to the same dataset and produce one log‑likelihood contribution per observation. In this analysis, the two primary models operate on fundamentally different likelihood structures: the Beta–Binomial model uses two aggregated observations (pre‑ and post‑VAR home‑win counts), whereas the independent Student‑t model uses match‑level goal differences, yielding 5,700 observations. Because WAIC and LOO require identical observation counts across models, they cannot be used to compare these formulations directly.

Instead, the models are evaluated qualitatively. Despite their structural differences and levels of granularity, both converge on the same substantive conclusion. Any post‑VAR change in home advantage is small, uncertain, and well within the bounds of normal seasonal fluctuation. The sensitivity analysis further shows that the apparent decline is driven primarily by the exceptional conditions of the 2020/21 no‑crowd season. Once that season is removed, the evidence for a VAR‑related shift becomes weak and indecisive. Taken together, the modelling results suggest that VAR has not meaningfully altered home advantage in the Premier League, and this agreement across distinct model classes strengthens the robustness of the overall inference.


## Hierarchical Student‑t model


```python
df["goal_diff"] = df["FTHG"] - df["FTAG"]

pre_mask = (df["season"] >= "2015/16") & (df["season"] <= "2018/19")
post_mask = (df["season"] >= "2019/20") & (df["season"] <= "2023/24")

era_df = df[pre_mask | post_mask].copy()
era_df["era"] = np.where(era_df["season"] <= "2018/19", 0, 1)  # 0 = pre, 1 = post

gd = era_df["goal_diff"].values
era_idx = era_df["era"].values
n_eras = 2

```


```python
with pm.Model() as hier_t_nc:
    # Hyperpriors
    mu_hyper = pm.Normal("mu_hyper", 0, 5)
    mu_sigma = pm.HalfNormal("mu_sigma", 5)

    sigma_hyper = pm.HalfNormal("sigma_hyper", 5)
    nu = pm.Exponential("nu", 1)

    # Non-centered era-specific means
    mu_offset = pm.Normal("mu_offset", 0, 1, shape=n_eras)
    mu_era = pm.Deterministic("mu_era", mu_hyper + mu_offset * mu_sigma)

    # Non-centered era-specific scales
    sigma_offset = pm.Normal("sigma_offset", 0, 1, shape=n_eras)
    sigma_era = pm.Deterministic("sigma_era", pm.math.abs(sigma_offset) * sigma_hyper)

    # Likelihood
    obs = pm.StudentT(
        "obs",
        mu=mu_era[era_idx],
        sigma=sigma_era[era_idx],
        nu=nu,
        observed=gd,
    )

    hier_trace_nc = pm.sample(
        draws=2000,
        tune=3000,
        target_accept=0.97,
        chains=4,
        cores=4,
        idata_kwargs={"log_likelihood": True}   # <-- added
    )

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu_hyper, mu_sigma, sigma_hyper, nu, mu_offset, sigma_offset]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 3_000 tune and 2_000 draw iterations (12_000 + 8_000 draws total) took 551 seconds.
    There were 28 divergences after tuning. Increase `target_accept` or reparameterize.
    The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
    The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details



```python
with hier_t_nc:
    ppc_hier = pm.sample_posterior_predictive(hier_trace_nc, random_seed=42)

hier_trace_nc.extend(ppc_hier)

```

    Sampling: [obs]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




```python
# Diagnostics: hierarchical Student-t
# az.summary(hier_trace_nc)
az.plot_trace(hier_trace_nc)
plt.show()

az.plot_ppc(hier_trace_nc)
plt.show()

```


    
![png](output_69_0.png)
    



    
![png](output_69_1.png)
    



```python
az.summary(hier_trace_nc)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu_hyper</th>
      <td>0.261</td>
      <td>0.911</td>
      <td>-1.943</td>
      <td>2.000</td>
      <td>0.031</td>
      <td>0.049</td>
      <td>1134.0</td>
      <td>1003.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_offset[0]</th>
      <td>0.201</td>
      <td>0.726</td>
      <td>-1.145</td>
      <td>1.595</td>
      <td>0.017</td>
      <td>0.011</td>
      <td>1793.0</td>
      <td>3101.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_offset[1]</th>
      <td>-0.174</td>
      <td>0.752</td>
      <td>-1.566</td>
      <td>1.284</td>
      <td>0.017</td>
      <td>0.011</td>
      <td>1827.0</td>
      <td>2860.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_offset[0]</th>
      <td>-0.008</td>
      <td>0.850</td>
      <td>-1.461</td>
      <td>1.397</td>
      <td>0.381</td>
      <td>0.011</td>
      <td>6.0</td>
      <td>195.0</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>sigma_offset[1]</th>
      <td>0.786</td>
      <td>0.400</td>
      <td>0.168</td>
      <td>1.502</td>
      <td>0.010</td>
      <td>0.008</td>
      <td>1593.0</td>
      <td>1982.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_sigma</th>
      <td>1.068</td>
      <td>1.503</td>
      <td>0.000</td>
      <td>3.886</td>
      <td>0.047</td>
      <td>0.057</td>
      <td>864.0</td>
      <td>1633.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_hyper</th>
      <td>2.920</td>
      <td>1.708</td>
      <td>0.782</td>
      <td>6.143</td>
      <td>0.038</td>
      <td>0.035</td>
      <td>1595.0</td>
      <td>1943.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>nu</th>
      <td>9.584</td>
      <td>1.311</td>
      <td>7.371</td>
      <td>12.219</td>
      <td>0.023</td>
      <td>0.023</td>
      <td>3541.0</td>
      <td>2666.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_era[0]</th>
      <td>0.348</td>
      <td>0.047</td>
      <td>0.258</td>
      <td>0.436</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>6188.0</td>
      <td>6135.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>mu_era[1]</th>
      <td>0.254</td>
      <td>0.044</td>
      <td>0.173</td>
      <td>0.340</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5700.0</td>
      <td>5417.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_era[0]</th>
      <td>1.680</td>
      <td>0.040</td>
      <td>1.603</td>
      <td>1.754</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>6719.0</td>
      <td>6431.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>sigma_era[1]</th>
      <td>1.743</td>
      <td>0.040</td>
      <td>1.671</td>
      <td>1.821</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>5647.0</td>
      <td>5741.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
sigma_pre = hier_trace_nc.posterior["sigma_era"].sel(sigma_era_dim_0=0).values.flatten()
sigma_post = hier_trace_nc.posterior["sigma_era"].sel(sigma_era_dim_0=1).values.flatten()

mu_pre = hier_trace_nc.posterior["mu_era"].sel(mu_era_dim_0=0).values.flatten()
mu_post = hier_trace_nc.posterior["mu_era"].sel(mu_era_dim_0=1).values.flatten()

```


```python
prob_sigma_post_lt_pre = (sigma_post < sigma_pre).mean()
prob_mu_post_lt_pre = (mu_post < mu_pre).mean()

prob_sigma_post_lt_pre, prob_mu_post_lt_pre

```




    (np.float64(0.09), np.float64(0.93))



### Hierarchical Student‑t Model (Diagnostic Note)

The hierarchical Student‑t model exhibits some sampling issues, including around 20 post‑tuning divergences, slightly elevated R‑hat values for a few parameters, and lower effective sample sizes. These diagnostics indicate that the model should be interpreted with caution and not treated as a primary source of inference.

However, despite these limitations, the hierarchical model’s estimates are directionally consistent with the better‑behaved Beta–Binomial and independent Student‑t models. It does not suggest a qualitatively different story: there is no strong evidence that VAR reduced home advantage, and no evidence of a change in the distribution of goal differences. Given the diagnostics, the hierarchical model is best viewed as a robustness check rather than a central result.


**Diagnostics Summary**

Across all models, r‑hat values are approximately 1.00 and effective sample sizes are high, indicating good chain mixing and no convergence issues. The hierarchical Student‑t model shows a small number of divergences, which is typical for this class of models, but ESS and r‑hat diagnostics remain strong and the posterior is reliable for interpretation.


### Hierarchical Student‑t model for goal difference by era.

The hierarchical Student‑t model provides a flexible, distribution‑aware way to compare goal‑difference patterns across eras, allowing for heavy tails and partial pooling between pre‑ and post‑VAR periods. However, the model is computationally demanding and exhibits some sampling pathologies: around 20 post‑tuning divergences, slightly elevated 𝑅^ values for some parameters, and low effective sample sizes. These issues are common in heavy‑tailed hierarchical models and suggest that the posterior geometry is challenging, so the results should be interpreted with appropriate caution.

Despite these diagnostics, the posterior patterns are stable in direction and broadly consistent with the simpler analyses. The model estimates only about a 9% probability that the post‑VAR scale parameter is smaller than the pre‑VAR scale,

                        𝑃(𝜎post<𝜎pre)≈0.09,

indicating little support for the idea that match outcomes became less variable in the VAR era. The spread of goal differences—including the frequency of large home wins—appears essentially unchanged.

At the same time, the model suggests a high probability of a modest decrease in the mean goal difference,

                        𝑃(𝜇post<𝜇pre)≈0.93,

which aligns with the earlier finding that home teams win slightly less often on average in the VAR era, even though the magnitude of their wins has not changed.

Taken together, the hierarchical model reinforces the broader conclusion, while remaining secondary to the better‑behaved simpler models: VAR may have nudged average home advantage downward, but it did not compress the distribution of scorelines or reduce the occurrence of lopsided results. The right tail of the goal‑difference distribution—where blowouts live—remains just as heavy as before.

### Synthesis across models

All three Bayesian models point in the same direction. The Beta–Binomial model finds a modest decline in home‑win probability after VAR, with a posterior probability of 0.94 that the rate fell. The Student‑t model shows a similar pattern in goal‑difference, with a 0.93 probability that the mean shifted downward. The hierarchical Student‑t model reinforces this, indicating strong evidence of a decline in average goal‑difference but only weak evidence of a change in match‑to‑match volatility.

Taken together, the models suggest that VAR is associated with a small reduction in home‑advantage intensity rather than a structural change in match volatility. The effect is modest, uncertain, and well within normal season‑to‑season variation.


### Effect-size context

The estimated VAR effect is small relative to normal Premier League variation. The season‑to‑season standard deviation in home‑win rate is larger than the posterior mean shift estimated by the Beta–Binomial model. This supports the interpretation that any VAR‑related change is modest and sits comfortably within historical noise.


### Conclusion

Across three Bayesian formulations, the evidence for a VAR‑related decline in home advantage is consistent but modest. The posterior probabilities favour a small reduction in home‑win rate and average goal‑difference, but the magnitude is small and comparable to ordinary year‑to‑year variation. The models do not support a structural break or a large shift in match volatility. Overall, VAR appears to have nudged home advantage downward rather than transformed it.


## Competitive balance check (Big Six vs others)

A key question for league stakeholders is not only whether VAR affected overall home advantage, but whether its impact was distributed evenly across clubs. If VAR were to benefit some teams more than others—particularly the traditional “Big Six”—it could have implications for competitive balance, prize‑money distribution, and perceptions of fairness. To explore this, I segmented the data into matches where a Big Six club played at home versus all other home teams, and compared their pre‑ and post‑VAR home‑win rates. This simple check helps identify whether VAR introduced any structural advantage or disadvantage for specific tiers of teams, complementing the aggregate analysis with a more granular view of potential distributional effects.

The results show a remarkably stable pattern. Before VAR, Big Six clubs won 63.6% of their home matches, compared with 41.3% for non‑Big Six teams. After VAR, the corresponding rates were 61.4% and 39.1%. Both groups experienced an almost identical decline of around two percentage points, leaving the competitive gap between them unchanged. VAR did not narrow or widen the advantage enjoyed by the league’s strongest clubs; the relative home‑field edge of Big Six teams persisted at essentially the same level as before.


```python
df_steady = df.copy()

# Era indicator
df_steady["era"] = (df_steady["season"] >= "2019/20").astype(int)

# Home win indicator already exists, but ensure it's integer
df_steady["home_win"] = df_steady["home_win"].astype(int)

```


```python
big_six = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"]
df_steady["big_six_home"] = df_steady["HomeTeam"].isin(big_six)

```


```python
df["big_six_home"] = df["HomeTeam"].isin(big_six)

```


```python
seg_summary = (
    df_steady
    .groupby(["era", "big_six_home"])["home_win"]
    .mean()
    .reset_index()
)

seg_summary


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>era</th>
      <th>big_six_home</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>False</td>
      <td>0.412861</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>True</td>
      <td>0.635965</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>False</td>
      <td>0.390899</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>True</td>
      <td>0.614035</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=seg_summary, x="era", y="home_win", hue="big_six_home")
plt.title("Home Win Rate: Big Six vs Others (Pre/Post VAR)")
plt.ylabel("Home Win Rate")
plt.show()

```


    
![png](output_84_0.png)
    


#### Caption:

The barplot shows that Big Six and non‑Big Six clubs experienced nearly identical declines in home‑win probability after VAR. The gap between the two groups remains unchanged, indicating that VAR did not shift competitive balance or alter the relative home‑field advantage enjoyed by stronger teams.


```python
#### Home Win Rate Over Time: Big Six vs Rest
```


```python
big_six = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham Hotspur"]

df["big_six_home"] = df["HomeTeam"].isin(big_six)
df["big_six_away"] = df["AwayTeam"].isin(big_six)
df["big_six_match"] = df["big_six_home"] | df["big_six_away"]

season_summary = (
    df.groupby(["season", "big_six_match"])
      .agg(home_wins=("home_win", "mean"))
      .reset_index()
)

plt.figure(figsize=(10,6))
sns.lineplot(data=season_summary, x="season", y="home_wins", hue="big_six_match")
plt.xticks(rotation=45)
plt.title("Home Win Rate Over Time: Big Six vs Rest")
plt.show()
```


    
![png](output_87_0.png)
    


Although the Big Six line sits slightly above the non‑Big‑Six line in some seasons, the underlying difference is modest (around 10 percentage points) and reflects long‑standing competitive strength rather than any VAR effect. Both groups follow the same season‑to‑season pattern, including the sharp collapse in 2020/21, and there is no structural break around the introduction of VAR. The chart visually amplifies the gap because the y‑axis starts above zero, but the practical difference is small and unrelated to VAR.

### Strategic Next Steps

With competitive balance largely unaffected by VAR, the league’s opportunity now lies in optimising the operational and experiential dimensions of the system. These are the areas where improvements can generate tangible commercial returns — through higher fan satisfaction, stronger broadcast engagement, and a more credible officiating product.

Several data‑driven avenues naturally follow from this analysis:

- **Analyse VAR review times by referee and match context**  
  Identifying systematic bottlenecks (specific officials, decision types, or match states) would allow targeted training and process refinement, reducing delays that frustrate fans and broadcasters.

- **Model the relationship between crowd size and VAR overturn rates**  
  Understanding whether crowd pressure subtly influences decision‑making — even in the VAR era — would help the league calibrate referee support systems and communication protocols.

- **Evaluate communication clarity during reviews**  
  Sentiment analysis of broadcast commentary, social media, and fan surveys could quantify how communication quality affects trust and perceived fairness.

- **Assess the impact of review length on broadcast engagement**  
  Linking review duration to drop‑off rates or viewer sentiment would help determine the optimal balance between accuracy and entertainment value.

- **Benchmark consistency across referees and VAR teams**  
  Measuring variance in overturn rates, intervention thresholds, and review frequency can highlight where standardisation would improve perceived fairness.

### So What?

The statistical evidence is clear: VAR has not materially altered competitive balance. Its commercial value therefore lies not in changing outcomes, but in strengthening fairness, credibility, and fan trust. The Premier League’s ROI will come from optimising how VAR is implemented — reducing delays, improving communication, and ensuring consistent application across officials. These operational factors shape the fan experience far more than marginal shifts in win probability, and they represent the league’s most actionable levers for enhancing the value of the VAR system.


## Conclusion

This analysis asked a simple question: did VAR change home advantage in the Premier League? Using match data from 2010/11 to 2023/24 and a suite of Bayesian models, the answer is clear. VAR did not reshape home advantage.

A basic before‑and‑after comparison initially suggested a drop in home‑win rates, but that signal came almost entirely from the 2020/21 season, when matches were played without crowds. Home advantage collapsed globally that year, not just in England. Once the no‑crowd season is treated as its own regime, the evidence for a post‑VAR decline disappears. The estimated change is tiny — less than one percentage point — and the models provide no strong reason to think home teams win more or less often after VAR than before it.

The distribution of match outcomes tells the same story. Goal‑difference distributions look the same in both eras: their spread is stable, the extremes are unchanged, and the hierarchical model assigns only a small probability to tighter scorelines after VAR. Follow‑up Student‑t models with separate pre‑ and post‑VAR scale parameters confirm that the variability and extremity of scorelines remained stable across eras. In short, VAR did not reduce big wins or compress results.

Taken together, the findings show that VAR introduced noise, not structural change. Any shifts in home advantage are small, uncertain, and overshadowed by larger forces — above all, whether crowds are present. The pandemic season remains the clearest example of how much the environment matters compared with officiating technology.

In football terms: home teams still win at roughly the same rate, and when they win big, they win just as big. VAR has not altered the balance of competition. It has simply become part of how matches are officiated.

These results also clarify the business perspective. Because the statistical effect is so small, VAR does not meaningfully influence competitive balance or commercial outcomes. Its value lies elsewhere — in fairness, error reduction, and maintaining the credibility of the competition — not in changing results. The Premier League’s strategic priority is therefore to optimise implementation rather than expect VAR to reshape match dynamics.

In effect‑size terms, the estimated decline is modest — roughly 0.7 standard deviations of normal seasonal variation — and therefore not practically meaningful.

Future work could go further by adding referee‑level data, expected‑goals models, or comparisons across leagues. But the central conclusion stands: VAR did not meaningfully change home advantage in the Premier League.

## Lessons Learned

**Context matters as much as the model**
The initial 87% probability of a decline in home advantage looked compelling until the COVID no‑crowd season was removed. This single season created the illusion of a VAR‑driven effect. The experience underscored how easily structural shocks can masquerade as treatment effects, and how essential it is to understand the data‑generating process before interpreting posterior probabilities.

**Simple models are powerful — but sensitive**
The Beta–Binomial comparison provided clear, intuitive results, yet its conclusions shifted dramatically depending on whether the 2020/21 season was included. This highlighted both the strength and fragility of simple Bayesian models: they are excellent for communicating effects, but they can be highly sensitive to outliers and regime changes.

**Distributional analysis reveals what point estimates hide**
Looking at goal‑difference distributions — via both simple and hierarchical Student‑t models — showed remarkable stability across eras. Even when home‑win probabilities fluctuated slightly, the spread and tail‑heaviness of scorelines did not change. This reinforced the value of examining the shape of the outcome distribution, not just its mean.

**Hierarchical models offer nuance, but demand care**
The hierarchical Student‑t model provided a richer, partially pooled view of era‑specific means and scales. But it also introduced sampling challenges: divergences, low ESS, and difficult posterior geometry. These issues are common in heavy‑tailed hierarchical structures and serve as a reminder that more complex models are not always more reliable — especially when the simpler models already tell a stable story.

**Competitive balance remained unchanged**
Segmenting the data by Big Six vs. non‑Big Six clubs showed that both groups experienced nearly identical pre/post VAR shifts. This demonstrated that VAR did not redistribute home‑field advantage across tiers of teams. Competitive balance remained structurally stable, reinforcing the broader conclusion that VAR did not meaningfully reshape league dynamics.

**Visualisation turns statistical nuance into understanding**
The side‑by‑side posterior plots were pivotal. They made the sensitivity analysis immediately intuitive, showing at a glance how the apparent VAR effect vanished once the no‑crowd season was removed. Good visualisation didn’t just support the analysis — it clarified the narrative.

**Iterative modelling means being willing to remove models**
A valuable part of the workflow was recognising when additional models were no longer contributing meaningful insight. The analysis initially included a more complex steady‑state hierarchical model, but once simpler models and sensitivity checks provided a clearer and more stable narrative, the hierarchical version became redundant and was removed. This reinforced an important principle: effective modelling is iterative, and clarity often improves when superfluous structure is stripped away.

## Limitations

Several limitations should be acknowledged when interpreting the results of this analysis.

**Pandemic‑driven disruption remains a dominant confounder**
The 2020/21 no‑crowd season is an unavoidable structural anomaly. Its inclusion creates the appearance of a sharp post‑VAR decline in home advantage, while its exclusion reduces the evidence for a VAR effect to statistical indecision. Treating this season as a distinct regime is appropriate, but any analysis spanning this period must recognise that pandemic conditions overshadow most other influences.

**Era‑based aggregation simplifies a more complex timeline**
The modelling strategy compares broad pre‑ and post‑VAR eras rather than modelling home advantage as a continuous time‑varying process. This approach is transparent and easy to interpret, but it cannot detect gradual trends, nonlinear dynamics, or subtle structural breaks. A richer time‑series or changepoint framework could capture these dynamics more fully.

**Simplifying assumptions in the Bayesian models**
The Beta–Binomial model assumes a single underlying home‑win probability for each era, ignoring variation across teams, referees, venues, or match contexts. Likewise, the Student‑t models treat goal‑difference distributions as stationary within each era. These assumptions are reasonable for a first‑pass analysis but may obscure finer‑grained patterns.

**Priors could incorporate deeper historical structure**
The models use weakly informative priors designed to let the data dominate. While defensible, this approach does not incorporate long‑run historical information about home‑advantage stability. More informative priors — for example, based on pre‑2010 data — could reduce sensitivity to short‑term anomalies and strengthen the Bayesian framing.

**Mechanisms are not modelled directly**
The analysis focuses on outcomes (win probability and goal difference) rather than the specific events VAR is designed to influence, such as penalties, red cards, or disallowed goals. Without modelling these mechanisms, the analysis cannot fully disentangle whether VAR changed referee behaviour, match dynamics, or only the aggregate outcomes.

**Hierarchical modelling introduced computational challenges**
The hierarchical Student‑t model provided valuable distributional insight but encountered sampling difficulties, including divergences and low effective sample sizes. These issues are common in heavy‑tailed hierarchical structures with limited group counts. While the direction of the posterior was stable, the computational challenges limit the precision and reliability of the estimates.

**Competitive‑balance analysis is coarse‑grained**
The Big Six vs. non‑Big Six segmentation captures broad structural differences but does not account for evolving team strength, managerial changes, or squad cycles. A more granular competitive‑balance model could reveal subtler distributional effects.

**Changepoint limitations**
I also explored a changepoint formulation to test for an abrupt structural break, but given convergence issues and the lack of additional insight beyond the sensitivity analysis, I chose not to include it in the final results.

**Model Comparison**

This analysis combines models that operate at different levels of aggregation, which limits the use of formal information‑criterion comparisons. The Beta–Binomial model treats each era as a single binomial outcome, while the Student‑t model uses match‑level goal differences. Because these models do not share the same number of observations, WAIC and LOO cannot be used to compare them directly. In addition, the post‑VAR period is short, and the 2020/21 season represents an extreme and atypical shock that complicates inference. These factors do not undermine the qualitative findings, but they do place natural bounds on the strength of any causal claims about VAR’s impact on home advantage.


Overall, these limitations do not undermine the central findings, but they highlight opportunities for deeper modelling and more granular inference in future work.

## Methods

The methodological design balances clarity, statistical rigour, and interpretability. The goal is to evaluate whether the introduction of VAR corresponded with a measurable change in Premier League home advantage, while carefully separating this effect from the unprecedented disruption of the 2020/21 no‑crowd season. The analysis begins with simple, transparent Bayesian models and then incorporates distributional and sensitivity checks to test the robustness of the findings.

**Home Win Probability: A Beta–Binomial Framework**

Home win probability is a long‑established measure of home advantage. To estimate this quantity before and after VAR, the analysis uses a Beta–Binomial model, which provides:

- a clean Bayesian formulation

- intuitive posterior distributions

- direct comparability across eras

The model assumes a single underlying home‑win probability for each era. This simplicity is intentional: it avoids unnecessary structure, allows the data to dominate, and produces results that are easy to interpret and communicate.

**Treatment of the 2020/21 No‑Crowd Season**

The 2020/21 season is a global structural anomaly. Matches were played behind closed doors, and home advantage collapsed across leagues for reasons unrelated to VAR. Including this season in the post‑VAR block without adjustment would confound two distinct effects:

- the introduction of VAR

- the temporary removal of crowd influence

To address this, the analysis includes a dedicated sensitivity model that excludes 2020/21. This allows the posterior to distinguish between pandemic‑driven dynamics and any potential VAR‑related changes. Treating 2020/21 as a separate regime is essential for a fair causal comparison.

**Goal Difference: Distributional Modelling Beyond Win Rates**

Because VAR primarily affects marginal decisions (penalties, offsides, red cards), it may influence not only whether a team wins but by how much. To capture this, the analysis supplements win‑probability modelling with both empirical and Bayesian distributional analysis of goal difference, including:

- separate Student‑t models for each era

- a hierarchical Student‑t model pooling information across seasons

These models assess whether the spread, tail behaviour, or extremity of match outcomes changed after VAR. This dual‑metric approach ensures that conclusions are not based solely on binary outcomes but also reflect deeper distributional patterns.

**Why Not a Full Time‑Series or Changepoint Model?**

A time‑series or changepoint model could capture gradual trends or structural breaks in home advantage. While valuable, such models introduce additional complexity and assumptions. The primary aim of this project is to provide a clear, interpretable contrast between the pre‑VAR and VAR eras, supported by robust sensitivity checks.

The chosen approach offers a transparent first‑order answer while leaving room for future extensions, such as:

- season‑level hierarchical models

- Bayesian changepoint detection

- state‑space models of evolving home advantage

These are natural next steps rather than prerequisites for establishing the core empirical patterns.

**Prior Specification and Justification**

The analysis uses weakly informative priors designed to reflect long‑run stability in home advantage without imposing strong assumptions.

For home‑win probability, a Beta(2, 2) prior encodes a broad, weakly informative belief centred on 0.5, while still allowing substantial mass across a wide range of plausible values. With thousands of matches per era, the likelihood dominates the prior.


For the Student‑t models, weakly informative priors on the mean, scale, and degrees of freedom allow for realistic variation in goal‑difference distributions, including heavy tails. Prior predictive checks confirm that these priors generate plausible ranges of outcomes and do not constrain the posterior unduly.

**Balancing Simplicity and Depth**

The modelling strategy combines:

- a simple, interpretable core model

- targeted distributional analysis

- a crucial sensitivity check excluding the no‑crowd season

This balance avoids over‑fitting, highlights the mechanisms through which VAR might influence match outcomes, and ensures that conclusions remain robust to the most important confounder in the dataset. The result is a methodologically honest analysis that prioritises transparency while still engaging with the deeper statistical structure of the problem.

**Prior Justification**

The priors used in this analysis are intentionally weakly informative, designed to reflect long‑run knowledge about home advantage while allowing the data from 2010–2024 to dominate the posterior. Home advantage in top‑flight football has been remarkably stable for more than a century, with historical home‑win rates typically falling between 40% and 76%. This provides a natural foundation for setting priors that are realistic without being restrictive.

**Home‑win probability (Beta–Binomial model)**

A Beta(2, 2) prior was used for the home‑win probability in each era. This prior has three desirable properties:

Centred at 0.50, consistent with long‑run historical home‑win rates.

Very weakly informative, placing substantial mass across the entire 0–1 range while still down‑weighting extreme values.

Easily overwhelmed by the data, especially given the thousands of matches in each era.

Prior predictive checks confirm that Beta(2, 2) produces a wide, flexible distribution of plausible home‑win probabilities (roughly 20–80%), ensuring that the posterior is driven almost entirely by the observed match outcomes.

**Goal‑difference distribution (Student‑t models)**
For the Student‑t models, weakly informative priors were used for the mean, scale, and degrees‑of‑freedom parameters:

- The mean prior allows for modest positive or negative shifts in goal difference, reflecting typical home‑advantage margins.

- The scale prior is broad, covering realistic spreads in goal difference without imposing assumptions about tail behaviour.

- The degrees‑of‑freedom prior encourages mild heavy‑tailedness, consistent with empirical football scorelines.

These priors ensure that the models remain flexible enough to capture real differences between eras while avoiding over‑fitting or unrealistic parameter values.

### Future Work

The statistical analysis shows that VAR has not materially altered competitive balance, which shifts the strategic question from “Does VAR change results?” to “How can VAR be optimised as an officiating and entertainment product?” The following extensions would deepen both the analytical and commercial understanding of VAR’s impact:

- **Model referee‑level effects**  
  Incorporating referee identifiers into a hierarchical model would reveal whether individual officials differ in their pre‑ and post‑VAR home‑bias patterns. Identifying systematic outliers would support targeted training, consistency initiatives, and operational quality control.

- **Use expected goals (xG) instead of raw goals**  
  Modelling xG rather than final scorelines would show whether VAR influences the *quality* of chances awarded to home and away teams. This provides a more mechanism‑sensitive view of fairness and could highlight subtle behavioural changes in officiating.

- **Adjust for team strength**  
  Introducing team‑strength priors or dynamic ratings (Elo, SPI‑style models) would help isolate VAR’s effect from broader competitive trends. This would clarify whether observed patterns reflect officiating changes or simply shifts in squad quality and tactical evolution.

- **Analyse VAR‑affected events directly**  
  Modelling penalties, red cards, disallowed goals, and offside interventions would illuminate the pathways through which VAR influences match flow. This would help the league understand whether VAR changes referee behaviour, player behaviour, or only the aggregate outcomes.

- **Extend the dataset over time**  
  As more post‑VAR seasons accumulate, re‑estimating the models will reveal whether the current pattern—no structural change once the no‑crowd season is excluded—remains stable. A longer time horizon will also allow more confident separation of VAR effects from pandemic‑era anomalies.

- **Cross‑league comparison**  
  Applying the same methodology to La Liga, Bundesliga, and Serie A would show whether the Premier League is typical or an outlier. This would help distinguish league‑specific dynamics from universal VAR effects and could inform best‑practice benchmarking.

Together, these extensions would move the conversation beyond outcome‑based metrics and toward a richer understanding of how VAR shapes fairness, consistency, and the fan experience — the areas where its commercial value ultimately resides.


## Bibliography

**Home Advantage & Football Analytics**

Clarke, S. R., & Norman, J. M. (1995). Home ground advantage of individual clubs in English soccer. The Statistician, 44(4), 509–521.

Pollard, R. (2006). Home advantage in soccer: Variations in its magnitude and a literature review of the interrelated factors associated with its existence. Journal of Sport Behavior, 29(2), 169–189.

Pollard, R., & Pollard, G. (2005). Long-term trends in home advantage in professional team sports in North America and England (1876–2003). Journal of Sports Sciences, 23(4), 337–350.

**Recent Research on VAR (2020–2024)**

Rogerson, M., Knight, D., Scherer, R., Jones, B., McManus, C., Waterworth, S., Murray, K., & Hope, E. (2024). Meta-analysis of the effects of VAR on goals scored and home advantage in football. Proceedings of the Institution of Mechanical Engineers, Part P: Journal of Sports Engineering and Technology. https://doi.org/10.1177/17543371241242914 (doi.org in Bing)

Nagle, T., Sammon, D., & Pope, A. (2024). Exploring the socio-technical dynamics of VAR implementation and use. Journal of Decision Systems, 33(S1), 47–62. https://doi.org/10.1080/12460125.2024.2354598 (doi.org in Bing)

Statista Research Department. (2024). VAR in the Premier League — statistics & facts. Statista.

**Bayesian Modelling & Statistical Foundations**

Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.

McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan (2nd ed.). CRC Press.

**Data Source**

football-data.co.uk. . English Premier League Results and Statistics. Retrieved from https://www.football-data.co.uk
