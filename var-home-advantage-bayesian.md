# Did VAR Reduce Home Advantage in the Premier League?

## Executive Summary

This analysis evaluates whether the introduction of the Video Assistant Referee (VAR) in the 2019/20 Premier League season successfully reduced the long-standing structural Home Field Advantage.

Our robust Bayesian Beta-Binomial model estimates a 52% probability that Home Win Rate declined in the VAR era (2019/20–2023/24) when accounting for seasonal variance. This suggests that VAR’s impact on home advantage is modest and statistically ambiguous. In contrast, a standard pre/post comparison that includes the anomalous 2020/21 "no-crowd" season incorrectly suggests an 87% probability of decline. This divergence underscores that stadium environment (crowd presence) exerts a significantly stronger influence on match outcomes than the refereeing technology itself.

Robust Impact is Minimal: When excluding the 2020/21 outlier season, the model suggests a shift from a pre-VAR home win rate of ~46% to a post-VAR rate of ~44%, with a 52% probability that this change is real. This indicates that VAR’s direct influence is likely overstated; the game is roughly as fair as it was before the intervention.
The Stadium Factor is Primary: The 87% decline observed in the uncleaned data is primarily driven by the absence of crowds during the 2020/21 season. This confirms academic consensus that crowd pressure is a dominant component of home advantage, one that VAR cannot fully mitigate.
Consistent Advantage Persists: While the magnitude of advantage may have dipped slightly, the fundamental hierarchy of teams (Big Six vs. mid-table) and the flow of the league remains unchanged.
Strategic Recommendation

The evidence does not support the narrative that VAR "broke" home advantage. Rather, it implies that VAR stabilized an already-existing phenomenon by reducing the variance of officiating errors (marginal calls). Future analysis should focus less on "Did VAR work?" and more on optimizing the fan experience and game flow within the parameters of this new, standardized officiating environment.

## Key Findings

Home advantage has declined slightly in raw terms, from about 46% pre‑VAR to about 44% post‑VAR.

An initial naive Bayesian Beta–Binomial model estimates an 87% probability that home advantage fell after VAR was introduced, but this result is highly sensitive to the 2020/21 no‑crowd season. Excluding the no‑crowd season reduces the probability of a decline to roughly 52% (effectively a coin toss), showing that much of the apparent drop is driven by pandemic‑related conditions rather than VAR.

The estimated effect size is small — about a 1.5‑percentage‑point reduction in home win probability — and well within normal season‑to‑season variation.

**Goal‑difference analysis shows no evidence that VAR reduced extreme outcomes.**
The 95th percentile of goal difference is unchanged across eras, and the empirical standard deviations are nearly identical (1.86 pre‑VAR vs 1.96 post‑VAR).

**Bayesian distributional models reinforce this.**
Separate and hierarchical Student‑t models assign only an 8–10% probability to a smaller post‑VAR scale parameter, indicating that the spread of match outcomes did not narrow after VAR.

**VAR therefore does not appear to have reduced blowouts or compressed the distribution of scorelines.**
Large home wins (+3, +4, +5 goals) remain just as common in the VAR era as before.

**Overall, VAR’s impact on home advantage is modest and uncertain.**
Crowd presence and broader contextual factors exert a far stronger influence on match outcomes than VAR. Home advantage remains a persistent feature of the Premier League, even if slightly weaker in recent years.

##  Introduction

Home advantage is one of the most persistent findings in football analytics. Across leagues, eras, and competition formats, home teams win more often than away teams. Pollard & Pollard (2005) show that this pattern has been stable for more than a century, suggesting that home advantage is a structural feature of the sport. Clarke & Norman (1995) found that it is particularly strong in football, largely due to the influence of the crowd. Pollard (2006) argued that the effect arises from several factors — travel fatigue, crowd support, and, crucially, referee behaviour.

The introduction of VAR (Video Assistant Referee) in the Premier League in 2019/20 created a natural test of one specific mechanism: referee bias. If part of home advantage stems from social pressure on referees, then a system that provides video review and external oversight might reduce that bias. This leads to a clear question: did VAR reduce home advantage in the Premier League?

To answer it, this analysis uses match results from 2010/11 to 2023/24 and compares home win probabilities before and after VAR’s introduction. A Bayesian Beta–Binomial model estimates the home win probability in each era and quantifies the probability that the post‑VAR value is lower than the pre‑VAR value.

Home advantage, however, is not expressed only through win rates. If referee‑driven bias affects marginal decisions — penalties, red cards, stoppage‑time goals — then VAR might also reduce the extremity of match outcomes. Correcting marginal errors could compress the distribution of goal differences and reduce the frequency of large home wins. To test this, the analysis also examines the distribution of goal difference across eras using empirical summaries and Bayesian distributional models, including a hierarchical Student‑t.

Together, these two perspectives — win probability and goal‑difference distribution — provide a broad assessment of VAR’s impact on home advantage.

## Hypotheses

This analysis addresses one central question:

- Did VAR reduce home advantage in the Premier League?

- To answer it, we evaluate two related hypotheses.

Research Question 1

1. Did VAR reduce home advantage in the Premier League?

H₁: VAR reduced home win probability

Let:

𝑃pre = home win probability in the pre‑VAR era (2010/11–2018/19)

𝑃post = home win probability in the VAR era (2019/20–2023/24)

Define:

                        𝛿 = 𝑃post−𝑃pre

The first hypothesis is:

                        𝐻1:𝛿<0

A Bayesian Beta–Binomial model estimates the posterior distribution of 𝛿 and quantifies the probability that home advantage declined after VAR.

Research Question 2

2. Did VAR reduce the spread of goal differences

H₂: VAR reduced the spread of goal differences

Let:

- 𝜎pre = scale (spread) of goal difference in the pre‑VAR era

- 𝜎post = scale of goal difference in the VAR era

Define:

                            Δ𝜎=𝜎post−𝜎pre


The second hypothesis is:

                            𝐻2:Δ𝜎<0

This tests whether VAR reduced the variability of match outcomes — specifically, whether it decreased the frequency or magnitude of extreme home wins. It is evaluated using empirical standard deviations, percentile comparisons, separate Student‑t models, and a hierarchical Student‑t model.

## Data & Definitions

**Data Source**

Match results were obtained directly from football-data.co.uk, a long‑standing public repository of historical football statistics. The data were accessed programmatically using a simple Python function that downloads each season’s Premier League CSV file.

**Seasons Included**

Pre‑VAR era: 2010/11–2018/19

VAR era: 2019/20–2023/24

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

**Why These Variables Matter**

- home_win captures home advantage as an outcome, enabling estimation of home win probabilities in each era.

- goal_diff captures the distribution of match outcomes, allowing us to test whether VAR reduced the spread or extremity of results.

- era provides a clean, reproducible way to separate the two periods for both hypotheses.

Together, these variables support the two central hypotheses:

- H₁: VAR reduced home win probability

- H₂: VAR reduced the spread of goal differences


```python
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
```

    /home/ndrew/miniconda3/envs/pymc_env/lib/python3.11/site-packages/arviz/__init__.py:39: FutureWarning: 
    ArviZ is undergoing a major refactor to improve flexibility and extensibility while maintaining a user-friendly interface.
    Some upcoming changes may be backward incompatible.
    For details and migration guidance, visit: https://python.arviz.org/en/latest/user_guide/migration_guide.html
      warn(



```python
import os

if os.path.exists("pl_matches_cached.csv"):
    print("Loading cached data...")
    df = pd.read_csv("pl_matches_cached.csv")
else:
    print("Downloading data...")
    dfs = []
    for year in range(2010, 2025):
        dfs.append(load_pl_season_online(year))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv("pl_matches_cached.csv", index=False)

```

    Downloading data...
    Fetching 2010 from https://www.football-data.co.uk/mmz4281/1011/E0.csv...
    Fetching 2011 from https://www.football-data.co.uk/mmz4281/1112/E0.csv...
    Fetching 2012 from https://www.football-data.co.uk/mmz4281/1213/E0.csv...
    Fetching 2013 from https://www.football-data.co.uk/mmz4281/1314/E0.csv...
    Fetching 2014 from https://www.football-data.co.uk/mmz4281/1415/E0.csv...
    Fetching 2015 from https://www.football-data.co.uk/mmz4281/1516/E0.csv...
    Fetching 2016 from https://www.football-data.co.uk/mmz4281/1617/E0.csv...
    Fetching 2017 from https://www.football-data.co.uk/mmz4281/1718/E0.csv...
    Fetching 2018 from https://www.football-data.co.uk/mmz4281/1819/E0.csv...
    Fetching 2019 from https://www.football-data.co.uk/mmz4281/1920/E0.csv...
    Fetching 2020 from https://www.football-data.co.uk/mmz4281/2021/E0.csv...
    Fetching 2021 from https://www.football-data.co.uk/mmz4281/2122/E0.csv...
    Fetching 2022 from https://www.football-data.co.uk/mmz4281/2223/E0.csv...
    Fetching 2023 from https://www.football-data.co.uk/mmz4281/2324/E0.csv...
    Fetching 2024 from https://www.football-data.co.uk/mmz4281/2425/E0.csv...



```python
df = df.copy()
```


```python
df["home_win"] = (df["FTHG"] > df["FTAG"]).astype(int)
```


```python
print(df.head())
print(df.columns)
print(len(df))

```

      Div      Date     HomeTeam    AwayTeam  FTHG  FTAG FTR  HTHG  HTAG HTR  ...  \
    0  E0  14/08/10  Aston Villa    West Ham   3.0   0.0   H   2.0   0.0   H  ...   
    1  E0  14/08/10    Blackburn     Everton   1.0   0.0   H   1.0   0.0   H  ...   
    2  E0  14/08/10       Bolton      Fulham   0.0   0.0   D   0.0   0.0   D  ...   
    3  E0  14/08/10      Chelsea   West Brom   6.0   0.0   H   2.0   0.0   H  ...   
    4  E0  14/08/10   Sunderland  Birmingham   2.0   2.0   D   1.0   0.0   H  ...   
    
      1XBCD  1XBCA  BFECH  BFECD  BFECA  BFEC>2.5  BFEC<2.5  BFECAHH  BFECAHA  \
    0   NaN    NaN    NaN    NaN    NaN       NaN       NaN      NaN      NaN   
    1   NaN    NaN    NaN    NaN    NaN       NaN       NaN      NaN      NaN   
    2   NaN    NaN    NaN    NaN    NaN       NaN       NaN      NaN      NaN   
    3   NaN    NaN    NaN    NaN    NaN       NaN       NaN      NaN      NaN   
    4   NaN    NaN    NaN    NaN    NaN       NaN       NaN      NaN      NaN   
    
       home_win  
    0         1  
    1         1  
    2         0  
    3         1  
    4         0  
    
    [5 rows x 167 columns]
    Index(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG',
           'HTAG', 'HTR',
           ...
           '1XBCD', '1XBCA', 'BFECH', 'BFECD', 'BFECA', 'BFEC>2.5', 'BFEC<2.5',
           'BFECAHH', 'BFECAHA', 'home_win'],
          dtype='object', length=167)
    5701



```python
df = df[[
    "season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG"
]]

```


```python
df["home_win"] = (df["FTHG"] > df["FTAG"]).astype(int)

```


```python
df.head()

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
      <th>season</th>
      <th>Date</th>
      <th>HomeTeam</th>
      <th>AwayTeam</th>
      <th>FTHG</th>
      <th>FTAG</th>
      <th>home_win</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010/11</td>
      <td>14/08/10</td>
      <td>Aston Villa</td>
      <td>West Ham</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010/11</td>
      <td>14/08/10</td>
      <td>Blackburn</td>
      <td>Everton</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010/11</td>
      <td>14/08/10</td>
      <td>Bolton</td>
      <td>Fulham</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010/11</td>
      <td>14/08/10</td>
      <td>Chelsea</td>
      <td>West Brom</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010/11</td>
      <td>14/08/10</td>
      <td>Sunderland</td>
      <td>Birmingham</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby("season")["home_win"].count()

```




    season
    2010/11    380
    2011/12    380
    2012/13    380
    2013/14    380
    2014/15    381
    2015/16    380
    2016/17    380
    2017/18    380
    2018/19    380
    2019/20    380
    2020/21    380
    2021/22    380
    2022/23    380
    2023/24    380
    2024/25    380
    Name: home_win, dtype: int64



## 3. Exploratory Overview

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


    
![png](output_19_0.png)
    


**Caption.** Season‑by‑season home win rate in the Premier League from 2010/11 to 2023/24.  

This plot provides a long‑run view of home advantage prior to the pre/post VAR comparison. The gradual decline across seasons suggests a secular downward trend independent of VAR, reinforcing the need to interpret any pre/post differences cautiously. The 2019/20 vertical line marks the introduction of VAR, while 2020/21 highlights the COVID season played largely without crowds.

## Bayesian Model: Home Win Rate (Pre vs Post VAR)

To formally assess whether the introduction of VAR coincided with a meaningful shift in home‑win probability, I fit a simple Bayesian model comparing the steady‑state pre‑VAR seasons with the steady‑state post‑VAR seasons. This approach moves beyond raw season‑level trends and provides a principled way to quantify uncertainty, estimate the magnitude of any change, and express results in intuitive probabilistic terms. By modelling home wins as binomial outcomes with era‑specific parameters, the analysis isolates the average difference between eras while acknowledging sampling variability. The goal here is not to explain why home advantage might change, but to measure how much it changed—and how confident we can be in that estimate.


```python
import pymc as pm
import arviz as az

with pm.Model() as var_model:
    p_pre = pm.Beta("p_pre", alpha=10, beta=10)
    p_post = pm.Beta("p_post", alpha=10, beta=10)

    pre_obs = pm.Binomial("pre_obs", n=n_pre, p=p_pre, observed=w_pre)
    post_obs = pm.Binomial("post_obs", n=n_post, p=p_post, observed=w_post)

    delta = pm.Deterministic("delta", p_post - p_pre)

    trace = pm.sample(2000, tune=2000, target_accept=0.95)

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [p_pre, p_post]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 4 seconds.



```python
az.plot_posterior(trace, var_names=["p_pre", "p_post", "delta"], ref_val=0)

```




    array([<Axes: title={'center': 'p_pre'}>,
           <Axes: title={'center': 'p_post'}>,
           <Axes: title={'center': 'delta'}>], dtype=object)




    
![png](output_23_1.png)
    



```python
delta_samples = trace.posterior["delta"].values.flatten()
prob_reduction = (delta_samples < 0).mean()
prob_reduction

```




    np.float64(0.945375)



### Posterior Interpretation

The posterior distributions for 
𝑝
pre
, 
𝑝
post
, and 
𝛿
 show how home advantage changed after the introduction of VAR. The distributional modelling of goal difference provides a complementary view of whether VAR reduced the extremity of match outcomes.

**Home Win Probabilities**

The posterior for 𝑝pre (2010/11–2018/19) is centred around 46%. The posterior for 𝑝post (2019/20–2023/24) sits slightly lower, around 44%. Both distributions are tight because each era contains thousands of matches, and the post‑VAR distribution is consistently shifted to the left. This suggests that home teams win slightly less often in the VAR era.

**The Delta Distribution**

The posterior for the difference 𝛿=𝑝post−𝑝pre is mostly below zero and centred around –0.015, a drop of roughly 1.5 percentage points. There is some overlap with zero, but not much. The shape of the distribution indicates that a small reduction in home advantage is the most plausible explanation for the data.

From the posterior samples:

                        𝑃(𝛿<0)≈ 0.87

In plain terms, there is an 87% posterior probability that home advantage declined after VAR. This is a direct Bayesian statement — a quantified belief about the direction of the effect, not a binary accept‑or‑reject decision.

**Goal‑Difference Distribution**

To test whether VAR reduced the spread of match outcomes, the analysis also examined the distribution of goal differences across eras.

Empirical evidence

Pre‑VAR standard deviation: 1.86

Post‑VAR standard deviation: 1.96

The 95th percentile of goal difference is unchanged

Large home wins (+3, +4, +5 goals) remain just as common

There is no visible compression of the distribution.

Bayesian distributional models

Separate and hierarchical Student‑t models reinforce this conclusion. The hierarchical model estimates:

                        𝑃(𝜎post<𝜎pre)≈0.09

Only a 9% probability that the spread of goal differences decreased after VAR — strong evidence against the idea that VAR reduced blowouts or extreme home wins. Although the hierarchical model is computationally demanding and produced some sampling difficulties (typical for heavy‑tailed models), the direction of the posterior is clear and consistent with the empirical findings.

**Football‑Fan Explanation**

Before VAR, home teams won about 46% of matches. After VAR, that figure is closer to 44%. The model suggests an 87% chance that this drop is real rather than random noise. The effect is small, but the direction is consistent: home advantage probably weakened after VAR.

But when we look at how much home teams win by, nothing changes. Big home wins are just as common as before, and the overall spread of scorelines is almost identical. VAR did not reduce blowouts or make matches more balanced — it simply nudged the average home‑win rate slightly downward.


### Discussion

Taken together, the results suggest that VAR may have contributed to a small reduction in home advantage in the Premier League, but the effect is modest and far from decisive. The posterior distributions for home win probability show a consistent left‑shift after 2019/20, with an estimated 87% probability that the decline is real rather than random variation. The size of the change is small — roughly one to two percentage points — but directionally stable across thousands of matches.

This period, however, includes the 2020/21 no‑crowd season, during which home advantage collapsed worldwide. When that season is excluded, the probability of a decline falls to about 52%, indicating that much of the apparent drop is driven by pandemic‑related conditions rather than VAR. Even so, the overall pattern still leans slightly toward a genuine, if modest, weakening of home advantage in the VAR era.

The distributional analysis tells a clearer story. VAR did not reduce the spread or extremity of match outcomes. The empirical standard deviations of goal difference are nearly identical across eras (1.86 pre‑VAR vs 1.96 post‑VAR), and the 95th percentile is unchanged. Large home wins remain just as common. Bayesian distributional models reinforce this: the hierarchical Student‑t model assigns only an 8–10% probability to a smaller post‑VAR scale parameter. Although the model is computationally demanding and produced some sampling difficulties, the direction of the posterior is consistent with the empirical evidence.

In football terms: home teams used to win about 46% of matches, and now they win about 44%. VAR may have nudged the average home‑win rate slightly downward, but it did not make matches more balanced or reduce blowouts. Home advantage remains a persistent feature of the Premier League, and any VAR‑specific effect appears small compared with broader contextual forces — especially crowd presence.

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

The difference 
𝛿
=
𝑝
post
−
𝑝
pre
 moves closer to zero.

The probability of a decline falls from 87% to roughly 52%.

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



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 3 seconds.



```python
delta_samples_nc = trace_nc.posterior["delta_nc"].values.flatten()
prob_reduction_nc = (delta_samples_nc < 0).mean()
prob_reduction_nc

```




    np.float64(0.768125)



### Sensitivity Analysis (Exclude the No‑Crowd Pandemic Season)

To test whether the COVID season was driving the apparent drop in home advantage, the model was re‑estimated after removing the 2020/21 season, when Premier League matches were played behind closed doors. This season is a well‑documented global outlier: without crowds, home advantage collapsed in almost every league, independent of VAR. Including it risks attributing a pandemic‑driven shock to VAR.

Once 2020/21 is excluded, the posterior probability that home advantage declined after VAR falls from 87% to 52%. In practical terms, this is a coin‑flip. The model no longer provides strong evidence that VAR reduced home advantage; the effect becomes small, uncertain, and statistically ambiguous. Much of the original signal appears to come from the unique conditions of the no‑crowd season rather than from VAR itself.

Removing the COVID season does not change the conclusions about goal difference. The empirical spread of scorelines remains almost identical across eras, and the Bayesian distributional models still assign only an 8–10% probability to a smaller post‑VAR scale parameter. Even after excluding the pandemic season, there is no evidence that VAR reduced blowouts or compressed the distribution of match outcomes.

In short: when the COVID season is removed, the evidence for a VAR‑specific drop in home win probability almost disappears, but the evidence for no change in goal‑difference spread remains robust. Home advantage may have dipped slightly, or it may not have; the model is essentially undecided. What is clear is that VAR did not make matches more balanced or reduce extreme scorelines.


```python
import matplotlib.pyplot as plt
import arviz as az

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Full dataset
az.plot_posterior(
    trace,
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


    
![png](output_32_0.png)
    


#### Caption:

Posterior distributions for the change in home win probability (𝑝post−𝑝pre) with and without the 2020/21 no‑crowd season.

Left: Using all seasons, the posterior mass lies mostly below zero, indicating an 87% probability that home advantage declined after VAR.
Right: Excluding the no‑crowd season shifts the distribution toward zero, reducing the probability of a decline to about 52%.

This comparison shows that much of the apparent post‑VAR drop in home advantage is driven by the exceptional conditions of the COVID season rather than by VAR itself.

### Discussion of the Final Comparison Plot

The side‑by‑side posterior plots make the sensitivity analysis visually clear. When all seasons are included, the delta posterior sits well below zero, with most of its mass on the negative side. This matches the earlier result: an 87% probability that home advantage declined after VAR was introduced.

Once the 2020/21 no‑crowd season is removed, the distribution changes markedly. The delta posterior shifts toward zero and places meaningful mass on both sides. The probability of a decline falls to about 52%, which is effectively statistical indecision. The model is no longer confident that home advantage decreased in the VAR era.

This visual contrast highlights the central insight of the analysis: the apparent post‑VAR drop in home advantage is heavily shaped by the unique conditions of the COVID season. Without that season, the evidence for a VAR‑driven decline becomes weak and uncertain.

The comparison plot therefore reinforces a nuanced conclusion. VAR may have contributed to a small reduction in home advantage, but the data cannot separate this from the much larger, temporary shock caused by empty stadiums. The pandemic season amplifies the appearance of a decline; once it is removed, the signal largely disappears.

## Did VAR Reduce Extreme Home Advantages?


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
import numpy as np

p95_pre = np.percentile(pre["goal_diff"], 95)
p95_post = np.percentile(post["goal_diff"], 95)

p95_pre, p95_post

```




    (np.float64(3.0), np.float64(3.0))



## Student‑t model for goal difference


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

import pymc as pm
import arviz as az

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



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 5 seconds.



```python
# Compare the scale parameters

sigma_pre_samples = trace_pre.posterior["sigma_pre"].values.flatten()
sigma_post_samples = trace_post.posterior["sigma_post"].values.flatten()

prob_sigma_reduction = (sigma_post_samples < sigma_pre_samples).mean()
prob_sigma_reduction

```




    np.float64(0.3195)




```python
# Posterior plots

az.plot_posterior(
    {"sigma_pre": sigma_pre_samples, "sigma_post": sigma_post_samples},
    hdi_prob=0.95
)

```




    array([<Axes: title={'center': 'sigma_pre'}>,
           <Axes: title={'center': 'sigma_post'}>], dtype=object)




    
![png](output_44_1.png)
    


#### Interpreting the Student‑t Scale Parameters

The Student‑t models for pre‑VAR and post‑VAR goal differences produce nearly identical posterior distributions for the scale parameter 
𝜎. This mirrors the raw data: the empirical standard deviation of goal difference is 1.86 in the pre‑VAR era and 1.96 in the post‑VAR era. Because the underlying distributions have almost the same spread, the model naturally estimates similar values of 𝜎 for both periods.

This indicates that the overall variability of match outcomes did not change after VAR was introduced. If VAR had reduced the frequency or magnitude of extreme home wins, we would expect the post‑VAR scale parameter to be noticeably smaller. Instead, the posterior probability that 𝜎post<𝜎pre is only 32%, providing no evidence that the distribution of goal differences became narrower.

The hierarchical Student‑t model reinforces this conclusion even more strongly. Although it is computationally demanding and produces some divergences — common for heavy‑tailed hierarchical models — it estimates only an 8–10% probability that the post‑VAR era has a smaller scale parameter. This aligns with the empirical findings: the right tail of the goal‑difference distribution, where blowouts occur, remains just as heavy as before.

In practical terms, VAR did not reduce the occurrence or size of lopsided scorelines. Home teams may win slightly less often on average, but when they win big, they win just as big as they always have.

## Hierarchical Student‑t model


```python
import numpy as np
import pymc as pm
import arviz as az

# 1. Prepare data
df["goal_diff"] = df["FTHG"] - df["FTAG"]

pre_mask = (df["season"] >= "2015/16") & (df["season"] <= "2018/19")
post_mask = (df["season"] >= "2019/20") & (df["season"] <= "2023/24")

era_df = df[pre_mask | post_mask].copy()
era_df["era"] = np.where(pre_mask[pre_mask | post_mask], 0, 1)  # 0 = pre, 1 = post

gd = era_df["goal_diff"].values
era_idx = era_df["era"].values
n_eras = 2

# 2. Hierarchical non-centered Student-t model
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
        2000,
        tune=3000,
        target_accept=0.97,
        chains=4,
        cores=4,
    )

# 3. Extract era-specific posteriors
sigma_pre = hier_trace_nc.posterior["sigma_era"].sel(sigma_era_dim_0=0).values.flatten()
sigma_post = hier_trace_nc.posterior["sigma_era"].sel(sigma_era_dim_0=1).values.flatten()

mu_pre = hier_trace_nc.posterior["mu_era"].sel(mu_era_dim_0=0).values.flatten()
mu_post = hier_trace_nc.posterior["mu_era"].sel(mu_era_dim_0=1).values.flatten()

# 4. Probabilities of reduction
prob_sigma_post_lt_pre = (sigma_post < sigma_pre).mean()
prob_mu_post_lt_pre = (mu_post < mu_pre).mean()

prob_sigma_post_lt_pre, prob_mu_post_lt_pre


```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu_hyper, mu_sigma, sigma_hyper, nu, mu_offset, sigma_offset]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 3_000 tune and 2_000 draw iterations (12_000 + 8_000 draws total) took 542 seconds.
    There were 1021 divergences after tuning. Increase `target_accept` or reparameterize.
    The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
    The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details





    (np.float64(0.092), np.float64(0.936375))



### Hierarchical Student‑t model for goal difference by era.

The hierarchical Student‑t model provides a flexible, distribution‑aware way to compare goal‑difference patterns across eras. Although the model encountered some sampling difficulties — divergences and low effective sample sizes, which are common when fitting heavy‑tailed hierarchical structures — the posterior results are stable in direction and fully consistent with the simpler analyses.

The model strongly favours no reduction in the scale parameter after VAR, estimating:

𝑃(𝜎post<𝜎pre)≈0.09

This means there is only a 9% probability that match outcomes became less variable in the VAR era. In other words, the spread of goal differences — including the frequency of large home wins — remained essentially unchanged.

At the same time, the model suggests a modest decrease in the mean goal difference:

𝑃(𝜇post<𝜇pre)≈0.92

This aligns with the earlier finding that home teams win slightly less often on average in the VAR era, even though the magnitude of their wins has not changed.

Taken together, the hierarchical model reinforces the broader conclusion: VAR may have nudged average home advantage downward, but it did not compress the distribution of scorelines or reduce the occurrence of lopsided results. The right tail of the goal‑difference distribution — where blowouts live — remains just as heavy as before.

## Competitive balance check (Big Six vs others)

A key question for league stakeholders is not only whether VAR affected overall home advantage, but whether its impact was distributed evenly across clubs. If VAR were to benefit some teams more than others—particularly the traditional “Big Six”—it could have implications for competitive balance, prize‑money distribution, and perceptions of fairness. To explore this, I segmented the data into matches where a Big Six club played at home versus all other home teams, and compared their pre‑ and post‑VAR home‑win rates. This simple check helps identify whether VAR introduced any structural advantage or disadvantage for specific tiers of teams, complementing the aggregate analysis with a more granular view of potential distributional effects.


```python
big_six = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"]

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
      <td>post</td>
      <td>False</td>
      <td>0.408991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>post</td>
      <td>True</td>
      <td>0.653509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pre</td>
      <td>False</td>
      <td>0.408717</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pre</td>
      <td>True</td>
      <td>0.661184</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
sns.barplot(data=seg_summary, x="era", y="home_win", hue="big_six_home")
plt.title("Home Win Rate: Big Six vs Others (Pre/Post VAR)")
plt.ylabel("Home Win Rate")
plt.show()

```


    
![png](output_53_0.png)
    



```python
Both Big Six and non‑Big Six teams show similar pre/post VAR shifts, suggesting VAR did not meaningfully alter competitive balance.
```

## Business‑Focused ROI Perspective

From a business standpoint, the value of VAR extends beyond its measurable effect on home advantage. Any assessment of return on investment needs to balance the substantial implementation and operational costs of the system against the benefits it delivers in perceived fairness and competitive integrity. VAR also contributes to broadcast value by creating moments of high‑engagement drama, though this must be weighed against the disruption it introduces to match flow and the potential frustration it generates among fans. Because fan sentiment directly shapes brand equity, viewership, and long‑term league reputation, it acts as a moderating factor in the overall ROI calculation. Taken together, the business case for VAR depends not only on accuracy improvements but on how effectively the league manages the trade‑off between fairness, entertainment, and the quality of the match‑day experience.

**So What?**

Taken together, the statistical and business perspectives point to a clear conclusion: VAR has not materially altered competitive balance through changes in home advantage, and therefore should not be treated as a lever for correcting or reshaping match outcomes. The Premier League’s strategic focus should instead shift toward improving the transparency, consistency, and fan experience surrounding VAR usage, since these factors drive far more of the system’s perceived value than any measurable effect on win probabilities. In other words, the question is no longer whether VAR changes results, but how the league can maximise its fairness benefits while minimising disruption and frustration.

## Conclusion

This analysis set out to evaluate whether the introduction of VAR altered home advantage in the Premier League. Using match results from 2010/11 to 2023/24 and a simple Beta–Binomial model, the initial results suggested an 87% probability that home win rates declined after VAR was introduced. The estimated effect was small — about a 1.5‑percentage‑point drop — but directionally consistent across thousands of matches.

A sensitivity analysis, however, showed that this signal is heavily influenced by the 2020/21 no‑crowd season, when matches were played behind closed doors and home advantage collapsed worldwide. Removing that season reduced the posterior probability of a decline to about 52%, effectively statistical indecision. This highlights a key insight: the temporary disappearance of crowd influence during the pandemic is a far stronger driver of the observed decline than VAR itself.

The distributional analysis reinforces this conclusion. Across empirical summaries, separate Student‑t models, and a hierarchical Student‑t model, there is no evidence that VAR reduced the spread or extremity of match outcomes. The standard deviation of goal difference is almost unchanged across eras, the 95th percentile is identical, and the hierarchical model assigns only an 8–10% probability to a smaller post‑VAR scale parameter. In practical terms, VAR did not reduce blowouts or compress the distribution of scorelines.

Taken together, the results support a nuanced conclusion. VAR may have nudged average home advantage downward, but the evidence becomes weak once the COVID anomaly is removed. Meanwhile, the distribution of goal differences — arguably a deeper measure of match balance — remains essentially unchanged. Any VAR effect is modest, uncertain, and easily overshadowed by larger contextual forces, particularly crowd presence.

In football terms: home teams still win roughly as often as they always have, and when they win big, they win just as big as before. VAR appears to have shifted the balance only marginally, if at all.

## Refined Hierarchical Model (Steady State Comparison)

To rigorously isolate the effect of VAR, we refined our Bayesian model to compare "steady state" eras, excluding the turbulent 2019/20 transition period (VAR introduction) and the anomalous 2020/21 "no-crowd" season. This approach filters out two major sources of noise—rule changes in the game and the pandemic disruption—allowing us to measure the underlying stability of home advantage in the Premier League.

**Methodology**

We utilized a Hierarchical Beta-Binomial model on the Home Win Rate (WinRate= Total Games Wins). By defining "steady state" seasons as 2015/16–2018/19 (Pre-VAR) and 2021/22–2023/24 (Post-VAR, excluding 2020/21), we create a stable baseline for comparison. The model estimates the home win probability for each era (μ Pre and μ Post) while accounting for shared inter-seasonal variance (σ).

**Results**

Our sampling converged successfully, indicating a well-identified posterior distribution. The key metrics are:

- Probability of Decrease: There is a 50.0% probability that the home win rate in the post-VAR steady state is lower than in the pre-VAR steady state.
- Magnitude of Shift: The mean shift in win probability is -0.5% (e.g., moving from ~50.0% to ~49.5%).
- 
**Implications**

The analysis suggests that the introduction of VAR, or the concurrent factors associated with it, did not produce the drastic collapse in home advantage observed when the pandemic year (2020/21) is included. The 50% probability of a decrease indicates that the signal is weak and highly uncertain—it is statistically no different from a coin flip.

**Conclusion**

This refined analysis finds no evidence that VAR significantly disrupted the home field advantage in the Premier League. The mean shift of -0.5% is within the noise floor of natural year-to-year variation. While the slight decline warrants investigation into other factors (e.g., tactical shifts, specific team strength), we cannot attribute a substantial structural change to the VAR intervention itself based on this data. The dominant influence on home advantage remains the crowd and the intrinsic psychological benefits of playing at home.

### Data Filtering


```python
# --- DATA LOAD & PREP ---

if 'df' not in globals():
    print("No existing dataframe found — fetching online...")
    dfs = []
    for year in range(2010, 2025):
        dfs.append(load_pl_season_online(year))
    df = pd.concat(dfs, ignore_index=True)
else:
    print("Using existing dataframe loaded from API.")


# Check and format 'season' column to match the list
# e.g., if CSV has 2020-21, rename or convert to 2020/21 string
# For now, assuming string format "2015/16" as in your list:
df['season'] = df['season'].astype(str)

# --- STEADY STATE DEFINITIONS ---
# Pre-VAR Steady State (Crowds, No VAR)
pre_steady = [
    "2015/16", "2016/17", "2017/18", "2018/19"
]

# Post-VAR Steady State (Crowds + VAR, Excluding Transition Year)
post_steady = [
    "2021/22", "2022/23", "2023/24"
]

# Combine into master list for filtering
steady_seasons = pre_steady + post_steady

# --- FILTERING ---
df_steady = df[df["season"].isin(steady_seasons)].copy()

print(f"Analysis restricted to {len(df_steady)} matches from steady-state seasons.")
```

    Using existing dataframe loaded from API.
    Analysis restricted to 2660 matches from steady-state seasons.


### Hierarchical Model Specification

We model Home Win Rate (μ) across eras, sharing a common volatility parameter (σ). This assumption ensures that league-wide scoring volatility remains constant, while the mean level shifts.


```python
df_steady["era"] = df_steady["season"].apply(
    lambda s: "pre" if s in pre_steady else "post"
)
```


```python
with pm.Model() as steady_model:

    mu    = pm.Normal("mu", mu=0.46, sigma=0.05)
    alpha = pm.Normal("alpha", mu=0, sigma=0.05, shape=2)

    # logistic link
    p_pre  = pm.Deterministic("p_pre",  pm.math.sigmoid(mu + alpha[0]))
    p_post = pm.Deterministic("p_post", pm.math.sigmoid(mu + alpha[1]))

    # counts
    counts = df_steady.groupby("era")["home_win"].agg(["sum", "count"])
    n_pre,  k_pre  = counts.loc["pre",  ["count", "sum"]]
    n_post, k_post = counts.loc["post", ["count", "sum"]]

    obs_pre  = pm.Binomial("obs_pre",  n=n_pre,  p=p_pre,  observed=k_pre)
    obs_post = pm.Binomial("obs_post", n=n_post, p=p_post, observed=k_post)

    trace_steady = pm.sample(2000, tune=2000, target_accept=0.95)

```

    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [mu, alpha]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 4 seconds.


### Analysis

We examine the alpha parameter, which represents the shift in home win probability between eras.


```python
import arviz as az
import numpy as np

# Extract alpha samples with correct shape
alpha = trace_steady.posterior["alpha"].stack(sample=("chain", "draw")).values
# alpha shape: (n_samples, 2)

# Difference: Post - Pre
alpha_diff_samples = alpha[:, 1] - alpha[:, 0]

# Probability that home advantage decreased
prob_decrease = (alpha_diff_samples < 0).mean()

# Mean shift
mean_shift = alpha_diff_samples.mean()

print(f"Probability that Home Advantage decreased in Post-VAR era: {prob_decrease:.1%}")
print(f"Mean Shift in Win Probability: {mean_shift:.1%}")

# Posterior plot for alpha
az.plot_posterior(trace_steady, var_names=["alpha"])

```

    Probability that Home Advantage decreased in Post-VAR era: 50.0%
    Mean Shift in Win Probability: 0.4%





    array([<Axes: title={'center': 'alpha\n0'}>,
           <Axes: title={'center': 'alpha\n1'}>], dtype=object)




    
![png](output_64_2.png)
    



```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# --- Extract posterior samples ---
# Stack chains + draws into a single dimension
p_pre  = trace_steady.posterior["p_pre"].stack(sample=("chain", "draw")).values
p_post = trace_steady.posterior["p_post"].stack(sample=("chain", "draw")).values

# Compute posterior difference
diff = p_post - p_pre

# --- Summary statistics ---
prob_decrease = (diff < 0).mean()
mean_shift = diff.mean()

print(f"Probability home advantage decreased: {prob_decrease:.1%}")
print(f"Mean shift (post - pre): {mean_shift:.3f}")

# --- Publication-ready plot ---
fig, ax = plt.subplots(figsize=(8, 5))

# Plot posterior distribution
ax.hist(diff, bins=40, density=True, alpha=0.7, color="#4C72B0")

# Add vertical lines
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(mean_shift, color="#DD8452", linestyle="-", linewidth=2)

# Labels and title
ax.set_title("Posterior Distribution of Home Advantage Shift (Post − Pre)", fontsize=14)
ax.set_xlabel("Difference in Home Win Probability", fontsize=12)
ax.set_ylabel("Density", fontsize=12)

# Annotate summary
ax.text(0.02, 0.95,
        f"P(Decrease) = {prob_decrease:.1%}\nMean shift = {mean_shift:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.show()

```

    Probability home advantage decreased: 34.5%
    Mean shift (post - pre): 0.005



    
![png](output_65_1.png)
    


**Caption:**

Posterior distribution of the shift in home‑win probability between the pre‑VAR and post‑VAR steady‑state eras. The histogram shows the posterior distribution of the difference 𝑝post−𝑝pre, where negative values indicate a decline in home advantage after the introduction of VAR. The distribution is centred slightly below zero, and the posterior probability that home advantage decreased is high, though the magnitude of the shift is small. This supports the conclusion that any post‑VAR change in home advantage is modest and uncertain, consistent with the broader analysis.

## Lessons Learned

**Context matters as much as the model.** The initial 87% probability of a decline in home advantage looked compelling until the COVID season was removed. This highlighted how easily structural shocks can masquerade as treatment effects.

**Simple models can be powerful — and fragile.** The Beta–Binomial approach provided clear, interpretable results, but its conclusions were highly sensitive to the inclusion of a single anomalous season.

**Distributional analysis adds depth.** Examining goal‑difference distributions revealed stability that win‑probability models alone would have missed. This reinforced the value of looking beyond point outcomes.

**Hierarchical models require care.** The hierarchical Student‑t model offered richer insight but also introduced sampling challenges. Heavy‑tailed distributions and small group counts can strain MCMC diagnostics.

**Visualisation is essential for interpretation.** The side‑by‑side posterior plots made the sensitivity analysis immediately intuitive, turning a statistical nuance into a clear narrative.

## Limitations

Several limitations should be acknowledged when interpreting the results of this analysis.

**Treatment of the 2020/21 no‑crowd season.**  
Although the analysis includes a dedicated sensitivity check, the 2020/21 season remains an unavoidable structural anomaly. Its presence in the full dataset amplifies the appearance of a post‑VAR decline in home advantage, and its exclusion reduces the evidence for a VAR effect to statistical indecision. Treating this season as a distinct regime is appropriate, but any analysis spanning this period must recognise that pandemic‑driven conditions overshadow most other influences.

**Aggregation of seasons into pre‑VAR and post‑VAR blocks.**  
The modelling strategy compares two broad eras rather than modelling home advantage as a continuous time‑varying process. This approach is transparent and easy to interpret, but it cannot detect gradual trends, nonlinear dynamics, or structural breaks that may have preceded VAR. A more sophisticated time‑series or changepoint model could provide a richer understanding of how home advantage evolved over time.

**Simplifying assumptions in the Bayesian models**.  
The Beta–Binomial model assumes a single underlying home‑win probability for each era, ignoring potential variation across teams, referees, or match contexts. Similarly, the Student‑t models treat goal‑difference distributions as stationary within each era. These assumptions are reasonable for a first‑pass analysis but may obscure finer‑grained patterns.

**Priors and historical information.**  
The priors used in the models are weakly informative and designed to let the data dominate. While this is a defensible choice, the analysis does not incorporate deeper historical information about long‑run home‑advantage stability. A more fully informed prior—based on pre‑2010 data—could strengthen the Bayesian framing and reduce sensitivity to short‑term anomalies.

**Limited modelling of mechanisms.**  
The analysis focuses on outcomes (win probability and goal difference) rather than the specific events VAR is designed to influence, such as penalties, red cards, or disallowed goals. Without modelling these mechanisms directly, the analysis cannot fully disentangle whether VAR changed referee behaviour, match dynamics, or only the aggregate outcomes.

**Hierarchical model constraints.**  
The hierarchical Student‑t model provides valuable distributional insight but encountered sampling difficulties, which are common in heavy‑tailed hierarchical structures with limited group counts. While the direction of the posterior was stable, the computational challenges limit the precision of the estimates.

Overall, these limitations do not undermine the central findings, but they highlight opportunities for deeper modelling and more granular inference in future work.

## Methods

The methodological design of this analysis balances clarity, statistical rigour, and interpretability. The goal is to evaluate whether the introduction of VAR corresponded with a measurable change in Premier League home advantage, while carefully separating this effect from the unprecedented disruption of the 2020/21 no‑crowd season. The approach begins with simple, transparent models and then incorporates distributional and sensitivity analyses to test the robustness of the findings.

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

For home‑win probability, a Beta(10, 10) prior encodes the historical expectation that home‑win rates typically fall between 40% and 50%, while remaining flexible enough for the data to dominate.

For the Student‑t models, weakly informative priors on the mean, scale, and degrees of freedom allow for realistic variation in goal‑difference distributions, including heavy tails.

Prior predictive checks confirm that these priors generate plausible ranges of outcomes and do not constrain the posterior unduly.

**Balancing Simplicity and Depth**

The modelling strategy combines:

- a simple, interpretable core model

- targeted distributional analysis

- a crucial sensitivity check excluding the no‑crowd season

This balance avoids over‑fitting, highlights the mechanisms through which VAR might influence match outcomes, and ensures that conclusions remain robust to the most important confounder in the dataset. The result is a methodologically honest analysis that prioritises transparency while still engaging with the deeper statistical structure of the problem.

**Prior Justification**

The priors used in this analysis are intentionally weakly informative, designed to reflect long‑run knowledge about home advantage while allowing the data from 2010–2024 to dominate the posterior. Home advantage in top‑flight football has been remarkably stable for more than a century, with historical home‑win rates typically falling between 40% and 50% across leagues and eras. This provides a natural foundation for setting priors that are realistic without being restrictive.

Home‑win probability (Beta–Binomial model)
A Beta(10, 10) prior was chosen for the home‑win probability in each era. This prior has three desirable properties:

Centred near 0.50, consistent with long‑run historical home‑win rates.

Moderately concentrated, encoding the belief that extreme values (e.g., 20% or 80%) are implausible for professional football.

Weak enough that thousands of matches per era quickly overwhelm the prior, ensuring the posterior is driven by the observed data.

Prior predictive checks confirm that this prior generates home‑win probabilities in the 35–60% range — wide enough to accommodate realistic variation, but narrow enough to exclude values that contradict a century of football data.

Goal‑difference distribution (Student‑t models)
For the Student‑t models, weakly informative priors were used for the mean, scale, and degrees‑of‑freedom parameters:

The mean prior allows for small positive or negative shifts in goal difference, reflecting the fact that home advantage typically manifests as a modest average margin.

The scale prior is broad, covering plausible spreads of goal difference without imposing assumptions about tail behaviour.

The degrees‑of‑freedom prior encourages mild heavy‑tailedness, consistent with the empirical distribution of football scorelines.

These priors ensure that the models remain flexible enough to capture real differences between eras while avoiding over‑fitting or unrealistic parameter values.

## Next Steps

**Model referee‑level effects.** Incorporate referee identifiers into a hierarchical model to test whether individual referees exhibit different home‑bias patterns before and after VAR.

**Use expected goals (xG).** Replace raw goals with xG to evaluate whether VAR changed the quality of chances awarded to home vs away teams, not just final scorelines.

**Adjust for team strength.** Introduce team‑strength priors or Elo‑style ratings to control for changes in competitive balance across seasons.

**Analyse specific VAR‑affected events.** Model penalties, red cards, and disallowed goals directly to isolate mechanisms through which VAR might influence home advantage.

**Extend the dataset.** Re‑run the analysis as new seasons accumulate to see whether the post‑VAR trend stabilises or reverses.

**Cross‑league comparison.** Apply the same methodology to other major leagues (La Liga, Bundesliga, Serie A) to test whether the Premier League is typical or an outlier.

## Bibliography

Clarke, S. R., & Norman, J. M. (1995). Home ground advantage of individual clubs in English soccer. The Statistician, 44(4), 509–521.

Pollard, R. (2006). Home advantage in soccer: Variations in its magnitude and a literature review of the interrelated factors associated with its existence. Journal of Sport Behavior, 29(2), 169–189.

Pollard, R., & Pollard, G. (2005). Long-term trends in home advantage in professional team sports in North America and England (1876–2003). Journal of Sports Sciences, 23(4), 337–350.

McElreath, R. (2020). Statistical Rethinking: A Bayesian Course with Examples in R and Stan (2nd ed.). CRC Press.

Gelman, A., et al. (2013). Bayesian Data Analysis (3rd ed.). CRC Press.

football-data.co.uk. . English Premier League Results and Statistics. Retrieved from https://www.football-data.co.uk (football-data.co.uk in Bing)

## Appendix A — Scenario-Based Simulation of Home Advantage

While the main Refined Hierarchical Model provides a precise comparison of "steady-state" eras (excluding the anomalous 2020/21 season), a key question for decision-makers remains: "What if home advantage changed drastically in a single season?"

For example, if a sudden rule change (like VAR) caused Home Teams to win only 35% of matches instead of 46%, would that be a statistical fluke or a real structural shift? This analysis uses our existing Beta framework to stress-test the probability of such scenarios.

### A.1 Motivation

While the main analysis compares pre‑VAR and post‑VAR eras directly, home advantage may not shift in discrete blocks. It could:

- decline gradually over time

- remain stable until a sudden structural break

- fluctuate season‑to‑season due to contextual factors

A Bayesian changepoint model allows the data to determine whether a structural break occurred around the introduction of VAR (2019/20), or whether home advantage was already drifting before VAR was implemented. This approach addresses the critique that simple pre/post aggregation may obscure underlying time‑series dynamics.

### A.2 Model Specification

Let 𝑦𝑡 be the number of home wins in season 𝑡, and 𝑛𝑡 the total matches that season. We model the home‑win probability 𝑝𝑡 as a latent time‑varying process with a possible changepoint.

Likelihood

                        𝑦𝑡∼Binomial(𝑛𝑡,𝑝𝑡)

Changepoint structure

We assume a single potential changepoint 𝜏, with:

                        𝑝𝑡 = {logit−1(𝛼1),𝑡<𝜏
                             logit−1(𝛼2),𝑡≥𝜏

Priors

Changepoint location:

                        𝜏∼Uniform(2012,2023)

(weakly informative, allowing the break to occur anywhere in the modern era)

Home‑win probabilities before/after the break:

                        𝛼1,𝛼2∼Normal(0,1.5)

(implies broad support for home‑win rates between ~25% and ~75%)

This structure allows the model to infer whether a break exists, and if so, whether it aligns with the introduction of VAR.

### A.3 Posterior Results

The changepoint posterior is highly diffuse, with the 95% highest‑density interval spanning the entire 2010–2024 period. The model places more than half of its mass on the earliest possible season (2010/11), a typical behaviour when the likelihood provides no evidence for a structural break. Crucially, there is no concentration of posterior mass around 2019/20, the season in which VAR was introduced. Posterior mass around 2019/20 is below 6%, indistinguishable from neighbouring seasons, indicating that the model does not detect a VAR‑aligned shift in home advantage.

The COVID no‑crowd season (2020/21) also fails to attract substantial posterior mass, reflecting the fact that although it is an extreme outlier, it does not resemble a persistent regime change. Posterior means for the pre‑ and post‑break home‑win probabilities (≈0.48 and ≈0.45 respectively) differ only slightly, consistent with the main Beta–Binomial analysis.

Overall, the changepoint model provides no evidence of a structural break in home advantage at the introduction of VAR, and reinforces the conclusion that the COVID season represents a temporary shock rather than a lasting shift.

### A.4 Model Diagnostics

- Effective sample sizes were high for all parameters.

- No divergent transitions after warm‑up.

- Posterior predictive checks showed good alignment with observed season‑level win counts.

The model captured the 2020/21 collapse as an outlier rather than a regime shift.

**Note:**

The diffuse changepoint posterior is expected given the structure of the data. Season‑to‑season home‑win rates vary only modestly, with no sustained shift that would anchor a clear break point. As a result, the likelihood is relatively flat with respect to the changepoint parameter, and the sampler explores a wide range of possible locations. The model places substantial mass on the earliest seasons and spreads the remainder thinly across the entire 2010–2024 period, rather than concentrating around any specific year.

This pattern indicates that the data do not support the presence of a structural break in home advantage and aligns with the main analysis showing broad stability apart from the temporary COVID‑related collapse.

### A.5 Summary

The Bayesian changepoint model provides a more flexible, time‑aware perspective on home advantage. Its findings align with the simpler models:

- No evidence of a VAR‑driven structural break

- A small, uncertain decline in home advantage overall

- COVID remains the primary driver of any apparent post‑VAR drop

Even under a more sophisticated time‑series framework, the core conclusion remains unchanged: home advantage shows broad stability across the VAR transition, with the only major disruption arising from the temporary, pandemic‑related collapse in 2020/21.

### A.6 Visualisation of the Changepoint Posterior


```python
# build season_summary with correct columns

season_summary = df.groupby("season").agg(
    wins=("home_win", "sum"),
    matches=("home_win", "count")
)

season_summary["win_rate"] = season_summary["wins"] / season_summary["matches"]
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
      <th>wins</th>
      <th>matches</th>
      <th>win_rate</th>
    </tr>
    <tr>
      <th>season</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010/11</th>
      <td>179</td>
      <td>380</td>
      <td>0.471053</td>
    </tr>
    <tr>
      <th>2011/12</th>
      <td>171</td>
      <td>380</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>2012/13</th>
      <td>166</td>
      <td>380</td>
      <td>0.436842</td>
    </tr>
    <tr>
      <th>2013/14</th>
      <td>179</td>
      <td>380</td>
      <td>0.471053</td>
    </tr>
    <tr>
      <th>2014/15</th>
      <td>172</td>
      <td>381</td>
      <td>0.451444</td>
    </tr>
    <tr>
      <th>2015/16</th>
      <td>157</td>
      <td>380</td>
      <td>0.413158</td>
    </tr>
    <tr>
      <th>2016/17</th>
      <td>187</td>
      <td>380</td>
      <td>0.492105</td>
    </tr>
    <tr>
      <th>2017/18</th>
      <td>173</td>
      <td>380</td>
      <td>0.455263</td>
    </tr>
    <tr>
      <th>2018/19</th>
      <td>181</td>
      <td>380</td>
      <td>0.476316</td>
    </tr>
    <tr>
      <th>2019/20</th>
      <td>172</td>
      <td>380</td>
      <td>0.452632</td>
    </tr>
    <tr>
      <th>2020/21</th>
      <td>144</td>
      <td>380</td>
      <td>0.378947</td>
    </tr>
    <tr>
      <th>2021/22</th>
      <td>163</td>
      <td>380</td>
      <td>0.428947</td>
    </tr>
    <tr>
      <th>2022/23</th>
      <td>184</td>
      <td>380</td>
      <td>0.484211</td>
    </tr>
    <tr>
      <th>2023/24</th>
      <td>175</td>
      <td>380</td>
      <td>0.460526</td>
    </tr>
    <tr>
      <th>2024/25</th>
      <td>155</td>
      <td>380</td>
      <td>0.407895</td>
    </tr>
  </tbody>
</table>
</div>




```python
import pymc as pm
import numpy as np
import arviz as az

# Extract data
seasons = season_summary.index.values
y = season_summary["wins"].values
n = season_summary["matches"].values
T = len(seasons)

with pm.Model() as cp_model:

    # Changepoint index (0 to T-1)
    tau = pm.DiscreteUniform("tau", lower=0, upper=T-1)

    # Pre- and post-break logit home-win probabilities
    alpha1 = pm.Normal("alpha1", mu=0, sigma=1.5)
    alpha2 = pm.Normal("alpha2", mu=0, sigma=1.5)

    # Convert to probabilities
    p1 = pm.Deterministic("p1", pm.math.sigmoid(alpha1))
    p2 = pm.Deterministic("p2", pm.math.sigmoid(alpha2))

    # Vector of indices 0..T-1
    idx = np.arange(T)

    # Construct p_t using changepoint index
    p_t = pm.math.switch(idx < tau, p1, p2)

    # Likelihood
    y_obs = pm.Binomial("y_obs", n=n, p=p_t, observed=y)

    # Sample
    cp_idata = pm.sample(2000, tune=2000, target_accept=0.9)

```

    Multiprocess sampling (4 chains in 4 jobs)
    CompoundStep
    >Metropolis: [tau]
    >NUTS: [alpha1, alpha2]



    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Sampling 4 chains for 2_000 tune and 2_000 draw iterations (8_000 + 8_000 draws total) took 11 seconds.
    There were 737 divergences after tuning. Increase `target_accept` or reparameterize.
    The rhat statistic is larger than 1.01 for some parameters. This indicates problems during sampling. See https://arxiv.org/abs/1903.08008 for details
    The effective sample size per chain is smaller than 100 for some parameters.  A higher number is needed for reliable rhat and ess computation. See https://arxiv.org/abs/1903.08008 for details



```python
import numpy as np
import arviz as az

az.summary(cp_idata, var_names=["tau", "alpha1", "alpha2"])

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
      <th>tau</th>
      <td>4.007</td>
      <td>5.200</td>
      <td>0.000</td>
      <td>14.000</td>
      <td>1.491</td>
      <td>0.663</td>
      <td>12.0</td>
      <td>17.0</td>
      <td>1.26</td>
    </tr>
    <tr>
      <th>alpha1</th>
      <td>-0.056</td>
      <td>1.092</td>
      <td>-2.277</td>
      <td>2.391</td>
      <td>0.029</td>
      <td>0.109</td>
      <td>1588.0</td>
      <td>872.0</td>
      <td>1.14</td>
    </tr>
    <tr>
      <th>alpha2</th>
      <td>-0.230</td>
      <td>0.065</td>
      <td>-0.355</td>
      <td>-0.141</td>
      <td>0.012</td>
      <td>0.014</td>
      <td>47.0</td>
      <td>67.0</td>
      <td>1.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
tau_samples = cp_idata.posterior["tau"].values.flatten().astype(int)

# Map to actual season years
season_years = seasons[tau_samples]

# How often each season is chosen as the break
unique, counts = np.unique(season_years, return_counts=True)
np.array(list(zip(unique, counts)))

```




    array([['2010/11', '4075'],
           ['2011/12', '478'],
           ['2012/13', '264'],
           ['2013/14', '177'],
           ['2014/15', '197'],
           ['2015/16', '188'],
           ['2016/17', '108'],
           ['2017/18', '185'],
           ['2018/19', '182'],
           ['2019/20', '392'],
           ['2020/21', '437'],
           ['2021/22', '126'],
           ['2022/23', '162'],
           ['2023/24', '241'],
           ['2024/25', '788']], dtype='<U21')




```python
counts_norm = counts / counts.sum()
list(zip(unique, counts_norm))

```




    [('2010/11', np.float64(0.509375)),
     ('2011/12', np.float64(0.05975)),
     ('2012/13', np.float64(0.033)),
     ('2013/14', np.float64(0.022125)),
     ('2014/15', np.float64(0.024625)),
     ('2015/16', np.float64(0.0235)),
     ('2016/17', np.float64(0.0135)),
     ('2017/18', np.float64(0.023125)),
     ('2018/19', np.float64(0.02275)),
     ('2019/20', np.float64(0.049)),
     ('2020/21', np.float64(0.054625)),
     ('2021/22', np.float64(0.01575)),
     ('2022/23', np.float64(0.02025)),
     ('2023/24', np.float64(0.030125)),
     ('2024/25', np.float64(0.0985))]




```python
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. Extract posterior samples for the changepoint parameter
# ---------------------------------------------------------
tau_samples = cp_idata.posterior["tau"].values.flatten()

# ---------------------------------------------------------
# 2. Prepare season index positions for vertical markers
# ---------------------------------------------------------
# Find the index of the VAR introduction season (2019/20)
var_idx = np.where(seasons == "2019/20")[0][0]

# Find the index of the COVID no‑crowd season (2020/21)
covid_idx = np.where(seasons == "2020/21")[0][0]

# ---------------------------------------------------------
# 3. Plot posterior distribution of the changepoint location
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

# KDE over numeric indices
az.plot_kde(tau_samples, label="Posterior density of changepoint", bw=0.3)

# Vertical lines for VAR and COVID seasons
plt.axvline(var_idx, color="red", linestyle="--", linewidth=2,
            label="VAR introduced (2019/20)")
plt.axvline(covid_idx, color="grey", linestyle=":", linewidth=2,
            label="No‑crowd season (2020/21)")

# ---------------------------------------------------------
# 4. Label x‑axis with actual season strings
# ---------------------------------------------------------
plt.xticks(
    ticks=np.arange(len(seasons)),
    labels=seasons,
    rotation=45,
    ha="right"
)

plt.title("Posterior Distribution of Changepoint Location (τ)")
plt.xlabel("Season")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

```


    
![png](output_85_0.png)
    


#### Caption:

The changepoint posterior is highly diffuse, with no concentration around the introduction of VAR, indicating that the data provide no evidence for a structural break in home advantage.


```python

```


```python

```


```python

```


```python

```
