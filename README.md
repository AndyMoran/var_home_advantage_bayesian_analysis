# VAR and Premier League Home Advantage: A Bayesian Analysis

This project investigates whether the introduction of Video Assistant Referees (VAR) in the Premier League meaningfully changed home advantage. Using match‑level data from 2010/11 to 2023/24, the analysis combines Bayesian modelling, sensitivity checks, segmentation, and business framing to provide a balanced, evidence‑driven assessment of VAR’s impact.

## Overview
Home advantage in football has been declining globally for years, and the introduction of VAR in 2019/20 raised questions about whether technology might further reduce referee bias toward home teams. This project evaluates that claim using:

- A Bayesian Beta–Binomial model comparing pre‑VAR and post‑VAR steady‑state eras

- A sensitivity analysis excluding the 2020/21 no‑crowd season

- A temporal trend diagnostic to check for long‑run secular decline

- A competitive balance segmentation (Big Six vs others)

- A business‑focused ROI and “So What?” section for strategic interpretation

The result is a clear, honest, and multi‑layered assessment of VAR’s effect on competitive balance.

## Key Findings

1. Apparent VAR effect is driven by the COVID no‑crowd season
A naïve Bayesian model suggests an 87% probability that home advantage declined after VAR was introduced.
However, this result collapses to ~52% (a coin toss) once the 2020/21 season is excluded.

This indicates that the pandemic, not VAR, explains most of the observed drop.

2. Long‑run trends show a gradual decline independent of VAR
Season‑by‑season home win rates reveal a slow downward drift beginning long before VAR.
This reinforces the need for caution when attributing changes to technology.

3. No evidence of competitive imbalance
Segmenting matches into Big Six vs non‑Big Six home teams shows similar pre/post shifts.
VAR did not disproportionately benefit or disadvantage elite clubs.

4. Business implications outweigh statistical effects
Since VAR does not meaningfully alter home advantage, the Premier League’s strategic focus should shift toward:

- improving transparency

- reducing disruption

- managing fan sentiment

- maximising broadcast value

The question is no longer “Does VAR change results?” but “How can VAR be implemented to improve fairness and experience?”

## Repository Structure

.
├── var-home-advantage-bayesian.ipynb   # Main analysis notebook
├── figures/                            # Exported plots
│   └── (trend, posterior, segmentation visuals)
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation

## Methods

**Bayesian Modelling**
A Beta–Binomial model estimates home‑win probabilities for:

- Pre‑VAR steady‑state seasons

- Post‑VAR steady‑state seasons

- Posterior differences quantify the probability that home advantage declined.

**Sensitivity Analysis**
The 2020/21 season is treated as a confounder due to empty stadiums.
Excluding it dramatically changes the posterior, revealing the fragility of naïve conclusions.

**Temporal Trend Check**
A season‑level trend plot provides context for secular decline and helps avoid misattribution.

**Competitive Balance Segmentation**
Home win rates are compared for:

- Big Six clubs

- All other clubs

This checks for distributional effects relevant to league fairness and revenue.

##Business Perspective

A structured ROI lens evaluates VAR beyond match outcomes:

Costs: implementation, operations, disruption to match flow

Benefits: fairness, accuracy, broadcast drama

Moderators: fan sentiment, league reputation

The analysis concludes that VAR’s value depends more on how it is implemented than on any measurable effect on home advantage.

##Conclusion

VAR did not meaningfully change Premier League home advantage.
The small, uncertain shift is dominated by pandemic‑related conditions and long‑run trends rather than technology.

The Premier League’s strategic priority should be optimising VAR’s transparency, consistency, and fan experience — not expecting it to reshape competitive balance.

How to Run

Install dependencies:

pip install -r requirements.txt

Then open the notebook:

jupyter lab

