# VAR and Premier League Home Advantage: A Bayesian Analysis

This project asks a simple question: Did VAR change home advantage in the Premier League?  
Using match data from 2010/11 to 2023/24, I use Bayesian modelling, sensitivity checks, and segmentation to separate real effects from noise, confounders, and long‑run trends.

The goal is clear, honest analysis.

## Overview

Home advantage has been drifting downward for years. When VAR arrived in 2019/20, many people assumed it would reduce referee bias and cut home wins. This project tests that idea using a:

- Bayesian Beta–Binomial model comparing pre‑VAR and post‑VAR seasons

- sensitivity analysis that removes the 2020/21 no‑crowd season

- trend check to see whether home advantage was already falling

- Big Six vs. non‑Big Six segmentation

- short business and fan‑experience discussion

The aim is to understand what changed, what didn’t, and what actually matters.

## Key Findings

1. The “VAR effect” disappears once you remove the COVID season
A naïve model suggests an 87% chance that home advantage fell after VAR.
Remove the 2020/21 no‑crowd season and that drops to ~52% — essentially a coin flip.

Conclusion: the pandemic, not VAR, explains the sharp dip.

2. Home advantage has been declining for a decade
Season‑by‑season trends show a slow, steady fall long before VAR existed.
This warns against blaming technology for a long‑running shift.

3. No sign of competitive imbalance
Big Six clubs and everyone else show the same pattern.
VAR did not tilt the league toward or against elite teams.

4. The business story matters more than the statistical one
Since VAR doesn’t change results much, the Premier League’s priorities should be:

- clearer communication

- fewer delays

- better fan experience

- stronger broadcast value

The real question isn’t “Does VAR change outcomes?”  
It’s “How do we make VAR work better for the sport?”

## Repository Structure
.
├── var-home-advantage-bayesian.ipynb   # Main analysis notebook

├── figures/                            # Exported plots

│   └── (trend, posterior, segmentation visuals)

├── data/ (optional)                    # Cached CSVs

├── requirements.txt                    # Python dependencies

└── README.md                           # Project documentation

## Methods

**Bayesian Model**

A Beta–Binomial model estimates home‑win probabilities for:

- pre‑VAR steady‑state seasons

- post‑VAR steady‑state seasons

Posterior differences give the probability that home advantage fell.

**Sensitivity Analysis**

The 2020/21 season had empty stadiums. It behaves like an outlier and overwhelms the signal. Removing it shows how fragile the naïve conclusion is.

**Trend Check**

A simple season‑level trend line shows a long‑term decline.
This helps avoid attributing structural changes to VAR.

**Competitive Balance**

I compare home‑win rates for:

- Big Six clubs

- all other clubs

This checks whether VAR shifted fairness or league balance.

**Business Perspective**

VAR affects more than match outcomes.

I look at its broader return on investment:

- Costs: implementation, operations, stoppages

- Benefits: fairness, accuracy, broadcast drama

- Moderators: fan sentiment, league reputation

The evidence suggests VAR’s value depends on how it is run, not on any measurable change in home advantage.

**Conclusion**

VAR did not meaningfully change home advantage in the Premier League. The small, uncertain shift is driven by the pandemic and long‑run trends, not by the technology itself.
The league’s real opportunity lies in improving transparency, consistency, and the fan experience — not expecting VAR to reshape competitive balance.

How to Run

Install dependencies:

pip install -r requirements.txt

Then open the notebook:

jupyter lab

