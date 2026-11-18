# ☕ The Coffee Premium
### A microeconomic reading of the specialty coffee shelf

> **Research question:** Is the price of a coffee listing more closely associated with its quality score, bean type, roast, or origin?

![Two-panel EDA chart](outputs/coffee_premium_dashboard.png)

**Python · Pandas · Seaborn · Statsmodels · Descriptive statistics · Market differentiation**

## Why this is an economics project
Coffee is a differentiated good: buyers observe labels such as origin, roast, and bean type, but may not fully observe quality. This miniature study explores **price dispersion**, **quality signaling**, and **product segmentation** rather than stopping at visualizations.

## Important data disclosure
**All 240 listings are synthetic.** `analysis.py` creates them from a fixed random seed and explicitly coded pricing assumptions. They are **not scraped offers, transactions, or evidence about actual coffee markets**. Prices are illustrative USD per 250 g. The simulated quality score is not a certified cupping score.

## Research design
- Unit of observation: one simulated coffee listing.
- Outcome: USD price per 250 g (standardized pack size).
- Characteristics: bean type, roast, origin, specialty label, quality score.
- Descriptive statistics: count, mean, median, sample standard deviation, coefficient of variation.
- Association: Pearson's r for quality and price.
- Multivariable OLS with HC3 robust standard errors for *conditional associations*, not causal effects.

## Run it
```bash
git clone https://github.com/YOUR_USERNAME/coffee-premium-eda.git
cd coffee-premium-eda
pip install -r requirements.txt
python analysis.py
```
The script generates `data/coffee_market_simulated.csv` if absent and saves charts and statistics under `outputs/`.

## Economic interpretation
1. **Dispersion:** A higher standard deviation means wider absolute price spread. The coefficient of variation (`SD / mean × 100`) compares relative dispersion across bean groups.
2. **Quality premium:** Pearson correlation measures linear co-movement, not the causal value of quality.
3. **Segmentation:** Mean prices across roast or bean groups may differ because of the mix of origins, labels, and quality scores.
4. **Conditional pricing:** The regression asks whether a quality-price association remains *after holding observed categories constant*. Omitted variables and synthetic construction prevent causal claims.

## Limits and next step
The pricing relationships are deliberately built into the generator. Finding them again is a **pipeline demonstration, not a discovery**. To turn this into market research, replace the simulated CSV with dated, verifiable retail listings, record vendor and pack size, normalize currencies, and document sampling bias.

## A question worth debating
> Is a coffee premium a reward for intrinsic quality, or a price for signals that consumers can more easily recognize?

**Suggested GitHub topics:** `python` `exploratory-data-analysis` `economics` `coffee` `pandas` `data-visualization`
