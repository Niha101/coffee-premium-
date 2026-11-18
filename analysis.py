"""The Coffee Premium: reproducible, economics-led exploratory analysis.
Data are SIMULATED and do not represent observed market prices.
Run: python analysis.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coffee_market_simulated.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", palette="crest")
plt.rcParams.update({"figure.facecolor":"#fbf8f2","axes.facecolor":"#fbf8f2",
                     "savefig.facecolor":"#fbf8f2","font.family":"DejaVu Sans"})
DARK = "#283b32"
rng = np.random.default_rng(42)

def generate_data(n=240):
    """Illustrative cross-sectional coffee listings with explicit assumptions."""
    origin = rng.choice(["Ethiopia","Colombia","Brazil","India","Kenya","Guatemala"],
                        n, p=[.19,.20,.19,.17,.12,.13])
    bean = rng.choice(["Arabica","Robusta","Blend"], n, p=[.57,.18,.25])
    roast = rng.choice(["Light","Medium","Dark"], n, p=[.30,.45,.25])
    specialty = rng.binomial(1, .42, n)
    quality = np.clip(np.round(rng.normal(82, 4.2, n) +
                     3*specialty + 1.1*(bean=="Arabica") -
                     1.8*(bean=="Robusta"),1), 68, 96)
    origin_effect = pd.Series(origin).map({"Ethiopia":2.5,"Colombia":1.8,"Brazil":-.8,
                       "India":-.3,"Kenya":2.1,"Guatemala":1.2}).to_numpy()
    # Synthetic price-setting process: designed for teaching, NOT estimated from real listings.
    price = (8 + .65*(quality-80) + 3.4*specialty +
             2.1*(bean=="Arabica") - 1.7*(bean=="Robusta") +
             1.1*(roast=="Light") + origin_effect + rng.normal(0,3.5,n))
    price = np.maximum(5.5, price).round(2)
    df = pd.DataFrame({"listing_id":[f"CP-{i:03d}" for i in range(1,n+1)],
            "origin":origin,"bean_type":bean,"roast":roast,
            "specialty_label":np.where(specialty==1,"Specialty","Standard"),
            "quality_score":quality,"price_usd_per_250g":price})
    df.to_csv(DATA,index=False)
    return df

def main():
    if not DATA.exists():
        generate_data()
    df = pd.read_csv(DATA)
    assert df["listing_id"].is_unique
    assert df["price_usd_per_250g"].gt(0).all()
    assert df[["price_usd_per_250g","quality_score","bean_type","roast"]].notna().all().all()
    p = "price_usd_per_250g"
    stats = df.groupby("bean_type")[p].agg(["count","mean","median","std"]).round(2)
    stats["cv_pct"] = (100*stats["std"]/stats["mean"]).round(1)
    stats.to_csv(OUT/"bean_descriptives.csv")
    corr = df[["quality_score",p]].corr().iloc[0,1]
    model = smf.ols("price_usd_per_250g ~ quality_score + C(bean_type) + C(roast) + C(origin) + C(specialty_label)",data=df).fit(cov_type="HC3")
    (OUT/"regression_summary.txt").write_text(model.summary().as_text(),encoding="utf-8")
    print("\nCOFFEE PREMIUM | SIMULATED DATA — NOT REAL MARKET EVIDENCE")
    print(f"Listings: {len(df)} | Mean: ${df[p].mean():.2f} | Median: ${df[p].median():.2f}")
    print(f"Std dev: ${df[p].std():.2f} | Quality-price Pearson r: {corr:.3f}")
    print("\nBean-level descriptive statistics:\n",stats.to_string())
    print("\nHC3 robust OLS coefficient on quality:",round(model.params["quality_score"],3))
    print("Caution: synthetic correlations are consequences of simulation assumptions.")

    fig,axs=plt.subplots(1,2,figsize=(13,5.5),layout="constrained")
    order=["Robusta","Blend","Arabica"]
    sns.boxplot(data=df,x="bean_type",y=p,order=order,ax=axs[0],color="#8eaa94",
                showfliers=False)
    axs[0].set(title="01 / Price dispersion by bean",xlabel="",ylabel="USD per 250 g")
    sns.regplot(data=df,x="quality_score",y=p,ax=axs[1],
                scatter_kws={"alpha":.45,"s":24,"color":"#4b7766"},
                line_kws={"color":"#bd7148"})
    axs[1].set(title=f"02 / Quality vs. price  |  r = {corr:.2f}",
               xlabel="Illustrative quality score",ylabel="USD per 250 g")
    fig.suptitle("THE COFFEE PREMIUM  /  What shapes a coffee price?",
                 fontsize=18,fontweight="bold",color=DARK)
    fig.savefig(OUT/"coffee_premium_dashboard.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

    means=df.groupby("roast")[p].mean().reindex(["Light","Medium","Dark"])
    fig,ax=plt.subplots(figsize=(8,4.7),layout="constrained")
    bars=ax.bar(means.index,means.values,color=["#bdad84","#72927b","#374e40"])
    ax.bar_label(bars,fmt="$%.2f",padding=5,fontsize=11)
    ax.set(title="Roast-level mean price | descriptive, not causal",
           ylabel="USD per 250 g",ylim=(0,means.max()*1.23))
    fig.savefig(OUT/"roast_price_comparison.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

if __name__=="__main__":
    main()
