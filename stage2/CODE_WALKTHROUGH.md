# Stage 2 notebook — line-by-line walkthrough

A companion to `Stage2_VIX_GARCH.ipynb`. Every cell, in order, with the Python
explained as well as the econometrics. Written to be read next to the notebook.

Cell numbers count from 0 and include markdown cells, so they match what you see
if you run `nbformat` over the file. Section numbers (1.1, 4.2, …) are the ones
in the notebook's own headings.

---

## Part 0 — Python idioms that recur throughout

Read this once and the rest of the notebook stops looking cryptic. These are the
constructs that show up in nearly every cell.

### f-strings

```python
print(f"Observations: {len(df):,}")
```

An `f` before the quote makes it a *formatted* string: anything inside `{}` is
evaluated as Python and pasted in. The part after `:` is a **format spec**:

| Spec | Meaning | Example output |
|---|---|---|
| `:,` | thousands separators | `3,680` |
| `:.4f` | fixed point, 4 decimals | `0.9573` |
| `:.3e` | scientific, 3 decimals | `1.548e-06` |
| `:+.4f` | always show the sign | `+0.4100` |
| `:6d` | integer, padded to width 6 | `   250` |
| `:16s` | string, padded to width 16 | `returns         ` |
| `:%d %b %Y` | date formatting | `16 Mar 2020` |

Padding is what makes the printed tables line up in columns.

### DataFrames and Series

A **DataFrame** is a table. A **Series** is one column of it. `df["VIX"]` pulls
out a Series; `df[["SP500", "VIX"]]` (note the double brackets) pulls out a
smaller DataFrame with two columns.

The **index** is the row labels. In this notebook the index is always the date,
which is why `df.loc["2012-01-01":"2026-08-21"]` works — you slice by *label*,
not position. That is `.loc`. Its counterpart `.iloc` slices by position, and is
barely used here.

A crucial property: pandas operations **align on the index automatically**. When
you write `sp_full["SP500"] - vix_full["VIX"]`, pandas matches rows by date, not
by row number. This is why the merge logic later can be so terse, and also why a
mismatched calendar silently produces `NaN` rather than a wrong answer.

### Method chaining

```python
df.index.to_series().diff().dt.days
```

Each `.something()` returns a new object, which the next call acts on. Read it
left to right: take the index → turn it into a Series → take successive
differences → extract the number of days. No intermediate variables needed.

### Vectorisation

```python
100 * np.log(sp_full["SP500"]).diff()
```

There is no loop. `np.log` applied to a Series returns a Series of logs;
`.diff()` subtracts each element from the one before; `100 *` scales all of them.
NumPy and pandas do the looping in C, which is both faster and less error-prone
than writing the loop yourself. Almost every calculation in this notebook is
vectorised.

### Boolean masks

```python
(df["gap"] > 0).sum()
```

`df["gap"] > 0` produces a Series of `True`/`False`, one per row. In Python
`True` counts as 1 and `False` as 0, so `.sum()` counts the `True`s and
`.mean()` gives the *proportion* — which is why `100 * (df['gap']>0).mean()`
yields a percentage. This trick appears repeatedly.

### List comprehensions

```python
[str(d.date()) for d in only_vix]
```

Builds a list by looping inline. Equivalent to a three-line `for` loop with
`.append()`. Read it as "give me `str(d.date())` for each `d` in `only_vix`".

### Tuple unpacking in loops

```python
for name, s in [("VIX", vix_full), ("SP500", sp_full)]:
```

The list holds pairs. Each turn of the loop unpacks one pair into two variables,
so the body runs once for VIX and once for the S&P 500. It is how the notebook
avoids copy-pasting the same block twice.

### `display()` vs `print()`

`print()` renders text. `display()` is a Jupyter function that renders a
DataFrame as a formatted HTML table. Using `display()` on the last line of a cell
is equivalent to just writing the variable name, but works anywhere in the cell.

---

## Section 1 — Data preparation

### Cell 0 (markdown) — title

States the scope: build a matched dataset, document its properties, estimate a
baseline GARCH. Explicitly notes *no forecasting at this stage*, which is what
keeps Stage 2 separable from Stage 3.

### Cell 1 (markdown) — data sources

The sourcing table and the justification for using two providers. The substance:
CBOE publishes the VIX free but not the S&P 500, so the underlying has to come
from elsewhere; Yahoo is that elsewhere. Liu et al. (2015 §4.1) hit the same
constraint and are cited for it.

### Cell 2 (code) — imports and configuration

```python
from pathlib import Path
import numpy as np
import pandas as pd

pd.set_option("display.float_format", "{:,.4f}".format)

CANDIDATES = [Path("data/raw"), Path("."), Path("stage2/data/raw"), Path("/content")]
DATA_DIR = next((p for p in CANDIDATES if (p / "sp500_raw.csv").exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "sp500_raw.csv not found in any of: " + ", ".join(str(p) for p in CANDIDATES))
print("Reading data from:", DATA_DIR.resolve())

START, END = "2012-01-01", "2026-08-21"
```

- `from pathlib import Path` gives an object-oriented way to handle file paths.
  `Path("data/raw") / "sp500_raw.csv"` joins them with the correct separator for
  the operating system — `/` on Linux and Colab, `\` on Windows. Using `/` as the
  join operator looks odd but is deliberate: `Path` overloads the division sign.
- `pd.set_option("display.float_format", ...)` sets how floats print *globally*.
  `"{:,.4f}".format` is a function passed without calling it (no parentheses) —
  pandas calls it later on each number. Note the consequence flagged in Section 2:
  a p-value of 1e-300 displays as `0.0000` under this format, which is why
  p-values are printed separately with `:.3e`.
- The `CANDIDATES` block is a **path-finder**. The notebook might be run from the
  repo root, from inside `stage2/`, or on Colab where files land in `/content`.
  Rather than hard-coding one path, it tries each in turn.
- `next((p for p in CANDIDATES if ...), None)` returns the first candidate that
  contains the file, or `None` if there is none. The `(p for p in ...)` part is a
  *generator expression*: it produces values lazily, so `next()` stops at the
  first match rather than checking all four.
- `raise FileNotFoundError(...)` fails loudly with a useful message. Failing fast
  beats letting a later cell throw a confusing error.
- `START, END = "2012-01-01", "2026-08-21"` assigns two variables on one line.

### Cell 3 (markdown) — §1.1 heading

### Cell 4 (code) — the loaders

```python
def load_yf(filename, colname):
    df = pd.read_csv(DATA_DIR / filename, index_col=0, parse_dates=True, skiprows=[1, 2])
    df.index.name = "Date"
    return df[["Close"]].rename(columns={"Close": colname})
```

This exists because of a quirk in how `yfinance` writes CSVs. The file starts:

```
Price,Close,High,Low,Open,Volume
Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC
Date,,,,,
2011-12-30,1257.60,...
```

Three header rows, not one. The arguments handle it:

- `index_col=0` — use the first column (the date) as the row index.
- `parse_dates=True` — convert that index from text to real timestamps. Without
  this, `.loc["2012":"2026"]` slicing and `.diff().dt.days` would not work.
- `skiprows=[1, 2]` — discard rows 1 and 2 (the `Ticker` and empty `Date` rows),
  keeping row 0 as the column names. Note this counts *data* rows after the
  header, which is why it is `[1, 2]` and not `[1, 2, 3]`.
- `df[["Close"]]` — double brackets keep it a DataFrame rather than collapsing to
  a Series, so `.rename(columns=...)` still applies.
- `.rename(columns={"Close": colname})` — so the VIX column is called `VIX` and
  the S&P column `SP500`, which matters once they are concatenated.

```python
def load_cboe(filename, colname):
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["DATE"]).set_index("DATE")
    df.index.name = "Date"
    return df[["CLOSE"]].rename(columns={"CLOSE": colname})
```

CBOE's file is a clean four-column CSV (`DATE,OPEN,HIGH,LOW,CLOSE`) with US-format
dates, so it needs no `skiprows`. `parse_dates=["DATE"]` names the column to
convert; `.set_index("DATE")` promotes it to the index afterwards. Renaming the
index to `Date` in both loaders is what lets the two frames align later.

```python
vix_cboe = load_cboe("VIX_History.csv", "VIX")        # primary VIX series
vix_yf   = load_yf("vix_raw.csv",   "VIX_YF")         # validation only
sp_hist  = load_yf("sp500_raw.csv", "SP500")          # S&P 500 prices

for name, s in [("VIX (CBOE)", vix_cboe), ("VIX (Yahoo)", vix_yf), ("S&P 500", sp_hist)]:
    print(f"{name:14s} {s.index.min().date()} -> {s.index.max().date()}   N = {len(s):,}")
```

Three series loaded, then a loop printing the span and row count of each. This is
a **sanity check**: if a file were truncated or misparsed, the dates or counts
would look wrong immediately. Expect the CBOE series to start in 1990 (its full
history) while the Yahoo files start wherever `datascript.py` asked.

### Cell 5 (markdown) — §1.2 heading

Explains *why* the audit exists: an inner join would drop mismatches silently, and
the brief asks for missing observations to be checked rather than dropped.

### Cell 6 (code) — calendar audit

```python
win_cboe = vix_cboe.loc[START:END]
win_yf   = vix_yf.loc[START:END]
win_sp   = sp_hist.loc[START:END]
```

Restricting each series to the analysis window first. Necessary because CBOE's
history runs from 1990: without this, every pre-2012 date would register as a
"mismatch" and swamp the comparison.

```python
vix_not_sp = win_cboe.index.difference(win_sp.index)
sp_not_vix = win_sp.index.difference(win_cboe.index)
```

`.difference()` is set subtraction on the index: dates in the first that are not
in the second. Running it both ways distinguishes *VIX has an extra day* from
*the S&P has an extra day* — different problems with different causes.

```python
print("\nInternal missing values:",
      {"VIX_CBOE": int(vix_cboe['VIX'].isna().sum()), "SP500": int(sp_hist['SP500'].isna().sum())})
print("Duplicate index entries:",
      {"VIX_CBOE": int(vix_cboe.index.duplicated().sum()), ...})
```

Two more failure modes. `.isna()` flags missing values, `.duplicated()` flags
repeated dates — a duplicate date would quietly double-count an observation in
every statistic that follows. `int(...)` converts NumPy's integer type to a plain
Python int so it prints as `0` rather than `np.int64(0)`.

### Cell 7 (markdown) — the finding

CBOE reports a VIX close on **32 dates** when the NYSE cash market was shut — US
holidays, concentrated from 2022 onward. These are genuine index values: VIX is
computed from SPX option quotes, which trade in sessions that do not always
coincide with equity hours. Since no matched S&P return can be formed on those
days, the inner join drops them. Nothing is wrong with either dataset.

### Cells 8–12 — §1.3, validating Yahoo against CBOE

```python
cmp = win_yf.join(vix_cboe, how="inner").round(2)
diff = (cmp["VIX_YF"] - cmp["VIX"]).abs().round(2)
```

`.join()` merges on the index (the date) by default. `how="inner"` keeps only
dates present in both. `.round(2)` before differencing matters: both providers
quote the VIX to two decimals, so comparing unrounded floats would flag
floating-point noise as disagreement.

```python
print(f"Identical to 2 dp          : {(diff == 0).sum():,}  ({100*(diff==0).mean():.2f}%)")
```

The boolean-mask trick again — `.sum()` for the count, `.mean()` for the share.

```python
print(cmp.assign(abs_diff=diff)[diff > 0.01]
         .sort_values("abs_diff", ascending=False)
         .to_string(float_format=lambda v: f"{v:.2f}"))
```

- `.assign(abs_diff=diff)` adds a column and returns a *new* DataFrame, leaving
  `cmp` untouched. Useful inside a chain.
- `[diff > 0.01]` filters rows using the boolean mask.
- `.sort_values(..., ascending=False)` orders worst-first.
- `.to_string(float_format=...)` overrides the global 4-decimal format for this
  one printout. A `lambda` is a small unnamed function: `lambda v: f"{v:.2f}"`
  takes a number and returns it as a 2-decimal string.

The result: 3,681 common dates, agreement to 2dp on 99.76%, correlation 0.99997,
nine days differing by more than 0.01. The largest is 2026-02-06, where Yahoo says
20.37 against CBOE's 17.76.

**Cell 11** settles which one is wrong by printing the surrounding week alongside
S&P returns. The index rose 1.95% that day — a move that should *lower* implied
volatility. CBOE's path 21.77 → 17.76 → 17.36 fits; Yahoo's 20.37 does not fit its
own neighbours or the return. So Yahoo carries a bad print, and CBOE is used
throughout. Note the reasoning pattern: an anomaly is resolved by checking it
against an independent series, not by assuming the official source is right.

### Cell 13 (markdown) — §1.4, why returns come before truncation

### Cell 14 (code) — log returns

```python
sp_hist["ret"] = 100 * np.log(sp_hist["SP500"]).diff()
sp_hist.loc["2011-12-28":"2012-01-05"]
```

The formula is `r_t = 100 * [ln(P_t) - ln(P_{t-1})]`. `.diff()` performs the
subtraction against the previous row. Log returns are used rather than simple
percentage changes because they add across time (a two-day return is the sum of
two daily log returns) and are closer to symmetric, which suits a model assuming
roughly Gaussian innovations.

The ordering is the point. Returns are computed on the **full** price history
*before* slicing to 2012. Slice first and the first in-sample return would be
`NaN`, wasting an observation; compute first and 2012-01-03's return is correctly
measured against the 2011-12-30 close. The printed slice is the proof — you can
see the boundary row.

### Cell 15 (markdown) — §1.5 heading

### Cell 16 (code) — merge and restrict

```python
df = pd.concat([sp_hist, vix_cboe], axis=1, join="inner").dropna()
df = df.loc[START:END, ["SP500", "VIX", "ret"]]
```

- `pd.concat(..., axis=1)` glues frames side by side (`axis=0` would stack them
  vertically). Alignment is on the index, so dates match up automatically.
- `join="inner"` keeps only dates present in both — this is where the 32
  holiday-only VIX observations are dropped.
- `.dropna()` removes any row with a missing value, which here is the very first
  return.
- `.loc[START:END, [...]]` takes rows by date range and columns by name in one
  step, and fixes the column order.

```python
gaps = df.index.to_series().diff().dt.days
```

Converts the index into a Series so `.diff()` can run on it, giving the interval
between consecutive observations as a timedelta; `.dt.days` extracts the number of
days as an integer. `gaps.max()` finds the largest, `gaps.idxmax()` the date where
it occurs.

Result: 3,680 rows, no missing values, largest gap five calendar days ending
2012-10-31 — the NYSE closure for Hurricane Sandy. A real market closure, so no
imputation; the return spanning it is a true multi-day return.

### Cell 17 (markdown) — the Hurricane Sandy note

### Cell 18 (code) — save the clean dataset

```python
df.to_csv(DATA_DIR / "stage2_clean.csv")
```

Writes the merged, cleaned data. Saving at this checkpoint means later sections
can be re-run without repeating the loading and cleaning.

---

## Section 2 — Descriptive statistics

### Cell 19 (markdown) — why not `.describe()`

The format follows Fernandes et al. (2014, Table 1). The key point: pandas'
built-in `.describe()` reports neither skewness nor kurtosis, and those are
precisely the moments that motivate a conditional-variance model.

### Cell 20 (code) — the describe function

```python
from scipy import stats

def describe(s):
    jb_stat, jb_p = stats.jarque_bera(s)
    return {
        "N": len(s), "Mean": s.mean(), "Median": s.median(),
        "Std. dev": s.std(), "Minimum": s.min(), "Maximum": s.max(),
        "Skewness": stats.skew(s),
        "Kurtosis": stats.kurtosis(s, fisher=False),
        "Jarque-Bera": jb_stat,
    }
```

Returns a **dictionary** — key/value pairs. Building a DataFrame from a dict of
dicts, as the next lines do, turns each inner dict into a column with the keys as
row labels.

Two details worth knowing:

- `stats.kurtosis(s, fisher=False)` gives **unstandardised** kurtosis, where a
  normal distribution scores 3. The default `fisher=True` would subtract 3 and
  report *excess* kurtosis. The notebook chooses `False` to match Fernandes et al.
  Reporting 18.9 when a reader expects excess kurtosis, or vice versa, is a
  classic and embarrassing error — hence the explicit note in the markdown.
- **Skewness** measures asymmetry. Positive means a long right tail.
- **Jarque–Bera** tests normality jointly through skewness and kurtosis. Under
  the null of normality it is chi-squared with 2 degrees of freedom.

```python
desc = pd.DataFrame({"VIX": describe(df["VIX"]),
                     "S&P 500 return (%)": describe(df["ret"])})
```

Two calls, two columns, aligned by their shared keys.

```python
for name, s in [("VIX", df["VIX"]), ("S&P 500 return", df["ret"])]:
    print(f"Jarque-Bera p-value, {name:15s}: {stats.jarque_bera(s)[1]:.3e}")
```

Printed separately in scientific notation because the global float format would
render these as `0.0000`. `[1]` picks the p-value out of the returned tuple.

### Cell 21 (markdown) — reading the table

VIX: mean 17.75 above median 16.18, skewness +2.86, max 82.69 (16 March 2020)
against min 9.14 — a series that sits low most of the time and spikes hard.
Kurtosis 18.9 against a normal 3 says those spikes are far more common than a
Gaussian allows.

Returns: mean 0.049% daily, sd 1.05%, skewness −0.64 — the mirror image, because
the big moves that drive volatility up are mostly falls. Kurtosis 19.3, worst day
−12.77%, best +9.09%.

---

## Section 3 — Plots

### Cell 22 (markdown) — why a shared style

### Cell 23 (code) — plotting setup

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

plt.rcParams.update({...})

def finish(ax, title, ylabel, xlabel="Year"):
    ax.set_title(title, color=INK, pad=8, loc="left", fontweight="medium")
    ax.set_ylabel(ylabel); ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    return ax
```

- `FIG_DIR.mkdir(exist_ok=True)` creates the folder; `exist_ok=True` stops it
  erroring if it already exists.
- `plt.rcParams` is matplotlib's global settings dictionary. Setting it once means
  every subsequent figure inherits the same fonts, colours, grid and spines — the
  figures read as one set rather than seven unrelated charts.
- Colours are named constants, so changing the palette is a one-line edit.
- `finish()` factors out the repeated title/label/margin work. `set_axisbelow(True)`
  draws gridlines *behind* the data. Returning `ax` allows chaining.

The two chart colours are a blue/orange pair chosen for colour-vision-deficiency
separation, which also means they survive greyscale printing.

### Cells 24–32 — Figures 1 to 3

The pattern is identical each time:

```python
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(df.index, df["VIX"], color=BLUE, linewidth=0.8)
finish(ax, "CBOE Volatility Index (VIX), daily close", "VIX (index points)")
fig.savefig(FIG_DIR / "fig1_vix.png"); plt.show()
```

`plt.subplots()` returns a figure (the canvas) and an axes (the plot area). Then
draw, label, save, show.

Figure 1 adds an annotation at the maximum:

```python
peak = df["VIX"].idxmax()
ax.annotate(f"{df['VIX'].max():.1f}  ({peak:%d %b %Y})",
            xy=(peak, df["VIX"].max()), xytext=(12, -6),
            textcoords="offset points", fontsize=8, color=INK_SOFT)
```

`.idxmax()` gives the *index label* (the date) of the maximum, as opposed to
`.max()` which gives the value. `xy` is where to point, `xytext` with
`textcoords="offset points"` shifts the label 12 points right and 6 down so it
does not sit on top of the spike.

Figure 2 plots returns and adds `ax.axhline(0, ...)` for a zero reference.
Figure 3 plots `df["ret"]**2` — squaring removes the sign so only magnitude
remains, which makes the clustering unmistakable. The bursts line up with the VIX
spikes in Figure 1: the same episodes seen through realised rather than implied
volatility. That correspondence is the premise of the whole project.

---

## Section 4 — Time-series properties

### Cell 34 (code) — §4.1, persistence of VIX

```python
from statsmodels.tsa.stattools import adfuller, acf

NLAGS = 250
acf_vix = acf(df["VIX"], nlags=NLAGS, fft=True)
band = 1.96 / np.sqrt(len(df))
```

- **Autocorrelation** at lag k is the correlation of the series with itself k
  periods earlier. `acf()` returns an array where element 0 is lag 0 (always 1.0),
  element 1 is lag 1, and so on — which is why later code indexes `acf_vix[L]`.
- `fft=True` computes it via the fast Fourier transform. Same answer, much faster
  at 250 lags.
- `band = 1.96/sqrt(N)` is the 95% confidence band under the null of white noise.
  If the series were pure noise, sample autocorrelations would fall inside ±band
  about 95% of the time. With N = 3,680 the band is ±0.0323 — very tight, because
  a long sample estimates autocorrelation precisely.

```python
ax.vlines(range(1, NLAGS + 1), 0, acf_vix[1:], color=BLUE, linewidth=0.9)
ax.axhspan(-band, band, color=ORANGE, alpha=0.16, zorder=0, label="95% band under white noise")
```

`vlines` draws the stems from 0 up to each autocorrelation — the standard
correlogram. `axhspan` shades the band; `zorder=0` puts it behind the stems.
`acf_vix[1:]` skips lag 0, whose value of 1.0 would dwarf everything else.

```python
outside = np.where(np.abs(acf_vix[1:]) > band)[0] + 1
print(f"ACF remains outside the band as far as lag {outside.max()} of {NLAGS}")
```

`np.where(condition)` returns the positions where the condition holds, as a
one-element tuple — hence the `[0]`. The `+ 1` corrects for having sliced off lag
0, converting positions back to lag numbers.

**Result and why it matters.** ACF is 0.961 at lag 1, 0.563 at lag 22 (one month),
0.337 at lag 66 (a quarter), and still 0.046 at lag 250 — outside the ±0.032 band
across the entire range, a full trading year. The decay is hyperbolic, not
geometric: **long memory**.

Fernandes et al.'s explanation applies directly. VIX measures expected volatility
over the *next 30 calendar days* but is sampled *daily*, so today's and tomorrow's
values describe almost the same window. That overlap mechanically induces
persistence. The implication for the project is stated plainly in the notebook:
any proposed new predictor — GARCH volatility included — must add information
beyond an own-history baseline that is already extremely strong.

### Cell 37 (code) — §4.2, stationarity

```python
def adf_report(series, label, regression="c"):
    stat, p, lags, nobs, crit, _ = adfuller(series, regression=regression, autolag="AIC")
```

Six-way tuple unpacking; `_` is the conventional name for a value you intend to
ignore.

- The **Augmented Dickey–Fuller** test has a null hypothesis of a *unit root* —
  a random walk, where shocks never die out. Rejecting the null supports
  stationarity.
- `regression="c"` includes a constant but no time trend, appropriate for a series
  with a non-zero mean and no drift.
- `autolag="AIC"` selects how many lagged differences to include by minimising the
  Akaike Information Criterion, rather than fixing it by hand.

```python
adf_report(df["VIX"],         "VIX (level, constant)")
adf_report(np.log(df["VIX"]), "log(VIX) (constant)")
adf_report(df["ret"],         "S&P 500 returns (constant)")
```

log(VIX) is tested because that is Fernandes et al.'s dependent variable — useful
to have verified before Stage 3 chooses a specification.

The unit root is rejected for all three (VIX: −5.56, p ≈ 1.5e-6). **Stationary but
strongly persistent** is not a contradiction: stationarity says shocks eventually
die; the ACF says they take a very long time doing so.

### Cells 40–41 (code) — §4.3, volatility clustering

```python
L = 40
acf_ret = acf(df["ret"],    nlags=L, fft=True)
acf_sq  = acf(df["ret"]**2, nlags=L, fft=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, a, title in [(axes[0], acf_ret, "..."), (axes[1], acf_sq, "...")]:
    ...
```

`plt.subplots(1, 2)` makes one row of two panels; `axes` is then an array of two
axes objects. `sharey=True` forces a common vertical scale, without which the
visual comparison would be meaningless. The loop draws both panels with identical
code — the whole reason for the tuple-unpacking pattern.

```python
for name, s in [("returns", df["ret"]), ("squared returns", df["ret"]**2)]:
    lb = acorr_ljungbox(s, lags=[10, 22], return_df=True)
    for lag, row in lb.iterrows():
        print(f"   {name:16s} h={lag:3d}:  Q = {row['lb_stat']:10.2f}   p = {row['lb_pvalue']:.3e}")
```

- **Ljung–Box** tests whether *all* autocorrelations up to lag h are jointly zero,
  rather than testing each lag separately. `return_df=True` gives a DataFrame;
  `.iterrows()` walks it row by row, unpacking each into its index label and data.

```python
for lag in [5, 10, 22]:
    lm, lm_p, _, _ = het_arch(df["ret"], nlags=lag)
```

- **Engle's ARCH-LM** test regresses squared residuals on their own lags and tests
  whether those coefficients are jointly zero. Rejecting means the variance is
  predictable from its own past — ARCH effects.

**The numbers.** ACF of raw returns: −0.116, +0.072, +0.034, −0.034, −0.078 —
small and alternating, close to unforecastable, as market efficiency suggests.
ACF of squared returns: +0.449, +0.469, +0.290, +0.218, +0.065 — positive
throughout and an order of magnitude larger. Ljung–Box Q(10) = 3,955 on squared
returns against 236 on raw. ARCH-LM = 1,109 at five lags, p ≈ 1e-237.

The direction of tomorrow's move is unpredictable; its **magnitude** is highly
predictable from recent magnitudes. This is the empirical precondition for GARCH —
the model is not an arbitrary choice, it is the natural description of what these
tests and Figures 2, 3 and 5 all show.

---

## Section 5 — Estimating GARCH(1,1)

### Cell 43 (markdown) — the model

$$r_t = \mu + \epsilon_t, \qquad \epsilon_t = \sigma_t z_t, \qquad
\sigma_t^2 = \omega + \alpha\epsilon_{t-1}^2 + \beta\sigma_{t-1}^2$$

In words: today's variance is a weighted combination of a long-run level (ω), the
size of yesterday's surprise (α ε²), and yesterday's variance (β σ²). The mean
equation is just a constant μ; all the structure is in the variance.

`z_t` is a standardised innovation — mean 0, variance 1 — so `ε_t = σ_t z_t` says
the shock is the innovation scaled by the current volatility level. That is the
whole idea: the *distribution* of returns is the same shape every day, but its
*width* changes with σ_t.

### Cell 44 (code) — fitting

```python
from arch import arch_model

model = arch_model(df["ret"], mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
res = model.fit(disp="off")
print(res.summary())
```

- `mean="Constant"` — the mean equation is μ alone.
- `vol="GARCH"`, `p=1, q=1` — one ARCH lag and one GARCH lag, i.e. GARCH(1,1).
  In this library's convention `p` counts lagged squared *residuals* and `q`
  counts lagged *variances*.
- `dist="normal"` — Gaussian innovations, the baseline. Revisited in cell 48.
- `.fit(disp="off")` runs maximum likelihood and suppresses the optimiser's
  iteration log. Two objects matter: `model` describes the specification, `res`
  holds the results.

**Why returns are pre-scaled by 100.** The notebook feeds in percentage returns
(0.5 for half a percent) rather than decimals (0.005). Squared decimals are around
1e-5, and optimisers handle badly-scaled likelihoods poorly — you get convergence
warnings or wrong standard errors. Working in percent keeps everything near 1.
It also means σ_t comes back in percent, which the annualisation below assumes.

### Cell 45 (code) — pulling out and interpreting the estimates

```python
w  = res.params["omega"]
al = res.params["alpha[1]"]
be = res.params["beta[1]"]
persistence = al + be
```

`res.params` is a Series indexed by parameter name. The `[1]` in `alpha[1]` is
part of the *name* — it denotes the first lag, and would be joined by `alpha[2]`
in a GARCH(2,1).

```python
params = pd.DataFrame({
    "Estimate":   [res.params[k]   for k in ["mu", "omega", "alpha[1]", "beta[1]"]],
    "Std. error": [res.std_err[k]  for k in [...]],
    "t-statistic":[res.tvalues[k]  for k in [...]],
}, index=["mu (mean)", "omega", "alpha", "beta"])
```

Three list comprehensions over the same parameter names build a tidy table. The
`index=` argument supplies friendlier row labels than the raw names.

```python
half_life = np.log(0.5) / np.log(persistence)
lr_var    = w / (1 - persistence)
lr_ann    = np.sqrt(lr_var) * np.sqrt(252)
```

Three derived quantities that do most of the interpretive work:

- **Long-run variance** `ω/(1−α−β)`. Set σ²_t = σ²_{t−1} = σ̄² in the variance
  equation and solve: σ̄² = ω + ασ̄² + βσ̄², so σ̄² = ω/(1−α−β). This only exists
  if α+β < 1, which is exactly the stationarity condition.
- **Half-life** `ln(0.5)/ln(α+β)`. A shock to variance decays by a factor of
  (α+β) each day, so after k days a fraction (α+β)^k remains. Set that equal to
  0.5 and solve for k.
- **Annualisation** `× sqrt(252)`. Variance scales with time under independence,
  so volatility scales with the square root of time. 252 is the conventional
  number of trading days in a year.

### Cell 46 (markdown) — interpretation

- **ω = 0.0431** — the floor of the variance equation. Not interpretable alone,
  but with the persistence it implies a long-run daily variance of 1.010, a daily
  sd of 1.005%, and an annualised 15.95 — reassuringly close to the sample sd of
  returns (1.051%). If those two disagreed badly, something would be wrong.
- **α = 0.167** — the reaction coefficient. A surprise raises today's variance by
  16.7% of the squared surprise. This governs how sharply the model responds to
  news.
- **β = 0.791** — the carry-over from yesterday's variance. The most precisely
  estimated parameter (t = 30.9) and the one that dominates: volatility today is
  mostly volatility yesterday.
- **α + β = 0.9573** — the headline. Below 1, so variance is mean-reverting and
  stationary; but high enough that the half-life is 15.9 trading days, about three
  trading weeks for half a shock to dissipate. This is the standard result for
  daily equity index returns, where α+β usually lands between 0.95 and 0.99, and
  it mirrors the persistence already found in VIX itself.

The diagnostic checks: α, β > 0 (non-negativity, so variance can never go
negative) and α + β < 1 (stationarity). Both hold.

### Cells 47–49 — Student-t robustness

```python
res_t = arch_model(df["ret"], mean="Constant", vol="GARCH", p=1, q=1,
                   dist="t").fit(disp="off")
```

The only change is `dist="t"`. The motivation is honest: Section 2 measured return
kurtosis at 19.3, so assuming Gaussian innovations is questionable, and a result
that depended on that assumption would be fragile.

```python
compare = pd.DataFrame({
    "Normal":    [..., res.loglikelihood, res.bic, np.nan],
    "Student-t": [..., res_t.loglikelihood, res_t.bic, res_t.params["nu"]],
}, index=["omega", "alpha", "beta", "alpha + beta", "Log-likelihood", "BIC", "nu (d.o.f.)"])
```

`np.nan` is a placeholder in the Normal column, since a Gaussian has no
degrees-of-freedom parameter — the columns must be the same length.

**BIC** (Bayesian Information Criterion) compares models on fit while penalising
extra parameters; lower is better. Here 9,019 for Student-t against 9,247 for
Normal — a decisive preference.

**What survives.** α is essentially identical (0.167 either way); persistence
rises from 0.957 to 0.984; ν = 5.5 confirms strongly fat tails. So the persistence
*conclusion* does not depend on the distributional assumption, which is the point
of the exercise. The Gaussian model is retained because the brief specifies a
basic GARCH(1,1) — and the notebook says plainly that the Student-t is the one to
defend if a single specification has to be chosen later.

---

## Section 6 — Conditional volatility and annualisation

### Cell 51 (code)

```python
df["garch_vol_daily"] = res.conditional_volatility
df["garch_vol_ann"]   = df["garch_vol_daily"] * np.sqrt(252)
```

`res.conditional_volatility` is the fitted σ_t series — one number per day, the
model's estimate of that day's volatility given information through the previous
day. Assigning it to `df` works because it carries the same index.

Units are the crux. `arch` returns σ_t in the units of the input, which are daily
percent. VIX is quoted as an *annualised* percentage. Multiplying by √252 puts
them on the same footing — and that is what allows the overlay in Section 7 to
use a single vertical axis. Without this step the comparison would be
meaningless.

The summary table reuses the `describe()` function from Section 2, then `.T`
transposes it so the two series are rows rather than columns.

---

## Section 7 — VIX versus GARCH volatility

### Cell 53 — the overlay (Figure 6)

Two `ax.plot()` calls on the same axes, one per series, distinguished by colour
and legend label.

### Cell 55 — the gap statistics

```python
df["gap"] = df["VIX"] - df["garch_vol_ann"]
```

Element-wise subtraction, aligned on date.

```python
print(f"   as a share of GARCH mean    : {100*df['gap'].mean()/df['garch_vol_ann'].mean():.1f}%")
print(f"Days with VIX > GARCH          : {(df['gap']>0).sum():,} of {len(df):,} "
      f"({100*(df['gap']>0).mean():.2f}%)")
print(f"\nLargest GARCH overshoot        : {df['gap'].min():.2f} on {df['gap'].idxmin().date()}")
```

Note the two adjacent f-strings on separate lines — Python concatenates string
literals written next to each other, a tidy way to break a long line.

`.idxmin()` returns the date of the minimum, `.min()` the value. Both are needed:
one for *when*, one for *how much*.

### Cell 56 — the gap chart (Figure 7)

```python
ax.fill_between(df.index, 0, df["gap"], where=df["gap"] >= 0,
                color=BLUE, alpha=0.75, linewidth=0, label="VIX above GARCH")
ax.fill_between(df.index, 0, df["gap"], where=df["gap"] < 0,
                color=ORANGE, alpha=0.75, linewidth=0, label="GARCH above VIX")
```

`fill_between` shades between 0 and the gap. Calling it twice with complementary
`where=` masks colours positive and negative regions differently, so the
one-sidedness is visible at a glance rather than having to be inferred.

### Cell 57 (markdown) — the variance risk premium

VIX exceeds annualised GARCH volatility on **86.7%** of days, by **3.19 points on
average, about 22% of the GARCH mean**. A gap that one-sided is not estimation
error. It is the **variance risk premium**: VIX is a *risk-neutral* expectation
extracted from option prices, so it embeds the compensation investors demand for
bearing volatility risk. GARCH, estimated from realised returns alone, delivers
only the *physical* forecast. The difference between the two measures is the
premium.

Liu, Guo & Qiao (2015) report the same phenomenon: their GARCH-computed VIX ran
20–30% below the market VIX before September 2003 and 10–13% below afterwards,
interpreted the same way.

> **Correction to note.** The notebook and report say our 22% "sits in the same
> range". It does not, quite. Our sample (2012–2026) is entirely inside Liu et
> al.'s *post*-2003 regime, where their figure is 10–13% — so 22% is roughly
> double the relevant benchmark, and coincides instead with their pre-2003 range.
> They are also measuring a slightly different object: a one-day-ahead GARCH
> *forecast of VIX*, not annualised conditional volatility. The safe claim is
> that this is the same phenomenon at a larger magnitude, with the difference
> attributable to construction and sample.

**The exception is instructive.** GARCH exceeds VIX on 13.3% of days, and the
largest overshoot — 44 points on 17 March 2020 — follows directly from the
−12.77% crash of 16 March. Because σ²_t depends on ε²_{t−1}, one enormous return
is mechanically propagated into the next day's variance, pushing annualised GARCH
volatility to 120 while VIX, already pricing eventual normalisation, stood at
75.9. **GARCH extrapolates the recent past; VIX anticipates.**

### Cell 59 (code) — correlations

```python
lv = df["garch_vol_ann"].corr(df["VIX"])
sp = df["garch_vol_ann"].corr(df["VIX"], method="spearman")
lg = np.log(df["garch_vol_ann"]).corr(np.log(df["VIX"]))
```

Three views of the same relationship. **Pearson** measures linear association and
is sensitive to the 2020 outliers. **Spearman** correlates the *ranks*, so it is
robust to them — the gap between 0.83 and 0.73 is a measure of how much the
extremes are doing. The **log-log** version is what you want if the relationship
is closer to proportional than additive, which for volatility it usually is.

```python
d_g, d_v = df["garch_vol_ann"].diff(), df["VIX"].diff()
print(f"   Pearson  : {d_g.corr(d_v):.4f}")
```

Correlating **daily changes** rather than levels. Two series can be highly
correlated in levels simply because both trend or both spike in the same episodes;
correlating first differences asks the harder question of whether they move
together day to day.

```python
for k in range(-2, 4):
    print(f"   k = {k:+d} : {d_g.shift(-k).corr(d_v):+.4f}")
```

The **lead–lag profile**. `.shift(-k)` moves the GARCH change series backward by
k days, so the printed number is the correlation between the GARCH change at
`t+k` and the VIX change at `t`. Positive k asks: does GARCH move *after* VIX?

```python
for lo, hi in [(2012, 2015), (2016, 2019), (2020, 2022), (2023, 2026)]:
    m = (df.index.year >= lo) & (df.index.year <= hi)
    print(f"   {lo}-{hi} : {df.loc[m, 'garch_vol_ann'].corr(df.loc[m, 'VIX']):+.4f}")
```

`df.index.year` extracts the year from every date. The `&` combines two boolean
masks element-wise (single `&`, not `and` — `and` does not work on arrays), and
the parentheses are required because `&` binds more tightly than `>=`.

### Cell 60 (markdown) — the key result

**In levels:** Pearson 0.83, Spearman 0.73, log-log 0.79. By sub-period: 0.66
(2012–15), 0.80 (2016–19), 0.89 (2020–22), 0.70 (2023–26) — strongest in the
turbulent window, weakest in the calm stretches, which is what you expect when
both series are compressed into a narrow range.

**In daily changes:** −0.007, effectively zero. That looks contradictory until
the lead–lag profile resolves it:

| k | −2 | −1 | 0 | **+1** | +2 |
|---|---:|---:|---:|---:|---:|
| corr(ΔGARCH_{t+k}, ΔVIX_t) | +0.00 | +0.03 | −0.01 | **+0.41** | +0.03 |

Near zero contemporaneously, **+0.41 at a one-day lag**. The reason is structural,
not statistical: σ_t is a function of information dated t−1, so the GARCH series
*cannot* respond to today's news until tomorrow, while VIX is priced off today's
option quotes and responds immediately. The two agree about the *level* of
volatility, but GARCH learns about changes exactly one day after the options
market has priced them.

**Why this is the most important result of Stage 2.** A predictor that merely
tracked VIX would be redundant in a forecasting model. GARCH volatility does not:
it is strongly related in levels, reacts on a different schedule, and carries a
systematic risk-premium wedge that VIX contains and GARCH structurally cannot.
That is what makes it a candidate for *incremental* predictive power — and
Section 4.1 has already shown the own-history baseline it must beat is demanding.

---

## Section 8 — Summary and save

### Cell 61 (markdown)

Answers the five questions in the brief directly, each backed by the specific
statistic that settles it. Worth reading as a model of how to close: claim,
number, source of the number.

### Cell 62 (code)

```python
df.to_csv(DATA_DIR / "stage2_final.csv")
print(f"Saved {DATA_DIR / 'stage2_final.csv'}   ({len(df):,} rows, {df.shape[1]} columns)")
print("Columns:", list(df.columns))
```

Writes the final dataset with all derived columns: `SP500`, `VIX`, `ret`,
`garch_vol_daily`, `garch_vol_ann`, `gap`. `df.shape` is a `(rows, columns)`
tuple, so `df.shape[1]` is the column count. This file is the input to Stage 3.

---

## Appendix — questions you might be asked, and where the answer lives

| Question | Where |
|---|---|
| Why log returns rather than simple returns? | Cell 14 — additive across time, closer to symmetric |
| Why compute returns before truncating the sample? | Cell 13/14 — otherwise the first in-sample return is lost |
| Why is the VIX from CBOE but the S&P from Yahoo? | Cells 1, 8–12 — CBOE publishes no index history |
| Why 2012 and not the full CBOE history from 1990? | Matches Wen et al.'s window; avoids the Sept 2003 VIX redefinition |
| Why does CBOE have 32 days the S&P doesn't? | Cell 7 — SPX options trade on some equity holidays |
| Why unstandardised kurtosis? | Cell 20 — matches Fernandes et al., normal = 3 |
| Why is ±0.032 the white-noise band? | Cell 34 — 1.96/√N, tight because N = 3,680 |
| Stationary *and* persistent — isn't that contradictory? | Cell 38 — shocks die out, but slowly |
| Why scale returns by 100 before fitting? | Cell 44 — keeps the likelihood well conditioned |
| Where does the half-life formula come from? | Cell 45 — solve (α+β)^k = 0.5 |
| Why √252? | Cell 50 — variance scales with time, volatility with its square root |
| Why is VIX above GARCH almost always? | Cell 57 — variance risk premium (risk-neutral vs physical measure) |
| Why did GARCH hit 120 in March 2020? | Cell 57 — σ²_t depends on ε²_{t−1}; the −12.77% crash propagates |
| Levels correlate 0.83 but changes −0.007. Why? | Cell 60 — GARCH conditions on t−1, VIX on today; +0.41 at lag 1 |
| Why keep the Gaussian model when BIC prefers Student-t? | Cell 49 — the brief specifies basic GARCH(1,1); noted explicitly |
