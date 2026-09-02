**To:** Shobhana
**Cc:** Dr. Anadi
**Subject:** Stage 2 — data preparation and GARCH(1,1)

Dear Ma'am,

Please find attached the Stage 2 report, along with the Python notebook, the
figures, and the cleaned dataset. Monday 12–1 works for me.

**Sample.** As discussed, I have used 2012–2026 for both series rather than the
periods in the two base papers, which overlap by only two years. The final
dataset has 3,680 matched trading days, 3 January 2012 to 21 August 2026, with
no missing values.

**On the data source.** I checked what each paper actually uses, and none of
them resolves the question cleanly:

- Fernandes et al. (2014) state no data source anywhere in the paper.
- Wen et al. (2024) use OptionMetrics IvyDB and Tonghuashun. Both are paid
  subscriptions, and both are options databases rather than index sources, so
  neither would give us the VIX index or the S&P 500 level in any case.
- Liu et al. (2015) — the paper you sent for the GARCH estimation — address this
  directly in Section 4.1. They note that CBOE does not publish S&P 500 index
  history, so they take both series from Yahoo Finance, and report that
  comparing Yahoo's VIX against CBOE's shows "only very minor differences for a
  few days."

So the constraint is real: CBOE publishes the VIX free of charge but not the
S&P 500, and no free provider covers both.

Given your concern, I have taken the **VIX from CBOE's official file** rather
than Yahoo, and used Yahoo only for the S&P 500, where there is no free
alternative. I also ran Liu et al.'s comparison on our own sample as a check.
Across the 3,681 common dates the two providers agree exactly, to two decimal
places, on 99.76% of observations, with a correlation of 0.99997 and only nine
days differing by more than 0.01. The largest of those is 6 February 2026, where
Yahoo reports 20.37 against CBOE's 17.76 — the S&P 500 rose 1.95% that day, and
CBOE's path (21.77 → 17.76 → 17.36) is the one consistent with that rally, so
Yahoo appears to carry a bad print there. Nine bad observations in 3,681 would
not have changed the results, but since CBOE is free there seemed no reason to
accept them.

One other cleaning note: CBOE reports a VIX close on 32 dates when the NYSE cash
market was shut (US holidays, mostly from 2022 onward). These are genuine index
values, since VIX comes from SPX option quotes that trade in sessions not always
matching equity hours, but no matched return can be formed on those days, so I
have excluded them.

**Main results.**

- VIX is highly persistent: autocorrelation 0.961 at lag 1, still 0.563 after a
  month, and outside the white-noise band as far as lag 250. It is nonetheless
  stationary (ADF −5.56, p = 1.5×10⁻⁶). This is the long-memory pattern
  Fernandes et al. document.
- Volatility clustering is unambiguous. Squared returns are autocorrelated at
  0.449 at lag 1 against −0.116 for raw returns; Ljung–Box gives Q(10) = 3,955
  versus 236, and ARCH-LM returns 1,109.
- GARCH(1,1) gives ω = 0.0431, α = 0.1666, β = 0.7907, so **α + β = 0.9573** —
  stationary but persistent, with a shock half-life of about 15.9 trading days.
- Annualised GARCH volatility and VIX correlate **0.83 in levels but −0.007 in
  daily changes**. The lead–lag correlation is **+0.41 at one day**: since σ_t
  depends on information through t−1 while VIX is priced off current quotes,
  GARCH picks up the same news one day later. I think this is the most useful
  result for what comes next, since it suggests GARCH is not simply a proxy for
  VIX.
- VIX exceeds annualised GARCH volatility on 86.7% of days, by about 22% on
  average. This matches the variance risk premium Liu et al. report (20–30%
  before 2003, 10–13% after), which was a useful check that the estimation is
  behaving sensibly.

**One question for Monday.** Return kurtosis is 19.3, so I also estimated the
model with Student-t innovations as a robustness check. α is essentially
unchanged at 0.167, but persistence rises to 0.984 (ν = 5.48) and BIC prefers
that specification decisively. I have reported the Gaussian model as the main
result since the brief specified a basic GARCH(1,1), but I would like to ask
which specification we should carry into the forecasting stage.

I have not started the LSTM, as instructed.

Thank you,
Satvik
