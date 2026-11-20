# 5-minute presentation — slide plan and script

Slides: Zhang Qi. Recording: all four together in one online session (Zoom/Teams), each presenting the
slides of their own part; D records, edits and uploads. Speakers: Zhang Qi slides 1–2, B slide 3, C slide 5,
D slides 4 and 6 (C may join slide 4 for trees).

Replace every **[X]** with your numbers. About 650 spoken words in total.

### Slide 1 — Question (Zhang Qi, 0:00–0:35)
> Expected stock returns are the core object of asset pricing, but they are buried in noise. Gu, Kelly and Xiu ask whether
> machine learning can measure them better. We replicated their study: 920 predictors built from 94 firm characteristics,
> 8 macro variables and industry dummies, for all U.S. stocks from 1957 to 2016.

### Slide 2 — Design (Zhang Qi, 0:35–1:15)
*Visual:* timeline of the recursive scheme; feature construction $z = x \otimes c$.
> Every year from 1987 to 2016 we re-train each model on all data up to 13 years earlier, tune it on the following 12 years,
> and forecast the next year — 30 genuinely out-of-sample years. Characteristics are ranked into −1 to 1, and each one is
> interacted with the macro state, so effects can change over time.

### Slide 3 — Linear models (B, 1:15–2:15)
*Visual:* R² bars for OLS, OLS+H, OLS-3+H, ENet+H, PCR, PLS.
> Plain OLS with 920 inputs over-fits: its out-of-sample R² is [X] percent. Huber loss helps by [X]. Regularisation and
> dimension reduction turn the sign: elastic net, PCR and PLS reach [X], [X] and [X] percent, with only [X] components —
> the linear signal is low-dimensional.

### Slide 4 — Trees and networks (C + D, 2:15–3:30)
*Visual:* full R² table vs. paper; DM highlights.
> Allowing non-linearity helps: random forest and boosted trees reach [X] and [X]; neural networks [X] to [X], best is [X].
> Deeper is not better beyond [X] layers. Compared with the paper's [X] for NN3 we are [close / lower because …]. Gains are
> [larger] for [small / large] stocks, and the DM tests show [X] significantly beats [X].

### Slide 5 — What drives the forecasts (C, 3:30–4:20)
*Visual:* variable-importance heatmap.
> All models agree on a small set of signals: [short-term reversal, momentum, size/liquidity and volatility measures].
> Accounting valuation ratios matter much less. Among macro variables, [X] matters most.

### Slide 6 — Takeaways (D, 4:20–5:00)
> Three takeaways: regularisation is essential with many predictors; non-linear models add [X]% of R² on top of the best
> linear model; and the dominant predictors are price-based. Limitations: [fewer ensembles, re-tuning every five years].
> Code is on our GitHub. Thank you.
