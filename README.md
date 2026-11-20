# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tatsuyakenta77-arch/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tatsuyakenta77-arch/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, offline forecaster for every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, built for make.fun's ["Forecast how many GTA VI players earn each trophy in week one" bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250 in USDC).

## The model

`main.ts` (Node 22, standard library only) reads a case file and writes five quantiles (5/25/50/75/95%) of each trophy's earn rate at the target, scored by pinball loss on the logit scale.

- **Logit-trend fit (primary):** per trophy, `logit(rate)` is regressed on `ln(1 + hours since release)`. Log-time is the right clock for a launch — rates move fast in the first hours, then decelerate. Slopes are shrunk 25% toward the trophy tier's median slope and clamped, and the forecast averages the OLS prediction with a last-anchored prediction (last logit + slope × distance), which protects against Exophase's stepwise refreshes biasing the intercept.
- **Count-space nudge:** `ln(earners)` and `ln(players tracked)` are fitted separately and the implied rate is blended in at 25% weight, deviation clamped to ±0.75 logit — this is what lets a trophy's rate *fall* when new, less-progressed players flood in faster than existing players earn it (platinum, long grinds).
- **Honest spreads:** the logit sigma combines the random-walk term (0.5·√horizon in days), slope standard error × extrapolation distance, fit residual variance, and a refresh term; clamped to [0.30, 1.80].
- **Stale-snapshot branch:** when the whole history is one identical snapshot spanning ≥ 2 h, the last read is a stale cohort, not the present — Exophase sat unchanged at 638 players on launch night, then refreshed to 3,868 with every rate but the 100% one lower. The median is cut (bronze ×0.30, silver/gold ×0.22, platinum ×0.18) with a wide spread.
- The 100% opening-mission trophy gets a near-100 hedge `[99.5, 99.9, 99.98, 100, 100]`.

## Validation

On the kit's practice cases (p00/p01): 55 rows, non-decreasing quantiles in 0–100, 2,736-byte outputs, byte-identical across repeated runs, ~0.15 s runtime (limit 8 s). Against a practice answer reconstructed from the house's documented refresh (3,868 players, 4,276 trophies earned, new earners proportional to old counts), the forecaster scores ≈ 0.24 pinball loss vs ≈ 1.40 for the house baseline. On a synthetic final-like case (two days of stepwise-refresh history, 4.5-day horizon, known ground truth) it scores 0.31 vs the baseline's 0.43.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

This reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory. No network, no randomness, no dependencies — the same input always produces the same bytes.

---

Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
