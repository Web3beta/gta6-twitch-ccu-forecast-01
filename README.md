# GTA VI launch-day Twitch CCU forecaster (V2)

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Web3beta/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Web3beta/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), hours 2 to 24, given everything
recorded up to the end of hour 1. For each hour it outputs five quantiles
(5%, 25%, 50%, 75%, 95%) of that hour's viewer count.

Model (all in log space): the median is an equal-weight blend of three
per-hour models — a ridge regression on [ln hour-1, ln(hour-1 / pre-24h
peak)], a size-conditioned log-ratio table (median launch-day shape among
past launches of similar hour-1 size), and a global log-ratio median over
all rolling-midnight comparables. The spread is the per-hour leave-one-out
residual standard deviation of the blended median, so the 90% band lands
near nominal coverage. Final-case-only signals (minute slope inside hour 1,
top-stream concentration) are used when present, with graceful fallback.

Run locally: `MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out
python3 -P -s main.py` — writes `<case-id>.out` for the case found in the
input directory. Standard library only, deterministic, no network.
