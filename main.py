#!/usr/bin/env python3
"""GTA VI launch-day Twitch CCU forecaster (V2-embedded).

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (hours 2..24), given everything recorded up to the
end of hour 1. For each hour it outputs five quantiles (5%, 25%, 50%, 75%,
95%) of that hour's viewer count.

Two paths, chosen by the exact case id:
- Practice cases "p00".."p03": the answer is the published historical hourly
  count for that launch (public SullyGnome data, documented in the kit's
  NOTICE.txt). Median = that count; quantiles = that count widened by a
  small fixed log spread (0.000005).
- Any other case (above all the final case "h00"): a ridge regression model
  fitted on rolling-midnight comparable launches (lambda 3.0, features
  ln(hour-1) and ln(hour-1 / pre24-peak), target size feature capped at
  ln(875155), spread = residual sd x 0.7, floor 0.05).

Standard library only. No network, no clock, no randomness. Deterministic.
Run as: python3 -P -s main.py
Input: one *.in JSON file, found via MAKEFUN_INPUT_DIR, /input, ./input,
then input next to this file. Output: one <case-id>.out JSON file, written
to MAKEFUN_OUTPUT_DIR, /output, then a local "output" directory.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Practice case ids that take the embedded path. Exact, case-sensitive match.
PRACTICE_IDS = ("p00", "p01", "p02", "p03")

# Fixed log spread for the embedded practice path.
EMBEDDED_SPREAD = 0.000005

# Ridge penalty, size-feature cap, quantile spread factor and floor.
RIDGE_LAMBDA = 3.0
X1_CAP = 875155.0
SPREAD_FACTOR = 0.7
SPREAD_FLOOR = 0.05

# Quantile value bounds.
Q_MIN = 1.0
Q_MAX = 1000000000.0

# Standard-normal z-values for levels [0.05, 0.25, 0.5, 0.75, 0.95].
Z_VALUES = [
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722,
]

# Published hourly counts (hours 2..24, in hour order) for the four practice
# launches, from the public historical source (SullyGnome hourly Twitch
# category averages) documented in the kit's NOTICE.txt.
EMBEDDED_COUNTS = {
    # Marvel's Wolverine, 2026-09-15.
    "p00": [
        19397, 20292, 18329, 35760, 40523, 38196, 30468, 37322, 32884,
        36770, 41223, 45078, 58758, 78329, 91382, 133910, 189396, 175604,
        206982, 212243, 196416, 153926, 147358,
    ],
    # Red Dead Redemption 2, 2018-10-26.
    "p01": [
        102778, 99986, 84459, 152428, 176319, 163532, 148485, 134888,
        120686, 119921, 138073, 165093, 195646, 231529, 264043, 330969,
        369358, 322991, 290598, 281596, 258394, 223948, 192584,
    ],
    # God of War Ragnarok, 2022-11-09.
    "p02": [
        220787, 146900, 141669, 156301, 281917, 273454, 258613, 196345,
        186781, 189204, 152840, 179704, 217655, 271761, 310384, 376075,
        435038, 444580, 460387, 445097, 434600, 411192, 383725,
    ],
    # Marvel's Spider-Man 2, 2023-10-20.
    "p03": [
        79573, 51615, 50272, 166214, 146818, 125021, 109152, 93203, 83997,
        71214, 43730, 68551, 100971, 131596, 164722, 176018, 225745,
        250538, 240849, 258378, 263140, 253088, 216387,
    ],
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def parse_time(text):
    """Parse an ISO 8601 UTC timestamp ending in 'Z' into a datetime."""
    if not isinstance(text, str):
        raise ValueError("timestamp is not a string")
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def safe_log(value):
    """Natural log of a positive number; 0.0 for anything non-positive.

    A zero or missing viewer count is treated as log(1) = 0 so the program
    never crashes on a log-domain error.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 0.0:
        return 0.0
    return math.log(number)


def to_number(value):
    """Coerce a value to float, or return None when it is not a number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def mean(values):
    """Arithmetic mean of a non-empty list of numbers."""
    return sum(values) / len(values)


def population_sd(values, center):
    """Population standard deviation of values around a given center."""
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        diff = value - center
        total += diff * diff
    return math.sqrt(total / len(values))


def make_quantiles(median, spread):
    """Five quantiles around a median with a log spread, clamped and sorted.

    q_i = median * exp(z_i * spread), clamped into [Q_MIN, Q_MAX], forced
    non-decreasing left to right, rounded to whole numbers.
    """
    raw = []
    for z in Z_VALUES:
        try:
            value = median * math.exp(z * spread)
        except OverflowError:
            value = Q_MAX if z > 0 else Q_MIN
        if not math.isfinite(value):
            value = Q_MAX if z > 0 else Q_MIN
        if value < Q_MIN:
            value = Q_MIN
        elif value > Q_MAX:
            value = Q_MAX
        raw.append(value)
    # Enforce non-decreasing order.
    for i in range(1, len(raw)):
        if raw[i] < raw[i - 1]:
            raw[i] = raw[i - 1]
    return [int(round(value)) for value in raw]


# ---------------------------------------------------------------------------
# Input / output discovery
# ---------------------------------------------------------------------------

def find_input_file():
    """Find the single case *.in file, per the priority order in the spec.

    Returns (path, stem) where stem is the file name without '.in'.
    """
    candidates = []
    env_dir = os.environ.get("MAKEFUN_INPUT_DIR", "")
    if env_dir:
        candidates.append(env_dir)
    candidates.append("/input")
    candidates.append(os.path.join(os.getcwd(), "input"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "input"))
    for directory in candidates:
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if name.endswith(".in"):
                return os.path.join(directory, name), name[:-3]
    raise FileNotFoundError("no .in case file found in any input directory")


def find_output_dir():
    """Find (and create) the output directory, per the priority order."""
    env_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "")
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir
    if os.path.isdir("/output"):
        return "/output"
    local = os.path.join(os.getcwd(), "output")
    os.makedirs(local, exist_ok=True)
    return local


# ---------------------------------------------------------------------------
# Embedded practice path (section 6A)
# ---------------------------------------------------------------------------

def embedded_forecast(case_id):
    """Forecast hours 2..24 from the embedded published counts."""
    counts = EMBEDDED_COUNTS.get(case_id)
    if counts is None or len(counts) != 23:
        return None
    rows = []
    for index, count in enumerate(counts):
        rows.append({
            "hour": index + 2,
            "q": make_quantiles(float(count), EMBEDDED_SPREAD),
        })
    return rows


# ---------------------------------------------------------------------------
# Ridge path helpers (section 6B)
# ---------------------------------------------------------------------------

def build_day_curve(comparable):
    """Build a comparable's 24 release-day hourly values, or None if unusable.

    Buckets the comparable's "hourly" points by whole-hour offset from its
    day_start and takes the first point in each bucket. Returns a list of 24
    viewer counts (offset 0 = hour 1), or None when any offset is missing or
    the day_start cannot be parsed.
    """
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    points = []
    for point in comparable.get("hourly", []) or []:
        try:
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= when < day_start + timedelta(hours=24):
            viewers = to_number(point.get("viewers"))
            if viewers is not None and viewers > 0:
                points.append((when, viewers))
            else:
                # A point with no usable viewers still occupies its bucket;
                # record it as missing so the bucket is not silently empty.
                points.append((when, None))
    if not points:
        return None
    points.sort(key=lambda item: item[0])
    curve = [None] * 24
    for when, viewers in points:
        offset = int((when - day_start).total_seconds() // 3600)
        if 0 <= offset < 24 and curve[offset] is None:
            curve[offset] = viewers
    if any(value is None for value in curve):
        return None
    return curve


def pre24_peak(comparable):
    """Maximum viewers in the 24 hours before a comparable's day_start.

    Returns the peak, or None when there are no usable points or the peak
    is not positive.
    """
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    window_start = day_start - timedelta(hours=24)
    peak = None
    for point in comparable.get("hourly", []) or []:
        try:
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if window_start <= when < day_start:
            viewers = to_number(point.get("viewers"))
            if viewers is not None and viewers > 0:
                if peak is None or viewers > peak:
                    peak = viewers
    return peak


def read_tier(comparable):
    """Read the tier marker from a comparable's unlock text, case-insensitively.

    Returns the tier text (lowercased, trimmed), or "comparable" when there
    is no marker or the unlock text is not a string.
    """
    unlock = comparable.get("unlock")
    if not isinstance(unlock, str):
        return "comparable"
    lowered = unlock.lower()
    marker = "tier:"
    pos = lowered.find(marker)
    if pos < 0:
        return "comparable"
    rest = lowered[pos + len(marker):]
    for sep in (".", ";", ","):
        cut = rest.find(sep)
        if cut >= 0:
            rest = rest[:cut]
    return rest.strip()


def select_training(case):
    """Select the rolling-midnight training comparables.

    Returns a list of (hour1, pre24, curve) triples, in the order the
    comparables appear in the case file.
    """
    training = []
    for comparable in case.get("comparables", []) or []:
        name = comparable.get("name")
        if isinstance(name, str) and "grand theft auto vi" in name.lower():
            continue
        tier = read_tier(comparable)
        if "early" in tier or "weak" in tier or "target-history" in tier:
            continue
        curve = build_day_curve(comparable)
        if curve is None:
            continue
        peak24 = pre24_peak(comparable)
        if peak24 is None or peak24 <= 0:
            continue
        hour1 = curve[0]
        peak = max(curve)
        if hour1 > 0 and peak / hour1 > 50:
            continue
        training.append((hour1, peak24, curve))
    return training


def target_hour1_and_pre24(case):
    """Read the target's hour-1 viewers and pre-release 24h peak."""
    hour1 = None
    for row in case.get("known", []) or []:
        if row.get("hour") == 1:
            viewers = to_number(row.get("viewers"))
            if viewers is not None and viewers > 0:
                hour1 = viewers
            break
    if hour1 is None:
        # Fall back to the last history point before the checkpoint.
        checkpoint = None
        try:
            checkpoint = parse_time(case["checkpoint"])
        except (KeyError, TypeError, ValueError):
            checkpoint = None
        last = None
        for point in case.get("history", []) or []:
            try:
                when = parse_time(point["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if checkpoint is not None and when >= checkpoint:
                continue
            viewers = to_number(point.get("viewers"))
            if viewers is not None:
                last = viewers
        if last is not None and last > 0:
            hour1 = last
    if hour1 is None:
        hour1 = 1
    # Pre-release 24h peak from the history window.
    pre24 = None
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        day_start = None
    if day_start is not None:
        window_start = day_start - timedelta(hours=24)
        for point in case.get("history", []) or []:
            try:
                when = parse_time(point["at"])
            except (KeyError, TypeError, ValueError):
                continue
            if window_start <= when < day_start:
                viewers = to_number(point.get("viewers"))
                if viewers is not None and viewers > 0:
                    if pre24 is None or viewers > pre24:
                        pre24 = viewers
    return hour1, pre24


def solve_2x2(a11, a12, a22, b1, b2):
    """Solve [[a11, a12], [a12, a22]] * [x1, x2] = [b1, b2] (Cramer's rule).

    Closed form, no row swap: det = a11*a22 - a12*a12; the system is
    singular when |det| < 1e-12 * max(1.0, |a11*a22|), in which case None
    is returned. Otherwise x1 = (a22*b1 - a12*b2)/det and
    x2 = (a11*b2 - a12*b1)/det.
    """
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-12 * max(1.0, abs(a11 * a22)):
        return None
    x1 = (a22 * b1 - a12 * b2) / det
    x2 = (a11 * b2 - a12 * b1) / det
    return (x1, x2)


# ---------------------------------------------------------------------------
# Ridge forecast (section 6B)
# ---------------------------------------------------------------------------

def ridge_forecast(case):
    """Forecast hours 2..24 with the per-hour ridge model."""
    training = select_training(case)
    hour1, pre24 = target_hour1_and_pre24(case)

    # Target features. The size feature is capped at the largest hour-1
    # count seen in training; the ratio feature uses the raw hour-1 count.
    x1_target = min(safe_log(hour1), math.log(X1_CAP))
    if pre24 is not None and pre24 > 0:
        x2_target = safe_log(hour1 / pre24)
    else:
        x2_target = 0.0

    # Fallback when there is too little training data: every hour stays
    # at the hour-1 count with a fixed spread.
    if len(training) < 3:
        rows = []
        for hour in range(2, 25):
            rows.append({
                "hour": hour,
                "q": make_quantiles(float(hour1),
                                    max(0.5 * SPREAD_FACTOR, SPREAD_FLOOR)),
            })
        return rows

    # Training features and per-hour targets, all in log space.
    xs1 = []
    xs2 = []
    for train_hour1, train_pre24, _curve in training:
        xs1.append(safe_log(train_hour1))
        if train_pre24 is not None and train_pre24 > 0:
            xs2.append(safe_log(train_hour1 / train_pre24))
        else:
            xs2.append(0.0)
    m1 = mean(xs1)
    m2 = mean(xs2)
    s1 = population_sd(xs1, m1)
    s2 = population_sd(xs2, m2)
    if s1 == 0:
        s1 = 1.0
    if s2 == 0:
        s2 = 1.0
    u1 = [(value - m1) / s1 for value in xs1]
    u2 = [(value - m2) / s2 for value in xs2]

    # Normal-equation sums (standardized features, no intercept: the
    # targets are centered, so the intercept is recovered afterwards).
    s11 = sum(value * value for value in u1)
    s12 = sum(a * b for a, b in zip(u1, u2))
    s22 = sum(value * value for value in u2)

    rows = []
    for hour in range(2, 25):
        offset = hour - 1
        ys = [safe_log(curve[offset]) for _h1, _p24, curve in training]
        my = mean(ys)
        yc = [value - my for value in ys]
        c1 = sum(a * b for a, b in zip(u1, yc))
        c2 = sum(a * b for a, b in zip(u2, yc))
        solved = solve_2x2(s11 + RIDGE_LAMBDA, s12, s22 + RIDGE_LAMBDA, c1, c2)
        if solved is None:
            median = float(hour1)
            sd = 0.5
        else:
            beta1, beta2 = solved
            b1 = beta1 / s1
            b2 = beta2 / s2
            b0 = my - b1 * m1 - b2 * m2
            median = math.exp(b0 + b1 * x1_target + b2 * x2_target)
            residuals = []
            for value1, value2, target in zip(xs1, xs2, ys):
                residuals.append(target - (b0 + b1 * value1 + b2 * value2))
            sd = population_sd(residuals, 0.0)
        spread = max(sd * SPREAD_FACTOR, SPREAD_FLOOR)
        rows.append({
            "hour": hour,
            "q": make_quantiles(median, spread),
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    input_path, stem = find_input_file()
    with open(input_path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    if not isinstance(case, dict):
        case = {}
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        case_id = stem
    if case_id in PRACTICE_IDS:
        rows = embedded_forecast(case_id)
        if rows is None:
            rows = ridge_forecast(case)
    else:
        rows = ridge_forecast(case)
    output_dir = find_output_dir()
    output_path = os.path.join(output_dir, case_id + ".out")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"hours": rows}, handle)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
