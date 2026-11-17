#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour (V2).

Reads one case (schema makefun/twitch-ccu-case/v1): everything recorded up to
the end of hour 1 of the release day, hour 1's viewer count included. Writes a
forecast for hours 2..24, five quantiles (5%, 25%, 50%, 75%, 95%) per hour.

Model (all in log space, natural log of viewer counts):
  Median: equal-weight blend of two per-hour models --
    (A) ridge regression on [ln hour-1, ln(hour-1 / pre-24h peak)], lambda 1.0;
    (B) k-nearest-neighbour log-ratio median: among the 5 comparables whose
        ln(hour-1) is closest to the target's, take the median of
        ln(curve[h] / curve[0]) and add ln(hour-1). A global or tercile
        ratio median was measured and rejected (it badly underpredicts
        late-peak launches; see DESIGN_SPEC_V2.md).
  Spread: per-hour leave-one-out MAD (median absolute deviation of the
  blended median's leave-one-out residuals, x1.4826 for normal
  consistency), so the 90% band lands near nominal 0.90 coverage. A raw
  LOO standard deviation was measured too wide; the MAD is robust to the
  few comparables with unusual shapes.
  Final-case-only signals (minute slope inside hour 1, top-stream
  concentration) are used when present, with graceful fallback when absent.

Training data: rolling-midnight past launches from the case's comparables
(tier filter, peak/hour-1 filter, exactly-24-value curve requirement).

Standard library only. Deterministic. No network, no clock, no randomness.
Writes only the forecast output file; never writes into the repository.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]
RIDGE_LAMBDA = 1.0
MIN_SPREAD = 0.05
MAX_SPREAD = 1.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp ending in Z."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def median_of(values):
    """Median of a non-empty list of numbers."""
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def pop_sd(values, mean):
    """Population standard deviation around a given mean."""
    n = len(values)
    if n == 0:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def solve2(mat, rhs):
    """Solve a 2x2 linear system by Gaussian elimination with partial pivoting.

    mat is [[a, b], [c, d]], rhs is [e, f]. Returns (x1, x2) or None if
    the system is singular.
    """
    a, b = mat[0]
    c, d = mat[1]
    e, f = rhs
    if abs(c) > abs(a):
        a, b, e, c, d, f = c, d, f, a, b, e
    if a == 0.0:
        return None
    factor = c / a
    d2 = d - factor * b
    f2 = f - factor * e
    if d2 == 0.0:
        return None
    x2 = f2 / d2
    x1 = (e - b * x2) / a
    return (x1, x2)


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


# --------------------------------------------------------------------------
# Input / output discovery
# --------------------------------------------------------------------------

def find_input_dir():
    """First candidate directory that exists and holds at least one .in file."""
    cands = []
    env = os.environ.get("MAKEFUN_INPUT_DIR")
    if env:
        cands.append(env)
    cands.append("/input")
    cands.append("./input")
    here = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.join(here, "input"))
    for d in cands:
        if not d or not os.path.isdir(d):
            continue
        try:
            names = os.listdir(d)
        except OSError:
            continue
        if any(n.endswith(".in") for n in names):
            return d
    return None


def find_output_dir():
    """MAKEFUN_OUTPUT_DIR, else /output if it exists, else ./output."""
    env = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if env:
        return env
    if os.path.isdir("/output"):
        return "/output"
    return "output"


def load_case(in_dir):
    """Load the first .in file (sorted name order) from the input directory."""
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    if not names:
        return None, None
    path = os.path.join(in_dir, names[0])
    with open(path, "r", encoding="utf-8") as f:
        case = json.load(f)
    case_id = case.get("id")
    if not case_id:
        case_id = os.path.splitext(names[0])[0]
    return case, case_id


# --------------------------------------------------------------------------
# Comparable selection and curve building
# --------------------------------------------------------------------------

def build_curve(comp):
    """Build a comparable's 24-value release-day curve, or None if unusable.

    Takes the hourly points in [day_start, day_start + 24h), buckets them by
    whole-hour offset, and takes the FIRST point in each bucket. Needs exactly
    one value per offset 0..23, all positive.
    """
    day_start = parse_ts(comp["day_start"])
    end = day_start + timedelta(hours=24)
    pts = []
    for p in comp.get("hourly", []):
        try:
            t = parse_ts(p["from"])
        except (KeyError, ValueError):
            continue
        if day_start <= t < end:
            pts.append((t, p["viewers"]))
    pts.sort(key=lambda x: x[0])
    vals = [None] * 24
    for t, v in pts:
        off = int((t - day_start).total_seconds() // 3600)
        if 0 <= off < 24 and vals[off] is None:
            vals[off] = v
    if any(v is None or v <= 0 for v in vals):
        return None
    return vals


def pre24_peak_from_hourly(hourly, day_start):
    """Max viewers in [day_start - 24h, day_start), or None if none/<=0."""
    lo = day_start - timedelta(hours=24)
    best = None
    for p in hourly:
        try:
            t = parse_ts(p["from"])
        except (KeyError, ValueError):
            continue
        if lo <= t < day_start:
            v = p.get("viewers")
            if v is not None and (best is None or v > best):
                best = v
    if best is None or best <= 0:
        return None
    return best


def read_tier(unlock):
    """Read the tier marker from an unlock string, or 'comparable' if none."""
    idx = unlock.find("Tier:")
    if idx < 0:
        return "comparable"
    rest = unlock[idx + len("Tier:"):]
    end = len(rest)
    for ch in (".", ";", ","):
        pos = rest.find(ch)
        if 0 <= pos < end:
            end = pos
    return rest[:end].strip()


def select_comparables(case):
    """Keep only the rolling-midnight comparables usable for training.

    Excludes: the GTA VI own-history entry (by name), tier text containing
    'early', 'weak' or 'target-history', launches whose release-day peak is
    more than 50x their hour-1 value, and launches with no valid 24-value
    curve. Returns a list of dicts: curve (24 values), h1, pre24.
    """
    usable = []
    for comp in case.get("comparables", []):
        name = (comp.get("name") or "").lower()
        if "grand theft auto vi" in name:
            continue
        tier = read_tier(comp.get("unlock") or "")
        tl = tier.lower()
        if "early" in tl or "weak" in tl or "target-history" in tl:
            continue
        curve = build_curve(comp)
        if curve is None:
            continue
        h1 = curve[0]
        peak = max(curve)
        if h1 > 0 and peak / h1 > 50:
            continue
        day_start = parse_ts(comp["day_start"])
        p24 = pre24_peak_from_hourly(comp.get("hourly", []), day_start)
        usable.append({"curve": curve, "h1": h1, "pre24": p24})
    return usable


def knn_ratio_median(train, hour, x1_t, k=5):
    """Component B: k-nearest-neighbour log-ratio median.

    Among the k comparables whose ln(hour-1) is closest to the target's,
    take the median of ln(curve[hour-1] / curve[0]) and add ln(hour-1).
    Returns None when fewer than 3 ratios are available.
    """
    scored = sorted(train, key=lambda t: abs(math.log(t["h1"]) - x1_t))
    ratios = [math.log(t["curve"][hour - 1] / t["curve"][0])
              for t in scored[:k]]
    if len(ratios) < 3:
        return None
    return x1_t + median_of(ratios)


def blend_log_median(train, hour, x1_t, x2_t):
    """Equal-weight blend of the available median components (log space).

    Averages (A) ridge and (B) kNN log-ratio median when both are available;
    falls back to whichever one is available. Returns None only when neither
    is available.
    """
    parts = []
    a = ridge_log_median(train, hour, x1_t, x2_t)
    if a is not None:
        parts.append(a)
    b = knn_ratio_median(train, hour, x1_t)
    if b is not None:
        parts.append(b)
    if not parts:
        return None
    return sum(parts) / len(parts)


def loo_spread(train, hour):
    """Per-hour spread: LOO MAD of the blended median's residuals.

    For each training comparable i, refit the blended median without i and
    predict i's own log value; the residual is actual minus predicted.
    The spread is 1.4826 x the median absolute deviation of those residuals
    around their median (a robust, outlier-resistant scale estimate),
    clamped to [MIN_SPREAD, MAX_SPREAD]. Returns 0.5 when too few
    residuals exist.
    """
    residuals = []
    for i in range(len(train)):
        subset = train[:i] + train[i + 1:]
        ti = train[i]
        x1_i = math.log(ti["h1"])
        if ti["pre24"] is not None and ti["pre24"] > 0:
            x2_i = math.log(ti["h1"] / ti["pre24"])
        else:
            x2_i = 0.0
        pred = blend_log_median(subset, hour, x1_i, x2_i)
        if pred is None:
            continue
        actual = math.log(ti["curve"][hour - 1])
        residuals.append(actual - pred)
    if len(residuals) < 2:
        return 0.5
    med = median_of(residuals)
    mad = median_of([abs(r - med) for r in residuals])
    return clamp(1.4826 * mad, MIN_SPREAD, MAX_SPREAD)


# --------------------------------------------------------------------------
# Median model components (all in log space)
# --------------------------------------------------------------------------

def ridge_log_median(train, hour, x1_t, x2_t):
    """Component A: per-hour ridge regression on [ln h1, ln(h1/pre24)].

    Fits ln(curve[hour-1]) ~ b0 + b1*x1 + b2*x2 over the training set with
    penalty lambda, on standardized features, solved by 2x2 Gaussian
    elimination. Returns the predicted log-median for the target, or None
    when there are fewer than 3 training points or the system is singular.
    """
    xs1, xs2, ys = [], [], []
    for t in train:
        xs1.append(math.log(t["h1"]))
        if t["pre24"] is not None and t["pre24"] > 0:
            xs2.append(math.log(t["h1"] / t["pre24"]))
        else:
            xs2.append(0.0)
        ys.append(math.log(t["curve"][hour - 1]))
    n = len(ys)
    if n < 3:
        return None
    m1 = sum(xs1) / n
    m2 = sum(xs2) / n
    my = sum(ys) / n
    s1 = pop_sd(xs1, m1) or 1.0
    s2 = pop_sd(xs2, m2) or 1.0
    u1 = [(v - m1) / s1 for v in xs1]
    u2 = [(v - m2) / s2 for v in xs2]
    yc = [v - my for v in ys]
    a11 = sum(a * a for a in u1) + RIDGE_LAMBDA
    a12 = sum(a * b for a, b in zip(u1, u2))
    a22 = sum(b * b for b in u2) + RIDGE_LAMBDA
    r1 = sum(a * c for a, c in zip(u1, yc))
    r2 = sum(b * c for b, c in zip(u2, yc))
    sol = solve2([[a11, a12], [a12, a22]], [r1, r2])
    if sol is None:
        return None
    beta1, beta2 = sol
    b1 = beta1 / s1
    b2 = beta2 / s2
    b0 = my - b1 * m1 - b2 * m2
    return b0 + b1 * x1_t + b2 * x2_t


# --------------------------------------------------------------------------
# Target features and final-case-only signals
# --------------------------------------------------------------------------

def target_hour1(case):
    """Hour-1 viewer count: known row, else last history point, else 1."""
    for row in case.get("known", []):
        if row.get("hour") == 1:
            v = row.get("viewers")
            if v is not None and v > 0:
                return v
    hist = case.get("history", [])
    if hist:
        v = hist[-1].get("viewers")
        if v is not None and v > 0:
            return v
    return 1


def target_pre24(case):
    """Target's pre-24h peak: max history viewers in [day_start-24h, day_start)."""
    day_start = parse_ts(case["day_start"])
    lo = day_start - timedelta(hours=24)
    best = None
    for p in case.get("history", []):
        try:
            t = parse_ts(p["at"])
        except (KeyError, ValueError):
            continue
        if lo <= t < day_start:
            v = p.get("viewers")
            if v is not None and (best is None or v > best):
                best = v
    if best is None or best <= 0:
        return None
    return best


def minute_slope(case):
    """Log-slope per hour of viewers inside hour 1, or None if unavailable.

    Uses history points in [checkpoint - 60 min, checkpoint). Requires at
    least 6 points spanning at least 30 minutes; practice cases have ~1
    point there (hourly spacing), so this returns None on practice and only
    fires on the final case's minute reads. Bounded by the caller.
    """
    checkpoint = parse_ts(case["checkpoint"])
    lo = checkpoint - timedelta(minutes=60)
    pts = []
    for p in case.get("history", []):
        try:
            t = parse_ts(p["at"])
        except (KeyError, ValueError):
            continue
        if lo <= t < checkpoint:
            v = p.get("viewers")
            if v is not None and v > 0:
                pts.append((t, math.log(v)))
    if len(pts) < 6:
        return None
    span = (pts[-1][0] - pts[0][0]).total_seconds()
    if span < 1800:
        return None
    t0 = pts[0][0]
    xs = [(t - t0).total_seconds() / 3600.0 for t, _ in pts]
    ys = [v for _, v in pts]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def stream_concentration(case):
    """Top-1 stream's share of top-20 stream viewers, or None if absent."""
    streams = case.get("streams")
    if not streams:
        return None
    vs = []
    for s in streams:
        v = s.get("viewers")
        if v is not None and v > 0:
            vs.append(v)
    if not vs:
        return None
    total = sum(vs)
    if total <= 0:
        return None
    return max(vs) / total


# --------------------------------------------------------------------------
# Main forecast
# --------------------------------------------------------------------------

def forecast(case):
    """Build the 23 forecast rows (hours 2..24) for one case."""
    h1 = target_hour1(case)
    pre24 = target_pre24(case)
    x1_t = math.log(h1) if h1 > 0 else 0.0
    x2_t = math.log(h1 / pre24) if pre24 else 0.0
    train = select_comparables(case)
    slope = minute_slope(case)
    conc = stream_concentration(case)

    rows = []
    for hour in range(2, 25):
        log_med = blend_log_median(train, hour, x1_t, x2_t)
        if log_med is None:
            log_med = math.log(h1) if h1 > 0 else 0.0
        # Final-case-only signal: minute slope inside hour 1, bounded and
        # decaying; only hours 2..6, at most +-0.125 log at hour 2.
        if slope is not None and 2 <= hour <= 6:
            s = clamp(slope, -0.5, 0.5)
            log_med += s * 0.25 * (7 - hour) / 5.0
        spread = loo_spread(train, hour)
        # Final-case-only signal: a very concentrated top stream widens the
        # band a little (one streamer leaving moves the category a lot).
        if conc is not None and conc > 0.5:
            spread = clamp(spread * 1.1, MIN_SPREAD, MAX_SPREAD)
        median = math.exp(log_med)
        qs = []
        for z in Z_VALUES:
            q = median * math.exp(z * spread)
            q = clamp(q, 1.0, 1000000000.0)
            qs.append(q)
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        qs = [round(q) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": hour, "q": qs})
    return {"hours": rows}


def main():
    in_dir = find_input_dir()
    if in_dir is None:
        sys.stderr.write("no input directory with a .in file found\n")
        return 1
    case, case_id = load_case(in_dir)
    if case is None:
        sys.stderr.write("no case file found\n")
        return 1
    out = forecast(case)
    out_dir = find_output_dir()
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        sys.stderr.write("cannot create output directory: %s\n" % exc)
        return 1
    out_path = os.path.join(out_dir, case_id + ".out")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
