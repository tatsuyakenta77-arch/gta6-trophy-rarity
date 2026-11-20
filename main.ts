// gta6-trophy-rarity-forecast-01: GTA VI (PS5) trophy earn-rate forecaster.
//
// Model (Node 22, standard library only, deterministic, offline):
//   The primary forecast is a per-trophy logit-trend fit (A), with a
//   count-space fit (B) blended in as a small bounded nudge.
//   (A) Logit-trend fit.  logit(rate) is regressed (weighted least squares)
//       on x = ln(1 + hours since the game's release, 2026-11-19T00:00Z).
//       Log-time is the right clock for a launch: earn rates move fast in
//       the first hours and then decelerate, so a line in ln-time captures
//       the slowdown that a line in raw time would miss.  The slope is
//       shrunk 25% toward the median slope of the trophy's tier (platinum /
//       gold / silver / bronze) and clamped, because a single trophy's
//       handful of distinct reads is noisy.  The forecast is the average of
//       the OLS prediction at the target and a "last-anchored" prediction
//       (last observed logit + slope * dx), which protects against an
//       intercept biased by Exophase's stepwise refreshes.
//
//   (B) Count-space fit.  Earn rates are ratios: rate = earners / players
//       tracked.  ln(earners) and ln(players tracked) are each regressed on
//       the same x and the target rate is exp(lnE - lnN).  It is blended in
//       at 25% weight with its deviation from (A) clamped to +/-0.75 logit,
//       because player growth is convex in log-time and extrapolating both
//       lines unchecked can badly overstate a rate.  It is what lets a
//       trophy's rate FALL when new, less-progressed players flood in faster
//       than existing players earn the trophy (the platinum, the long
//       grinds), instead of forcing every rate up.
//
//   Quantile spread (logit sigma) combines the baseline random-walk term
//   (0.5 * sqrt(horizon in days)), the slope's standard error times the
//   extrapolation distance, the fit's residual variance, and a refresh
//   term for Exophase's irregular refreshes.  It is clamped to [0.30, 1.80]
//   logit units, so a 4.5-day horizon stays honestly wide while a 90-minute
//   horizon stays tight.
//
//   Stale-snapshot (flat-history) case.  Exophase refreshes a game's counts
//   on its own schedule, not with every player: on launch night the page
//   sat unchanged at 638 players tracked for hours, then refreshed to
//   3,868 players with every rate but the one at 100% lower, because the
//   newly tracked players had mostly only finished the opening mission.
//   So when the WHOLE history is one identical snapshot spanning >= 2 h,
//   the last read is a stale cohort, not the present: the median is cut
//   (bronze x0.30, silver/gold x0.22, platinum x0.18 -- the rarest trophies
//   dilute most, the commonest least) with a wide sigma of 1.10 logit.
//   This branch can only fire on a perfectly flat history, which the
//   final case (two days of reads with refreshes) cannot have.  The wide
//   sigma of 0.90 logit reflects that the refresh's size is not in the case
//   file: the median is a best guess at the dilution, the band covers a
//   refresh half or twice as large.
//
//   The 100% trophy ("Just Like Paradise", the opening mission) is
//   special-cased with a near-100 hedge [99.5, 99.9, 99.98, 100, 100]:
//   it was 100% in every read, and every tracked player must have played
//   the opening to be tracked at all, but a small lower tail hedges a
//   refresh that brings in players who have not played it yet.
//
// Output: /output/<id>.out = {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]},
// one row per trophy, quantiles non-decreasing, 0..100, <= 65,536 bytes.
// Reads $MAKEFUN_INPUT_DIR else /input/, writes $MAKEFUN_OUTPUT_DIR else
// /output/.  No network, no randomness, no clock reads: same bytes on
// repeated runs.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: the origin of the log-time clock. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
  tier: string;
}

interface HistoryRead {
  at?: string;
  slot?: string;
  players_tracked?: number;
  rates: Array<number | null>;
}

interface Case {
  id?: string;
  at?: string;
  target?: string;
  levels: number[];
  horizon_minutes: number;
  trophies: Trophy[];
  history: HistoryRead[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clamp01 = (x: number): number => Math.min(100, Math.max(0, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const clampNum = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

/** Hours from the release to an ISO instant (falls back to the slot). */
function hoursSinceRelease(read: HistoryRead, fallbackMs: number): number {
  const ms = Date.parse(read.at ?? '') || Date.parse(read.slot ?? '') || fallbackMs;
  return Math.max(0, (ms - RELEASE_MS) / 3_600_000);
}

interface Point {
  x: number;
  y: number;
  w: number;
}

/** Weighted least squares y ~ a + b*x. Returns slope, intercept, residual
 *  variance and the slope's standard error. */
function wls(points: Point[]): {
  a: number;
  b: number;
  residVar: number;
  seB: number;
  n: number;
} {
  const n = points.length;
  let sw = 0;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sw += p.w;
    sx += p.w * p.x;
    sy += p.w * p.y;
  }
  const mx = sx / sw;
  const my = sy / sw;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    sxx += p.w * (p.x - mx) * (p.x - mx);
    sxy += p.w * (p.x - mx) * (p.y - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let rss = 0;
  for (const p of points) {
    const r = p.y - (a + b * p.x);
    rss += p.w * r * r;
  }
  const dof = Math.max(1, n - 2);
  // Residual variance on the unweighted scale (weights are ~1, so this is
  // the familiar RSS / dof, rescaled by the mean weight).
  const meanW = sw / n;
  const residVarU = rss / (meanW * dof);
  const seB = sxx > 1e-12 ? Math.sqrt(Math.max(0, residVarU) / sxx) : 0;
  return { a, b, residVar: residVarU, seB, n };
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((p, q) => p - q);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const horizonDays = c.horizon_minutes / 1440;
  const targetMs = Date.parse(c.target ?? '') || Date.parse(c.at ?? '') || RELEASE_MS;
  const xTarget = Math.log(1 + Math.max(0, (targetMs - RELEASE_MS) / 3_600_000));
  const checkpointMs = Date.parse(c.at ?? '') || RELEASE_MS;

  const history = (c.history ?? []).filter((h) => Array.isArray(h.rates));

  // Distinct snapshots: collapse consecutive identical reads (same players
  // tracked and same rates) -- Exophase's flat stretches between refreshes
  // carry no information and would otherwise overweight a stale page.
  interface Snap {
    x: number;
    ms: number;
    players: number;
    rates: Array<number | null>;
  }
  const snaps: Snap[] = [];
  for (const h of history) {
    const ms = Date.parse(h.at ?? '') || Date.parse(h.slot ?? '') || 0;
    const players = typeof h.players_tracked === 'number' ? h.players_tracked : 0;
    const prev = snaps[snaps.length - 1];
    if (
      prev !== undefined &&
      prev.players === players &&
      prev.rates.length === h.rates.length &&
      prev.rates.every((r, i) => r === h.rates[i])
    ) {
      // Same page again: extend this snapshot to the latest time the value
      // was seen (the snapshot's age is what the stale checks measure).
      prev.ms = ms;
      prev.x = Math.log(1 + hoursSinceRelease(h, checkpointMs));
      continue;
    }
    snaps.push({ x: Math.log(1 + hoursSinceRelease(h, checkpointMs)), ms, players, rates: h.rates });
  }
  const lastSnap = snaps[snaps.length - 1];
  const flatHistory =
    snaps.length <= 1 &&
    history.length >= 2 &&
    lastSnap !== undefined &&
    snaps.length > 0 &&
    history.length > 0 &&
    (lastSnap.ms - (Date.parse(history[0].at ?? '') || Date.parse(history[0].slot ?? '') || lastSnap.ms)) >= 120 * 60_000;
  // --- First pass: per-trophy logit-trend slopes, for tier shrinkage. ---
  interface TrophyFit {
    slope: number;
    olsPred: number;
    anchoredPred: number;
    residVar: number;
    seB: number;
    lastLogit: number;
    lastRate: number;
    countPred: number | null;
    nPts: number;
  }
  const fits: TrophyFit[] = c.trophies.map((t, i) => {
    const pts: Point[] = [];
    const countPts: Point[] = [];
    const nPts: Point[] = [];
    let lastRate = 1;
    for (const s of snaps) {
      const r = s.rates[i];
      if (typeof r !== 'number' || !Number.isFinite(r)) continue;
      lastRate = r;
      // Mild recency tilt: later reads weigh up to ~1.6x the earliest.
      const w = 1 + 0.6 * (snaps.length > 1 ? (s.x - snaps[0].x) / Math.max(1e-9, snaps[snaps.length - 1].x - snaps[0].x) : 0);
      pts.push({ x: s.x, y: logit(r), w });
      if (r > 0 && r < 100 && s.players > 0) {
        const earners = (r / 100) * s.players;
        if (earners > 0) {
          countPts.push({ x: s.x, y: Math.log(earners), w });
          nPts.push({ x: s.x, y: Math.log(s.players), w });
        }
      }
    }
    const lastLogit = logit(lastRate);
    if (pts.length < 2 || snaps.length < 2) {
      return {
        slope: 0,
        olsPred: lastLogit,
        anchoredPred: lastLogit,
        residVar: 0,
        seB: 0,
        lastLogit,
        lastRate,
        countPred: null,
        nPts: pts.length,
      };
    }
    const fit = wls(pts);
    const slope = clampNum(fit.b, -2, 3.5);
    const olsPred = fit.a + fit.b * xTarget;
    const anchoredPred = lastLogit + fit.b * (xTarget - snaps[snaps.length - 1].x);
    let countPred: number | null = null;
    if (countPts.length >= 2 && nPts.length >= 2) {
      const fe = wls(countPts);
      const fn = wls(nPts);
      const logRate = fe.a + fe.b * xTarget - (fn.a + fn.b * xTarget);
      // logRate is ln(rate as a fraction); convert to logit of percent.
      const pct = clampNum(Math.exp(logRate) * 100, 0.01, 99.99);
      countPred = logit(pct);
    }
    return {
      slope,
      olsPred,
      anchoredPred,
      residVar: fit.residVar,
      seB: fit.seB,
      lastLogit,
      lastRate,
      countPred,
      nPts: pts.length,
    };
  });

  // Tier-median slopes for shrinkage.
  const tierSlopes = new Map<string, number>();
  for (const tier of ['platinum', 'gold', 'silver', 'bronze']) {
    const vals = fits
      .filter((f, i) => c.trophies[i].tier === tier && f.nPts >= 2)
      .map((f) => f.slope);
    if (vals.length > 0) tierSlopes.set(tier, medianOf(vals));
  }

  return c.trophies.map((t, i) => {
    const f = fits[i];

    // The 100% trophy: near-100 hedge (see header).
    if (f.lastRate >= 99.99 && snaps.every((s) => (s.rates[i] ?? 0) >= 99.99)) {
      return { id: t.id, q: [99.5, 99.9, 99.98, 100, 100] };
    }

    // Stale-snapshot branch: whole history is one flat snapshot.
    if (flatHistory) {
      const factor = t.tier === 'platinum' ? 0.18 : t.tier === 'bronze' ? 0.3 : 0.22;
      const m = logit(f.lastRate * factor);
      const sigma = 0.9;
      const q = Z.map((z) => round4(clamp01(invLogitPct(m + z * sigma))));
      // Enforce non-decreasing after rounding.
      for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
      return { id: t.id, q };
    }

    // Shrink the slope 25% toward the tier median.
    const tierMed = tierSlopes.get(t.tier) ?? 0;
    const b = f.slope * 0.75 + tierMed * 0.25;
    const dx = xTarget - (snaps.length > 0 ? snaps[snaps.length - 1].x : xTarget);
    const olsPred = f.olsPred + (b - f.slope) * xTarget;
    const anchoredPred = f.lastLogit + b * dx;
    const trendPred = 0.5 * olsPred + 0.5 * anchoredPred;

    // Blend in the count-space forecast as a small bounded nudge (25%):
    // fitting ln(earners) and ln(players) separately in log-time is fragile
    // -- player growth is convex in log-time, so extrapolating both lines
    // can badly overstate the rate -- so its deviation from the trend
    // forecast is clamped to +/-0.75 logit.  The logit-trend fit is the
    // primary forecast; it already contains the dilution the history saw.
    let medLogit = trendPred;
    if (f.countPred !== null) {
      const bounded = clampNum(f.countPred, trendPred - 0.75, trendPred + 0.75);
      medLogit = 0.75 * trendPred + 0.25 * bounded;
    }
    // Do not let the blend wander absurdly far from the last read on a
    // short horizon: cap the move at 2.5 logit units per sqrt(day) of
    // horizon plus 0.5, a guard against a wild slope on few points.
    const maxMove = 0.5 + 2.5 * Math.sqrt(Math.max(horizonDays, 0.01));
    medLogit = clampNum(medLogit, f.lastLogit - maxMove, f.lastLogit + maxMove);

    // Spread: random walk + slope uncertainty + residual + refresh term.
    const rw = 0.5 * Math.sqrt(horizonDays);
    const slopeTerm = f.seB * Math.max(0, dx);
    const residTerm = Math.sqrt(clampNum(f.residVar, 0, 1));
    // Extra uncertainty when the last distinct snapshot is old: Exophase
    // may have refreshed since, unseen. (Flat histories are handled above.)
    const staleHours =
      snaps.length > 0 ? (checkpointMs - snaps[snaps.length - 1].ms) / 3_600_000 : 0;
    const refreshVar = horizonDays > 1 ? 0.09 : 0.04;
    const staleVar = staleHours > 6 ? 0.09 : 0;
    const sigma = clampNum(
      Math.sqrt(rw * rw + slopeTerm * slopeTerm + residTerm * residTerm + refreshVar + staleVar),
      0.3,
      1.8,
    );

    const q = Z.map((z) => round4(clamp01(invLogitPct(medLogit + z * sigma))));
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  const out = `${JSON.stringify({ trophies })}\n`;
  if (Buffer.byteLength(out, 'utf8') > 65536) throw new Error(`${file} output exceeds 65,536 bytes`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), out);
}
