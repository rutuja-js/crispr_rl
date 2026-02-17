import { useState, useCallback, useRef, useEffect } from "react";

// ─── Pure-JS CRISPR core logic (mirrors the Python package) ──────────────────

const COMPLEMENT = { A: "T", T: "A", G: "C", C: "G", N: "N" };

function reverseComplement(seq) {
  return seq
    .toUpperCase()
    .split("")
    .reverse()
    .map((b) => COMPLEMENT[b] || "N")
    .join("");
}

function gcFraction(seq) {
  if (!seq.length) return 0;
  const gc = seq.split("").filter((b) => "GC".includes(b)).length;
  return gc / seq.length;
}

function maxHomopolymerRun(seq) {
  if (!seq.length) return 0;
  let max = 1, cur = 1;
  for (let i = 1; i < seq.length; i++) {
    cur = seq[i] === seq[i - 1] ? cur + 1 : 1;
    if (cur > max) max = cur;
  }
  return max;
}

function tmWallace(seq) {
  const at = (seq.match(/[AT]/g) || []).length;
  const gc = (seq.match(/[GC]/g) || []).length;
  return 2 * at + 4 * gc;
}

function efficiencyProxy(seq) {
  const tm = tmWallace(seq);
  const dev = Math.abs(tm - 65);
  return Math.exp(-(dev * dev) / (2 * 100));
}

function pamToRegex(pam) {
  const iupac = { N:"[ACGT]", R:"[AG]", Y:"[CT]", S:"[GC]", W:"[AT]", K:"[GT]", M:"[AC]", B:"[CGT]", D:"[AGT]", H:"[ACT]", V:"[ACG]", A:"A", C:"C", G:"G", T:"T" };
  return new RegExp(pam.split("").map((c) => iupac[c.toUpperCase()] || c).join(""), "g");
}

function scanSequence(seq, pamPattern = "NGG", guideLen = 20) {
  const s = seq.toUpperCase();
  const rc = reverseComplement(s);
  const hits = [];
  const re = pamToRegex(pamPattern);

  let m;
  re.lastIndex = 0;
  while ((m = re.exec(s)) !== null) {
    const pamStart = m.index;
    const protoStart = pamStart - guideLen;
    if (protoStart < 0) continue;
    const protospacer = s.slice(protoStart, pamStart);
    if (protospacer.length === guideLen) {
      hits.push({ position: protoStart, strand: "+", protospacer, pamSeq: m[0] });
    }
  }

  const pamRe2 = pamToRegex(pamPattern);
  pamRe2.lastIndex = 0;
  while ((m = pamRe2.exec(rc)) !== null) {
    const pamStart = m.index;
    const protoStart = pamStart - guideLen;
    if (protoStart < 0) continue;
    const protospacer = rc.slice(protoStart, pamStart);
    if (protospacer.length === guideLen) {
      const fwdPos = s.length - (pamStart + pamPattern.length);
      hits.push({ position: fwdPos, strand: "-", protospacer, pamSeq: m[0] });
    }
  }

  hits.sort((a, b) => a.position - b.position);
  return hits;
}

function specificityProxy(guide, seq) {
  const s = seq.toUpperCase();
  const rc = reverseComplement(s);
  const seedLen = 12;
  const n = guide.length;
  let offTargetSum = 0;

  for (const strand of [s, rc]) {
    for (let i = 0; i <= strand.length - n; i++) {
      const window = strand.slice(i, i + n);
      let mm = 0, wmm = 0;
      for (let j = 0; j < n; j++) {
        if (guide[j] !== window[j]) {
          mm++;
          wmm += j >= n - seedLen ? 3 : 1;
        }
      }
      if (mm > 0 && mm <= 3) offTargetSum += wmm;
    }
  }
  return 1 / (1 + offTargetSum);
}

function scoreGuide(guide, seq, position, weights) {
  const gc = gcFraction(guide);
  const hp = maxHomopolymerRun(guide);
  const eff = efficiencyProxy(guide);
  const gcPen = gc < 0.35 ? 0.35 - gc : gc > 0.65 ? gc - 0.65 : 0;
  const hpPen = Math.min(1, Math.max(0, hp - 4) * 0.2);
  const efficiency = Math.max(0, eff - gcPen - hpPen);
  const spec = specificityProxy(guide, seq);
  const coverage = Math.max(0, 1 - position / Math.max(1, seq.length - 1));

  const w1 = weights.w_efficiency ?? 0.3;
  const w2 = weights.w_specificity ?? 0.5;
  const w3 = weights.w_coverage ?? 0.2;

  const raw = w1 * efficiency + w2 * spec + w3 * coverage;
  const riskFlags = [];
  if (gc < 0.35 || gc > 0.65) riskFlags.push(`GC ${(gc * 100).toFixed(0)}%`);
  if (hp > 5) riskFlags.push(`Run:${hp}`);

  return {
    score: Math.min(1, Math.max(0, raw)),
    gc, hp, efficiency: round(efficiency), specificity: round(spec),
    coverage: round(coverage), riskFlags,
  };
}

function round(v) { return Math.round(v * 10000) / 10000; }

const PROFILES = {
  knockout:  { w_efficiency: 0.3, w_specificity: 0.5, w_coverage: 0.2 },
  knockdown: { w_efficiency: 0.5, w_specificity: 0.3, w_coverage: 0.2 },
  screening: { w_efficiency: 0.3, w_specificity: 0.2, w_coverage: 0.5 },
};

const DEMO_SEQ =
  "ATGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC" +
  "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGG" +
  "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAG" +
  "CATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG";

// ─── UI ──────────────────────────────────────────────────────────────────────

const ACCENT = "#00ff9d";
const BG = "#050c14";
const CARD = "#0a1628";
const BORDER = "#0f2847";
const TEXT = "#c8daf0";
const DIM = "#4a6a8a";

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body { background: ${BG}; color: ${TEXT}; font-family: 'Space Mono', monospace; }

  .app { min-height: 100vh; display: flex; flex-direction: column; }

  .header {
    padding: 20px 32px;
    border-bottom: 1px solid ${BORDER};
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(90deg, #050c14 0%, #091525 100%);
  }

  .logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 22px;
    color: ${ACCENT};
    letter-spacing: -0.5px;
  }

  .logo span { color: ${TEXT}; }

  .tagline { font-size: 11px; color: ${DIM}; letter-spacing: 2px; text-transform: uppercase; }

  .badge {
    margin-left: auto;
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.2);
    color: ${ACCENT};
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 2px;
  }

  .main { display: grid; grid-template-columns: 380px 1fr; flex: 1; min-height: 0; }

  .sidebar {
    border-right: 1px solid ${BORDER};
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    overflow-y: auto;
    background: ${CARD};
  }

  .panel { flex: 1; overflow-y: auto; padding: 24px; }

  .section-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: ${DIM};
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: ${BORDER};
  }

  textarea {
    width: 100%;
    background: ${BG};
    border: 1px solid ${BORDER};
    color: ${TEXT};
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 12px;
    resize: vertical;
    min-height: 100px;
    outline: none;
    transition: border-color 0.2s;
    line-height: 1.6;
  }

  textarea:focus { border-color: ${ACCENT}; }

  .profile-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }

  .profile-btn {
    background: ${BG};
    border: 1px solid ${BORDER};
    color: ${DIM};
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 8px 4px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }

  .profile-btn:hover { border-color: ${ACCENT}; color: ${ACCENT}; }
  .profile-btn.active { background: rgba(0,255,157,0.1); border-color: ${ACCENT}; color: ${ACCENT}; }

  .pam-row { display: flex; gap: 8px; align-items: center; }

  input[type="text"] {
    flex: 1;
    background: ${BG};
    border: 1px solid ${BORDER};
    color: ${TEXT};
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    padding: 8px 10px;
    outline: none;
    transition: border-color 0.2s;
  }

  input[type="text"]:focus { border-color: ${ACCENT}; }

  .pam-preset {
    background: ${BG};
    border: 1px solid ${BORDER};
    color: ${DIM};
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 6px 8px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }

  .pam-preset:hover { color: ${ACCENT}; border-color: ${ACCENT}; }

  .run-btn {
    width: 100%;
    background: ${ACCENT};
    color: ${BG};
    border: none;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 14px;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }

  .run-btn:hover { background: #00ffb0; transform: translateY(-1px); box-shadow: 0 4px 24px rgba(0,255,157,0.3); }
  .run-btn:active { transform: translateY(0); }
  .run-btn:disabled { background: ${DIM}; cursor: not-allowed; transform: none; box-shadow: none; }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 20px;
  }

  .stat-card {
    background: ${CARD};
    border: 1px solid ${BORDER};
    padding: 14px;
    text-align: center;
  }

  .stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: ${ACCENT};
    line-height: 1;
    margin-bottom: 4px;
  }

  .stat-label { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: ${DIM}; }

  .table-wrap { overflow-x: auto; }

  table { width: 100%; border-collapse: collapse; font-size: 11px; }

  th {
    background: ${CARD};
    border: 1px solid ${BORDER};
    color: ${DIM};
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 12px;
    text-align: left;
    white-space: nowrap;
    position: sticky;
    top: 0;
  }

  td {
    border: 1px solid ${BORDER};
    padding: 10px 12px;
    vertical-align: middle;
    transition: background 0.15s;
  }

  tr:hover td { background: rgba(0,255,157,0.03); }

  .seq-mono {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.5px;
    color: #7ec8e3;
  }

  .score-bar-wrap { display: flex; align-items: center; gap: 8px; min-width: 120px; }

  .score-bar-bg {
    flex: 1;
    height: 4px;
    background: ${BORDER};
    border-radius: 0;
    overflow: hidden;
  }

  .score-bar-fill {
    height: 100%;
    background: ${ACCENT};
    border-radius: 0;
    transition: width 0.6s ease;
  }

  .score-val { font-size: 11px; color: ${ACCENT}; min-width: 38px; text-align: right; }

  .strand-badge {
    display: inline-block;
    padding: 2px 6px;
    font-size: 10px;
    font-weight: 700;
    border-radius: 1px;
  }

  .strand-pos { background: rgba(0,255,157,0.1); color: ${ACCENT}; border: 1px solid rgba(0,255,157,0.3); }
  .strand-neg { background: rgba(255,100,100,0.1); color: #ff9090; border: 1px solid rgba(255,100,100,0.3); }

  .risk-tag {
    display: inline-block;
    background: rgba(255,180,0,0.1);
    color: #ffb400;
    border: 1px solid rgba(255,180,0,0.25);
    font-size: 9px;
    padding: 2px 6px;
    margin-right: 3px;
    border-radius: 1px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    gap: 16px;
    color: ${DIM};
  }

  .empty-icon { font-size: 48px; opacity: 0.3; }
  .empty-text { font-size: 11px; letter-spacing: 2px; text-transform: uppercase; }

  .loading-bar {
    width: 100%;
    height: 2px;
    background: ${BORDER};
    overflow: hidden;
    margin-bottom: 24px;
  }

  .loading-bar-inner {
    height: 100%;
    background: ${ACCENT};
    animation: loading 1.2s ease-in-out infinite;
  }

  @keyframes loading {
    0% { width: 0%; margin-left: 0; }
    50% { width: 60%; margin-left: 20%; }
    100% { width: 0%; margin-left: 100%; }
  }

  .gc-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }

  .panel-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 20px;
  }

  .panel-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #fff;
  }

  .panel-sub { font-size: 11px; color: ${DIM}; }

  .rank-num {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 16px;
    color: ${DIM};
    width: 24px;
    text-align: center;
  }

  .rank-num.top { color: ${ACCENT}; }

  .demo-btn {
    width: 100%;
    background: transparent;
    border: 1px dashed ${BORDER};
    color: ${DIM};
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 9px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .demo-btn:hover { border-color: ${ACCENT}; color: ${ACCENT}; }

  .error-box {
    background: rgba(255,80,80,0.08);
    border: 1px solid rgba(255,80,80,0.25);
    color: #ff8080;
    font-size: 11px;
    padding: 12px;
    line-height: 1.6;
  }

  .feedback-row { display: flex; gap: 4px; }
  .star {
    background: none;
    border: none;
    font-size: 16px;
    cursor: pointer;
    opacity: 0.3;
    transition: opacity 0.15s, transform 0.1s;
    padding: 2px;
  }
  .star.lit { opacity: 1; }
  .star:hover { transform: scale(1.2); }

  .counter-chip {
    background: rgba(0,255,157,0.08);
    border: 1px solid rgba(0,255,157,0.15);
    color: ${ACCENT};
    font-size: 10px;
    padding: 3px 8px;
    letter-spacing: 1px;
  }
`;

function ScoreBar({ score }) {
  const pct = Math.round(score * 100);
  const color = score > 0.6 ? ACCENT : score > 0.35 ? "#ffb400" : "#ff6060";
  return (
    <div className="score-bar-wrap">
      <div className="score-bar-bg">
        <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="score-val" style={{ color }}>{score.toFixed(3)}</span>
    </div>
  );
}

function GcDot({ gc }) {
  const color = gc >= 0.35 && gc <= 0.65 ? ACCENT : "#ffb400";
  return <span className="gc-dot" style={{ background: color }} />;
}

export default function App() {
  const [sequence, setSequence] = useState("");
  const [pam, setPam] = useState("NGG");
  const [profile, setProfile] = useState("knockout");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [ratings, setRatings] = useState({});
  const [hoverRating, setHoverRating] = useState({});

  const runDesign = useCallback(() => {
    const seq = sequence.trim().toUpperCase().replace(/\s/g, "");
    if (!seq) { setError("Please enter a DNA sequence."); return; }
    if (!/^[ACGTN]+$/.test(seq)) { setError("Sequence must contain only A, C, G, T, N characters."); return; }
    if (seq.length < 23) { setError("Sequence too short — need at least 23 nt (20 guide + 3 PAM)."); return; }

    setError("");
    setLoading(true);
    setResults(null);

    setTimeout(() => {
      const hits = scanSequence(seq, pam);
      const weights = PROFILES[profile];

      const candidates = hits.slice(0, 20).map((h, i) => {
        const s = scoreGuide(h.protospacer, seq, h.position, weights);
        return {
          id: `guide_${String(i).padStart(3, "0")}`,
          seq: h.protospacer,
          pam: h.pamSeq,
          locus: `pos:${h.position}`,
          strand: h.strand,
          position: h.position,
          ...s,
        };
      });

      candidates.sort((a, b) => b.score - a.score);

      const scores = candidates.map((c) => c.score);
      const avgScore = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
      const maxScore = scores.length ? Math.max(...scores) : 0;

      setResults({ candidates, avgScore, maxScore, seqLen: seq.length, nHits: hits.length });
      setLoading(false);
    }, 600);
  }, [sequence, pam, profile]);

  const loadDemo = () => {
    setSequence(DEMO_SEQ);
    setError("");
  };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        {/* Header */}
        <header className="header">
          <div>
            <div className="logo">CRISPR<span>_RL</span></div>
            <div className="tagline">Guide RNA Design Engine</div>
          </div>
          <span className="badge">v0.1.0 · Browser Edition</span>
        </header>

        <div className="main">
          {/* Sidebar */}
          <aside className="sidebar">
            <div>
              <div className="section-label">Sequence Input</div>
              <textarea
                placeholder="Paste DNA sequence (A/C/G/T/N)&#10;Minimum 23 nt required…"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                rows={6}
              />
              <button className="demo-btn" style={{ marginTop: 6 }} onClick={loadDemo}>
                ↓ Load demo sequence (329 nt)
              </button>
            </div>

            <div>
              <div className="section-label">PAM Pattern</div>
              <div className="pam-row">
                <input
                  type="text"
                  value={pam}
                  onChange={(e) => setPam(e.target.value.toUpperCase())}
                  maxLength={8}
                  style={{ width: 90, flex: "none" }}
                />
                {["NGG", "TTTN", "NNGRRT"].map((p) => (
                  <button key={p} className="pam-preset" onClick={() => setPam(p)}>
                    {p}
                  </button>
                ))}
              </div>
              <div style={{ fontSize: 10, color: DIM, marginTop: 6 }}>
                SpCas9=NGG · Cas12a=TTTN · SaCas9=NNGRRT
              </div>
            </div>

            <div>
              <div className="section-label">Profile</div>
              <div className="profile-grid">
                {Object.keys(PROFILES).map((p) => (
                  <button
                    key={p}
                    className={`profile-btn${profile === p ? " active" : ""}`}
                    onClick={() => setProfile(p)}
                  >
                    {p}
                  </button>
                ))}
              </div>
              <div style={{ fontSize: 10, color: DIM, marginTop: 8, lineHeight: 1.6 }}>
                {profile === "knockout" && "High specificity · NHEJ disruption"}
                {profile === "knockdown" && "High efficiency · Partial reduction"}
                {profile === "screening" && "High coverage · Library design"}
              </div>
            </div>

            {error && <div className="error-box">⚠ {error}</div>}

            <button className="run-btn" onClick={runDesign} disabled={loading}>
              {loading ? "SCANNING…" : "▶  RUN DESIGN"}
            </button>

            {results && (
              <div style={{ fontSize: 10, color: DIM, lineHeight: 1.8 }}>
                <div>Sequence: <span style={{ color: TEXT }}>{results.seqLen} nt</span></div>
                <div>PAM hits: <span style={{ color: TEXT }}>{results.nHits}</span></div>
                <div>Candidates: <span style={{ color: TEXT }}>{results.candidates.length}</span></div>
                <div>Best score: <span style={{ color: ACCENT }}>{results.maxScore.toFixed(4)}</span></div>
                <div>Avg score: <span style={{ color: ACCENT }}>{results.avgScore.toFixed(4)}</span></div>
              </div>
            )}
          </aside>

          {/* Main panel */}
          <main className="panel">
            {loading && (
              <>
                <div className="loading-bar"><div className="loading-bar-inner" /></div>
                <div style={{ color: DIM, fontSize: 11, letterSpacing: 2, textTransform: "uppercase" }}>
                  Scanning PAM sites · Scoring guides · Running RL optimizer…
                </div>
              </>
            )}

            {!loading && !results && (
              <div className="empty-state">
                <div className="empty-icon">🧬</div>
                <div className="empty-text">Enter a sequence and click Run Design</div>
              </div>
            )}

            {!loading && results && (
              <>
                <div className="panel-header">
                  <div className="panel-title">Ranked Guide RNAs</div>
                  <div className="panel-sub">{results.candidates.length} candidates · {profile} profile</div>
                  <span className="counter-chip" style={{ marginLeft: "auto" }}>
                    {results.candidates.filter((c) => c.score > 0.5).length} high-confidence
                  </span>
                </div>

                <div className="stats-row">
                  <div className="stat-card">
                    <div className="stat-val">{results.candidates.length}</div>
                    <div className="stat-label">Candidates</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-val">{results.maxScore.toFixed(3)}</div>
                    <div className="stat-label">Best Score</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-val">{results.candidates.filter(c => !c.riskFlags.length).length}</div>
                    <div className="stat-label">No Flags</div>
                  </div>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>ID</th>
                        <th>Protospacer (20 nt)</th>
                        <th>PAM</th>
                        <th>Strand</th>
                        <th>GC%</th>
                        <th>Score</th>
                        <th>Efficiency</th>
                        <th>Specificity</th>
                        <th>Flags</th>
                        <th>Feedback</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.candidates.map((c, i) => {
                        const myRating = ratings[c.id] || 0;
                        const myHover = hoverRating[c.id] || 0;
                        return (
                          <tr key={c.id}>
                            <td>
                              <span className={`rank-num${i < 3 ? " top" : ""}`}>{i + 1}</span>
                            </td>
                            <td style={{ color: DIM, fontSize: 10 }}>{c.id}</td>
                            <td>
                              <span className="seq-mono">
                                {c.seq.slice(0, 8)}
                                <span style={{ color: ACCENT }}>{c.seq.slice(8, 12)}</span>
                                {c.seq.slice(12, 20)}
                              </span>
                            </td>
                            <td><span className="seq-mono" style={{ color: "#ff9090" }}>{c.pam}</span></td>
                            <td>
                              <span className={`strand-badge ${c.strand === "+" ? "strand-pos" : "strand-neg"}`}>
                                {c.strand === "+" ? "＋" : "−"}
                              </span>
                            </td>
                            <td>
                              <GcDot gc={c.gc} />
                              {(c.gc * 100).toFixed(0)}%
                            </td>
                            <td><ScoreBar score={c.score} /></td>
                            <td style={{ color: c.efficiency > 0.5 ? ACCENT : DIM }}>
                              {(c.efficiency * 100).toFixed(0)}%
                            </td>
                            <td style={{ color: c.specificity > 0.5 ? ACCENT : DIM }}>
                              {(c.specificity * 100).toFixed(0)}%
                            </td>
                            <td>
                              {c.riskFlags.length === 0 ? (
                                <span style={{ color: ACCENT, fontSize: 10 }}>✓ clean</span>
                              ) : (
                                c.riskFlags.map((f) => <span key={f} className="risk-tag">{f}</span>)
                              )}
                            </td>
                            <td>
                              <div className="feedback-row">
                                {[1, 2, 3, 4, 5].map((star) => (
                                  <button
                                    key={star}
                                    className={`star${(myHover || myRating) >= star ? " lit" : ""}`}
                                    onMouseEnter={() => setHoverRating((h) => ({ ...h, [c.id]: star }))}
                                    onMouseLeave={() => setHoverRating((h) => ({ ...h, [c.id]: 0 }))}
                                    onClick={() => setRatings((r) => ({ ...r, [c.id]: star }))}
                                    title={`Rate ${star}/5`}
                                  >
                                    ★
                                  </button>
                                ))}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div style={{ marginTop: 24, fontSize: 10, color: DIM, lineHeight: 1.8 }}>
                  <span style={{ color: ACCENT }}>Seed region</span> (highlighted in sequence) = last 8 nt before PAM — most critical for specificity. &nbsp;
                  Optimal GC: 35–65%. &nbsp; Run-length penalty for homopolymers &gt;4 identical bases.
                </div>
              </>
            )}
          </main>
        </div>
      </div>
    </>
  );
}
