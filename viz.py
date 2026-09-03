#!/usr/bin/env python3
"""Build a self-contained HTML replay of a run.

    python3 viz.py runs/seed1

Produces runs/seed1/replay.html -- one file, no server, no dependencies.
Open it in a browser and press play.
"""

import json
import os
import sys

TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root{
    --bg:#f7f6f3; --panel:#ffffff; --ink:#1c1b19; --muted:#6b6862;
    --line:#e2ded7; --accent:#3b6ea5; --grid:#eae6df;
  }
  @media (prefers-color-scheme: dark){
    :root:not([data-theme="light"]){
      --bg:#14151a; --panel:#1c1e25; --ink:#e9e7e2; --muted:#918d86;
      --line:#2b2e37; --accent:#7aa7d8; --grid:#22252d;
    }
  }
  :root[data-theme="dark"]{
    --bg:#14151a; --panel:#1c1e25; --ink:#e9e7e2; --muted:#918d86;
    --line:#2b2e37; --accent:#7aa7d8; --grid:#22252d;
  }
  body{background:var(--bg);color:var(--ink);
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
       margin:0;padding:18px;}
  h1{font-size:18px;margin:0 0 2px;font-weight:650;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:12.5px;margin-bottom:14px}
  .wrap{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
  .card{background:var(--panel);border:1px solid var(--line);
        border-radius:10px;padding:12px}
  canvas{display:block;border-radius:6px;max-width:100%}
  #stage{width:min(620px,92vw);height:auto;aspect-ratio:1;background:var(--grid)}
  .side{width:min(340px,92vw);display:flex;flex-direction:column;gap:12px}
  .controls{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}
  button{background:var(--accent);color:#fff;border:0;border-radius:6px;
         padding:6px 14px;font:inherit;font-weight:600;cursor:pointer}
  button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
  input[type=range]{accent-color:var(--accent)}
  #scrub{flex:1;min-width:160px}
  .k{color:var(--muted);font-size:12px;text-transform:uppercase;
     letter-spacing:.06em;margin:0 0 8px;font-weight:600}
  .row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}
  .swatch{width:10px;height:10px;border-radius:3px;flex:none}
  .bar{flex:1;height:7px;background:var(--grid);border-radius:4px;overflow:hidden}
  .bar > i{display:block;height:100%}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .log{height:150px;overflow:auto;font-size:12px;line-height:1.45}
  .log div{margin:2px 0;color:var(--muted)}
  .log b{color:var(--ink);font-weight:600}
  .stat{display:flex;justify-content:space-between;font-size:13px;margin:3px 0}
  .stat span:last-child{font-variant-numeric:tabular-nums;font-weight:600}
  #share{width:100%;height:70px;display:block}
  .note{color:var(--muted);font-size:12px;margin-top:12px;max-width:980px}
</style>

<h1>__TITLE__</h1>
<div class="sub">__SUB__</div>

<div class="wrap">
  <div class="card">
    <canvas id="stage" width="720" height="720"></canvas>
    <div class="controls">
      <button id="play">Play</button>
      <button id="reset" class="ghost">Restart</button>
      <input type="range" id="scrub" min="0" max="1" value="0">
      <span class="mono" id="tick">t 0</span>
    </div>
    <div class="controls">
      <label class="k" style="margin:0">Speed</label>
      <input type="range" id="speed" min="1" max="60" value="18">
      <label class="k" style="margin:0 0 0 10px">
        <input type="checkbox" id="trails" checked> trails
      </label>
    </div>
  </div>

  <div class="side">
    <div class="card">
      <p class="k">Most-repeated sound, per agent</p>
      <div id="legend"></div>
    </div>
    <div class="card">
      <p class="k">Share of population, over time</p>
      <canvas id="share" width="640" height="140"></canvas>
    </div>
    <div class="card">
      <p class="k">Now</p>
      <div class="stat"><span>Speaking this tick</span><span id="nspeak">0</span></div>
      <div class="stat"><span>Groups (within vision)</span><span id="nclust">0</span></div>
      <div class="stat"><span>Largest group</span><span id="biggest">0</span></div>
    </div>
    <div class="card">
      <p class="k">Overheard</p>
      <div class="log mono" id="log"></div>
    </div>
  </div>
</div>

<p class="note">Every dot is one agent. Colour is the sound that agent has heard
or said most often &mdash; the closest thing it has to a belief. No agent was
given a goal, a role, or a reward; colour agreement, clumping and silence are
whatever happened on their own.</p>

<script>
const DATA = __DATA__;
const M = DATA.meta, F = DATA.frames, NAMES = M.names, TOKENS = M.tokens;
const cv = document.getElementById('stage'), cx = cv.getContext('2d');
const sh = document.getElementById('share'), sx = sh.getContext('2d');
let i = 0, playing = false, acc = 0, last = 0;

function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function hue(t){ return t < 0 ? null : (t * 137.508) % 360; }
function colorOf(t){ const h = hue(t); return h === null ? css('--muted') : `hsl(${h.toFixed(0)} 62% 55%)`; }

function draw(){
  const f = F[i], W = M.width, H = M.height;
  const s = cv.width / W;
  if (document.getElementById('trails').checked){
    cx.fillStyle = css('--grid'); cx.globalAlpha = .22;
    cx.fillRect(0,0,cv.width,cv.height); cx.globalAlpha = 1;
  } else {
    cx.fillStyle = css('--grid'); cx.fillRect(0,0,cv.width,cv.height);
  }
  // speech first, behind the dots
  cx.lineWidth = 1.2;
  for (const [spk, tgt, txt] of f.s){
    const a = f.a[spk];
    cx.beginPath();
    cx.arc((a[0]+.5)*s, (a[1]+.5)*s, s*3.2, 0, 6.2832);
    cx.strokeStyle = colorOf(a[2]); cx.globalAlpha = .30; cx.stroke();
    cx.globalAlpha = 1;
    if (tgt >= 0){
      const b = f.a[tgt];
      cx.beginPath();
      cx.moveTo((a[0]+.5)*s,(a[1]+.5)*s); cx.lineTo((b[0]+.5)*s,(b[1]+.5)*s);
      cx.strokeStyle = colorOf(a[2]); cx.globalAlpha = .45; cx.stroke();
      cx.globalAlpha = 1;
    }
  }
  for (const a of f.a){
    cx.fillStyle = colorOf(a[2]);
    cx.beginPath();
    cx.arc((a[0]+.5)*s, (a[1]+.5)*s, Math.max(2.2, s*0.42), 0, 6.2832);
    cx.fill();
  }
  document.getElementById('tick').textContent = 't ' + f.t;
  document.getElementById('scrub').value = i;
  document.getElementById('nspeak').textContent = f.s.length;
  stats(f); legend(f); shareLine();
  const log = document.getElementById('log');
  for (const [spk, tgt, txt] of f.s.slice(0,3)){
    const d = document.createElement('div');
    d.innerHTML = '<b>' + NAMES[spk] + '</b>' +
      (tgt >= 0 ? ' &rarr; ' + NAMES[tgt] : '') + ' &ldquo;' + txt + '&rdquo;';
    log.prepend(d);
  }
  while (log.childNodes.length > 60) log.removeChild(log.lastChild);
}

function stats(f){
  // connected groups, agents linked when within vision range
  const n = f.a.length, p = new Array(n).fill(0).map((_,k)=>k), R = M.vision;
  const find = k => { while (p[k]!==k){ p[k]=p[p[k]]; k=p[k]; } return k; };
  for (let x=0;x<n;x++) for (let y=x+1;y<n;y++){
    if (Math.abs(f.a[x][0]-f.a[y][0])<=R && Math.abs(f.a[x][1]-f.a[y][1])<=R){
      const rx=find(x), ry=find(y); if (rx!==ry) p[rx]=ry;
    }
  }
  const cnt = {}; for (let x=0;x<n;x++){ const r=find(x); cnt[r]=(cnt[r]||0)+1; }
  const sizes = Object.values(cnt);
  document.getElementById('nclust').textContent = sizes.length;
  document.getElementById('biggest').textContent = Math.max(...sizes);
}

function legend(f){
  const c = {}; for (const a of f.a) c[a[2]] = (c[a[2]]||0)+1;
  const top = Object.entries(c).sort((a,b)=>b[1]-a[1]).slice(0,7);
  const n = f.a.length;
  document.getElementById('legend').innerHTML = top.map(([t,k])=>{
    const tid = +t, nm = tid < 0 ? '(silent)' : TOKENS[tid];
    const col = colorOf(tid);
    return `<div class="row"><i class="swatch" style="background:${col}"></i>
      <span class="mono" style="width:74px">${nm}</span>
      <span class="bar"><i style="width:${(100*k/n).toFixed(1)}%;background:${col}"></i></span>
      <span style="width:26px;text-align:right">${k}</span></div>`;
  }).join('');
}

const SERIES = __SERIES__;   // [{token, values[]}] shares over time
const PEAK = Math.max(0.05, ...SERIES.flatMap(s => s.values)) * 1.12;
function shareLine(){
  sx.clearRect(0,0,sh.width,sh.height);
  const W = sh.width, H = sh.height, L = F.length, pad = 14;
  const y0 = H - 12, span = y0 - 6;
  sx.strokeStyle = css('--line'); sx.lineWidth = 1;
  sx.beginPath(); sx.moveTo(pad,y0+.5); sx.lineTo(W,y0+.5); sx.stroke();
  sx.beginPath(); sx.moveTo(pad,6.5); sx.lineTo(W,6.5); sx.stroke();
  sx.fillStyle = css('--muted'); sx.font = '10px ui-monospace,monospace';
  sx.fillText((PEAK*100).toFixed(0)+'%', 0, 10);
  sx.fillText('0', 0, y0);
  for (const s of SERIES){
    sx.beginPath();
    for (let k=0;k<L;k++){
      const x = pad + (W-pad)*k/(L-1||1), y = y0 - span*(s.values[k]/PEAK);
      k ? sx.lineTo(x,y) : sx.moveTo(x,y);
    }
    sx.strokeStyle = colorOf(s.token); sx.lineWidth = 1.8; sx.stroke();
  }
  const x = pad + (W-pad)*i/(L-1||1);
  sx.strokeStyle = css('--ink'); sx.globalAlpha=.5; sx.lineWidth=1;
  sx.beginPath(); sx.moveTo(x,0); sx.lineTo(x,H); sx.stroke(); sx.globalAlpha=1;
}

function loop(ts){
  if (playing){
    const fps = +document.getElementById('speed').value;
    if (!last) last = ts;
    acc += (ts-last)/1000; last = ts;
    while (acc > 1/fps){ acc -= 1/fps; i = (i+1) % F.length; }
    draw();
  } else last = 0;
  requestAnimationFrame(loop);
}

const scrub = document.getElementById('scrub');
scrub.max = F.length - 1;
scrub.addEventListener('input', e => { i = +e.target.value; draw(); });
document.getElementById('play').addEventListener('click', e => {
  playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play';
});
document.getElementById('reset').addEventListener('click', () => { i = 0; draw(); });
document.getElementById('trails').addEventListener('change', draw);
const jump = /t=(\d+)/.exec(location.hash);   // replay.html#t=400
if (jump) i = Math.min(F.length - 1, Math.max(0, +jump[1]));
draw();
requestAnimationFrame(loop);
</script>
"""


def build(run_dir, out_name="replay.html"):
    with open(os.path.join(run_dir, "frames.json"), encoding="utf-8") as f:
        data = json.load(f)
    meta, frames = data["meta"], data["frames"]
    n = len(meta["names"])

    # Share of the population holding each token as its most-repeated sound.
    totals = {}
    for fr in frames:
        for a in fr["a"]:
            totals[a[2]] = totals.get(a[2], 0) + 1
    top = [t for t, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:8]
           if t >= 0]
    series = []
    for t in top:
        vals = []
        for fr in frames:
            c = sum(1 for a in fr["a"] if a[2] == t)
            vals.append(round(c / n, 4))
        series.append({"token": t, "values": vals})

    title = "Emergence &mdash; %d agents, no goals" % n
    sub = ("%d&times;%d closed world &middot; %d ticks &middot; vision %d "
           "&middot; seed %d &middot; %d distinct sounds uttered"
           % (meta["width"], meta["height"], meta["ticks"], meta["vision"],
              meta["seed"], len(meta["tokens"])))

    html = (TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUB__", sub)
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__SERIES__", json.dumps(series, separators=(",", ":"))))
    out = os.path.join(run_dir, out_name)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "runs/seed1"))
