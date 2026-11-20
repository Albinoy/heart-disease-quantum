"""CardiaQ visual layer: page styles, the hero, and scroll-animated charts.

Each chart is a small self-contained HTML page shown with
streamlit.components.v1.html. It plays its animation when it scrolls into
view and replays it when you scroll back to it (IntersectionObserver).
People who have reduced motion turned on see the final state straight away.

Design tokens
  ink    #14213D  text, classical model
  red    #D7263D  ECG trace, higher risk, missed cases
  teal   #0B7A7C  quantum model
  green  #5E8C61  lowers risk, correct cases
  amber  #D99A00  false alarms
  paper  #FFF8F8  ECG chart paper behind the hero
Type: Barlow Semi Condensed (headings, numbers), Source Sans 3 (text).
"""
import json
import math

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600"
         "&family=Source+Sans+3:wght@400;600&display=swap');")

INK, RED, TEAL, GREEN, AMBER, MUTED = "#14213D", "#D7263D", "#0B7A7C", "#5E8C61", "#D99A00", "#5B6578"
EASE = "cubic-bezier(.2,.8,.2,1)"

CSS = "<style>" + FONTS + """
:root{--ink:#14213D;--muted:#5B6578;--line:#E3D9D9;--red:#D7263D;--teal:#0B7A7C}
.stApp{font-family:'Source Sans 3',system-ui,sans-serif;color:var(--ink);background:#fff}
.stApp h1,.stApp h2,.stApp h3,.cq-brand{font-family:'Barlow Semi Condensed','Source Sans 3',sans-serif}
[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
.block-container{max-width:1080px;padding-top:1.2rem;padding-bottom:4rem}

.cq-hero{position:relative;overflow:hidden;border:1px solid #EBCFD3;border-radius:6px;
  padding:2.6rem 2.2rem 7rem;background-color:#FFF8F8;
  background-image:linear-gradient(rgba(215,38,61,.20) 1px,transparent 1px),
  linear-gradient(90deg,rgba(215,38,61,.20) 1px,transparent 1px),
  linear-gradient(rgba(215,38,61,.07) 1px,transparent 1px),
  linear-gradient(90deg,rgba(215,38,61,.07) 1px,transparent 1px);
  background-size:50px 50px,50px 50px,10px 10px,10px 10px}
.cq-brand{font-size:clamp(3rem,9vw,5.2rem);font-weight:600;line-height:.95;margin:0;letter-spacing:-.005em}
.cq-tag{max-width:34rem;font-size:1.2rem;line-height:1.45;margin:1rem 0 1.4rem}
.cq-go{display:inline-block;background:var(--ink);color:#fff!important;text-decoration:none!important;
  font-weight:600;padding:.6rem 1.3rem;border-radius:4px}
.cq-go:hover{background:var(--red)}
.cq-go:focus-visible{outline:3px solid var(--red);outline-offset:3px}
.cq-hero svg{position:absolute;left:0;right:0;bottom:12px;width:100%;height:96px}
.cq-hero path{fill:none;stroke:var(--red);stroke-width:2.4;vector-effect:non-scaling-stroke;stroke-linejoin:round}
.cq-hero .ghost{opacity:.2}
.cq-hero .sweep{stroke-dasharray:24 76;animation:cqsweep 6s linear infinite}
@keyframes cqsweep{from{stroke-dashoffset:24}to{stroke-dashoffset:-76}}
@media (prefers-reduced-motion:reduce){.cq-hero .sweep{animation:none;stroke-dasharray:none}}

.cq-note{border-left:4px solid var(--red);background:#FFF4F5;padding:.7rem 1rem;margin:1rem 0 0;
  font-size:.95rem;line-height:1.45}
.cq-sec{margin:3.2rem 0 1rem;border-top:2px solid var(--ink);padding-top:.7rem;scroll-margin-top:1rem}
.cq-sec h2{margin:0;padding:0;font-size:2.1rem;font-weight:600;line-height:1.1}
.cq-sec p{margin:.35rem 0 0;color:var(--muted);max-width:44rem;font-size:1.05rem}
.cq-sub{font-family:'Barlow Semi Condensed',sans-serif;font-size:1.4rem;font-weight:600;margin:1.8rem 0 .4rem}
.cq-read{font-size:1.12rem;line-height:1.55;max-width:46rem;margin:.6rem 0 1rem}
.cq-small{color:var(--muted);font-size:.9rem;max-width:46rem}
.cq-table{border-collapse:collapse;margin:.6rem 0 1rem;font-size:.95rem;max-width:100%;display:block;overflow-x:auto}
.cq-table th,.cq-table td{text-align:left;padding:.35rem .9rem .35rem 0;border-bottom:1px solid var(--line);white-space:nowrap}
.cq-table th{font-weight:600}
.cq-def{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:1rem 1.6rem;margin:.6rem 0 1.4rem}
.cq-def b{font-family:'Barlow Semi Condensed',sans-serif;font-size:1.2rem;font-weight:600}
.cq-def p{margin:.1rem 0 0;color:var(--muted);line-height:1.4}
.cq-steps{counter-reset:s;list-style:none;padding:0;margin:.6rem 0;display:grid;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:1.2rem}
.cq-steps li{counter-increment:s;border-top:3px solid var(--teal);padding-top:.6rem;line-height:1.45}
.cq-steps li:before{content:counter(s);display:block;font:600 1.6rem 'Barlow Semi Condensed',sans-serif;color:var(--teal)}
.cq-limits{padding-left:1.2rem;max-width:46rem;line-height:1.55}
.cq-foot{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.9rem}

[data-testid="stForm"]{border:1px solid #CFC4C4;border-radius:6px;background:#fff;padding:1.2rem 1.4rem}
.stButton>button,[data-testid="stFormSubmitButton"] button,.stDownloadButton>button{
  background:var(--ink);color:#fff;border:0;border-radius:4px;font-weight:600;padding:.6rem 1.5rem}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover,.stDownloadButton>button:hover{
  background:var(--red);color:#fff}
[data-testid="stExpander"] details{border-radius:6px}
</style>"""


def hero_html():
    beat = "l12 -4 l12 4 h40 l8 6 l12 -54 l14 72 l10 -24 h60 l12 -10 l16 10 h50"
    d = "M0 52 h40 " + " ".join([beat] * 6) + " h400"
    return f"""<div class="cq-hero">
<div class="cq-brand">CardiaQ</div>
<p class="cq-tag">Estimate heart-disease risk from routine test results with two models, one classical
and one quantum, and see how much each can be trusted.</p>
<a class="cq-go" href="#check-a-patient">Check a patient</a>
<svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">
<path class="ghost" d="{d}"/><path class="sweep" pathLength="100" d="{d}"/></svg></div>"""


def section(title, text="", anchor=""):
    p = f"<p>{text}</p>" if text else ""
    a = f' id="{anchor}"' if anchor else ""
    return f'<div class="cq-sec"{a}><h2>{title}</h2>{p}</div>'


# ---------------------------------------------------------------- chart pages
_BASE = FONTS + """
*{box-sizing:border-box}
body{margin:0;padding:4px 2px;font:14px 'Source Sans 3',system-ui,sans-serif;color:#14213D;background:transparent}
.num{font-family:'Barlow Semi Condensed',sans-serif;font-weight:600;font-variant-numeric:tabular-nums}
.k{color:#5B6578;font-size:13px}
"""

_JS = """
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
function onView(el,play,reset){
  if(!('IntersectionObserver' in window)){play();return;}
  let shown=false;
  new IntersectionObserver(es=>es.forEach(e=>{
    if(e.isIntersecting&&!shown){shown=true;play();}
    else if(!e.isIntersecting&&shown&&e.intersectionRatio===0){shown=false;reset&&reset();}
  }),{threshold:[0,.35]}).observe(el);
}
function count(el,to,suffix,ms,dec){
  dec=dec||0;
  if(RM){el.textContent=to.toFixed(dec)+suffix;return;}
  const t0=performance.now();
  (function f(t){const k=Math.min(1,(t-t0)/ms);el.textContent=(to*(1-Math.pow(1-k,3))).toFixed(dec)+suffix;
    if(k<1)requestAnimationFrame(f);})(t0);
}
"""


def _page(css, body, script, data):
    js = script.replace("__DATA__", json.dumps(data))
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{_BASE}{css}</style></head>"
            f"<body>{body}<script>{_JS}{js}</script></body></html>")


def ruler_html(p_rf, p_q, level, color):
    """Both risk estimates as markers on a low / moderate / high scale."""
    css = """
.top{display:flex;gap:2.6rem;align-items:flex-end;flex-wrap:wrap;margin-bottom:8px}
.big{font-size:60px;line-height:1}
.lvl{margin-left:auto;text-align:right}.lvl .num{font-size:32px}
.ruler{position:relative;margin:40px 8px 0;height:18px}
.zones{display:flex;height:18px;border-radius:3px;overflow:hidden}
.zones i{display:block;position:relative;font-style:normal}
.zones i span{position:absolute;left:8px;top:1px;font-size:11.5px;color:#3d4659}
.pin{position:absolute;top:0;width:0;left:0;transition:left 1.4s __E__}
.pin span{position:absolute;white-space:nowrap;transform:translateX(-50%);font-size:12.5px;font-weight:600}
.pin:before{content:"";position:absolute;left:-8px;border:8px solid transparent}
.p1 span{top:-34px;color:#14213D}.p1:before{top:-16px;border-top-color:#14213D;border-bottom:0}
.p2 span{top:38px;color:#0B7A7C}.p2:before{top:18px;border-bottom-color:#0B7A7C;border-top:0}
.ticks{position:relative;height:16px;margin:44px 8px 0;color:#5B6578;font-size:12px}
.ticks span{position:absolute;transform:translateX(-50%)}
.ticks span:first-child{transform:none}.ticks span:last-child{transform:translateX(-100%)}
@media(max-width:560px){.top{flex-wrap:nowrap;gap:1.1rem}.big{font-size:40px}.lvl .num{font-size:22px}.k{font-size:12px}}
""".replace("__E__", EASE)
    body = f"""<div id="r">
<div class="top"><div><div class="k">Classical model</div><div class="num big" id="v1">0%</div></div>
<div><div class="k">Quantum model</div><div class="num big" id="v2" style="color:{TEAL}">0%</div></div>
<div class="lvl"><div class="k">Combined signal</div><div class="num" style="color:{color}">{level}</div></div></div>
<div class="ruler"><div class="zones"><i style="flex:30;background:#DCEBDD"><span>Low</span></i>
<i style="flex:30;background:#F6E7BD"><span>Moderate</span></i><i style="flex:40;background:#F4C9CD"><span>High</span></i></div>
<div class="pin p1" id="m1"><span>Classical</span></div><div class="pin p2" id="m2"><span>Quantum</span></div></div>
<div class="ticks"><span style="left:0">0%</span><span style="left:30%">30%</span><span style="left:60%">60%</span><span style="left:100%">100%</span></div></div>"""
    js = """
const D=__DATA__,m1=document.getElementById('m1'),m2=document.getElementById('m2');
function play(){m1.style.left=D.a*100+'%';m2.style.left=D.b*100+'%';
  count(document.getElementById('v1'),D.a*100,'%',1300);count(document.getElementById('v2'),D.b*100,'%',1300);}
function reset(){[m1,m2].forEach(m=>{m.style.transition='none';m.style.left='0%';m.offsetWidth;m.style.transition=RM?'none':'';});}
if(RM){m1.style.transition=m2.style.transition='none';}
onView(document.getElementById('r'),play,reset);
"""
    return _page(css, body, js, {"a": p_rf, "b": p_q}), 210


def bars_html(items, diverging=False, axis=None, vmax=None):
    """Horizontal bars that grow when scrolled into view.

    items: dicts with label, sub (optional), value, display, color.
    diverging: bars grow left or right from a centre line (for +/- effects).
    axis: optional (left text, right text) shown under a diverging chart.
    vmax: value that fills the whole track (e.g. 1.0 for scores). Default: the largest value.
    """
    mx = vmax or max([abs(i["value"]) for i in items] + [1e-9])
    rows = ""
    for i in items:
        w = abs(i["value"]) / mx * (50 if diverging else 100)
        side = "neg" if (diverging and i["value"] < 0) else "pos"
        sub = f'<small>{i["sub"]}</small>' if i.get("sub") else ""
        rows += (f'<div class="row"><div class="lbl">{i["label"]}{sub}</div>'
                 f'<div class="track"><div class="bar {side}" data-w="{w:.2f}" style="background:{i["color"]}"></div></div>'
                 f'<div class="val num">{i["display"]}</div></div>')
    css = """
.row{display:grid;grid-template-columns:200px 1fr 70px;gap:14px;align-items:center;min-height:46px}
.lbl{line-height:1.2;font-size:14.5px}.lbl small{display:block;color:#5B6578;font-size:12px}
.track{position:relative;height:16px;background:#F1EBEB;border-radius:3px}
.bar{position:absolute;top:0;bottom:0;width:0;transition:width 1s __E__}
.bar.pos{left:__L__}.bar.neg{right:50%}
.val{text-align:right;font-size:19px}
.mid{position:absolute;left:50%;top:-5px;bottom:-5px;width:2px;margin-left:-1px;background:#14213D}
.axis{display:grid;grid-template-columns:200px 1fr 70px;gap:14px;font-size:12px;color:#5B6578;margin-top:6px}
.axis div{display:flex;justify-content:space-between}
@media(max-width:560px){.row,.axis{grid-template-columns:118px 1fr 54px;gap:8px}.lbl{font-size:13px}}
""".replace("__E__", EASE).replace("__L__", "50%" if diverging else "0")
    ax = (f'<div class="axis"><span></span><div><span>{axis[0]}</span><span>{axis[1]}</span></div><span></span></div>'
          if (diverging and axis) else "")
    body = f'<div id="b">{rows}{ax}</div>'
    js = """
const bars=[...document.querySelectorAll('.bar')];
if(__DATA__)document.querySelectorAll('.track').forEach(t=>t.insertAdjacentHTML('beforeend','<div class="mid"></div>'));
function play(){bars.forEach((b,i)=>{b.style.transitionDelay=RM?'0s':(i*70)+'ms';b.style.width=b.dataset.w+'%';});}
function reset(){bars.forEach(b=>{b.style.transition='none';b.style.width='0';b.offsetWidth;b.style.transition=RM?'none':'';});}
if(RM)bars.forEach(b=>b.style.transition='none');
onView(document.getElementById('b'),play,reset);
"""
    return _page(css, body, js, bool(diverging)), 50 * len(items) + (36 if ax else 12)


def percentile_html(rows):
    """Where one patient sits among the training patients. rows: label, text, pct, note."""
    out = ""
    for r in rows:
        out += (f'<div class="p"><div class="h"><span>{r["label"]}</span><span class="num">{r["text"]}</span></div>'
                f'<div class="tr"><i class="dot" data-p="{r["pct"]:.1f}"></i></div>'
                f'<div class="n">{r["note"]}</div></div>')
    css = """
.p{margin:0 0 18px}.h{display:flex;justify-content:space-between;font-size:15px}
.h .num{font-size:20px}
.tr{position:relative;height:10px;margin:8px 9px 5px;border-radius:3px;background:#E9E1E1}
.dot{position:absolute;top:-5px;left:0;width:20px;height:20px;margin-left:-10px;border-radius:50%;
  background:#14213D;border:3px solid #fff;box-shadow:0 0 0 1px #14213D;transition:left 1.3s __E__}
.n{color:#5B6578;font-size:13px}
""".replace("__E__", EASE)
    body = f'<div id="w">{out}</div>'
    js = """
const dots=[...document.querySelectorAll('.dot')];
function play(){dots.forEach((d,i)=>{d.style.transitionDelay=RM?'0s':(i*90)+'ms';d.style.left=d.dataset.p+'%';});}
function reset(){dots.forEach(d=>{d.style.transition='none';d.style.left='0%';d.offsetWidth;d.style.transition=RM?'none':'';});}
if(RM)dots.forEach(d=>d.style.transition='none');
onView(document.getElementById('w'),play,reset);
"""
    return _page(css, body, js, 0), 101 * len(rows) + 6


def waffle_html(models):
    """Test results as 'out of 100 patients' dot grids.

    models: list of dicts with name, color, confusion {tn, fp, fn, tp}.
    """
    blocks = []
    for m in models:
        c = m["confusion"]
        sick, well = max(1, c["tp"] + c["fn"]), max(1, c["tn"] + c["fp"])
        caught = round(c["tp"] / sick * 100)
        cleared = round(c["tn"] / well * 100)
        blocks.append({"name": m["name"], "color": m["color"],
                       "grids": [{"title": "Of 100 people with heart disease", "good": caught,
                                  "goodText": f"{caught} caught", "badText": f"{100 - caught} missed",
                                  "bad": RED},
                                 {"title": "Of 100 people without it", "good": cleared,
                                  "goodText": f"{cleared} correctly cleared",
                                  "badText": f"{100 - cleared} false alarms", "bad": AMBER}]})
    css = """
.model{display:grid;grid-template-columns:150px 1fr 1fr;gap:24px;align-items:start;margin-bottom:22px}
.mname{font:600 21px 'Barlow Semi Condensed',sans-serif;padding-top:22px}
.g h4{margin:0 0 8px;font-weight:600;font-size:13.5px;color:#5B6578}
.grid{display:grid;grid-template-columns:repeat(20,1fr);gap:3px;max-width:330px}
.grid i{aspect-ratio:1;border-radius:2px;background:#ECE4E4;transform:scale(.4);opacity:.35;
  transition:transform .35s __E__,opacity .35s,background .35s}
.leg{display:flex;gap:16px;margin-top:8px;font-size:13px;flex-wrap:wrap}
.leg span:before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;
  background:var(--c);vertical-align:-1px}
@media(max-width:620px){.model{grid-template-columns:1fr 1fr;gap:6px 14px;margin-bottom:14px}.mname{grid-column:1/-1;padding-top:0;font-size:18px}.grid{gap:2px}.g h4{font-size:12px}.leg{gap:2px 10px;font-size:12px}}
""".replace("__E__", EASE)
    body = '<div id="wf"></div>'
    js = """
const D=__DATA__,root=document.getElementById('wf');
D.forEach(m=>{
  let h='<div class="model"><div class="mname" style="color:'+m.color+'">'+m.name+'</div>';
  m.grids.forEach(g=>{
    h+='<div class="g"><h4>'+g.title+'</h4><div class="grid">';
    for(let i=0;i<100;i++)h+='<i data-c="'+(i<g.good?m.color:g.bad)+'"></i>';
    h+='</div><div class="leg"><span style="--c:'+m.color+'">'+g.goodText+'</span><span style="--c:'+g.bad+'">'+g.badText+'</span></div></div>';
  });
  root.insertAdjacentHTML('beforeend',h+'</div>');
});
const cells=[...document.querySelectorAll('.grid i')];
function play(){document.querySelectorAll('.grid').forEach(gr=>[...gr.children].forEach((c,i)=>{
  c.style.transitionDelay=RM?'0s':((i%20)*14+Math.floor(i/20)*40)+'ms';
  c.style.background=c.dataset.c;c.style.transform='scale(1)';c.style.opacity=1;}));}
function reset(){cells.forEach(c=>{c.style.transition='none';c.style.background='';c.style.transform='';c.style.opacity='';
  c.offsetWidth;c.style.transition=RM?'none':'';});}
if(RM)cells.forEach(c=>c.style.transition='none');
onView(root,play,reset);
"""
    return _page(css, body, js, blocks), 176 * len(models) + 8


def vbars_html(labels, metrics, colors, ymax=1.0):
    """Vertical bars with error whiskers and metric tabs.

    labels: one per bar. metrics: {tab name: [(mean, std), ...]}.
    """
    css = """
.tabs{display:flex;gap:6px;margin-bottom:14px;flex-wrap:wrap}
.tabs button{font:600 13.5px 'Source Sans 3',sans-serif;border:1px solid #14213D;background:#fff;color:#14213D;
  padding:6px 15px;border-radius:4px;cursor:pointer}
.tabs button.on{background:#14213D;color:#fff}
.tabs button:focus-visible{outline:3px solid #D7263D;outline-offset:2px}
.plot{position:relative;height:250px;margin-left:36px;border-bottom:2px solid #14213D;
  background:repeating-linear-gradient(to top,#ECE4E4 0,#ECE4E4 1px,transparent 1px,transparent 62.5px)}
.cols,.lbls{display:flex;gap:12px}
.col{position:relative;flex:1;height:250px}
.b{position:absolute;bottom:0;left:50%;width:62%;max-width:64px;transform:translateX(-50%);height:0;
  border-radius:3px 3px 0 0;transition:height 1s __E__}
.w{position:absolute;left:50%;width:2px;margin-left:-1px;background:#14213D;opacity:0;transition:opacity .4s}
.v{position:absolute;left:0;right:0;text-align:center;font-size:18px;opacity:0;transition:opacity .4s}
.lbls{margin:8px 0 0 36px}.l{flex:1;text-align:center;font-size:12.5px;color:#3d4659;line-height:1.25}
.y{position:absolute;left:-36px;font-size:11px;color:#5B6578;width:30px;text-align:right}
""".replace("__E__", EASE)
    body = ('<div id="c"><div class="tabs" id="t" role="tablist"></div><div class="plot">'
            '<span class="y" style="bottom:-7px">0</span><span class="y" style="bottom:118px">__MID__</span>'
            '<span class="y" style="bottom:243px">__TOP__</span><div class="cols" id="cols"></div></div>'
            '<div class="lbls" id="lb"></div></div>').replace("__MID__", f"{ymax / 2:g}").replace("__TOP__", f"{ymax:g}")
    js = """
const D=__DATA__,H=250,names=Object.keys(D.m);let cur=names[0];
const cols=document.getElementById('cols'),lb=document.getElementById('lb'),tb=document.getElementById('t');
D.labels.forEach((l,i)=>{cols.insertAdjacentHTML('beforeend','<div class="col"><div class="b" style="background:'+D.colors[i]+
  '"></div><div class="w"></div><div class="v num"></div></div>');lb.insertAdjacentHTML('beforeend','<div class="l">'+l+'</div>');});
names.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.setAttribute('role','tab');
  b.onclick=()=>{cur=n;show(true);};tb.appendChild(b);});
function show(anim){
  [...tb.children].forEach(b=>{const on=b.textContent===cur;b.className=on?'on':'';b.setAttribute('aria-selected',on);});
  [...cols.children].forEach((c,i)=>{
    const [m,s]=D.m[cur][i],bar=c.children[0],w=c.children[1],v=c.children[2],k=H/D.ymax;
    bar.style.transition='none';bar.style.height='0';w.style.opacity=v.style.opacity=0;bar.offsetWidth;
    bar.style.transition=RM?'none':'';
    w.style.bottom=Math.max(0,m-s)*k+'px';w.style.height=Math.max(0,2*s*k)+'px';
    v.style.bottom=(m+s)*k+8+'px';v.textContent=m.toFixed(2);
    setTimeout(()=>{bar.style.height=m*k+'px';},RM?0:30+i*70);
    setTimeout(()=>{w.style.opacity=1;v.style.opacity=1;},RM?0:900+i*70);
  });
}
function reset(){[...cols.children].forEach(c=>{c.children[0].style.transition='none';c.children[0].style.height='0';
  c.children[1].style.opacity=c.children[2].style.opacity=0;});}
onView(document.getElementById('c'),()=>show(true),reset);
"""
    data = {"labels": labels, "m": metrics, "colors": colors, "ymax": ymax}
    return _page(css, body, js, data), 400


def line_html(xs, ys, xlabel, ylabel, unit=" s"):
    """A line that draws itself on a log y-axis. ys must be positive."""
    W, H, L, B = 640, 270, 56, 44
    lo, hi = math.log10(min(ys)) - .15, math.log10(max(ys)) + .15
    pts = [(L + 20 + i * (W - L - 60) / max(1, len(xs) - 1),
            H - B - (math.log10(v) - lo) / (hi - lo) * (H - B - 30)) for i, v in enumerate(ys)]
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    marks = "".join(
        f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
        f'<text class="pl num" x="{x:.1f}" y="{y - 14:.1f}">{ys[i]:.0f}{unit}</text>'
        f'<text class="xl" x="{x:.1f}" y="{H - B + 20}">{xs[i]}</text>' for i, (x, y) in enumerate(pts))
    css = """
svg{width:100%;max-width:760px;height:auto;overflow:visible;display:block}
.ln{fill:none;stroke:#0B7A7C;stroke-width:3;stroke-dasharray:100;stroke-dashoffset:100;stroke-linejoin:round;
  transition:stroke-dashoffset 1.6s __E__}
.pt{fill:#fff;stroke:#0B7A7C;stroke-width:3;opacity:0;transition:opacity .3s}
.pl{font-size:15px;fill:#14213D;text-anchor:middle;opacity:0;transition:opacity .3s}
.xl,.ax{font-size:12.5px;fill:#5B6578;text-anchor:middle}
""".replace("__E__", EASE)
    body = (f'<svg id="s" viewBox="0 0 {W} {H}" role="img" aria-label="{ylabel} by {xlabel}">'
            f'<line x1="{L}" y1="{H - B}" x2="{W - 10}" y2="{H - B}" stroke="#14213D" stroke-width="2"/>'
            f'<line x1="{L}" y1="16" x2="{L}" y2="{H - B}" stroke="#ECE4E4" stroke-width="1"/>'
            f'<text class="ax" x="{(W + L) / 2}" y="{H - 4}">{xlabel}</text>'
            f'<text class="ax" x="16" y="{(H - B) / 2}" transform="rotate(-90 16 {(H - B) / 2})">{ylabel} (log scale)</text>'
            f'<path class="ln" pathLength="100" d="{d}"/>{marks}</svg>')
    js = """
const ln=document.querySelector('.ln'),els=[...document.querySelectorAll('.pt,.pl')];
function play(){ln.style.strokeDashoffset=0;els.forEach((e,i)=>setTimeout(()=>e.style.opacity=1,RM?0:450+i*120));}
function reset(){ln.style.transition='none';ln.style.strokeDashoffset=100;els.forEach(e=>e.style.opacity=0);
  ln.getBoundingClientRect();ln.style.transition=RM?'none':'';}
if(RM)ln.style.transition='none';
onView(document.getElementById('s'),play,reset);
"""
    return _page(css, body, js, 0), 336
