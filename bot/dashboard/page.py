"""Polished dark "Console" dashboard page (polls /api/status, no secrets, no CDN)."""

PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Forecasting bot · monitor</title>
<style>
:root{--bg:#0b0e14;--panel:#121826;--line:rgba(255,255,255,.07);--text:#e7ebf2;
--muted:#8b95a7;--dim:#5d6678;--accent:#5ad1c4;--ok:#52d39a;--warn:#e6b15e;--bad:#e0667a;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-size:14px;
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:22px 20px 48px}
header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:11px}
.brand .title{font-size:16px;font-weight:600}
.brand .sub{color:var(--dim);font-size:12px}
.pulse{width:9px;height:9px;border-radius:50%;background:var(--ok);animation:p 2s infinite}
@keyframes p{0%{box-shadow:0 0 0 0 rgba(90,209,196,.45)}70%{box-shadow:0 0 0 7px rgba(90,209,196,0)}100%{box-shadow:0 0 0 0 rgba(90,209,196,0)}}
.fresh{color:var(--dim);font-size:12px;font-variant-numeric:tabular-nums}
.strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px;margin-bottom:14px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:13px 15px}
.kpi .l{color:var(--muted);font-size:12px;margin-bottom:7px}
.kpi .v{font-size:23px;font-weight:600;font-family:var(--mono);letter-spacing:-.5px}
.kpi .s{color:var(--dim);font-size:12px;margin-top:3px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.panel h2{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
letter-spacing:.06em;margin:0 0 10px;display:flex;justify-content:space-between;align-items:center}
.panel h2 .meta{font-weight:400;text-transform:none;letter-spacing:0;color:var(--dim);font-size:12px}
.row{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:7px 0;
border-top:1px solid var(--line);font-size:13px}
.row:first-of-type{border-top:none}
.r{font-family:var(--mono);color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:1px}
.chip{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;font-family:var(--mono)}
.unavail{color:var(--warn);font-size:13px}
.muted{color:var(--muted)}
.span2{grid-column:1/-1}
.bar{height:5px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:10px}
.bar>span{display:block;height:100%;background:var(--accent)}
</style></head><body>
<div class="wrap">
 <header>
  <div class="brand"><span class="pulse" id="pulse"></span>
   <div><div class="title">Forecasting bot</div><div class="sub">mopsus · Summer 2026 FutureEval</div></div></div>
  <div class="fresh" id="fresh">connecting…</div>
 </header>
 <div class="strip" id="strip"></div>
 <div class="grid" id="grid"></div>
</div>
<script>
const $=(id)=>document.getElementById(id);
const usd=(n)=>n==null?'—':'$'+Number(n).toFixed(2);
const pct=(n)=>n==null?'—':Math.round(n)+'%';
const DOT={ok:'var(--ok)',warn:'var(--warn)',bad:'var(--bad)',idle:'var(--dim)'};
const dot=(s)=>'<span class="dot" style="background:'+(DOT[s]||DOT.idle)+'"></span>';
const esc=(s)=>(s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmtZ=(iso)=>{if(!iso)return '—';const d=new Date(iso);return String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0')+'Z';};
const dur=(s)=>{if(s==null)return '—';const m=Math.floor(s/60),x=s%60;return m?m+'m'+String(x).padStart(2,'0'):x+'s';};
function kpi(label,val,sub,color){return '<div class="kpi"><div class="l">'+label+'</div><div class="v"'+(color?' style="color:'+color+'"':'')+'>'+val+'</div><div class="s">'+(sub||'')+'</div></div>';}
function panel(title,meta,body){return '<div class="panel"><h2>'+title+(meta?'<span class="meta">'+meta+'</span>':'')+'</h2>'+body+'</div>';}
const unavail=(p)=>'<div class="unavail">unavailable'+(p&&p.error?(' · '+esc(p.error)):'')+'</div>';
const covTotal=(c,k)=>(!c||!c.available)?0:Object.values(c.tournaments||{}).reduce((a,t)=>a+(t[k]||0),0);
const NAMES={'33022':'FutureEval 33022','minibench':'MiniBench'};
let lastGen=null;
function render(s){
 const P=s.panels||{};lastGen=s.generated_at?new Date(s.generated_at):null;
 const armed=P.armed||{},cost=P.cost||{},runs=P.runs||{};
 const degraded=Object.values(P).some(p=>p&&p.available===false)||runs.cadence_stale;
 const down=armed.available&&!armed.run_bot_active;
 const st=down?'bad':(degraded?'warn':'ok');$('pulse').style.background=DOT[st];
 const gap=covTotal(P.coverage,'gap_count'),open=covTotal(P.coverage,'open_count');
 const last=(runs.runs&&runs.runs[0])||null;
 const costColor=cost.pct_used>95?'var(--bad)':(cost.pct_used>80?'var(--warn)':null);
 $('strip').innerHTML=
   kpi('Status',down?'Down':(degraded?'Degraded':'Live'),
       armed.run_bot_active?(armed.submit_prediction==='true'?'armed · submitting':'armed · dry-run'):'workflow off',DOT[st])
  +kpi('Cost · week',usd(cost.window_used),cost.available?('of '+usd(cost.limit)+' · '+pct(cost.pct_used)):'unavailable',costColor)
  +kpi('Coverage gap',(P.coverage&&P.coverage.available)?String(gap):'—',
       (P.coverage&&P.coverage.available)?(open+' open now'):'unavailable',gap>0?'var(--bad)':'var(--ok)')
  +kpi('Last run',last?fmtZ(last.created_at):'—',last?((last.conclusion||'?')+' · '+dur(last.duration_s)):'no runs',
       last&&last.conclusion&&last.conclusion!=='success'?'var(--bad)':null);
 let g='';
 g+=panel('Coverage','',covBody(P.coverage));
 g+=panel('Recent runs',runsMeta(runs),runsBody(runs));
 g+=panel('Resolved · our score',resolvedMeta(P.resolved),resolvedBody(P.resolved));
 g+=panel('Armed state','',armedBody(armed));
 g+=panel('Cost detail','',costBody(cost));
 g+='<div class="span2">'+panel('Alerts','',alertsBody(P))+'</div>';
 $('grid').innerHTML=g;
}
function covBody(c){
 if(!c||c.available===false)return unavail(c);
 const T=c.tournaments||{};let h='';
 for(const id in T){const t=T[id],g=t.gap_count||0;
  h+='<div class="row"><span>'+dot(g>0?'bad':'ok')+(NAMES[id]||id)+'</span><span class="r">'+t.open_count+' open · '+g+' gap</span></div>';
  if(g>0)(t.gap||[]).slice(0,4).forEach(q=>{h+='<div class="row"><span class="muted" style="padding-left:15px">'+esc(q.title||q.post_id)+'</span><span class="r">'+esc(q.type||'')+'</span></div>';});}
 return h||'<div class="muted">no tournaments</div>';
}
function runsMeta(r){if(!r||!r.available)return '';if(r.cadence_stale)return 'stale '+r.last_success_minutes_ago+'m';return r.last_success_minutes_ago!=null?(r.last_success_minutes_ago+'m ago'):'';}
function runsBody(r){if(!r||r.available===false)return unavail(r);
 return (r.runs||[]).slice(0,6).map(x=>'<div class="row"><span class="r">'+fmtZ(x.created_at)+'</span><span>'+dot(x.conclusion==='success'?'ok':(x.conclusion?'bad':'idle'))+(x.conclusion||x.status||'?')+' · '+dur(x.duration_s)+'</span></div>').join('')||'<div class="muted">no runs</div>';}
function resolvedMeta(r){if(!r||!r.available)return '';const sc=(r.resolved||[]).map(x=>x.score&&x.score.peer_score).filter(v=>v!=null);if(!sc.length)return (r.total||0)+' total';const a=sc.reduce((x,y)=>x+y,0)/sc.length;return 'peer avg '+(a>=0?'+':'')+a.toFixed(0);}
function fmtOurs(x){if(x.type==='binary'&&x.summary!=null)return 'we said '+Math.round(x.summary*100)+'%';
 if((x.type==='numeric'||x.type==='date')&&x.median!=null)return 'we said '+Number(x.median).toLocaleString();
 if(x.type==='multiple_choice'&&x.top_option)return 'we said '+esc(x.top_option);return '';}
function resolvedBody(r){if(!r||r.available===false)return unavail(r);
 const rows=(r.resolved||[]).slice(0,6);if(!rows.length)return '<div class="muted">none resolved yet</div>';
 return rows.map(x=>{const peer=x.score&&x.score.peer_score;const c=peer==null?'idle':(peer>=0?'ok':'bad');const ours=fmtOurs(x);
  return '<div class="row"><span>'+dot(x.resolution?'ok':'idle')+esc(x.title||x.question_id||x.post_id)+'</span><span>'+(ours?'<span class="muted" style="margin-right:8px">'+ours+'</span>':'')+(peer!=null?'<span class="chip" style="background:rgba(255,255,255,.06);color:'+DOT[c]+'">peer '+(peer>=0?'+':'')+peer.toFixed(0)+'</span>':'')+'</span></div>';}).join('');}
function armedBody(a){if(!a||a.available===false)return unavail(a);
 return '<div class="row"><span>run-bot workflow</span><span style="color:'+(a.run_bot_active?'var(--ok)':'var(--bad)')+'">'+(a.run_bot_active?'active':'off')+'</span></div>'
  +'<div class="row"><span>submit_prediction</span><span style="color:'+(a.submit_prediction==='true'?'var(--ok)':'var(--warn)')+'">'+esc(a.submit_prediction)+'</span></div>'
  +'<div class="row"><span>secrets</span><span class="r">'+((a.secrets||[]).length)+' set</span></div>';}
function costBody(c){if(!c||c.available===false)return unavail(c);
 const p=Math.min(100,Math.round(c.pct_used||0));
 return '<div class="row"><span>today</span><span class="r">'+usd(c.today)+'</span></div>'
  +'<div class="row"><span>this week</span><span class="r">'+usd(c.week)+' / '+usd(c.limit)+'</span></div>'
  +'<div class="row"><span>this month</span><span class="r">'+usd(c.month)+'</span></div>'
  +'<div class="row"><span>lifetime</span><span class="r">'+usd(c.lifetime)+'</span></div>'
  +'<div class="bar"><span style="width:'+p+'%;background:'+(p>95?'var(--bad)':(p>80?'var(--warn)':'var(--accent)'))+'"></span></div>';}
function alertsBody(P){const a=[];const runs=P.runs||{};
 if(runs.cadence_stale)a.push(['warn','No run-bot run in '+runs.last_success_minutes_ago+'m']);
 if(runs.recent_failures>0)a.push(['bad',runs.recent_failures+' recent run failure(s)']);
 const rs=P.run_summary||{};if(rs.available&&rs.total_truncated>0)a.push(['bad',rs.total_truncated+' question(s) truncated by the per-run cap']);
 const c=P.cost||{};if(c.available&&c.pct_used>80)a.push(['warn','Weekly budget '+pct(c.pct_used)+' used']);
 const gap=covTotal(P.coverage,'gap_count');if(gap>0)a.push(['bad',gap+' open question(s) not yet forecast']);
 Object.entries(P).forEach(([k,p])=>{if(p&&p.available===false)a.push(['warn',k+' data unavailable']);});
 if(!a.length)return '<div class="row"><span>'+dot('ok')+'No alerts — all nominal</span></div>';
 return a.map(([s,m])=>'<div class="row"><span>'+dot(s)+esc(m)+'</span></div>').join('');}
async function tick(){try{const r=await fetch('/api/status');render(await r.json());}catch(e){$('fresh').textContent='fetch error';}}
function freshTick(){if(!lastGen)return;const sec=Math.max(0,Math.round((Date.now()-lastGen.getTime())/1000));$('fresh').textContent='updated '+(sec<60?sec+'s':Math.floor(sec/60)+'m')+' ago';}
tick();setInterval(tick,10000);setInterval(freshTick,1000);
</script></body></html>"""
