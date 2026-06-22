"""Static HTML+JS for the dashboard (polls /api/status, no secrets embedded)."""

PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Forecasting bot monitor</title>
<style>
 body{font-family:system-ui,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:20px}
 h1{font-size:18px} .stamp{color:#8a93a2;font-size:12px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:14px}
 .card{background:#171a21;border:1px solid #232833;border-radius:10px;padding:14px}
 .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#9aa4b2;margin:0 0 8px}
 .bad{color:#e0667a}.ok{color:#5bd6a0}.warn{color:#e7b75f}
 table{width:100%;border-collapse:collapse;font-size:13px}td{padding:2px 4px}
</style></head><body>
<h1>Forecasting bot monitor <span class="stamp" id="stamp"></span></h1>
<div class="grid" id="grid"></div>
<script>
async function tick(){
 try{
  const r = await fetch('/api/status'); const s = await r.json();
  document.getElementById('stamp').textContent = 'data as of '+(s.generated_at||'—');
  const g = document.getElementById('grid'); g.innerHTML='';
  for(const [name,p] of Object.entries(s.panels||{})){
   const d=document.createElement('div'); d.className='card';
   const body = p && p.available===false
     ? '<span class="bad">unavailable'+(p.error?(' — '+p.error):'')+'</span>'
     : '<pre style="white-space:pre-wrap;margin:0;font-size:12px">'+
        JSON.stringify(p,null,1)+'</pre>';
   d.innerHTML='<h2>'+name+'</h2>'+body; g.appendChild(d);
  }
 }catch(e){ document.getElementById('stamp').textContent='fetch error: '+e; }
}
tick(); setInterval(tick, 10000);
</script></body></html>"""
