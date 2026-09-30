from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from src.sentiment_analysis.train import analyze_sentiment

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#10131f">
  <title>Sentiment Studio</title>
  <style>
    :root{color-scheme:dark;--bg:#10131f;--panel:#191d2d;--muted:#a3aabd;--ink:#f5f6fb;--line:#30364b;--accent:#a996ff}
    *{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(ellipse at 12% 0%,#39305e 0,transparent 35%),radial-gradient(ellipse at 100% 95%,#17434b 0,transparent 30%),var(--bg);color:var(--ink);font:16px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    main{width:min(100% - 32px,780px);margin:0 auto;padding:68px 0 48px}.brand{display:flex;align-items:center;gap:11px;color:#c9bdff;font-size:13px;font-weight:700;letter-spacing:.13em;text-transform:uppercase}.mark{display:grid;width:34px;height:34px;place-items:center;border:1px solid #7160b8;border-radius:11px;background:#282342;font-size:19px}
    h1{margin:28px 0 10px;font-size:clamp(38px,8vw,64px);line-height:1.03;letter-spacing:-.055em}h1 span{color:#b8aaff}.intro{max-width:530px;margin:0;color:var(--muted);font-size:17px}
    .card{margin-top:34px;padding:clamp(20px,5vw,34px);border:1px solid #ffffff16;border-radius:24px;background:#191d2dea;box-shadow:0 24px 80px #0005;backdrop-filter:blur(12px)}
    label{display:block;margin-bottom:11px;font-size:14px;font-weight:650}textarea{display:block;width:100%;min-height:166px;resize:vertical;padding:17px;border:1px solid var(--line);border-radius:14px;outline:none;background:#111522;color:var(--ink);font:inherit;line-height:1.6;transition:border-color .2s,box-shadow .2s}textarea:focus{border-color:#9885ff;box-shadow:0 0 0 3px #9885ff26}textarea::placeholder{color:#727b90}
    .row{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-top:13px;color:var(--muted);font-size:13px}.btn{border:0;border-radius:12px;padding:12px 19px;background:linear-gradient(110deg,#b8aaff,#81ddcf);color:#141623;font:inherit;font-size:14px;font-weight:700;cursor:pointer;transition:transform .15s,filter .15s}.btn:hover{transform:translateY(-1px);filter:brightness(1.08)}.btn:disabled{opacity:.65;cursor:wait}
    .examples{display:flex;flex-wrap:wrap;align-items:center;gap:9px;margin-top:22px}.examples span{margin-right:3px;color:var(--muted);font-size:13px}.chip{padding:7px 11px;border:1px solid var(--line);border-radius:999px;background:transparent;color:#d8dbea;font:inherit;font-size:12px;cursor:pointer}.chip:hover{border-color:#9483e6;color:#fff}
    #result{display:none;margin-top:18px;padding:20px;border:1px solid var(--line);border-radius:16px;background:#111522}#result.show{display:block;animation:pop .25s ease-out}.result-top{display:flex;align-items:center;justify-content:space-between;gap:14px}.badge{display:inline-flex;align-items:center;gap:8px;font-size:19px;font-weight:750;text-transform:capitalize}.badge:before{content:"";width:10px;height:10px;border-radius:50%;background:currentColor;box-shadow:0 0 16px currentColor}.positive{color:#69dfb4}.negative{color:#ff8496}.neutral{color:#c0adff}.confidence{color:var(--muted);font-size:13px}.track{height:7px;margin-top:17px;overflow:hidden;border-radius:10px;background:#30364b}.fill{width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,#ff8496,#c0adff,#69dfb4);transition:width .5s}.error{margin-top:15px;color:#ff9ca8;font-size:14px}footer{margin-top:26px;color:#798198;font-size:12px}
    @keyframes pop{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}@media(max-width:540px){main{padding-top:38px}.row{align-items:stretch;flex-direction:column-reverse}.btn{width:100%}.card{border-radius:19px}}
  </style>
</head>
<body>
<main>
  <div class="brand"><span class="mark">✳</span> Sentiment Studio</div>
  <h1>Read the <span>feeling.</span></h1>
  <p class="intro">Understand the tone behind a review, message, or post—in a moment.</p>
  <section class="card" aria-label="Sentiment analyzer">
    <form id="form">
      <label for="text">Your text</label>
      <textarea id="text" maxlength="5000" placeholder="“The support team was kind, but the wait was frustrating…”" required></textarea>
      <div class="row"><span id="count">0 / 5,000</span><button class="btn" id="submit" type="submit">Analyze sentiment&nbsp; →</button></div>
    </form>
    <div class="examples"><span>Try:</span><button class="chip" type="button" data-text="The staff were incredibly helpful and friendly.">Helpful service</button><button class="chip" type="button" data-text="I'm not happy with the long delay and rude support.">Bad experience</button><button class="chip" type="button" data-text="The product arrived today.">Neutral tone</button></div>
    <section id="result" aria-live="polite">
      <div class="result-top"><span id="badge" class="badge"></span><span id="confidence" class="confidence"></span></div>
      <div class="track"><div id="fill" class="fill"></div></div>
    </section>
    <div id="error" class="error" role="alert"></div>
  </section>
  <footer>Runs locally · No text is stored or sent to an external service.</footer>
</main>
<script>
  const form=document.querySelector('#form'),input=document.querySelector('#text'),count=document.querySelector('#count'),result=document.querySelector('#result'),badge=document.querySelector('#badge'),confidence=document.querySelector('#confidence'),fill=document.querySelector('#fill'),error=document.querySelector('#error'),submit=document.querySelector('#submit');
  input.addEventListener('input',()=>count.textContent=`${input.value.length.toLocaleString()} / 5,000`);
  document.querySelectorAll('.chip').forEach(button=>button.addEventListener('click',()=>{input.value=button.dataset.text;input.dispatchEvent(new Event('input'));input.focus()}));
  form.addEventListener('submit',async event=>{event.preventDefault();error.textContent='';result.classList.remove('show');submit.disabled=true;submit.textContent='Analyzing…';try{const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:input.value})}),data=await response.json();if(!response.ok)throw Error(data.error||'Could not analyze text.');badge.className=`badge ${data.sentiment}`;badge.textContent=data.sentiment;confidence.textContent=`${data.confidence}% signal`;fill.style.width=`${data.confidence}%`;result.classList.add('show')}catch(err){error.textContent=err.message||'Could not reach the analyzer. Please try again.'}finally{submit.disabled=false;submit.innerHTML='Analyze sentiment&nbsp; →'}});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    value = payload.get("text", request.form.get("text", ""))
    if not isinstance(value, str) or not value.strip():
        return jsonify({"error": "Please provide some text to analyze."}), 400
    text = value.strip()
    return jsonify({"text": text, **analyze_sentiment(text)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
