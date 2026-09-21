"""Render the portable technical report; requires Markdown 3.x."""
from pathlib import Path
import json
import re
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site/training'
text = (OUT / 'report.md').read_text()
md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'], extension_configs={'toc': {'toc_depth': '2-3'}})
body = md.convert(text.split('\n', 1)[1])
body = re.sub(r'<table>', '<div class="table-scroll" tabindex="0" role="region" aria-label="Scrollable technical table"><table>', body)
body = body.replace('</table>', '</table></div>')
body = body.replace('<table>\n<thead>\n<tr>\n<th>Reward term</th>', '<table class="reward-table">\n<thead>\n<tr>\n<th>Reward term</th>')
body = re.sub(r'<img alt="([^"]*)" src="([^"]*)"\s*/>', r'<a class="figure-link" href="\2" aria-label="Open full-size figure: \1"><img alt="\1" src="\2" loading="lazy"></a>', body)
result = json.loads((OUT / 'appendices/comparison.json').read_text())
assert result['soil_task_failures'] == {'start': 56, 'scheduled': 53, 'target': 54}
summary = dict(report_date='2026-09-21', study_date='2026-09-20', study_status='completed',
               checkpoint_promoted=False, terrain_ready=False,
               failures=result['soil_task_failures'], aggregate=result['aggregate'],
               acceptance=result['acceptance'], retained_rollouts_per_arm=256,
               retained_transitions_per_arm=24576, evaluation_seconds_per_family=30,
               evaluation_seed=23, evaluation_clips=3,
               starting_checkpoint_sha256=result['plans']['scheduled']['checkpoint_sha256'])
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sand & mud training report · Newton Terrain Lab</title>
<meta name="description" content="Complete SONIC terrain-training report: rewards, reset events, PPO, G1 actuation, MPM sand/mud physics, curriculum experiments and measured failures. Updated 21 September 2026.">
<meta name="theme-color" content="#0a141b"><link rel="icon" href="../favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="report.css"><script src="report.js" defer></script></head><body>
<a class="skip" href="#report">Skip to report</a>
<header class="site-header"><a class="wordmark" href="../">NEWTON<span>TERRAIN LAB</span></a>
<nav aria-label="Report actions"><a href="../#training">Showcase</a><a href="report.md" download>Markdown ↓</a><a href="report.pdf" download>PDF ↓</a><button id="print-report" type="button">Print</button></nav></header>
<main>
<section class="report-hero" aria-labelledby="report-title"><p class="eyebrow">TRAINING LOG / 21 SEPTEMBER 2026</p>
<h1 id="report-title">Motion tracking<br>on <span>sand and mud.</span></h1>
<p class="lead">Two controlled training runs completed. The terrain policy still falls short of its tracking and stability requirements.</p>
<p class="status"><span class="status-dot" aria-hidden="true"></span>Study complete · No checkpoint promoted</p>
<div class="score-grid" aria-label="Original-depth soil task failures"><div><span>Starting policy</span><strong>56</strong><small>soil task failures</small></div><div><span>Depth curriculum</span><strong>53</strong><small>5.36% fewer · acceptance failed</small></div><div><span>Direct training</span><strong>54</strong><small>3.57% fewer · acceptance failed</small></div></div>
<p class="denominator">Original 14 cm soil · 3 development clips · seed 23 · 30 s per terrain family.<br>Each trained arm added 256 retained rollouts. Counts include automatic resets.</p>
</section>
<section class="kernel" aria-labelledby="kernel-title"><div><p class="eyebrow">INSPECT THE PROGRAMMED OBJECTIVE</p><h2 id="kernel-title">How quickly does root tracking reward fade?</h2><p>Change position error to inspect the actual baseline kernel. This is a formula illustration, not a predicted learning result.</p><code>0.5 × exp(−XY² / 0.30² − Z² / 0.45²)</code></div><div class="kernel-controls"><label for="xy-error">Horizontal error <output id="xy-label">0.00 m</output></label><input id="xy-error" type="range" min="0" max="1.5" step="0.01" value="0"><label for="z-error">Vertical error <output id="z-label">0.00 m</output></label><input id="z-error" type="range" min="0" max="0.6" step="0.01" value="0"><div class="kernel-result"><span>Weighted rate <output id="kernel-rate" aria-live="polite">0.500000</output></span><span>20 ms contribution <output id="kernel-step">0.010000</output></span></div><noscript>The formula above remains available; enable JavaScript to adjust its inputs.</noscript></div></section>
<div class="report-layout"><aside class="contents"><details open><summary>In this report</summary>{md.toc}</details><a class="evidence-link" href="evidence-manifest.json">Evidence SHA256 manifest ↗</a><a class="evidence-link" href="appendices/comparison.json">Exact measured results ↗</a></aside>
<article id="report" class="report-body">{body}</article></div>
</main><footer><a href="../">← Newton Terrain Lab</a><p>Completed experiments, preserved failures, explicit uncertainty.<br>External weights, motion assets and raw trajectories are not distributed here.</p><a href="#report-title">Back to top ↑</a></footer>
</body></html>'''
(OUT / 'index.html').write_text(page)
print(json.dumps(dict(report_words=len(text.split()), sections=len(re.findall(r'^## ', text, re.M)), bytes=len(page.encode()))))
