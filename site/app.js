const $ = (s) => document.querySelector(s);
const colors = { ground: '#607581', sand: '#c29a58', mud: '#694537', rocks: '#adc0ca' };
let evidence;
function updateClipStats(name){ if(evidence){ $('#clip-dones').textContent=evidence.clips[name].automatic_done_count + ' / 10 s'; } }
const cases = {
  mixed: {video: 'mixed.mp4', poster: 'mixed-poster.jpg', caption: 'Starts on sand + mud + rocks. The curriculum demotes after native failures; tile and episode changes remain visible.'},
  sand: {video: 'sand.mp4', poster: 'sand-poster.jpg', caption: 'Sand and surrounding ground · measured policy motion and reconstructed granular material.'},
  mud: {video: 'mud.mp4', poster: 'mud-poster.jpg', caption: 'Mud and mixed ground · measured policy motion and reconstructed viscoplastic material.'}
};
document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => {
  const item = cases[button.dataset.case];
  document.querySelectorAll('[data-case]').forEach(b => { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); });
  const video = $('#case-video'); video.pause(); video.poster = 'media/' + item.poster; video.src = 'media/' + item.video; video.load();
  $('#video-caption').textContent = item.caption;
  updateClipStats(button.dataset.case);
  video.play().catch(() => {});
}));
// Video stays user-controlled; reduced-motion and mobile visitors never get forced autoplay.
let course;
function selectTile(id) {
  const cell = course.cells[id];
  document.querySelectorAll('.map-tile').forEach(b => { const active = Number(b.dataset.tile) === id; b.classList.toggle('selected', active); b.setAttribute('aria-pressed', String(active)); });
  $('#tile-id').textContent = `TILE ${String(id).padStart(3,'0')} / LEVEL ${cell.level}`;
  $('#tile-name').textContent = cell.name.split('+').join(' + ');
  $('#tile-origin').textContent = `${cell.origin[0]}, ${cell.origin[1]} m`;
  $('#tile-rocks').textContent = cell.rocks.length;
  const svg = $('#tile-plan'); svg.replaceChildren();
  function rect(lo, hi, color) { const r = document.createElementNS('http://www.w3.org/2000/svg','rect'); r.setAttribute('x',lo[0]); r.setAttribute('y',6-hi[1]); r.setAttribute('width',hi[0]-lo[0]); r.setAttribute('height',hi[1]-lo[1]); r.setAttribute('fill',color); svg.append(r); }
  rect([-2,-2],[8,8], '#334953'); cell.patches.forEach(p => rect(p.lo,p.hi,colors[p.material])); cell.rocks.forEach(r => rect(r.lo,r.hi,colors.rocks));
  const title=document.createElementNS('http://www.w3.org/2000/svg','title'); title.textContent=`Tile ${id}: ${cell.name}, ${cell.rocks.length} fixed rock proxies`; svg.prepend(title);
}
fetch('course-map.json').then(r => {if(!r.ok)throw Error('Map unavailable');return r.json();}).then(data => {
  course=data; const grid=$('#map-grid');grid.replaceChildren();
  // North-up display: reverse rows while preserving each cell's original ID.
  const columns=16;
  for(let row=15;row>=0;row--) for(let col=0;col<columns;col++) {
    const c=course.cells[row*columns+col]; const b=document.createElement('button'); b.className='map-tile';b.dataset.tile=c.id;b.dataset.level=c.level;
    const mats=[...new Set(c.patches.map(p=>p.material))];
    b.style.background=mats.length===1?colors[mats[0]]:`linear-gradient(90deg,${mats.map((m,i)=>`${colors[m]} ${i*100/mats.length}% ${(i+1)*100/mats.length}%`).join(',')})`;
    b.classList.toggle('has-rocks',c.rocks.length>0); b.setAttribute('aria-label',`Tile ${c.id}, ${c.name}, level ${c.level}`);b.title=`${c.id} · ${c.name} · level ${c.level}`;
    b.addEventListener('click',()=>selectTile(c.id));grid.append(b);
  }
  selectTile(11);
}).catch(() => { $('#map-grid').textContent='The map could not be loaded. Please reload or open the generator on GitHub.'; });
$('#level-filter').addEventListener('change',e=>document.querySelectorAll('.map-tile').forEach(b=>b.classList.toggle('dimmed',e.target.value!=='all'&&b.dataset.level!==e.target.value)));
const commands = {
 generate: () => `git clone https://github.com/linjiw/newton-terrain-lab.git\ncd newton-terrain-lab\npython3.11 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\npython dry_course.py --output assets/dry-map --seed 23\npython export_mujoco.py --course assets/dry-map/course.json \\\n  --output assets/dry-map/terrain.xml`,
 evaluate: n => `# In the cloned repo, with the map generated:\npip install -r requirements-sim.txt\ncp local.example.json local.json\n# Edit local.json: native Python, SONIC, robot, model & data.\npython prepare_dry_run.py --output outputs/eval-01 \\\n  --num-envs ${n} --initial-level 3 --steps 500 --surfaces\npython run_dry.py outputs/eval-01`,
 train: n => `# After native integration and a short evaluation pass:\npython prepare_dry_run.py --output outputs/train-01 \\\n  --num-envs ${n} --training-iterations 100 \\\n  --rollout-steps 24 --ppo-epochs 1\n# Review train.yaml. Training starts at ground level 0.\npython run_dry.py outputs/train-01 --mode train`,
 render: n => `# After native integration setup:\npip install -r requirements-render.txt\npip install --no-deps pyrender==0.1.45\npython prepare_dry_run.py --output outputs/video-01 \\\n  --num-envs ${n} --steps 500 --record-surfaces\npython run_dry.py outputs/video-01\npython render_dry_showcase.py outputs/video-01 \\\n  --env 0 --output outputs/video-01/render`
};
const notes={generate:'CPU-only geometry export. No teacher checkpoint or Isaac installation required.',evaluate:'Requires the compatible SONIC TRL fork and your assets. Two/four-env native runs have been tested; larger batches need validation.',train:'A fresh optimizer warm start, not checkpoint resume. Map capacity is 64 envs per level; throughput and learning at that scale are unvalidated.',render:'Records native joint/root states and material surfaces. Offline rendering does not step physics. Episode resets remain visible.'};
function updateCommands(){const mode=$('#workflow').value,n=$('#env-count').value;$('#env-output').value=n;$('#env-count').disabled=mode==='generate';$('#commands').textContent=commands[mode](n);$('#workflow-note').textContent=notes[mode];$('#command-title').textContent=mode.toUpperCase();$('#copy-status').textContent='';}
$('#workflow').addEventListener('change',updateCommands);$('#env-count').addEventListener('input',updateCommands);updateCommands();
$('#copy-command').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#commands').textContent);$('#copy-status').textContent='Copied. Follow the setup notes before launching.';}catch{$('#copy-status').textContent='Select the command text and copy it manually.';}});
fetch('evidence.json').then(r=>r.json()).then(e=>{evidence=e;updateClipStats(document.querySelector('[data-case].active').dataset.case);$('#run-status').textContent=e.native_runs.every(r=>r.exit_code===0)?'Both exited successfully':'See measurement record';}).catch(()=>{$('#run-status').textContent='See measurement record';});
