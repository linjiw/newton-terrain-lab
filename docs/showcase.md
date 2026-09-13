# Simulation showcase and recording

The public [GitHub Pages site](https://linjiw.github.io/newton-terrain-lab/) includes three native policy replay videos, an interactive view of the generated map, and workflow command examples. The website is a static showcase, not a browser physics engine or a remotely hosted training service.

## Which policy is shown?

The showcase uses the frozen SONIC teacher at step 8,000, SHA256 `48b3a1c04cdbbcd9ffe8ad10b2591aff781c253c03c61ccb8683348d862c54ed`. In the available local comparison on 20 repaired development motions at seed 91231, it completed 11/20 with mean progress 71.48%. Repaired snapshots at steps 3,400 / 4,700 / 6,200 / 12,500 completed 2 / 4 / 3 / 1 motions respectively. These checkpoints have different training exposure; the result identifies the strongest measured candidate in that comparison, not a universal best SONIC model or a soil-navigation benchmark. The latest checkpoint was not assumed to be the best.

The original toolkit smoke tests used the separately configured repaired motion2scene teacher. The showcase checkpoint is selected only through an environment override; this does not replace the configured teacher or change future training data. Teacher weights and motion/robot assets are not bundled.

## Run and record

Complete the integration guide first. Install rendering dependencies in the worker environment:

```bash
pip install -r requirements-render.txt
pip install --no-deps pyrender==0.1.45
python terrain_doctor.py --native
```

The explicit pyrender install avoids its historical PyOpenGL pin; the renderer uses the PyOpenGL supplied with MuJoCo. The checked path is headless EGL on Linux/NVIDIA. This is optional: normal terrain generation and simulation do not require pyrender.

Select a compatible teacher without changing `local.json`:

```bash
TERRAIN_CHECKPOINT=/absolute/path/checkpoint.pt python prepare_dry_run.py \
  --output outputs/video-01 --num-envs 2 --steps 500 \
  --eval-tiles 1,5 --record-surfaces
python run_dry.py outputs/video-01
python render_dry_showcase.py outputs/video-01 --env 0 \
  --model-label "My evaluated teacher" --output outputs/video-01/sand-render
python render_dry_showcase.py outputs/video-01 --env 1 \
  --model-label "My evaluated teacher" --output outputs/video-01/mud-render
```

Use a fresh output directory. Default tiles 1 and 5 provide sand and mud. `--eval-tiles` holds these evaluation conditions through resets; omit it to visualize curriculum changes. It never changes the generated training runtime's random curriculum allocation.

The callback records root poses, joint positions/names, actions, rewards, done flags, tile IDs, episode generations and control timestamps. At 10 Hz the worker's reconstructed surfaces are saved as compressed numeric arrays. The renderer uses measured 50 Hz robot poses sampled to 25 fps and the most recent surface from the same tile and episode. It never crosses a reset with a stale material mesh. The external coupling introduces a one-physics-step (5 ms in the checked configuration) pose/material timing approximation. No physics is advanced during rendering.

Videos show native automatic failures/resets; an episode counter and tile label make the transitions visible. Smooth surfaces are visualization only. The default camera orbits the active local tile; relocating the tile to local coordinates for display does not imply robots physically teleported to a different map position outside native resets.

Outputs: `motion.mp4`, `poster.jpg`, and a render receipt. Raw simulation archives stay under ignored `outputs/`. The web-hosted files contain only compressed videos, posters, the generated map description and sanitized measurement records. The showcase's fixed stopping-motion reference is diagnostic; these videos are not a multi-motion held-out terrain benchmark.

## Publish changes to the site

Edit `site/`, validate JavaScript with `node --check site/app.js` and validate local assets with `python validate_site.py`. Push to `main`; `.github/workflows/pages.yml` deploys only `site/` through GitHub Pages. It does not publish local configuration, checkpoints, source logs or datasets. The Python package tests run in a separate workflow.
