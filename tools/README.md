# Training report build

The published report source is `site/training/report.md`. To regenerate its HTML and machine-readable summary:

```bash
python3 -m pip install -r tools/requirements-report.txt
python3 tools/build_training_report.py
python3 validate_site.py
node --check site/training/report.js
```

The PDF is a separate browser print export using the report print stylesheet. Regenerate it when report content changes, and resolve its relative links against the public report URL before exporting. The evidence manifest hashes the downloadable appendices and figures; it does not include generated HTML or PDF files.
