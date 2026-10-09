#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e '.[test]'
# Separate paths preserve bundled artifacts and measured run.
python -m searchsense.cli ingest data/raw/aol_shard02_first250k.tsv.gz --database data/reproduced.sqlite
python -m searchsense.cli train --database data/reproduced.sqlite --artifacts artifacts_reproduced --epochs 5 --vocab-size 6000 --max-groups 12000
python -m searchsense.cli evaluate --database data/reproduced.sqlite --artifacts artifacts_reproduced --output reports/reproduced_evaluation.json
python -m pytest -q
