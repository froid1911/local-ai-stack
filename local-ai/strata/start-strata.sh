#!/bin/sh
# Start Strata. The generated run-unsloth-ud-iq4_xs.sh hardcodes --port 8080,
# which is the Ethereum beacon-chain node here, and --open (browser popup).
# GPUs come from the config's "gpu" key: [0,1,2] on the current three-card host.
# The formerly faulty card (serial 1324221009624) is removed.
cd /home/froid/strata
exec /home/froid/strata/.venv/bin/python /home/froid/strata/serve/server.py \
  --engine strata \
  --config /home/froid/strata/strata-unsloth-ud-iq4_xs.json \
  --port 8090
