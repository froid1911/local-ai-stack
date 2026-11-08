#!/bin/sh
# Start Strata on IQ3_S. GPUs come from the config's "gpu" key [0,1,2].
# The formerly faulty card (serial 1324221009624) is removed; this host now
# has three 3090s at indices 0,1,2.
cd /home/froid/strata
exec /home/froid/strata/.venv/bin/python /home/froid/strata/serve/server.py \
  --engine strata \
  --config /home/froid/strata/strata-iq3_s.json \
  --port 8090
