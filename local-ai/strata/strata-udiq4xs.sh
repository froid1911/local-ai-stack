#!/bin/bash
# Prepare Strata on the UD-IQ4_XS GGUFs already on /seal1tb -- no download.
# Dropped --low-ram: with 188 GB of RAM it buys nothing here and would make
# setup write a separate 59.5 GB experts.bin (setup.py:4739). Without it the
# pack is built in seconds and the experts are read from the GGUF.
# GPU 2 is now the healthy former GPU 3: the faulty card (serial 1324221009624)
# is removed, so the current three-card layout is 0,1,2.
set -u
cd /home/froid/strata || exit 1
export STRATA_NVCC=/usr/local/cuda-12.9/bin/nvcc
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
export CUDAToolkit_ROOT=/usr/local/cuda-12.9 CUDA_PATH=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
LOG=/home/froid/strata/local/udiq4xs.log
{
  echo "=== started $(date -Is) ==="
  ./setup.sh \
    --family unsloth \
    --model UD-IQ4_XS \
    --gguf-dir /seal1tb/models/Qwen3.8-Flash-Next/UD-IQ4_XS \
    --gpus 0,1,2 \
    --vram-reserve-mib 256 \
    --vision no \
    --data-dir /home/froid/Strata-data \
    --no-start \
    --yes
  echo "=== exit=$? at $(date -Is) ==="
} >>"$LOG" 2>&1
