# Local inference

Production set only: the Strata DeepSeek V4.1 Flash container, the Strata
Qwen3.8-Flash-Next variants, vLLM 27B W4A16, llama.cpp LFM3, and the GPU-stop
script. Experimental launch variants from `~/models` are not included.

## Services

| Script | Port | GPUs | What it starts |
|---|---:|---|---|
| `strata/start-dsv41-container.sh` | 18080 | 0,1,2 | DeepSeek-V4.1-Flash-1017 IQ2_XS in container `strata-dsv41` |
| `strata/start-strata-iq3s.sh` | 8090 | 0,1,2 | Qwen3.8-Flash-Next IQ3_S via Strata Python server |
| `strata/start-strata.sh` | 8090 | 0,1,2 | Qwen3.8-Flash-Next UD-IQ4_XS via Strata Python server |
| `vllm/start-27b-w4a16.sh` | 8003 | 0,1 | Qwen3.8-27B W4A16 AutoRound, TP=2 |
| `llama.cpp/start-lfm3.sh` | 8310 | 0 | LFM3-1.7B Q8_0 |
| `stop-inference.sh` | — | all | Kills GPU holders by PID |

## Order and exclusivity

Only one model can occupy a given GPU set. Start by stopping whatever holds
the cards:

```bash
./stop-inference.sh
./strata/start-dsv41-container.sh   # or one of the other servers
```

The Strata container is already running on the reference host and is named
`strata-dsv41`; starting the script again fails because the name exists.
Remove the container first if you want to restart it.

## Host paths

The scripts are written for the reference host:

- `/home/froid/strata` — Strata engine checkout and Python server
- `/home/froid/Strata-data` — packs, native GGUFs, MTP data
- `/home/froid/models` — vLLM/GGUF checkpoints
- `/home/froid/vllm-env-028` — vLLM virtualenv
- `~/llama-cpp-cuda/build/bin/llama-server` — llama.cpp master build with
  `src/models/lfm3.cpp`

Change these paths before running elsewhere. Model weights are not included.

## Strata configs

The JSON files are the server configs: engine binary, pack path, GPU list,
context length, KV type, port. `strata-iq3_s.json` and
`strata-unsloth-ud-iq4_xs.json` were updated from the old `[0,1,3]` GPU list
to the current `[0,1,2]`; `strata-deepseek-v4.1-flash-ist-das-2.25bpw.json`
already used `0,1,2`.

## DeepSeek container

`start-dsv41-container.sh` reproduces the running container:

- image `strata-base:24.04`, host network, `$HOME` bind-mounted
- `strata launch --model ist-das-2.25bpw --family deepseek41 --gpu 0,1,2
  --context 256k --vision off --tune quick --access local --port 18080`
- keeps the container alive with `exec sleep infinity`

The original container also set a Tailscale `STRATA_ALLOWED_HOSTS` value; it
is omitted here because it is host-specific network configuration.
