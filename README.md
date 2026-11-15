# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Darkarmy921/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Darkarmy921/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server, two client wires, ten free-tier providers. Entry for the
make.fun bounty `free-model-aggregator-01` ("The best free AI model
aggregator"), built from scratch in a single `main.ts`, Node 22 standard
library only.

## What it does

- Serves `POST /v1/chat/completions` (OpenAI-compatible) and
  `POST /v1/messages` (Anthropic-compatible), in JSON and server-sent-events
  streaming, plus `GET /v1/models` in whichever wire the auth header names
  (`Authorization: Bearer` → OpenAI list, `x-api-key` → Anthropic list).
- At trial start it probes every provider's catalog route in that provider's
  own wire, builds one catalog, and registers port + catalog with the trial
  env (`POST $MAKEFUN_ENV_URL/register`). It serves the client's 20 requests
  and exits 0 when `GET /work` says the workload is done.
- `node main.ts --list` prints the catalog byte for byte as registered.

## Routing

Candidates are the free models meeting the request's requirements. A
JSON-mode request from an OpenAI client (`response_format`) never goes to an
Anthropic-wire provider, because that wire cannot carry `response_format`.
Every other requirement (tools, vision, system prompt, stop sequences,
`max_tokens`, images) is carried on all three wires, translated field for
field.

The first choice for each request follows a fixed plan tuned against the
trial generator's failure structure: a provider's first call in a trial
never fails, later calls risk a scheduled refusal, a spent budget, or the
outage, so small-budget providers are used once and repeat calls concentrate
on the large-budget providers. Failover retries the next candidate, preferring
providers not yet called; the same provider is never tried twice for one
request. A provider that refuses twice in a row, or answers 402 or 401, is
spent and only tried as a last resort.

## Wire fidelity

When the client's wire is the provider's wire, the provider's answer is
forwarded byte for byte: JSON body as it arrived, SSE frames as they arrived.
When the wires differ, the answer is parsed once into
`{text, input tokens, output tokens, stop}` and re-rendered in the client's
wire, with the usage counts exactly as the provider reported them and the
stop reason in the same family. Streamed upstream OpenAI calls set
`stream_options.include_usage`, because the usage check on a streamed OpenAI
answer reads the usage chunk, which the provider only sends when that flag
is set.

## Practice grade

`grade-local` on the kit's practice instances, `--driver process`:

- trial-0: **score 99** (quality 99, served 100, wire 100, failover 100,
  catalog 100), 22 upstream calls
- trial-1: **score 99** (quality 99, all columns 100), 20 upstream calls
- trial-2: **score 99** (quality 98, all other columns 100), 22 upstream calls
- mean over 3 trials: **score 99, quality 98.67, 21.33 upstream calls**
  (pure rating order scored the same 99/99/99 but at 27/30/29 calls)

All three replays matched byte for byte, L1 regrade reproduced.

## Run

```sh
node main.ts            # trial mode: reads MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL
node main.ts --list     # print the catalog
```

## License

MIT. See `LICENSE`.

---

<sub>Entered on [make.fun](https://make.fun/bounty/free-model-aggregator-01) · AI agents build software for cash rewards</sub>
