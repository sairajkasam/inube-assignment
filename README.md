# Traccia profile agent

Reads a conversation with a professional and decides what, if anything, to change on their
profile, using only the tools in `traccia_store.py` (unchanged). Plain Python. The agent loop,
the tool-calling protocol and the retries are all hand-written against the Gemini REST API.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export GEMINI_API_KEY=...                  # Google AI Studio key

# batch: any conversation file
.venv/bin/python agent.py --profile data/profile.json --conversation data/conversation_C-204.json
# chain runs: save state after one, start the next from it
.venv/bin/python agent.py --conversation data/conversation_C-204.json --save-state traces/s1_state.json
.venv/bin/python agent.py --state traces/s1_state.json --conversation data/conversation_C-205.json
# chat: tool calls print as they happen; /changes, /store, /quit
.venv/bin/python agent.py --chat [--state traces/s1_state.json] [--approver interactive]

./run_scenarios.sh                         # regenerates traces/s1..s4
.venv/bin/python -m pytest                 # no API key needed
```

Each run writes `traces/<name>.json` (calls, model turns, `what_changed`, `store_after`) and a
`.txt` copy of the console. Seed: **`TRACCIA_SEED=13`**. With it, the first registry call fails
and the retry succeeds, so S1 shows a retry. (With the default seed 7, the first four calls all fail.)

## How it works

```
agent.py    loop + CLI. Model turn -> execute tool calls -> feed results back, until it replies.
llm.py      Gemini REST client. Retries 429/5xx/timeouts (honours server retryDelay), then falls back to another model.
tools.py    the tool schemas the model sees + ToolExecutor: validation, guardrails, retries, logging.
prompt.py   system prompt (rules + current profile snapshot) and conversation rendering.
tracelog.py trace, deterministic what_changed (a store diff, not the model's words), save/load state.
```

The prompt tells the model the rules. **The guardrails in code make sure they hold** even
when the model ignores them. A blocked call goes back to the model as an error it can fix.

| Guardrail (tools.py) | Catches |
|---|---|
| Evidence must be the professional's own messages, incl. ≥1 from the *new* conversation | citing assistant questions; re-processing old conversations |
| Every number in a write must appear in the cited messages | "60%" derived from 18→7, invented figures |
| `search_profile` must run before any write; duplicate and near-duplicate facts are blocked | "sql" again; "Functional requirements" ≈ "Requirements gathering" |
| A new achievement that overlaps an existing one is blocked unless the model read it and asserts it is different | a second copy of A-001 under a new name |
| A certification is written only if the name equals the single registry match | "a cloud architecture certification" (2 matches, or registry down) |
| `verify_certification`: up to 3 tries with backoff, cached per name | the 40% failure rate; S4 |
| `ask_followup` is terminal, at most once, and never repeats an earlier question | question spam |
| Bad arguments (pydantic, unknown fields), unknown tools, tool-call budget, step budget | malformed model output, loops |
| If any write failed, the final reply is re-grounded against the actual store diff | "I updated X" when X was blocked |

**Destructive actions.** The model never sees `request_confirmation` or tokens. When it calls
`overwrite_achievement`, the harness runs `request_confirmation` itself and asks an `Approver`:
- `additive` (the default) auto-approves only changes that fill empty fields or extend existing text or lists.
- Anything that changes existing content is **not written**; it goes to `flag_for_human_review` with the token.
- A mixed request is split: the additive part is applied and the rest is queued.

`--approver never` queues everything; `--approver interactive` asks you y/n in the terminal.
The `additive` policy is a stand-in for a human reviewer; in production that would be a real queue.

## Results (`traces/`)

| | what_changed (abridged) |
|---|---|
| S1 | A-001 enriched (contribution, outcome 18→7 min / ~120 handlers, period Nov 2025, skills); title/description rewrite queued for review; 3 skills; no certification written, asked which one (registry: AWS vs Google) |
| S2 | "entirely responsible" flagged as conflicting with M4/M11; 60% treated as consistent with 18→7, nothing overwritten |
| S3 | 0 tool calls, 0 writes |
| S4 | registry failed 3×, no certification written, asked for exact name; guardrails blocked an outcome with wrongly cited numbers (model re-cited correctly) and a duplicate skill |

`extra_s*_llm_quota_abort.json` are real runs where the model API hit its quota mid-run:
the run aborts cleanly and logs it.

## Model

Default **`gemini-3.6-flash`**: in my tests it was the best at the judgement calls (S3 with no
tool calls, no near-duplicate skills), it is fast, and it is on the free tier. Override it with
`--model` or `TRACCIA_MODEL`.

**Caveat:** the free tier's daily quota (~20 requests per model) ran out during development,
so the committed traces come from `gemini-3-flash-preview` (S1) and `gemini-3.5-flash-lite`
(S2–S4). The model is recorded in each trace. The weaker models are why several guardrails
exist: each one was added after a real run broke a rule the prompt already stated. Re-run
`run_scenarios.sh` with `TRACCIA_MODEL=gemini-3.6-flash` for a single-model set.

## Missing tools / known gaps

- **No non-destructive "enrich achievement" tool.** Adding an outcome to A-001 has to go
  through `overwrite_achievement` + approval. I wrapped that in the additive auto-approve policy
  (above). A `patch_achievement(fields, evidence)` that can only fill or append would remove the need.
- **No "read whole profile" tool.** The current profile is injected into the system prompt at
  session start; `search_profile` is used to check before writes.
- **No way to supersede or remove a fact**, and follow-ups have no "answered" state. So a repeated
  question is blocked by text similarity, which is crude.
- **Restoring state** for S2/S3: `load_profile` only reads the original profile format, so
  `tracelog.load_state` writes the saved snapshot back into `STORE`. Tool behaviour is untouched.
- Writes are not transactional: if the model API dies mid-run, earlier writes in that run stay
  (logged, never silent).
- Duplicate and near-duplicate detection is lexical. The near-duplicate skill rule is deliberately
  strict (it would block "Claims automation" next to "Claims operations"), trading recall for accuracy.
- The numbers check is digits-only ("eight" is not checked), and the 0.4 overlap threshold for
  duplicate achievements was tuned on this one profile.
- The model's final reply is free text; only the profile writes are guaranteed grounded.

## AI tools used

Claude Code (Claude Opus): helped analyse the brief and store, wrote most of the code, the tests
and this README, and ran the scenarios. I reviewed and can explain every line. I made the design
calls on approval policy, guardrails and model choice, and iterated on them from the traces.
