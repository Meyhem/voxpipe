# Tech Spec: voxpipe (voice-changer CLI)

*Drafted 2026-10-08 from an interview with Michal. Status: **draft**.*
*Domain reference: [domain.md](domain.md) — rule IDs R-xx below refer to it.*

---

## 1. What this is

A single-user Python command-line tool for a Linux desktop running PipeWire. It has six subcommands:

- `list` — show microphones with their identifiers.
- `create` — create the one virtual microphone, or reuse it if it already exists.
- `run` — a foreground, real-time effect loop: source mic → effect chain configured by a profile → virtual mic.
- `tune` — an offline search that fits effect settings to target/reference sample pairs and writes a profile JSON.
- `render` — apply a profile to any audio or video file and write an audio file.
- `clean` — remove the virtual microphone.

The tool has no server, network, database or background process. It calls only PipeWire's own command-line tools and ffmpeg. It owns the profile JSON files and nothing else. Consuming apps such as Telegram see the virtual mic as an ordinary input.

---

## 2. Constraints

Given, not chosen.

- **C-01** — Linux with PipeWire only. *(R-10)*
- **C-02** — CLI with subcommands and options; no TUI or GUI. *(R-01)*
- **C-03** — Written in Python; run from the repository with `uv run`; no packaging or distribution. *(Interview, round 1.)*
- **C-04** — PipeWire is driven through its CLI tools (`pw-cli`, `pw-dump`, `pw-record`, `pw-play`), which are assumed to be present. No native bindings. `pactl` is not available. *(Interview, round 3.)*
- **C-05** — `run` runs in the foreground and stops on Ctrl+C. *(Interview, round 1; R-20.)*
- **C-06** — Profiles are strict: versioned, validated against a schema, fail fast. *(Interview, round 3; R-15 revised.)*
- **C-07** — `render` output must sound identical to what `run` produces live. *(Interview, round 2.)*
- **C-08** — The effect chain must be broad enough to converge on arbitrary target samples. The target voice style is not fixed in advance. *(Interview, round 2.)*
- **C-09** — `tune` may take minutes. Match quality beats speed. *(Interview, round 3.)*
- **C-10** — Development machine: 32-thread Ryzen 9 9950X, ffmpeg 6.1, Python 3.12, uv. *(Observed.)*

---

## 3. Quality targets

- **N-01** — End-to-end added latency (mic → virtual mic) of **≤ 50 ms**. *(Assumed: A-01.)*
- **N-02** — The engine processes one 10 ms block (480 samples at 48 kHz, mono) in **≤ 3 ms** on one core with the full chain enabled. This leaves headroom for scheduling jitter.
- **N-03** — No audible dropouts during a 30-minute call on the development machine under normal desktop load.
- **N-04** — Latency does not drift. Any input backlog beyond a fixed cap is discarded, so delay never accumulates over a long call.
- **N-05** — `tune` finishes with default settings in **≤ 10 minutes** for up to 10 sample pairs of ≤ 10 s each, and shows progress. *(Assumed: A-02.)*
- **N-06** — `tune` is deterministic: the same samples, settings and seed give the same profile.
- **N-07** — Degradation preference for `run`: a brief click or silence (dropout) is better than stopping the call. *(Interview, round 4.)*

---

## 4. Architecture

The tool has five components. The main boundary separates **pure audio processing**, which needs no PipeWire and is fully testable, from **PipeWire I/O**, which talks to the system.

**Effect engine.** The core component.
- An ordered chain of stateful effects. Each effect processes one block of float32 mono 48 kHz samples and keeps its state (filter memories, delay lines, oscillator phase) between blocks.
- Every effect declares its own parameters: name, range, default, and whether it can be bypassed. The profile schema and the tuning search space are both generated from these declarations, so the three can never disagree.
- `run`, `render` and `tune` all drive this same engine with the same block size (C-07). Offline processing just means feeding blocks faster than real time.
- The chain is broad, to meet C-08. The planned effect types:
  - input gain;
  - pitch shift;
  - formant shift;
  - high-pass and low-pass filters;
  - parametric EQ bands;
  - ring modulator;
  - comb filter / metallic resonator;
  - distortion and bit-crush;
  - chorus / doubling;
  - short reverb;
  - dry/wet mix;
  - output gain.

  Each effect has a mix or enable parameter, so tuning can effectively switch it off.

**Device layer.** Wraps the PipeWire CLI tools (C-04).
- Discovery parses `pw-dump` JSON to list `Audio/Source` nodes, including the virtual mic, and resolves identifiers (R-08). The number shown is the PipeWire object ID. The name is `node.name`.
- `create` and `clean` add and remove the virtual mic with `pw-cli`.
- Audio streams in through a `pw-record` child process and out through a `pw-play` child process, as raw float32 mono 48 kHz over pipes, both targeted at specific nodes.

**Live loop (`run`).**
- Reads 10 ms blocks from the `pw-record` pipe, runs them through the engine, and writes them to the `pw-play` pipe.
- Watches the backlog. If more than a few blocks are waiting, the oldest are dropped and counted (N-04, N-07).
- Ends on Ctrl+C, or when either child process exits (R-20, A-07). On exit it prints a summary: duration, dropouts, discarded blocks.

**Media decoding.** All file input (`tune` samples, `render` source) goes through an `ffmpeg` child process that decodes any audio or video format to float32 mono 48 kHz. `render` writes its output as WAV.

**Tuner.** An offline optimiser over the engine's normalised parameter space (§5, D-06, D-07).

**Why this layout.** Keeping PipeWire behind a narrow device layer means the engine, profiles, tuner and `render` are all testable without any audio system. It also means switching later to native PipeWire bindings, if latency demands it (D-03), touches only that layer.

```
 pw-record ─pipe─▶ [live loop] ─▶ [effect engine] ─▶ [live loop] ─pipe─▶ pw-play ─▶ virtual mic ─▶ Telegram
                                       ▲
 ffmpeg ─▶ [render / tuner] ───────────┘        profile.json ◀── tuner
```

---

## 5. Data

**There is no database.** The only persistent artefacts are files the user owns.

**Profile JSON** — the source of truth for one effect configuration.
- Top-level fields:
  - `format: "voxpipe-profile"`
  - `version: <int>`
  - `name`
  - `created` (ISO timestamp)
  - `provenance`: tool version, seed, list of sample-pair IDs, similarity score, `partial` flag, tuning duration
  - `chain`: an ordered list of `{effect, params}`
- The chain order and the effect list live in the profile (D-04). In v1, `tune` always emits the fixed Mechanicus/universal chain order. Hand-built chains with any order are still valid.
- Validation is strict (C-06, R-15):
  - Load fails, naming the field, on wrong `format`, unknown `version`, unknown effect, unknown parameter, missing parameter, or out-of-range value.
  - Profiles from a newer version are refused.
  - Older versions are loaded only through an explicit migration step written when the version is bumped. Silent defaults are never applied.
- Default location is the current directory. Every path can be overridden (R-25, R-26). Writing an existing name overwrites it (R-24).
- Writes are atomic: write to a temporary file, then rename. A crash or Ctrl+C never leaves a truncated profile.

**Samples folder.**
- Read-only input.
- Pairs are `<id>-target.*` plus `<id>-reference.*` (R-09), in any format ffmpeg decodes.
- Incomplete pairs are skipped with a warning (R-16).
- A length mismatch beyond ±30 % between target and reference gives a warning (R-12; the threshold is an assumption, A-03).

**Virtual mic.**
- Lives as PipeWire state, not as tool data.
- Identified only by its fixed `node.name` (R-17).
- Lost on reboot (R-19).

**Internal audio format:** float32, mono, 48 kHz everywhere. All resampling and downmixing happens at the edges (in ffmpeg and in the `pw-*` tools).

---

## 6. Interfaces and contracts

| Interface | Consumers | Stability | Notes |
|---|---|---|---|
| CLI subcommands and flags | the user, shell scripts | Unstable in v1 | Exit code 0 on success, non-zero on any error; errors go to stderr. |
| Profile JSON | the user (hand edits), `run`, `render`, other users (sharing) | **Stable, versioned** | The hardest contract to change. Bump `version` on any incompatible change and ship a migration. |
| Virtual mic `node.name` | `create`, `run`, `clean`, consuming apps | Stable | Changing it orphans existing mics until reboot. |
| Samples folder naming | the user | Stable (R-09) | |
| `pw-*` and `ffmpeg` CLI output | device layer and decoder | External | Parse `pw-dump` JSON, never human-readable text. |

**Idempotency:**
- `create` reuses an existing virtual mic (R-13).
- `clean` succeeds when there is nothing to remove, and says so.
- `tune` overwrites its output (R-24).

---

## 7. External dependencies

| Dependency | Used for | Limits | Behaviour when missing or failing |
|---|---|---|---|
| PipeWire daemon | everything live | — | Every live command fails fast with "PipeWire not running". |
| `pw-dump`, `pw-cli` | discovery, create/clean | — | Startup check; clear error naming the missing tool. |
| `pw-record`, `pw-play` | live audio I/O | pipe buffering adds latency (risk K-01) | Child exits → `run` stops (R-20, A-07). |
| `ffmpeg` | decoding files for `tune` and `render` | — | Clear error naming the missing tool; a file it can't decode, or one with no audio stream, is an error naming the file. |
| Python numeric stack (numpy, scipy) and a JSON-schema validator | engine, tuner, validation | — | Pinned in `uv.lock`. |

---

## 8. Identity, access, security

Not applicable beyond the following:
- Single local user and no network (domain A-01).
- Profiles are data, never code. Loading a profile can never execute anything.
- File paths from the user go straight to `ffmpeg` and `pw-*` as argument lists, never through a shell.

---

## 9. Failure and recovery

| Situation | Behaviour |
|---|---|
| Source mic disappears mid-run | `pw-record` exits → `run` stops with a message and non-zero exit (R-20). |
| Virtual mic disappears mid-run (e.g. `clean`) | `pw-play` exits → `run` stops (A-07). |
| Processing late or CPU stall | The output gets silence for the gap, `run` continues, and dropouts are counted and reported at exit (N-07). |
| Input backlog grows | Blocks beyond the cap are discarded and counted, keeping latency bounded (N-04). |
| `run` before `create` | Error: "virtual microphone not found — run `create`". |
| Source and destination are the same device | Refused before starting (R-11). |
| Invalid profile | Refused before any audio starts, naming the field (R-15). |
| `run` without a profile | Error (R-22). |
| Ctrl+C during `tune` | The best profile so far is saved atomically with `provenance.partial = true` and its score. |
| `tune` finds only a poor match | The profile is saved anyway and the score is reported. The user judges it with `render` (interview, round 3). |
| No complete sample pairs | `tune` fails (R-16). |
| Crash mid-write | Atomic rename, so the previous profile file survives intact. |

---

## 10. Operations

- Only the developer runs the tool, from the repo with `uv run voxpipe …`. There are no environments, deploys or telemetry.
- Logging: human-readable progress on stderr. A `--verbose` flag adds per-second loop stats (block time, dropouts, backlog).
- Configuration comes only from CLI flags. There is no config file in v1.

---

## 11. Key decisions

- **D-01 — Python, numpy/scipy block-based DSP.**
  *Because:* C-03. Each 10 ms block is vectorised, so the interpreter's per-sample cost is avoided.
  *Rejected:* Rust (toolchain absent, slower to build); Node (weak signal-processing tooling).
  *Reversibility:* Moderate. The engine interface is small, so a hot effect could later move to a compiled extension.

- **D-02 — Virtual mic is a lingering PipeWire null-audio node.** It is created with `pw-cli create-node adapter` using `support.null-audio-sink`, `media.class=Audio/Source/Virtual`, `object.linger=true` and a fixed `node.name`.
  *Because:* It must outlive the `create` process (R-17), disappear on reboot (R-19), and be found by name.
  *Rejected:* loopback modules or Pulse null-sinks (no `pactl`, C-04); mics owned by the running process (would break the separate `create` and `run` commands).
  *Reversibility:* Two-way, but the `node.name` is a contract (§6).

- **D-03 — Live I/O through `pw-record` and `pw-play` child processes over pipes.**
  *Because:* C-04. It's the simplest robust way to target exact nodes from Python, and a child exiting is a clean "device gone" signal.
  *Rejected:* PortAudio/sounddevice (can't target specific PipeWire nodes cleanly); native libpipewire bindings (immature in Python, rejected in C-04).
  *Reversibility:* Two-way, contained in the device layer. **Revisit if N-01 is missed** (K-01).
  *Spike 2026-10-08 (PipeWire 1.0.5, `scripts/spike_pipewire.py`):*
  - The virtual node lingers after `pw-cli` exits. `pw-cli destroy` removes it.
  - `pw-record --target <node.name> -` writes raw f32 with no header, and links to the virtual mic's `capture_MONO`.
  - `pw-play --target <virtual mic name>` does **not** link: the stream falls back to the default sink. Start it with `--target 0` and a fixed `-P '{ node.name=… }'`, then link with `pw-link <stream>:output_MONO <node>:input_MONO`.
  - `pw-play` needs `--volume 1.0`. Without it the stream came up with channel volume 0.185 (WirePlumber restored state), attenuating the voice by about 14 dB.
  - Hop latency (write to `pw-play`, read from `pw-record`) is about 32 ms with `--latency 480` and about 10 ms with `--latency 120`.
  - With `-P '{ node.dont-reconnect = true }'`, `pw-record` exits when its target is destroyed instead of moving to another device.

- **D-04 — The profile stores an ordered chain of `{effect, params}` with a format version.**
  *Because:* New voice styles and hand-built chains should need no format change (interview, round 2).
  *Rejected:* a fixed chain in code with a flat parameter list.
  *Reversibility:* **One-way** once profiles are shared.

- **D-05 — One stateful block engine shared by `run`, `render` and `tune`.**
  *Because:* C-07. `render` and the tuner's objective must hear exactly what Telegram hears.
  *Rejected:* separate higher-quality offline effects.
  *Reversibility:* One-way in spirit. Every effect must stay streamable.

- **D-06 — Tuning objective: DTW-aligned spectral distance on paired speech.**
  - The target and the unprocessed reference are aligned once with dynamic time warping on log-mel/MFCC features. Effects in the chain don't change timing, so the alignment can be reused.
  - Each candidate's processed reference is scored on that fixed alignment with a weighted mix of three distances: frame-wise log-mel, pitch contour, and long-term average spectrum.
  - Overall level is not scored. The mean dB difference between processed and target over the scored frames is removed before the spectral distances (objective v4, 2026-10-08). A target's recording volume is arbitrary, and scoring it made the search chase loudness, up to clipping.
  - After the search, a trailing `gain` is set so processed speech is as loud as the user's own reference. This is a pure level change, so the score is unchanged.
  - The result is reported as a 0–100 similarity score.

  *Because:* "Speech as similar as possible to the target" (interview, round 2) needs moment-by-moment comparison. Reusing the alignment keeps each evaluation cheap.
  *Rejected:* an averaged spectrum only (too coarse for "as similar as possible"); re-aligning per candidate (too slow).
  *Reversibility:* Two-way. The score definition is versioned in provenance.

- **D-07 — Optimiser: seeded CMA-ES over normalised parameters, with evaluations spread across all CPU cores.**
  - CMA-ES follows its population mean, so a lucky early candidate can stay unbeaten for a long time. After 15 generations without a new best (or when CMA-ES stops on its own), the search restarts around the best point with a narrower step (σ 0.15) and the next seed (`seed + restart`). (2026-10-08.)

  *Because:* There are 20–40 continuous parameters, the objective is noisy and has no gradient, and the budget is minutes (C-09). Seeding and evaluating in a fixed order make it deterministic (N-06).
  *Rejected:* grid or random search (too weak at this many parameters); gradient-based search through differentiable DSP (much heavier; the GPU isn't needed).
  *Reversibility:* Two-way.

- **D-08 — Strict JSON-schema validation, with the schema generated from effect parameter declarations.**
  *Because:* C-06. A single source of truth means the schema and the engine can't drift apart.
  *Reversibility:* Two-way in code, but strictness is a promise to users.

- **D-09 — Internal format is float32 mono at 48 kHz.**
  *Because:* PipeWire's native rate and a voice-only signal. Conversion happens only at the edges.
  *Reversibility:* Hard once the golden test files exist.

---

## 12. Deferred decisions

- **Native PipeWire I/O.** Only needed if D-03 misses N-01. The device-layer seam keeps it local.
- **Auto-written previews from `tune`.** Domain Q-10b is still open; `render` covers the need today.
- **A config file or remembered devices.** Domain §12 defers these.
- **Moving effects to compiled code.** Only if N-02 isn't met.

---

## 13. Risks

- **K-01 — Pipe buffering in `pw-record`/`pw-play` exceeds the latency budget.**
  *Mitigation:* Request small node latency (e.g. `--latency 480`), keep pipe reads unbuffered, cap the backlog. Fallback: D-03 revisit.
  *Early signal:* Measured round-trip above 50 ms in the first spike.

- **K-02 — Low-latency pitch and formant shifting is hard.** Good-sounding pitch shifting usually needs 20–40 ms of look-ahead, which eats most of N-01.
  *Mitigation:* Use a short-window granular/PSOLA-style shifter and accept some artefacts. Report the effect's latency so the total stays visible.
  *Early signal:* The first pitch-shift prototype's latency.

- **K-03 — The broad chain still can't reach some targets** (e.g. heavy vocoder, layered voices).
  *Mitigation:* Report the score honestly (always saved). New effect types can be added without a format change (D-04).
  *Early signal:* Low scores on Michal's first real sample.

- **K-04 — The tuner overfits noise or reverb in the target** (domain §9).
  *Mitigation:* The user is responsible in v1. A per-pair score breakdown shows which pair is the outlier.

- **K-05 — Python GC or GIL pauses cause dropouts.**
  *Mitigation:* Allocate buffers up front, keep per-block work vectorised, and avoid allocation in the loop. Dropouts are counted, so the problem is visible.
  *Early signal:* Dropout counts in `--verbose`.

- **K-06 — Float results differ across numpy/scipy versions**, breaking golden tests.
  *Mitigation:* Compare with a tolerance, and pin versions in `uv.lock`.

---

## 14. Open questions

- ~~**Q-01** — Final product name~~ — resolved: **voxpipe**.
- **Q-02 — Domain Q-10b: should `tune` write preview audio automatically?** Doesn't block the architecture.

---

## 15. Assumptions

Not confirmed by the user.

- **A-01 — Latency target ≤ 50 ms end to end** (user said "do your best"). Revisit if Telegram calls feel laggy.
- **A-02 — Tuning budget ≤ 10 minutes for ≤ 10 pairs of ≤ 10 s.** The user said "minutes" and "best match". Revisit if the match quality still improves well past 10 minutes.
- **A-03 — R-12 length-mismatch warning threshold is ±30 %.**
- **A-04 — The PipeWire node latency from `pw-record`/`pw-play` can be set to about 10 ms on this machine.** **Verified 2026-10-08:** a hop of about 10 ms with `--latency 120` (see D-03). Not yet measured: mic capture latency and the full chain end to end.
- **A-05 — Mono is sufficient** for both input and the virtual mic.

---

## 16. Out of scope

- TUI or GUI; background daemon; packaging or distribution.
- Native PipeWire bindings, PulseAudio-only and JACK-only systems.
- Model-based voice conversion; GPU use.
- Recording samples inside the tool.
- Automated tests against a live PipeWire. Tests cover the engine, profile schema, sample pairing, tuner determinism, and golden-file `render` outputs. How it sounds and how it behaves with live devices are checked by hand. *(Interview, round 4.)*
- Monitoring of the processed voice (R-21).
