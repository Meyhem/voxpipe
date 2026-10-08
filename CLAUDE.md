# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Design phase: no code yet. The two specs in `design/` are the source of truth. Read them before building anything, and cite their IDs (R-xx, C-xx, N-xx, D-xx, K-xx, A-xx) in commits and discussion.

- `design/domain.md`: business rules (R-01…R-26). What the tool does, its commands, and edge cases.
- `design/tech-spec-voxpipe.md`: architecture. Constraints C-xx, quality targets N-xx, decisions D-xx, risks K-xx, assumptions A-xx.

If a change contradicts a confirmed rule or decision, raise it with the user rather than silently diverging. Anything listed under Assumptions is unconfirmed.

## Git workflow

After finishing a feature or fix (tests and `ruff check` pass, work committed), push to `origin` right away with `git push`, without asking. The remote is `git@github.com:Meyhem/voxpipe.git`, and the branch is `main` unless you're working on another branch. Never force-push. If a push is rejected, stop and report it rather than rebasing or overwriting.

## What voxpipe is

A Python CLI for Linux and PipeWire. It applies a real-time voice effect chain (Adeptus Mechanicus style) to a microphone and outputs the result to a virtual mic that apps like Telegram can select. Subcommands: `list`, `create`, `run`, `tune`, `render`, `clean`. There is no TUI, no daemon, and no network.

## Planned commands

- Run the tool: `uv run voxpipe <subcommand>`. It runs from the repo; there is no packaging (C-03).
- External tools assumed present: `pw-cli`, `pw-dump`, `pw-record`, `pw-play`, `ffmpeg`. `pactl` is **not** available (C-04).

Update this section with the real build, lint and test commands once the project is scaffolded.

## Architecture essentials

- **One stateful block effect engine shared by `run`, `render` and `tune` (D-05).** Audio is float32 mono 48 kHz in 10 ms blocks (D-09). `render` must sound identical to `run`, so never add offline-only effect variants. Every effect must stay streamable and keep its state between blocks.
- **Effects declare their own parameters** (range, default, bypass). The profile JSON schema and the tuner's search space are both generated from those declarations (D-08). Never maintain a separate schema by hand.
- **Profiles are a stable public contract (D-04).** A profile is an ordered `chain` of `{effect, params}` plus `format: "voxpipe-profile"`, `version`, and `provenance`. Validation is strict: any unknown, missing or out-of-range value is an error naming the field. Profiles from a newer version are refused. An incompatible change requires bumping the version and adding an explicit migration. Profile writes are atomic (temporary file, then rename).
- **The device layer is the only code that touches PipeWire (D-02, D-03).**
  - Discovery parses `pw-dump` JSON, never human-readable output.
  - The virtual mic is a lingering null-audio node (`Audio/Source/Virtual`, `object.linger=true`) with a fixed `node.name`, which is a contract.
  - Live I/O goes through `pw-record` and `pw-play` child processes over pipes. A child exiting means the device is gone, and `run` stops.
  - Pass arguments as lists, never through a shell.
- **`run` latency (N-01, assumed ≤ 50 ms, A-01).** Never let latency accumulate: discard backlog beyond a cap, fill late blocks with silence, and count dropouts. Avoid allocation in the per-block loop (K-05).
- **Tuner (D-06, D-07).**
  - DTW alignment of target vs unprocessed reference is computed once per pair; effects don't change timing.
  - Candidates are scored with a weighted log-mel, pitch and long-term-spectrum distance.
  - The optimiser is seeded CMA-ES over normalised parameters, parallel across cores, and must stay deterministic (N-06).
  - Ctrl+C saves the best result so far with `provenance.partial = true`.
- All file input is decoded by an `ffmpeg` child process into the internal format.

## Testing scope

Tests cover the engine, profile schema validation, sample pairing, tuner determinism, and golden-file `render` outputs (compared with a tolerance, K-06). Tests do **not** run against a live PipeWire; live behaviour and sound are checked by hand.
