# Domain Spec: Voice-Changer CLI (voxpipe)

*Drafted 2026-10-08 from a conversation with Michal. Revised the same day: profiles (effect configurations) are now part of v1; open questions Q-03–Q-06 and Q-09–Q-12 answered. Status: **draft**. The core workflow is settled; items marked as open questions or assumptions are not.*

---

## 1. Purpose and scope

A Linux command-line application that takes audio from a real microphone, applies a chain of audio effects in real time (initially a Warhammer Adeptus Mechanicus style voice), and routes the result into a virtual microphone that other applications (e.g. Telegram) can select as their input.

The CLI also makes the audio setup easy: listing microphones, creating the virtual microphone, and running the live effect loop from a chosen source to a chosen destination. A tuning step fits the effect settings to target voice samples and saves them as a **profile**, which the live loop then uses.

**Covers:** device discovery, virtual microphone creation and cleanup, the real-time effect loop, tuning against voice samples, profiles (saved effect configurations), and rendering a profile onto a file.

**Does not cover:** a terminal UI or GUI (explicitly not wanted), model-based voice conversion, profiles that also remember devices (deferred). Technical choices live in a separate technical spec.

---

## 2. Actors

| Actor | Who they are | What they do here |
|---|---|---|
| **User** | The person running the CLI (initially Michal) | Lists devices, creates the virtual mic, supplies samples, tunes profiles, runs the loop with a profile |
| **Consuming application** | Telegram or any app that records or transmits audio | Selects the virtual microphone as input; outside this tool's control |

---

## 3. Glossary

| Term | Meaning in this domain | Notes |
|---|---|---|
| **Microphone** | An audio input device on the system, real or virtual | |
| **Source microphone** | The microphone the user speaks into | Chosen explicitly when running the loop |
| **Virtual microphone** | An input device created by the CLI, seen by other apps as an ordinary mic | Receives the processed voice; named by the CLI's naming pattern so it can be recognised as ours |
| **Destination** | Where the processed audio goes — normally the virtual microphone | Chosen explicitly when running the loop |
| **Device identifier** | The handle shown by `list` that the user passes to other commands | The number from `list`; the device name also works |
| **Effect chain** | The ordered set of audio effects applied to the voice | v1: one Mechanicus chain |
| **Effect settings** | The parameter values (intensities) of each effect in the chain | |
| **Profile** | A saved effect configuration: a named set of effect settings, stored as a JSON file | Produced by `tune`, consumed by `run`; v1 profiles hold effect settings only, no devices |
| **Effect loop** | The live process: read source → apply effect chain with a profile → write destination | Runs until the user stops it |
| **Samples folder** | The dedicated folder where the user places voice recordings for tuning | User supplies recordings manually |
| **Target sample** | A recording of the voice to imitate (e.g. a Mechanicus clip) | |
| **Reference recording** | The user's own recording of **the same words** as the target sample | |
| **Sample pair** | A target sample plus its matching reference recording | Matched by file name: `<id>-target` + `<id>-reference` |
| **Tuning** | Searching for effect settings that make the user's processed reference recordings sound like the target samples | Result is saved as a profile |

---

## 4. Core concepts

### Microphone
- Any audio input on the system; has a human-readable name and a device identifier.
- `list` shows them so the user can choose a source and see the virtual mic.

### Virtual microphone
- Created by the CLI on request; appears to other apps as a normal input.
- Named using a fixed naming pattern, so later commands can rediscover and reuse it instead of creating a new one.
- Lives only until reboot; it is not made persistent (R-19).
- There is exactly one virtual microphone at a time (R-13).
- Removed by `clean`, even while it is in use (R-18).
- The usual destination of the effect loop.

### Profile
- A named set of effect settings, saved as a JSON file the user can keep, copy, and share.
- Created by **tune**; can also be edited by hand.
- Consumed by **run**: the loop applies the profile's settings to the live voice.
- Records how it was made (when, from which sample pairs, how close the match got), so the user can tell profiles apart.
- In v1 it contains **only effect settings** — not which microphone or destination to use.

### Effect loop
- Connects exactly one source microphone to exactly one destination through the effect chain, configured by one profile.
- Processes audio continuously in small blocks, with low enough delay for a live call or recording.
- Runs until the user stops it, or until the source microphone disappears — then it stops (R-20).
- No monitoring: the user does not hear their own processed voice through the loop (R-21).

### Samples folder and sample pairs
- A dedicated folder the user fills manually.
- Each pair is a target sample and a reference recording of the same words, matched by file name (e.g. `01-target.wav`, `01-reference.wav`).

---

## 5. Relationships

- An effect loop has exactly **one source**, **one destination** and **one profile**.
- Source and destination must be different devices (R-11).
- A sample pair has exactly one target sample and one reference recording, containing the same words.
- One tuning run consumes one or more sample pairs and produces exactly **one profile**.
- A profile can be used by any number of runs; a run never changes the profile.

---

## 6. Workflows

### A. Tune a profile
1. User places sample pairs into the samples folder.
2. User runs **tune**, giving the profile a name.
3. The CLI compares the user's processed reference recordings against the target samples, searches for the effect settings that minimise the difference, and saves them as a profile JSON file.
4. User may listen to the result with `render` (apply the profile to an audio or video file, get an audio file back) and refine the profile by ear (hand editing is allowed).

### B. Set up and run
1. **List** — user lists microphones and sees their identifiers.
2. **Create** — user creates the virtual microphone (once per boot; if one already exists, it is reused). Optional: `run` creates it when it is missing (R-27).
3. **Run** — user starts the effect loop, naming the source, the destination, and the profile.
4. In Telegram (or any app), the user picks the virtual microphone as input.
5. User stops the loop when done.
6. **Clean** (optional) — user removes the virtual microphone.

---

## 7. Commands (user-facing surface)

Subcommands with options; no interactive TUI.

| Command | Purpose | Status |
|---|---|---|
| `list` | Show microphones with identifiers | Confirmed |
| `create` | Create the virtual microphone | Confirmed |
| `run` | Run the effect loop from a source to a destination **using a profile** | Confirmed |
| `tune` | Fit effect settings to the sample pairs and **save them as a profile JSON** | Confirmed |
| `clean` | Remove the virtual microphone (found by naming pattern) | Confirmed |
| `render` | Apply a profile to a source file (any audio or video format) and write an audio file | Confirmed |

*Exact flag names and output formats belong to the technical spec.*

---

## 8. Rules and invariants

Confirmed by the user:

- **R-01** — The application is a subcommand- and option-based CLI with **no terminal UI**. *(hard)*
- **R-02** — The live workflow is list → create virtual mic → run. *(hard)*
- **R-03** — `run` takes an explicit source microphone and an explicit destination. *(hard)*
- **R-04** — The processed voice is delivered to other applications via a virtual microphone. *(hard)*
- **R-05** — Voice samples are supplied manually by the user into a dedicated samples folder. *(hard)*
- **R-06** — `tune` produces a profile (effect configuration) and saves it as a JSON file. *(hard)*
- **R-07** — `run` takes a profile and applies its effect settings to the live voice. *(hard)*
- **R-08** — A device can be identified by the number shown in `list` or by its name. *(hard)*
- **R-09** — Sample pairs are matched by file name: `<id>-target` with `<id>-reference`. *(hard)*
- **R-10** — v1 supports PipeWire only. *(hard — scope)*
- **R-11** — `run` refuses to use the same device as both source and destination (feedback loop). *(hard)*
- **R-12** — A reference recording must contain the same words as its target sample; tuning warns when lengths differ a lot. *(soft)*
- **R-13** — There is only one virtual microphone; `create` does not create a second one, it reports and reuses the existing one (see R-17). *(hard)*
- **R-14** — A profile contains effect settings only; devices are always given on the command line in v1. *(hard)*
- **R-15** — Profiles are strictly validated against a versioned schema: a missing, unknown or out-of-range value is an error naming the field, and the profile is refused. *(hard — revised from lenient during tech grill)*
- **R-16** — A target sample without its reference recording (or vice versa) is skipped with a warning; tuning fails only if no complete pair remains. *(soft)*
- **R-17** — Virtual microphones are named with a fixed naming pattern; `create` and `run` rediscover an existing one by that pattern and reuse it. *(hard)*
- **R-18** — `clean` removes the virtual microphone matching the naming pattern, even if it is in use by a running loop or another app. *(hard)*
- **R-19** — The virtual microphone does not survive a reboot. *(hard)*
- **R-20** — If the source microphone disappears while the loop runs, the loop stops. *(hard)*
- **R-21** — `run` has no monitoring (no playback of the processed voice to the user). *(hard)*
- **R-22** — `run` without a profile is an error; there is no built-in default profile. *(hard)*
- **R-23** — `render` takes a source file in any audio or video format and writes an audio file with the profile applied. *(hard)*
- **R-24** — When `tune` writes a profile whose name already exists, it overwrites it. *(hard)*
- **R-25** — Profiles are read from and written to the current working directory by default. *(hard)*
- **R-26** — Every path the CLI uses (samples folder, profiles location, render input and output) can be set by the user; the current working directory is the default. *(hard)*
- **R-27** — When `run`'s destination names the virtual microphone and it does not exist, `run` creates it (as `create` would) instead of failing. *(hard — added 2026-10-08 at the user's request)*

---

## 9. Edge cases and exceptions

- Source microphone is unplugged while the loop runs → loop stops (R-20).
- `run` is started before the virtual microphone exists, with the virtual microphone as destination → `run` creates it first (R-27). Any other unknown destination is an error.
- The virtual microphone is removed (`clean`) while the loop or an app is still using it → it is removed anyway (R-18); the running loop then loses its destination (A-07).
- Samples folder is empty or contains only incomplete pairs.
- Sample recordings in different formats or sample rates.
- Target samples with heavy reverb or background music (degrades tuning; user's responsibility in v1).
- `tune` would write a profile with a name that already exists → overwritten (R-24).
- `render` is given a video file with no audio track, or a file it cannot decode → clear error.
- Profile file is missing, malformed, or not a voxpipe profile → clear error, loop does not start.
- `run` given no profile → error (R-22).

---

## 10. Open questions

- **Q-10b** — Should `tune` also write preview audio automatically, or is `render` enough?

Resolved since the first draft: device identifiers (R-08), sample pairing (R-09), PipeWire only (R-10), where tuned settings live (R-06), virtual mic naming and cleanup (Q-03 → R-17, R-18), source loss (Q-04 → R-20), persistence (Q-05 → R-19), monitoring (Q-06 → R-21), missing profile (Q-09 → R-22), render (Q-10 → R-23), name collisions (Q-11 → R-24), profile location (Q-12 → R-25).

---

## 11. Assumptions

- **A-01** — Single user on a personal Linux machine. *Not verified.*
- **A-03** — v1 has one effect chain (Mechanicus); profiles differ only in its settings. *Not verified.*
- **A-04** — Effects only, no voice-conversion model in v1. *Follows the pure-effects decision in conversation.*
- **A-05** — The CLI does not record samples itself in v1; the user records them with other tools. *Not verified.*
- **A-06** — Final adjustments to a tuned profile are made by ear, by editing the JSON. *Not verified.*
- **A-07** — When the destination disappears mid-loop (e.g. after `clean`), the loop stops, just as it does when the source disappears (R-20). *Not verified.*

---

## 12. Out of scope (v1)

- Terminal UI and GUI.
- Profiles that remember devices (source/destination) — possible later.
- Model-based voice conversion (RVC, Seed-VC).
- PulseAudio-only or JACK-only systems.
- Cloning or impersonating real people's voices.
- Anything inside the consuming application (Telegram).
