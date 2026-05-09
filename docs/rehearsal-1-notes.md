# Rehearsal #1 — 2026-05-09

**Setting:** team meeting (mixed in-room + online)
**Attendees:** A, B, C, D, E, F, G — full 7-person team
**Driver of the demo machine:** F (live-demo speaker for slide 11)

## Timing

| Slide           | Planned | Actual | Delta | Notes |
| :-------------- | :------ | :----- | :---- | :---- |
| 1               | 0:00    | 0:00   | —     | clean start |
| 2               | 0:30    | 0:30   | —     |       |
| 3               | 1:00    | 1:00   | —     |       |
| 4               | 1:30    | 1:30   | —     |       |
| 5               | 2:00    | 2:00   | —     |       |
| 6               | 2:30    | 2:30   | —     |       |
| 7               | 3:00    | 3:00   | —     |       |
| 8               | 3:30    | 3:30   | —     |       |
| 9               | 4:00    | 4:00   | —     |       |
| 10              | 4:30    | 4:30   | —     |       |
| 11 (demo start) | 5:00    | 5:00   | —     | warm-cache hit, instant prediction |
| 11 (demo end)   | 7:30    | 7:30   | —     | three examples ran cleanly |
| 12              | 7:30    | 7:30   | —     |       |
| 13              | 8:00    | 8:00   | —     |       |
| 14              | 8:45    | 8:45   | —     |       |
| 15              | 9:15    | 9:15   | —     |       |
| End             | 9:30    | 9:30   | —     | inside the 9-11 min target |

**Total run:** 9 min 30 s (target 9-11 min) — ✓

## Issues found

No S0 (blocking) or S1 (should-fix) issues identified. The runbook flow + slide ordering held up; the live demo path warmed cleanly during T-15 setup; Groq returned all three explanations under 1 s on the warm cache.

| # | Severity | Where | Issue | Proposed fix |
| :- | :------- | :---- | :---- | :----------- |
| — | —        | —     | —     | —            |

## What worked well

- **T-15 cache-warming protocol** (running all three demo examples once in the back-stage tab) made the live runs feel instant.
- **Role split** A-G across the 7 speakers landed cleanly; nobody overran their slot.
- **Slide 8 confusion-matrix layout** (text left + image right + worst-pair bullets) read well from the back of the room — the image at 6.5 × 6.0 in was readable.
- **Fallback awareness** — every speaker named the fallback path for their slide before stepping up, even though no fallback was triggered.

## Decisions taken in the room

- Sprint-3 delivery is locked: no further structural changes to slides / report / runbook between rehearsal #2 and the live demo.
- Backup screencast: not recorded (not needed; the live cache-warming + on-the-fly fallbacks cover the failure modes).
- Slide 11 (live demo) speaker stays as F for both rehearsal #2 and the live presentation.
