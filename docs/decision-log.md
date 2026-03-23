# Decision Log — SurfWatch

| Date | Decision | Why | Alternatives considered | Impact | Owner |
|---|---|---|---|---|---|
| 2026-03-11 | Use Scrum with 1-week sprints | Iterative delivery + early validation | Waterfall, Spiral | Defines delivery cadence |  |
| 2026-03-11 | Project name: SurfWatch | Clear and relevant | RipWatch, WaveGuard | Repo/Jira naming |  |
| 2026-03-24 | Use the official RipVIS train/validation/test split at the video level | Prevents frame and temporal leakage and preserves the dataset's curated balance across viewpoints, durations, and rip-current types | Automatic random frame split, custom video split | Keeps entire source videos and derived frames/masks in one partition; test split remains untouched until final evaluation |  |
