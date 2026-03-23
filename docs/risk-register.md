# Risk Register — SurfWatch

| ID | Date added | Risk | Likelihood | Impact | Mitigation (prevent) | Contingency (if happens) | Owner | Status/Notes |
|---|---|---|---|---|---|---|---|---|
| R1 | 2026-03-11 | Insufficient model performance on unseen beach conditions | Med | High | Use public dataset + baseline model; validate on held-out set; conservative thresholds | Human-in-the-loop review mode; label “decision support only” |  | Open |
| R2 | 2026-03-11 | Scope creep (too much ML + too much app) | Med | High | Lock MVP; prioritise backlog; timebox extras | Drop non-core features; focus on demo workflow |  | Open |
| R3 | 2026-03-11 | Demo failure (environment/model issues) | Low | High | Seed demo data; rehearse; keep stable model version | Fallback recorded demo + screenshots |  | Open |
| R4 | 2026-03-24 | Data leakage or biased evaluation caused by splitting frames from the same video across train, validation, and test | Med | High | Preserve the official RipVIS train/validation/test split at the video level; keep all derived frames, masks, and annotations in the same source-video partition | Rebuild the dataset partitions from the original RipVIS split metadata and rerun affected experiments |  | Open |
