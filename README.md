# SurfWatch

SurfWatch is a semantic video segmentation project focused on detecting rip-current regions in beach footage. The current project direction is to build a leakage-safe machine learning pipeline around the RipVIS dataset and evaluate models on fully unseen videos rather than mixed frame samples.

## Project Goal

The goal of SurfWatch is to support rip-current detection from video in a way that is practical for real-world beach conditions and honest about model limitations. At this stage, the project is centered on:

- preparing a reproducible dataset pipeline
- converting RipVIS instance masks into a single semantic `rip` class
- training and validating baseline segmentation models
- evaluating only on held-out videos
- documenting decisions, risks, and assumptions clearly

## Dataset And Evaluation Strategy

SurfWatch uses the official RipVIS train/validation/test split. The split is defined at the video level, which means all frames, annotations, and derived masks from a source video must remain in the same partition.

This choice matters because frame-level random splitting can create:

- frame leakage between train and evaluation sets
- temporal leakage from near-duplicate consecutive frames
- misleadingly optimistic metrics
- imbalances across viewpoints, durations, and rip-current types

The current planned split sizes are:

- Train: 112 videos
- Validation: 36 videos
- Test: 36 videos

Further detail is documented in [docs/split-strategy.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/split-strategy.md).

## Current Project Decisions

The repo currently records a few key decisions that shape the project:

- SurfWatch is the project name
- delivery follows 1-week Scrum sprints
- the official RipVIS split is preserved at the video level for all experiments

See [docs/decision-log.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/decision-log.md) for the running decision history.

## Risks And Constraints

The main risks identified so far are:

- insufficient model performance on unseen beach conditions
- scope creep across ML and app work
- demo instability from environment or model issues
- data leakage or biased evaluation from incorrect splitting

Current mitigations include conservative scoping, validation on held-out data, stable demo preparation, and strict preservation of source-video partitions throughout preprocessing and evaluation.

See [docs/risk-register.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/risk-register.md) for the detailed register.

## Literature Context

The project is informed by background review material on:

- RipVIS as the core dataset and benchmark context
- Rip-current detection work relevant to mobile or deployable use cases

Supporting review documents are stored here:

- [docs/lit-review/ripvis-lit-review.pdf](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/lit-review/ripvis-lit-review.pdf)
- [docs/lit-review/RipFinder_mobile_lit-Review.pdf](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/lit-review/RipFinder_mobile_lit-Review.pdf)

## Repository Docs

- [docs/split-strategy.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/split-strategy.md): dataset split policy and leakage controls
- [docs/decision-log.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/decision-log.md): important project decisions
- [docs/risk-register.md](/Users/rashmikecaldera/Developer/curtin/CSP/surfwatch/docs/risk-register.md): tracked delivery and technical risks

## Status

SurfWatch is currently in the planning and documentation phase. The README will expand as the data pipeline, training code, evaluation scripts, and application components are added to the repository.
