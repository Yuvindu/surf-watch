# SurfWatch — Dataset Card

> **SCRUM-20** · Parent: SCRUM-18 Dataset & Data Preparation  
> Last updated: 2026-04-04

---

## Dataset: RipVIS v1.8.4

### Source
- **Name:** RipVIS — Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety
- **URL:** https://huggingface.co/datasets/Irikos/RipVIS
- **Website:** https://ripvis.ai
- **Paper:** https://arxiv.org/abs/2504.01128 (accepted at CVPR 2025)
- **Provider / Authors:** Andrei Dumitriu, Florin Tatui, Florin Miron, Aakash Ralhan, Radu Tudor Ionescu, Radu Timofte — University of Würzburg (Computer Vision Laboratory) & University of Bucharest (Faculty of Mathematics and Computer Science / Faculty of Geography)
- **Access method:** Public download via HuggingFace Datasets

---

### Licence
| Field | Detail |
|---|---|
| Licence name | Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) |
| Commercial use permitted? | **No** — non-commercial use only (commercial licensing available on request via andrei.dumitriu@uni-wuerzburg.de) |
| Attribution required? | Yes |
| Redistribution restrictions | Full dataset hosting/mirroring not permitted without written permission; RipVIS may constitute ≤ 20% of any derivative dataset without permission |
| Full licence URL | https://creativecommons.org/licenses/by-nc/4.0/ |

---

### Size
| Field | Detail |
|---|---|
| Total videos | 184 (150 with rip currents, 34 without rip currents) |
| Additional training images | 2,466 annotated images from Dumitriu et al. (CVPRW 2023) — included in `train_with_additional_data.json` |
| Train split | 112 videos (90 rip + 22 no-rip) |
| Validation split | 36 videos (30 rip + 6 no-rip) |
| Test split | 36 videos (30 rip + 6 no-rip) |
| Approx. storage size | Not stated — video dataset; see HuggingFace for current file sizes |
| Split method | Manual split by two rip current domain experts to prevent data leakage and ensure balanced distribution |

---

### Annotation Type
- **Format:** Instance segmentation — COCO JSON (`.json`) and YOLO (`.txt`) formats provided
- **Classes:** `rip_current` (annotated instances), `no_rip_current` (NR videos — no annotations)
- **Annotation tool:** Roboflow
- **Annotation level:** Per-frame polygon instance segmentation on sampled frames from each video
- **Annotators:** Domain experts (rip current researchers from University of Würzburg and University of Bucharest)
- **Evaluation scripts:** `compute_coco_ap.py` (AP50, AP[50:95]) and `compute_pr_f1_f2.py` (F1, F2) included

---

### Viewpoint Diversity
| Attribute | Coverage |
|---|---|
| Data modality | Video sequences (shore-facing fixed camera) + sampled frames |
| Rip current types | Multiple morphologies across 150 rip-current videos |
| Non-rip examples | 34 dedicated no-rip videos for negative training signal |
| Video duration & sampling rate | Varies across videos; expert-balanced across splits for duration and orientation diversity |
| Geographic locations | Multiple locations (international collaboration dataset) |
| Associated challenge | AIM 2025 RipSeg Challenge at ICCV 2025 — indicating broad community benchmark coverage |

---

### Strengths
- **CVPR 2025 benchmark** — peer-reviewed, high-quality expert-annotated dataset purpose-built for rip current detection
- **Video + frame data** — enables both video instance segmentation and still-image detection approaches
- **Dual annotation formats** — COCO JSON and YOLO formats available out of the box, compatible with common training pipelines (YOLOv8, Detectron2, MMDetection, etc.)
- **Expert train/val/test split** — manually split by rip current domain experts to prevent data leakage and ensure realistic evaluation
- **Additional training data included** — extra 2,466 annotated images from CVPRW 2023 predecessor paper available for augmenting the training set
- **Negative examples included** — 34 no-rip videos provide critical negative class signal
- **Actively maintained** — v1.8.4 as of September 2025, with future fixes and challenges planned

---

### Limitations & Known Issues
1. **Jittery annotations in some videos:** Several videos were annotated using Roboflow's default "Newest" file sorting rather than by filename order, causing annotations to appear visually jittery when overlaid on video. This does not affect quantitative metrics but is a visual issue. Affected videos: RipVIS-001, 003, 004, 006, 007, 011, 012, 014, 016, 017, 018, 020, 021, 022, 024, 033, 034, 035, 036, 037, 039, 040.
2. **Frame number mismatch:** For some videos, frame numbers in filenames do not correspond to the actual frame position in the video due to annotation pipeline issues. The correct frame can be estimated using sampling rate and total frame count. A fix is planned in a future release.
3. **Missing annotated frames:** A small number of annotated sampled frames were removed due to invalid annotation formatting introduced during the Roboflow pipeline. A fix is planned.
4. **No-rip videos are unannotated:** NR (no-rip) videos contain no annotations by design — models must infer the absence of rip currents from video content alone.
5. **Licence restricts commercial deployment:** CC BY-NC 4.0 means the dataset cannot be used to train models deployed commercially without separate written permission from the authors.

---

### Training Suitability for SurfWatch MVP

SurfWatch MVP requires real-time rip current detection from shore-based camera feeds.

| Criterion | Assessment |
|---|---|
| **Relevance** | ✅ Dataset is purpose-built for shore-facing rip current detection — directly aligned with SurfWatch's use case |
| **Annotation quality** | ✅ Expert-annotated instance segmentation by rip current domain specialists; peer-reviewed at CVPR 2025 |
| **Format compatibility** | ✅ YOLO format provided; compatible with YOLOv8/YOLO11 training pipelines |
| **Scale** | ⚠️ 184 videos with ~112 for training; relatively small for training from scratch — fine-tuning a pretrained backbone is recommended |
| **Negative examples** | ✅ Includes dedicated no-rip videos for balanced training |
| **Licence compatibility** | ⚠️ CC BY-NC 4.0 — suitable for MVP research and development, but **commercial deployment requires written permission** from authors |
| **Supplementary data** | ✅ Additional 2,466 images from CVPRW 2023 can be incorporated to increase training set size |

> **Overall verdict:** ✅ Suitable as the **primary dataset** for SurfWatch MVP training and evaluation. The YOLO annotation format, expert-validated split, and domain-specific focus make it the strongest available public dataset for this task. Licence must be reviewed before any commercial release.

---

### Citation

```bibtex
@inproceedings{dumitriu2025ripvis,
  author    = {Dumitriu, Andrei and Tatui, Florin and Miron, Florin and Ralhan, Aakash and Ionescu, Radu Tudor and Timofte, Radu},
  title     = {RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring and Safety},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2025},
  pages     = {3427--3437}
}
```
