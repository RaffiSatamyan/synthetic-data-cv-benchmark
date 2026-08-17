# Segmentation SOTA Survey — with a Synthetic-Data Lens

**Author:** Suren Saghatelyan
**Project:** SynData4CV / SCOPE — *When does synthetic data help computer-vision models?*
**Scope of this document:** State of the art for the three segmentation sub-tasks (semantic, instance, panoptic), the standard datasets and their scale, the size/scale of the SOTA models, and — most importantly for our paper — **which generative models are used to produce synthetic training data for segmentation and how much it actually helps.**
**Compiled:** August 2026. Leaderboards move; treat numbers as representative, not exact, and re-check on [Papers with Code](https://paperswithcode.com) before citing in the paper.

---

## 0. Why segmentation is the hard case for synthetic data (read this first)

For **classification** (the task the rest of the team is running first), a synthetic sample only needs a correct *image-level label*. For **segmentation** every synthetic sample needs a **pixel-accurate mask** aligned to the generated image. That alignment is the whole game:

- A GAN/diffusion model can make a beautiful street scene, but you still need per-pixel labels for it.
- So segmentation synthetic-data work splits into two families: **(a) graphics simulators** that render image + perfect ground-truth for free, and **(b) generative models (GAN/diffusion) that are extended to emit an aligned mask** (via cross-attention, a trained perception decoder, or a SAM-style auto-labeler).

This "annotation fidelity" bottleneck is the main reason segmentation is a more interesting — and more open — testbed for our benchmark than classification.

---

## 1. Segmentation sub-tasks and their standard benchmarks

| Sub-task | What it predicts | Primary datasets | Dataset scale | Primary metric |
|---|---|---|---|---|
| **Semantic** | class label per pixel (no instances) | Cityscapes; ADE20K; PASCAL VOC 2012; COCO-Stuff | Cityscapes 5k fine (+20k coarse), 19 cls · ADE20K ~20k train/2k val, 150 cls · VOC ~1.5k (+10k aug), 21 cls | **mIoU** |
| **Instance** | mask + class per object instance | COCO; LVIS; Cityscapes | COCO ~118k train / 5k val, 80 cls · LVIS ~1.2k cls long-tail | **mask AP** |
| **Panoptic** | semantic (stuff) + instance (things) unified | COCO-Panoptic; Cityscapes; ADE20K; Mapillary Vistas | COCO 133 cls (80 things + 53 stuff) | **PQ** (panoptic quality) |

---

## 2. SOTA architectures per sub-task (Table A)

Representative top-of-leaderboard entries and the *scale* of the model and data used to reach them.

### 2a. Semantic segmentation

| Model | Backbone / scale | Dataset | Metric (approx.) | Notes |
|---|---|---|---|---|
| **ONE-Peace / BEiT-3** | giant ViT, ~1.5–3B params | ADE20K | ~**62–63 mIoU** | foundation-model backbones; pretraining dominates |
| **InternImage-H** | ~1.1B params (DCNv3) | ADE20K / Cityscapes | ~**62.9 / 86.1 mIoU** | strong CNN-style backbone |
| **Mask2Former** | Swin-L (~215M) | ADE20K / Cityscapes | ~**57.7 / 84.3 mIoU (val)** | unified mask-classification decoder; the practical default |
| **SegFormer** | MiT-B5 (~85M) | Cityscapes | ~**84.0 mIoU (val)** | efficient, no positional-encoding, popular baseline |
| **DeepLabV3 / V3+** | ResNet-50/101 | Cityscapes | ~**79–82 mIoU** | the repo's current scaffolding; older but standard baseline |

### 2b. Instance segmentation (COCO `test-dev`, mask AP)

| Model | Backbone / scale | Metric (approx.) | Notes |
|---|---|---|---|
| **EVA / Co-DETR-based** | ViT-g + heavy pretraining | ~**55–56 AP** | academic ceiling; huge models |
| **Mask DINO** | Swin-L (~223M) | ~**54.5 AP** | unified detection + instance/panoptic/semantic |
| **Mask2Former** | Swin-L | ~**50.1 AP** | universal decoder |
| **RF-DETR (seg, 2025)** | real-time | ~**44.3 mAP** | current SOTA *real-time* instance seg |
| **Mask R-CNN** | ResNet-50-FPN | ~**37–39 AP** | the classic baseline everyone compares to |

### 2c. Panoptic segmentation (COCO `val`, PQ)

| Model | Backbone / scale | Metric (approx.) | Notes |
|---|---|---|---|
| **Mask DINO** | Swin-L | ~**59 PQ** | |
| **OneFormer** | ConvNeXt-XL / Swin-L | ~**58.0 / 57.9 PQ** | one model, trained once, all three tasks |
| **Mask2Former** | Swin-L | ~**57.8 PQ** | |
| **kMaX-DeepLab** | ConvNeXt-L | ~**58 PQ** | k-means mask transformer |
| **Panoptic-DeepLab** | Xception-71 | ~**40–42 PQ** | bottom-up baseline |

**Universal / foundation models worth naming in the paper:** **Mask2Former**, **OneFormer**, **Mask DINO** (all three tasks with one architecture), and **SAM / SAM 2** (promptable, class-agnostic masks — great as an *auto-annotator* for synthetic images, not a labeled-segmentation model by itself).

---

## 3. Synthetic data FOR segmentation — the core table (Table B)

This is the table that matters for our thesis. Two generator families, plus how each one obtains masks and how much it helps.

### 3a. Graphics simulators (image + perfect GT for free)

| Source | Scale | Classes shared w/ Cityscapes | Typical use |
|---|---|---|---|
| **GTA5** | 24,966 imgs | 19 | synthetic→real UDA source |
| **SYNTHIA (RAND-CITYSCAPES)** | 9,400 imgs | 16 | synthetic→real UDA source |
| **Virtual KITTI 2, Synscapes** | ~few×10k | driving cls | driving-scene synthetic |

**Best-known result path (GTA5/SYNTHIA → Cityscapes, unsupervised domain adaptation):**
DAFormer → **HRDA** → **MIC**. HRDA+MIC reaches roughly **~75–76 mIoU** on GTA5→Cityscapes (vs. ~46 mIoU for naive source-only training). This is the canonical "synthetic can replace a lot of real labels" evidence for segmentation.

### 3b. Generative models that emit image **+ mask** (GAN / diffusion)

| Method | Base generator | How masks are produced | Task / dataset | Reported effect |
|---|---|---|---|---|
| **DatasetGAN / SemanticGAN / EditGAN** (2021) | StyleGAN2 | small annotated latent set → label branch | semantic (faces, cars, parts) | near–real quality from ~16 labels |
| **DiffuMask** (ICCV 2023) | Stable Diffusion | text–image **cross-attention** → mask | semantic (VOC, Cityscapes) | competitive with real-data training |
| **DatasetDM** (NeurIPS 2023) | Stable Diffusion | trained **perception decoder** (mask/depth/pose) | semantic / instance / depth | one decoder, many perception tasks |
| **FreeMask** (NeurIPS 2023) | FreestyleNet (mask→image) | mask-conditioned synthesis + hard-sample sampling | semantic (ADE20K, COCO-Stuff) | boosts segmenters, filters noisy masks |
| **DGInStyle** (ECCV 2024) | Latent Diffusion + style/semantic control | style-swap + multi-res latent fusion | domain-generalizable street scenes | consistently lifts DG methods over prior SOTA |
| **Dataset-Diffusion** (2024) | Stable Diffusion + LLM prompts | attention + self-attention refinement | semantic (VOC, COCO) | strong text-only mask generation |
| **"What Makes Synthetic Data Effective in Image Segmentation"** (2025/26) | **Flux (12B), SD3.5 (8.1B), Sana (4.8B)** | mask-conditioned + composition control | semantic (Cityscapes, COCO, ADE20K) | see §4 numbers |
| **Training-free Synthetic Data Selection** (2501.15201, 2025) | diffusion | selects *which* synthetic samples to keep | semantic | quality-filtering beats raw volume |

**SAM/SAM2 as the annotator:** an increasingly common recipe is *generate images with a diffusion model → auto-mask them with SAM 2 → train the segmenter.* Worth flagging as a design option for our pipeline.

---

## 4. When does synthetic data actually help segmentation? (key quantitative findings)

From the most recent, directly-on-topic study (*What Makes Synthetic Data Effective in Image Segmentation*, 2025/26), at a **2:1 synthetic-to-real** ratio:

| Dataset | Segmenter | mIoU gain from synthetic |
|---|---|---|
| Cityscapes | DPT (DINOv2) | **+2.54** |
| COCO | DPT (DINOv2) | **+1.21** |
| ADE20K | Mask2Former (DINOv3-L) | **+1.64** |

Plus a finding that maps straight onto **our x0.5/x1/x2/x5/x10 ratio axis**: **scene compositional complexity matters** — dense synthetic scenes gave **66.56 vs 61.81 mIoU** over sparse ones on Cityscapes.

**Consolidated takeaways for our benchmark design:**
1. **Low-real-data regime is where synthetic wins.** Gains are largest at 1–10% real and shrink as real data grows — exactly why our nested 1→100% real-fraction axis is the right experiment.
2. **There is an optimal ratio, and more is not better.** Reported sweet spots cluster around **1:1–2:1** synthetic:real; heavy synthetic (our x5/x10) often *hurts* from domain gap. Finding where the curve turns over is a publishable result.
3. **Mask/annotation fidelity dominates.** For segmentation the generator's *mask alignment* matters more than raw image realism — a lever classification doesn't have.
4. **Filtering/selection beats volume.** FreeMask and the 2025 selection paper both show that culling bad synthetic samples outperforms just adding more.

---

## 5. Mapping this survey onto our repo / experiment plan

- The repo already scaffolds **DeepLabV3 + Cityscapes** ([`configs/models/deeplabv3.yaml`](../configs/models/deeplabv3.yaml), [`configs/datasets/cityscapes.yaml`](../configs/datasets/cityscapes.yaml)). For a competitive segmentation arm we'd want **Mask2Former / OneFormer** added as a model config.
- Our **generator axis** currently lists `stable_diffusion` and `stylegan2` ([`configs/generators/`](../configs/generators/)). For segmentation these must be upgraded to **mask-emitting** variants (DiffuMask/DatasetDM-style, or SD + SAM2 auto-labeling) — a plain SD image has no mask.
- The **real-fraction × synthetic-ratio × seed** grid in [`experiments/full.yaml`](../experiments/full.yaml) is exactly the sweep the literature says is under-explored for segmentation. Add GTA5/SYNTHIA as a "simulator" generator option for the domain-adaptation comparison.

---

## 6. Fill-in for the PPTX "Research Tasks" tables

**Slide 5 checklist status:**
- [x] Search up for a task → *segmentation (semantic / instance / panoptic)*
- [x] Identify SOTA architectures + datasets used → *§1, §2*
- [x] Size/scale of SOTA model + dataset → *§1, §2 (params & image counts)*
- [x] Find models used for synthetic-data generation → *§3*
- [ ] Fill the shared comparison table → *paste Table A + Table B into the group deck*

---

## Sources

- [Papers with Code — Semantic Segmentation on Cityscapes](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes)
- [Papers with Code — Semantic Segmentation on ADE20K](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k)
- [Mask2Former (arXiv 2112.01527)](https://arxiv.org/pdf/2112.01527) · [OneFormer (SHI-Labs)](https://github.com/SHI-Labs/OneFormer) · [Mask DINO (arXiv 2206.02777)](https://arxiv.org/pdf/2206.02777)
- [DiffuMask (arXiv 2303.11681)](https://arxiv.org/abs/2303.11681) · [DatasetDM (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/ab6e7ad2354f350b451b5a8e14d04f51-Paper-Conference.pdf)
- [DGInStyle (ECCV 2024)](https://dginstyle.github.io/) · [What Makes Synthetic Data Effective in Image Segmentation (arXiv 2605.19289)](https://arxiv.org/html/2605.19289v1) · [Training-free Synthetic Data Selection (arXiv 2501.15201)](https://arxiv.org/pdf/2501.15201)
- [RF-DETR Segmentation (Roboflow, 2025)](https://blog.roboflow.com/rf-detr-segmentation-preview/)
