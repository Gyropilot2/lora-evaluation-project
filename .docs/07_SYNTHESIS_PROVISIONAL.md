# 07 - Synthesis Provisional

This document covers everything derived from Evidence: the comparison layer,
metric structure, HeroMetric anatomy, aggregation rules, interpretation
principles, and design decisions made so far.

**Provisional status:** the concepts in Sec. 1-3 (Pair, Group, comparison
principles, HeroMetric anatomy) are stable - they describe how the system is
built. The metric interpretations in Sec. 4-6 are working interpretations based on
observed behavior, not final doctrine. Delta metrics (raw difference between
baseline and LoRA) are the most reliable readings. Compound and hero metrics
involve more judgment and should be treated as provisional unless stated
otherwise. Identity and Pose are the most reviewed hero packages; others are
less settled.

Nothing here should be read as "this metric definitively measures X." It should
be read as "this is our current best understanding of what this metric responds
to, and what to watch out for."

For what the raw measurement domains contain, see `06_EVIDENCE_REFERENCE.md`.
For canonical metric registration (IDs, units, polarity, bounds), see
`contracts/metrics_registry.py`. For the executable interpretation procedures
the review layer uses, see `contracts/procedures_registry.py`.

---

## 1 - Core Concepts

### Pair

One `(baseline_sample, lora_sample)` comparison. Both samples were produced by
the same Method (same prompt, model, settings) at the same seed. The only
controlled difference: one has a LoRA applied at a given strength, one does not.

Pairs are the unit of controlled measurement. A delta only means something
within a Pair - you cannot compare deltas across different Methods without
losing the controlled baseline.

### Group

The set of Pairs that share the same `(seed, lora_strength)` tuple across
multiple LoRAs. Within a group, the only difference between rows is LoRA
identity. Groups are the unit of fair cross-LoRA comparison.

A cross-LoRA comparison is only valid if both LoRAs were tested with identical
(seed, strength) pairs. Mixing seeds or strengths without normalization is not
a controlled comparison.

### LoRA Profile

The aggregate of a LoRA's results across all its Pairs. Used for consistency
analysis (within-LoRA variance across seeds) and cross-LoRA comparison
(relative position within controlled Groups).

### Review-layer building blocks

These terms belong to the review layer, not to raw Evidence storage:

- **Procedure** - a named callable with one narrow responsibility, such as
  gating, comparison, attribution, or aggregation support
- **Aggregation** - a rule for reducing several lower-level readings into one
  higher-level value
- **Recipe** - consumer policy describing what to surface and which
  aggregation strategy to use
- **Review transport** - the settled payload family used by the Operator App
  and local review dump

These surfaces work together, but they do different jobs. Procedure computes.
Aggregation reduces. Recipe chooses. Review transport carries the result.

---

## 2 - Comparison Principles

### Same-method, same-seed is the unit of fair comparison

If seed 2 produces three faces in the baseline and seed 7 produces one, pose
delta and identity delta will differ structurally between those seeds regardless
of the LoRA. Comparing a LoRA's seed-2 result against another LoRA's seed-7
result is not a controlled comparison.

Within a Group (same seed, same strength, multiple LoRAs), all LoRAs faced the
exact same baseline image. That controls for seed-driven confounds.

### Cross-LoRA comparison requires normalization

A LoRA tested at strength 1.0 on a dense constrained prompt will look different
from one tested at strength 0.25 on a loose prompt - even if both are doing
equally interesting things. Controlled comparison normalizes within each Group
before aggregating across Groups, so LoRA profiles are relative positions, not
raw values.

### Method quality constrains interpretation

A metric can be correctly implemented and still be weak evidence under a noisy
or loosely-constrained Method. Loose prompts introduce prompt-driven variance
that is not caused by the LoRA.

- Under constrained Methods: stronger inference is allowed when a metric path fails.
- Under loose/anarchy Methods: treat readings as ceilings or exploration signals,
  not measurements. Face and pose metrics are especially sensitive to method type.
- Baseline weakness matters: if the baseline cannot support a metric cleanly,
  LoRA-side failure on that same path is weaker evidence.

---

## 3 - HeroMetric Structure

A HeroMetric is a three-slot object that is the canonical shape for any primary
evaluation metric surfaced in the Operator App.

| Slot | Type | Meaning |
|---|---|---|
| `score` | `float \| null` | The metric value. Null when dropped. |
| `reliability` | `float \| int \| null` | Support channel paired with the score. Unit is metric-specific: may be a retention ratio, detector confidence, or count. Lower reliability is not itself a drop - it is the granular support channel. |
| `dropped` | `bool \| null` | True when the pipeline could not produce a score. False when it produced a score. Null when the drop concept is undefined (no baseline signal to lose). |

**Invariant:** `dropped == True` implies `score == null`. A score can exist with
low reliability. Do not infer drop from reliability alone.

### HeroMetric levels

**Sample level** - one HeroMetric per seed. `score`, `reliability`, and `dropped`
are computed directly from the measurement. This is the authoritative level.

**Strength level** - aggregated across seeds via `imputed_mean`:
- `score`: arithmetic mean of non-None sample scores. Dropped samples are excluded
  from both numerator and denominator. All-dropped -> null.
- `reliability`: same formula as score.
- `dropped`: does not exist at strength level. A null strength score means all
  samples were dropped or produced no value.

**Eval level** - aggregation rules not yet settled. Must be consistent across all
hero metrics when defined.

**Cross-hero consistency rule:** at any given level, the same aggregation method
must apply to all hero metrics. Pose and identity cannot use different aggregation
formulas at the same level - that would make them incomparable.

### HeroMetric bounds

Metrics may declare `value_min` / `value_max` in `contracts/metrics_registry.py`.
When present, the Operator App trend view may use them. When absent, the UI falls
back to dynamic peer-range scaling rather than inventing a fake ceiling.

| Slot | Preferred bound source |
|---|---|
| `score` | promoted metric `value_min`/`value_max` when declared, else peer range |
| `reliability` | reliability metric `value_min`/`value_max` when declared, else peer range |
| `dropped` | fixed semantic range (0-1 fraction) |

The UI may render a display-only `unreliability` line derived from `reliability`
so that "worse" rises upward. This is a visual transform - the stored slot
remains positive `reliability`.

### HeroMetric surface semantics

Important distinction:

- HeroMetric **shape and anatomy** are registry concerns
- HeroMetric **promotion and selection** are recipe concerns

A metric can declare hero anatomy in the registry without being promoted by the
current recipe. Promotion can change later without redefining the metric.

Operator surfaces must not silently invent hero semantics downstream. The app
consumes the package assembled from the registry, recipe, and review layer. It
does not recreate hero meaning from local fallback maps.

The same HeroMetric may appear in more than one presentation:
- trend views may render score and display-only unreliability channels
- the focused-item rail may render the same hero as grouped slots plus related
  component metrics
- a raw flat metric dump may still exist for exhaustive inspection

These are presentation choices over the same canonical HeroMetric. They do not
create a second metric system.

---

## 4 - Settled Hero Metrics

### Identity

Answers: did the LoRA shift the character's face and hair region?

| Slot | Sample level | Strength level |
|---|---|---|
| `score` | `identity_region_plus_arcface_exp` - mean of `face_cos_dist` + face/hair regional CLIP deltas, gated to None when no LoRA-side face anchor exists | `imputedMeanMetric` of sample scores |
| `reliability` | `det_score` - InsightFace detection confidence (0-1) | `imputedMeanMetric` of sample reliability scores |
| `dropped` | `face_detection_lost` - True when face_count == 0 on LoRA side AND baseline had >=1 face. None when baseline had no face. | N/A |

**Gate visibility:** when `face_count == 0` on the LoRA side, the live score stays
None but the payload also surfaces `identity_gate_status`,
`identity_region_drift_pre_gate_exp`, `identity_region_drift_usable`,
`identity_region_plus_arcface_pre_gate_exp`, and
`identity_region_plus_arcface_usable`. The Operator App can show whether a raw pre-gate
value existed, whether the metric is usable, and whether the face-anchor gate was
the reason it stayed null.

**Identity package composition:** The identity package combines ArcFace embedding
distance with face-region and hair-region CLIP semantic drift. ArcFace captures
identity embedding shift; region CLIP captures whether the face and hair *look*
semantically different even when ArcFace may be partially pose-confounded. The
package is stronger than either component alone because they agree on the same
event from different angles.

The region-only sub-package (`identity_region_drift_exp`) excludes ArcFace and
is sometimes the cleaner first-pass signal when ArcFace is elevated across many
LoRAs without strong visual correspondence. Both are surfaced; the full package
including ArcFace is the primary hero score.

### Pose

Answers: did the LoRA shift the body pose of the main subject?

| Slot | Sample level | Strength level |
|---|---|---|
| `score` | `pose_angle_drift` - selected-source mean angle drift across comparable joint triplets | `imputedMeanMetric` of sample scores |
| `reliability` | `pose_reliability` - selected-source joint-retention ratio against baseline person0 | `imputedMeanMetric` of sample reliability ratios |
| `dropped` | True when neither pose source can supply a headline score | N/A |

Per-source facts (`pose_openpose_*`, `pose_dw_*`) and the selected winner
(`pose_selected_source`) are always surfaced alongside the hero, so source choice
and support are visible.

### Background

Answers: did the LoRA alter the background region?

Score metric: `siglip_cos_bg`.

Fallback: when no character is detected (no main subject, mask broken/useless),
fall back to the full image instead of the background region. Without this
fallback, a broken mask produces near-zero drift that silently passes as good
background preservation.

No reliability or dropped slots - background segmentation always produces a value
when the pipeline runs; no binary failure mode defined.

### Composition

Answers: did the LoRA alter the spatial and depth structure of the scene?

Score metric: `bg_depth_diff`. Depth estimation always produces a value.

Note: `bg_depth_diff` currently captures camera/viewpoint geometry drift better
than scene-meaning drift. Background CLIP metrics (`siglip_cos_bg`) capture
semantic/aesthetic background change. These are different questions - do not
collapse them.

### Global Semantic

Answers: did the LoRA change the overall scene semantically at the full-image level?

Score metric: `clip_global_cos_dist` (full-frame SigLIP global embedding drift).

Key name uses `clip_`; the measurement uses SigLIP global
embeddings.

**Global CLIP dilution:** whole-image CLIP embeddings continue to see the same
subject framing, broad composition, and overall scene class across paired images.
Local changes (different clothing, different floor texture, different-looking
person) can remain small relative to the full-image signal. Global CLIP confirms
that *something* changed; it is weaker than region-aware metrics for answering
*what* changed. Treat it as broad context, not as the primary signal for specific
drift questions.

### Clothing

Answers: did the LoRA alter the clothing?

Score metric: `siglip_cos_cloth`.

---

## 5 - Interpretation Principles

### No free pass for dead signal

If a LoRA breaks a detector, mask, embedding, or measurement path, that is not
automatically neutral. Detector failure may itself be a bad outcome.

The burden is to decide, metric by metric, whether missingness should be:
- a meaningful penalty (drift-bearing metrics whose failure is itself evidence of
  LoRA-induced instability),
- a conditional failure (depends on method quality),
- or an unusable reading (method limitations prevent any inference).

"The metric failed, therefore the LoRA escapes judgment" is not a default.

### Reliability modulates inference weight, not blame

Missingness or detector weakness must not be blamed on the LoRA by default when
the Method itself is weak or underconstrained.

Strong, reliable methods support stronger inference when the LoRA breaks a metric
path. Weak-signal methods should reduce trust in the reading, not automatically
punish the LoRA.

Convergence across several moderately reliable metrics is stronger evidence than
any one metric alone.

### Metrics are not quality scores

Metrics record what changed. They do not judge whether the change is desirable.
A style LoRA is expected to push global CLIP distance high - that is the intended
behavior, not a failure. Interpretation requires knowing what the LoRA is for.

---

## 6 - Per-Metric Interpretation Rules

### `face_cos_dist`

- **Status:** conditional
- **Measures:** Identity drift - ArcFace cosine distance between primary face
  embedding in LoRA vs baseline image. 0 = identical, higher = more shift.
- **Valid range:** [0, 2]. Values > 1.0 are mathematically valid.
- **Failure semantics:** Face not detected -> metric absent. Absence is not neutral.
- **Method sensitivity:** HIGH. Vague prompts inflate this via prompt-driven pose
  and identity variance not caused by the LoRA. Constrained methods give more
  trustworthy readings.
- **Known confounds:** ArcFace embeddings are not fully pose-invariant. Pose LoRAs
  inflate `face_cos_dist` even when identity is preserved, because head angle
  changes the embedding space. Do not interpret high `face_cos_dist` as identity
  change for pose LoRAs without checking images and `head_rot_drift`.
- **Guidance:** Useful under constrained methods for style/character LoRAs.
  Unreliable for pose LoRAs without pose correction.

---

### `face_detection_lost`

- **Status:** provisional
- **Measures:** Whether the LoRA caused face detection to fail entirely. `True` =
  baseline had >= 1 face, LoRA has 0. `False` = LoRA still has >= 1 face. `None` =
  baseline had no face (metric undefined).
- **Failure semantics:** None-valued samples (no baseline face) are genuinely
  uninformative.
- **Design intent:** Not a standalone judgment metric. Read alongside
  `pose_dropped`, `pose_selected_source`, and per-source person0 facts. If face
  loss and total person0 loss coincide across both pose sources, the failure is
  likely broader subject collapse rather than face-only.
- **Known limitation:** Face detection failure does not prove subject collapse.
  Rear-facing characters, partial occlusion, or extreme head angle can produce
  `face_detection_lost = True` while the character is still present.

---

### `head_rot_drift`

- **Status:** conditional
- **Measures:** Overall head pose shift - L2 norm of (delta pitch, delta yaw, delta roll) in
  degrees. 0 deg = no pose change.
- **Failure semantics:** No face detected -> absent. Same policy as `face_cos_dist`.
- **Method sensitivity:** HIGH. The same LoRA measured 14.4 deg under a loose prompt
  and 1.3 deg under a constrained prompt. The 13 deg difference is prompt noise, not LoRA
  effect. Always anchor interpretation to method type.
- **Guidance:** Trustworthy under constrained methods. Under loose methods, treat
  as a ceiling, not a measurement.

---

### `cross_seed_face_dist`

- **Status:** retired from live production
- **Measures:** Identity consistency across seeds (mean pairwise ArcFace
  cosine distance at same eval+strength across different seeds). Retired because
  cross-seed consistency belongs to aggregation work, not the per-sample surface.
- **Critical distinction:** This metric answers "does this eval converge toward a
  similar face across seeds?" - not "does this eval preserve the baseline person?"
  A LoRA may score as self-consistent because it keeps producing the same alternate
  person. That can still be a bad identity outcome relative to the baseline. Do not
  confuse cross-seed consistency with baseline faithfulness.

---

### `lum_character_diff`

- **Status:** conditional
- **Measures:** Mean absolute luminance drift inside `masks.main_subject`.
- **Failure semantics:** None if `main_subject` mask unavailable.
- **Known confounds:** At high strength (>= 1.5), character and background luminance
  drift converge - the LoRA shifts overall scene tone and per-region distinction
  weakens. Useful discriminating range is low-to-mid strength. Face-only luminance
  (`lum_face_diff`) is sharper for face-specific brightness shifts.
- **Guidance:** Useful direction check. Expected pattern for style/identity LoRAs:
  `lum_character_diff` > `lum_bg_diff` at str <= 1.0. If background exceeds
  character, the LoRA is restructuring the background more than the subject.

---

### `siglip_cos_character` / `vitl_cos_character`

- **Status:** conditional
- **Measures:** Semantic drift of the main-subject region. Pooled CLIP spatial
  tokens over `masks.main_subject`.
- **Known confounds:** Whole-subject dilution. Pooling over the full main-subject
  region mixes face, clothing, skin, and boundary patches. Empirically runs at
  roughly 0.48-0.50x of `siglip_cos_face` at str=1.0. Rankings across LoRAs
  collapse at high strength compared to face-only.
- **Guidance:** For identity/character LoRAs, prefer face-only CLIP. Character CLIP
  may be more appropriate for full-body style LoRAs that change clothing while
  leaving the face unchanged.

---

## 7 - Settled Design Decisions

### 7.1 Structured Keypoints Superseded Image-Based Pose Metrics

All image-derived pose metrics (`pose_diff`, `pose_joint_root_drift`,
`pose_scale_drift`, `pose_root_shift`) have been removed. The finding that angle
change is the right question held; the measurement surface changed entirely.

What survived and was re-implemented using structured `pose_evidence` keypoints:
- `pose_angle_drift` - now computed from OpenPose/DWPose keypoints
- `pose_reliability` - selected-source joint-retention ratio against baseline person0
- `pose_dropped` - explicit boolean when neither source can supply a headline score

When a preprocessor exposes grouped structured keypoints upstream (`people[]` with
per-joint `x, y, confidence`), that grouped evidence is better than reverse-parsing
a raster pose image. The hardest pose selection problem ("which shoulder belongs to
which person?") is answered directly by upstream grouping; raster rescue only
approximates it.

`aux.pose` (the old colored dot visualization PNG) was fully removed from all DB
rows. It does not exist in any sample record.

### 7.2 Pose Source Selection: MainSubject_Mask Is The Primary Ownership Signal

When selecting which grouped person is the main subject across pose sources:
- `main_subject_overlap` is the first and strongest separator
- `densepose_overlap` is secondary support, not uniquely decisive
- `face_support` and body completeness (`recognized_joint_count`,
  `core_joint_count`) are corroboration layers
- InsightFace face presence is corroboration, not a requirement for a person to count

### 7.3 Processor Choice Is Case-Dependent

OpenPose and DWPose both produce grouped structured keypoints. Neither is a
universal winner. Both sources are stored and the review layer selects the winner
per sample via the support facts above. Do not hardcode a preferred source.

### 7.4 MainSubject Mask Convention

`masks.main_subject` is the `MainSubject_Mask`: white = subject foreground, black
= scene background. **True scene background is `1 - main_subject`.** Never invert
this. All review surfaces must use this polarity.

Verified on real constrained-method samples:
- `main_subject` covers approximately 15% of the portrait and tracks the main
  subject silhouette tightly
- its inverse covers approximately 85% and corresponds to the wall/floor/background

### 7.5 Soft Component Masks Are Not A Reliable Full-Subject Silhouette

Attempting to build a broader character region from the union of `face`, `skin`,
`clothing`, and `hair` via `np.maximum` does not produce a trustworthy
full-subject silhouette.

Root cause: component masks are sparse soft probability maps. Their combined
coverage (~17% of image area) barely exceeds the stored `main_subject` slot
(~15%), and `np.maximum` is dominated by the clothing detector which leaks soft
activations onto background pixels where wall and garment colors are similar.
The result approximates `main_subject` rather than producing a new region.

Rules:
- Use `main_subject` as the subject-region prior.
- Component masks are useful for targeted region questions (face, clothing, hair).
- Any broader "character/body region from component masks" approach requires
  explicit threshold calibration and edge-case review before it can be trusted.

---

## 8 - Metric Composition Rationale

This section explains why certain metrics are grouped as packages rather than
surfaced individually.

### Why an identity *package*?

Identity drift is not a single observable. ArcFace embeddings capture one axis
(are the high-dimensional face features similar?), but ArcFace is pose-sensitive -
a pose LoRA can inflate `face_cos_dist` while the identity is actually preserved.
Regional CLIP drift over the face and hair regions captures a different axis
(does the face region *look* semantically different?). Neither alone is complete.

The identity package combines both:
- `face_cos_dist` - embedding-space identity shift
- `siglip_cos_face`, `vitl_cos_face` - face-region semantic drift
- `siglip_cos_hair`, `vitl_cos_hair` - hair-region semantic drift

When these signals converge (ArcFace elevated, face and hair region CLIP both
elevated), the evidence for real identity drift is strong. When they disagree
(ArcFace elevated but region CLIP stable), it suggests pose or detector noise
rather than true identity change. The package exposes both the composed score and
its components so disagreement stays visible.

### Why a pose *package* with reliability?

Pose measurement can fail silently. A missing `pose_angle_drift` score might mean
the LoRA produced no detectable person, or it might mean the pose preprocessor
had insufficient keypoints to compute the metric. A simple score without reliability
would hide this distinction.

The pose package adds `pose_reliability` (joint-retention ratio - how much of the
baseline skeleton survived) and `pose_dropped` (explicit binary when no headline
score exists) so that score absence is always explained rather than hidden.

### Why separate background CLIP and background depth?

Background CLIP (`siglip_cos_bg`) answers: did the semantic meaning of the
background change - different wall, different furniture, different scene class?

Background depth (`bg_depth_diff`) answers: did the spatial geometry of the scene
change - different camera angle, different depth structure, different viewpoint?

A LoRA that keeps the same background class but shifts the camera will spike depth
without spiking CLIP. A LoRA that repaints the wall a different color will spike
CLIP without necessarily spiking depth. Collapsing these into one "background
changed" bucket loses that distinction. Both surfaces are kept separate.

---

## 9 - Extension Guidance

When adding new metrics or procedures to the review layer:

**Prefer answer-shaped metrics over generic similarity scores.** "Did clothing
change?" is more useful than "global distance increased." Region-constrained
procedures over masks are more useful than whole-image averages when masks exist.

**Prefer procedures that reuse stored Evidence.** New measurements should use what
is already in the DB before requiring new extractor work.

**Keep composites decomposable.** A package score should always expose its
components. Collapsing weak signals into a black-box score too early makes it
impossible to diagnose what the score is actually responding to.

**Do not invent a procedure only because a field exists.** Every new metric should
answer a question a reviewer would actually ask. If the question is not clear,
store the raw signal and wait.

**All interpretation rules that become settled belong in this document.** If the
review layer or Operator App behavior implies a rule that is not written here, the
rule should be reviewed and documented before it becomes implicit law.
