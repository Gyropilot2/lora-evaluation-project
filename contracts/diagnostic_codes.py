"""
contracts/diagnostic_codes.py — central diagnostic code registry.

Every code value passed to core.diagnostics.emit() must be listed here.
dev_healthcheck.py validates this at lint time.

Code style: dot-separated, uppercase tokens — DOMAIN.DETAIL
  Examples: SCHEMA.MISSING_FIELD, ASSET.WRITE_FAIL, ARCH.CHEAT

Rules:
  - Never use free-text in the code field. Always use a constant from this file.
  - Codes are stable identifiers — once published, rename only with a migration.
  - Add new codes here before using them in application code; healthcheck enforces this.
  - Codes are NOT removed when deprecated; mark them with a comment instead.

Usage:
    from contracts.diagnostic_codes import ALL_CODES
    # dev_healthcheck.py imports ALL_CODES to validate emit() calls
"""

# ---------------------------------------------------------------------------
# All valid diagnostic codes
# ---------------------------------------------------------------------------
# This variable name is required by dev_healthcheck.py.

ALL_CODES: set[str] = {

    # ---- Sample deduplication (Bouncer — 3-level model) ----
    # Same sample_hash + same latent_hash, no new measurement data. Discarded silently. DEBUG.
    "SAMPLE.DUPLICATE_RUN",

    # Same sample_hash + same latent_hash, new measurement slots present. Backend enriched. INFO.
    "SAMPLE.ENRICHED",

    # Same sample_hash + different latent_hash. Existing sample flagged dirty. Loud WARN.
    "SAMPLE.NOT_ZERO_DELTA",

    # ---- DEPRECATED (run/sample two-level model — replaced by 3-level model in 1.7.f) ----
    # DEPRECATED: Different run produced identical sample content. Use SAMPLE.DUPLICATE_RUN instead.
    "SAMPLE.DUPLICATE_CONTENT",

    # DEPRECATED: Run with same run_signature already exists. No longer used.
    "RUN.DUPLICATE",

    # DEPRECATED: run_id collision with different run_signature. No longer used.
    "DB.RUN_CONFLICT",

    # ---- Schema validation (Bouncer) ----
    # A required Evidence field is NULL or missing (non-vital).
    "SCHEMA.MISSING_FIELD",

    # A vital field (run_id, sample_id, run_signature, latent_hash, image_hash, vae_hash) is missing — hard refusal, nothing written.
    "SCHEMA.VITAL_MISSING",

    # Input record is structurally unreadable (not a dict, not valid JSON).
    "SCHEMA.PARSE_FAIL",

    # An unrecognised key was found in the incoming record; moved to _extras.
    "SCHEMA.UNKNOWN_KEY",

    # A field value has the wrong type or is otherwise malformed.
    "SCHEMA.INVALID_TYPE",

    # Candidate contains invalid non-storable fields and is hard-rejected before persistence.
    "SCHEMA.NON_STORABLE_REJECT",

    # ---- Asset management ----
    # Content hash mismatch when loading an asset (possible corruption).
    "ASSET.HASH_MISMATCH",

    # Failed to write an asset blob to the staging area or assets directory.
    "ASSET.WRITE_FAIL",

    # Failed to read an asset blob from disk.
    "ASSET.READ_FAIL",

    # Asset staging failed before commit (tmp write failure).
    "ASSET.STAGE_FAIL",

    # ---- Database / DataBank ----
    # DB insert or update failed.
    "DB.WRITE_FAIL",

    # DB read / query failed.
    "DB.READ_FAIL",

    # Schema migration applied successfully.
    "DB.MIGRATION_APPLIED",

    # ---- On-site Lab ----
    # Probe dump file written successfully.
    "LAB.DUMP_WRITTEN",

    # Failed to write a probe dump file to disk.
    "LAB.DUMP_WRITE_FAIL",

    # Probe encountered an unexpected error; workflow continues.
    "LAB.PROBE_ERROR",

    # Image-first pose-scene lab node hit an unexpected error; workflow continues.
    "LAB.POSE_SCENE_ERROR",

    # VAE decode failed inside the probe; fallback black image used; workflow continues.
    "LAB.DECODE_FAILED",

    # Binary attachment (e.g. luminance .npy) written alongside JSON dump.
    "LAB.ATTACHMENT_WRITTEN",

    # Failed to write a binary attachment file to disk.
    "LAB.ATTACHMENT_WRITE_FAIL",

    # ---- Extractor (Step 2.1) ----
    # Latent tensor hash computation failed (vital integrity field).
    "EXTRACTOR.LATENT_HASH_FAIL",

    # Image tensor hash computation failed (vital integrity field).
    "EXTRACTOR.IMAGE_HASH_FAIL",

    # Base model or VAE model weight hash computation failed.
    "EXTRACTOR.MODEL_HASH_FAIL",

    # Conditioning tensor hash computation failed.
    "EXTRACTOR.COND_HASH_FAIL",

    # lora_name was provided but the file could not be resolved via folder_paths.
    "EXTRACTOR.LORA_RESOLVE_FAIL",

    # stage_blob or commit_blob failed for an asset; ValueRef set to Invalid.
    "EXTRACTOR.ASSET_COMMIT_FAIL",

    # CLIP Vision encode_image() call failed non-fatally.
    "EXTRACTOR.CLIP_ENCODE_FAIL",

    # InsightFace analysis failed non-fatally; face_analysis domain set to Invalid.
    "EXTRACTOR.INSIGHTFACE_FAIL",

    # Latent tensor shape was unavailable; latent_width/height set to Invalid.
    "EXTRACTOR.LATENT_SHAPE_FAIL",

    # Live Extractor: pose_evidence build failed or inputs insufficient; pose_evidence omitted.
    "EXTRACTOR.POSE_EVIDENCE_FAIL",

    # Live Extractor: workflow_ref_json input was malformed; workflow_ref omitted.
    "EXTRACTOR.WORKFLOW_REF_PARSE_FAIL",

    # ---- SampleGuard (ComfyUI pre-flight node) ----
    # Hash computation raised an unexpected exception; guard is passing through (fail-open).
    "GUARD.HASH_FAIL",

    # Method hash already in DB — informational only, not a halt.
    "GUARD.METHOD_EXISTS",

    # Eval hash already in DB — informational only, not a halt.
    "GUARD.EVAL_EXISTS",

    # Sample hash already in DB — guard is halting the workflow.
    "GUARD.SAMPLE_EXISTS",

    # DB read failed during guard check; guard is passing through (fail-open).
    "GUARD.DB_ERROR",

    # ---- Replay / backfill (ComfyUI read-only + enrich transport) ----
    # Requested replay sample_hash not found in DB.
    "REPLAY.SAMPLE_NOT_FOUND",

    # Replay sample image asset could not be decoded from its ValueRef format.
    "REPLAY.IMAGE_DECODE_FAIL",

    # Replay guard halted because all requested enrichment paths already exist.
    "REPLAY.REQUEST_ALREADY_PRESENT",

    # Replay enrichment could not determine a MainSubject support mask.
    "REPLAY.MAIN_SUBJECT_MISSING",

    # ---- ComfyUI HTTP client ----
    # ComfyUI returned an application-level error in the /prompt response body.
    "COMFYUI.PROMPT.REJECTED",

    # ---- Batch runner (run / rerun) ----
    # A single run job (one LoRA × strength × seed) failed during submission or execution.
    "BATCH.RUN.JOB.FAIL",

    # ---- Batch replay (measurement backfill) ----
    # Replay plan built; informational summary.
    "BATCH.REPLAY.PLAN",

    # A single replay job completed successfully.
    "BATCH.REPLAY.JOB.OK",

    # A single replay job failed (ComfyUI rejection or transport error).
    "BATCH.REPLAY.JOB.FAIL",

    # Full replay batch finished (ok or with failures).
    "BATCH.REPLAY.DONE",

    # ---- Workflow onboarding ----
    # Raw ComfyUI export successfully tagged and written to workspace.
    "WORKFLOW.ONBOARD.OK",

    # ---- Operator review assembly ----
    # review_builder.py auto-installed Pillow into the active environment.
    "REVIEW.PILLOW_AUTO_INSTALL",

    # ---- Architecture / scaffolding ----
    # Intentional temporary rule-break. Must include 'why' and cleanup condition in ctx.
    # Emit with ctx={'cleanup': 'remove when X exists'}.
    "ARCH.CHEAT",

}


# ---------------------------------------------------------------------------
# Convenience accessor (optional — healthcheck reads ALL_CODES directly)
# ---------------------------------------------------------------------------


def is_valid_code(code: str) -> bool:
    """Return True if the given code is registered in ALL_CODES."""
    return code in ALL_CODES
