# Reproducibility snapshot

Run the environment capture on the experiment machine after the release commit is clean:

```bash
python scripts/capture_environment.py \
  --config configs/base.yaml \
  --output ../sclc-retrieval-environment.json
```

The generated JSON records the local Python/package/CUDA/GPU environment, Git state,
frozen-input hashes, QASPER source configuration, configured model metadata, and the local
Hugging Face cache state. For each configured model it reads the cached Hub reference when
available and records the resolved 40-character commit hash. If a cache reference cannot be
resolved, the script records that fact and lists the available local snapshots rather than
inventing a historical revision.

`reproducibility/frozen_inputs.json` stores the expected hashes and counts for the small frozen
inputs committed with the dissertation artefact. Run `python scripts/check_release.py` before
creating a release to verify those inputs and the release metadata.
