# VENDORED — do not edit here

This is a vendored copy of the `memorise_taxonomy` package
(https://github.com/ai-engine-memorise/memorise-taxonomy), copied into `src/` so
the recsys Docker image can `import memorise_taxonomy` without installing a
private, unpublished package.

Why vendored: `recsys/taxonomy.py` imports `memorise_taxonomy` (e.g. `to_tags`)
at module load. The package is a private repo, not on any index, and the recsys
Docker build context is `ai-engine/` only — so it cannot be pip-installed or
`COPY`'d from a sibling path. Without it, the container crashes on startup
(ImportError -> CrashLoopBackOff). The image already has `PYTHONPATH=/app/src` and
`COPY src ./src`, so dropping the package here makes it importable with no
Dockerfile change.

Source of truth is the upstream repo. To re-sync after upstream changes:

    cp -r ../memorise-taxonomy/src/memorise_taxonomy/{__init__.py,taxonomy.py,data} \
          ai-engine/src/memorise_taxonomy/

Better long-term fix: publish `memorise-taxonomy` to a package index (or a private
GHCR/GitLab pip index), add `"memorise-taxonomy>=X.Y"` to `Dockerfile.recsys`, and
delete this vendored copy.
