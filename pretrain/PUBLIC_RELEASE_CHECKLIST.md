# Public Release Checklist

Use this checklist before publishing the repository.

## Must Check

- Confirm that no private patient data, report text, or derived metadata is included.
- Confirm that all example files are safe to redistribute.
- Confirm that the pretrained weights can be shared publicly.
- Choose and add a repository license.
- Add the final paper citation, DOI, arXiv link, or project page when available.

## Strongly Recommended

- Remove `bert-base-uncased/` from version control if you do not need to ship a local copy.
- Keep large checkpoints and training outputs out of the repository.
- Add a short `CHANGELOG` or release note if this is the first public version.
- Verify the training command in `README.md` on a clean environment.
- Double-check relative paths in dataset manifests.

## Current Repository Notes

- `.gitignore` already excludes common model weights, outputs, and local artifacts.
- `train_CLIP.py` had a syntax issue and default path cleanup applied during this整理 pass.
- The repository still needs a final legal and data-sharing review before public release.
