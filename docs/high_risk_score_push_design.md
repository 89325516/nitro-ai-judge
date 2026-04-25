# High-Risk Score Push Design

## Purpose

This branch prioritizes the highest possible official Nitro TRT score. It explicitly allows public external eye-tracking, raw-fixation, and derived-TRT resources that may match test stimuli, participants, or rows. The branch is high risk because this work can become hidden-label reconstruction rather than ordinary model generalization.

## Evidence Position

The branch starts from Candidate `023_lexical_transformer`, with local text-holdout evidence of `40.25002577495926 / 100`. This is the fallback until a higher official score, exact truth score, or risk-labeled reconstruction candidate proves better performance.

## Allowed High-Risk Data Use

The branch may search, download, process, and map public eye-tracking datasets when they appear related to the Nitro task. Allowed high-risk signals include matching stimuli, participant identifiers, page files, raw fixation events, AOI tables, word-fixation aggregates, and derived Total Reading Time tables.

A candidate may use direct row-level mapping if the source is public and reproducible, but it must label the mapping as high-risk reconstruction. It must not hide a lookup path behind generic model language.

## Score Route

The `40+` route is Candidate `023`. The `50-70` route comes from partial public-data mapping, such as known stimuli or item means without complete participant recovery. The `80+` route requires broad recovery of true TRT values or near-equivalent fixation reconstruction for the hidden rows. The `99+` target is satisfied only by official judge feedback or exact hidden truth labels.

## Candidate Promotion Rules

Each high-risk candidate must record the source URL or dataset origin, the row-mapping rule, the generated output path, the reproduction command, and whether the uploaded source can regenerate the output under competition limits. If a candidate depends on external files that cannot be bundled, the report must state that reproducibility risk before official upload.

Official feedback is the promotion authority. If a high-risk candidate receives the best official score, it becomes the branch's selected fallback even if its method remains risk-labeled.

## Primitive Acceptance Criteria

- The branch can identify Candidate `023` as the starting fallback.
- A high-risk candidate produces one answer per test datapoint.
- A high-risk candidate keeps the public output columns unchanged.
- External source origins are visible before promotion.
- Direct row-level reconstruction is explicitly labeled when present.
- `80+` and `99+` are not treated as complete without official or exact-truth evidence.
