# Transformer Experiment Design

## Goal

Add an experiment-only path for contextual language features without replacing the stable raw-Ridge submission. The stable submission remains the fallback until a Transformer result beats the local text-holdout score of `38.388844 / 100`.

## Function

A Transformer provides context-aware word signals. The current Ridge baseline sees each word and simple surface features, but it does not know whether a word is predictable from surrounding words, syntactically hard, semantically unusual, or part of a difficult phrase.

## Principle

Transformer self-attention builds token vectors by mixing information from neighboring tokens. For this task, the experiment maps subword token vectors back to one vector per original word, so each reading-time row receives contextual features from its page.

## Implementation Method

Phase 1 uses frozen embeddings:

- group rows by `text` and page parsed from `word_id`;
- tokenize each page as pre-split words with a Hugging Face tokenizer;
- run a pretrained Romanian or multilingual model with gradients disabled;
- average subword hidden states into one vector per original word;
- reduce each word vector to compact numeric features;
- train raw-target Ridge on existing features plus Transformer features.

Phase 2 fine-tuning is optional. It may add a token-level regression head only after frozen features prove useful locally.

## Promotion Rule

Transformer results are local estimates until truth labels or judge feedback exist. A Transformer path must beat the raw-Ridge local estimate before it can replace the stable submission path.

## Primitive Acceptance Criteria

- The experiment report is labeled as a local estimate.
- Each input row receives exactly one Transformer feature vector.
- Subword vectors can be mapped back to original word rows.
- Stable `solution.py` and `submission.csv` remain unchanged by smoke tests.
- Generated feature caches are not tracked by git.
