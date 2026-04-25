from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.bert_two_head_data import WordChunk, chunk_rows, read_rows, write_submission
from experiments.bert_two_head_model import combine_predictions, first_token_positions, make_tiny_model, make_two_head_model

DEFAULT_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"

def select_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"

def make_backend(args: argparse.Namespace):
    import torch

    if args.backend == "tiny":
        tokenizer, model = make_tiny_model()
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = make_two_head_model(args.model, args.dropout, args.unfreeze_layers)
    device = select_device(torch, args.device)
    return torch, tokenizer, model.to(device), device

def encode_chunk(torch_module, tokenizer, chunk: WordChunk, device: str, max_length: int):
    encoded = tokenizer(list(chunk.words), is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
    positions = first_token_positions(encoded.word_ids(), len(chunk.words))
    tensors = {}
    for key, value in dict(encoded).items():
        tensor = value if torch_module.is_tensor(value) else torch_module.as_tensor(value)
        tensors[key] = (tensor.unsqueeze(0) if tensor.ndim == 1 else tensor).to(device)
    return tensors, torch_module.as_tensor(positions, dtype=torch_module.long, device=device)

def train_model(args: argparse.Namespace, chunks: list[WordChunk]):
    torch, tokenizer, model, device = make_backend(args)
    random.Random(args.seed).shuffle(chunks)
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.learning_rate)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for chunk in chunks:
            tensors, positions = encode_chunk(torch, tokenizer, chunk, device, args.max_length)
            read_logits, log_times = model(**tensors)
            read_word = read_logits[0, positions]
            time_word = log_times[0, positions]
            target = torch.as_tensor(chunk.targets, dtype=torch.float32, device=device)
            read_target = (target > 0.0).float()
            positive = target > 0.0
            time_loss = mse(time_word[positive], torch.log1p(target[positive])) if bool(positive.any()) else time_word.sum() * 0.0
            loss = args.skip_weight * bce(read_word, read_target) + args.time_weight * time_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return torch, tokenizer, model, device

def predict_chunks(args: argparse.Namespace, backend, chunks: list[WordChunk], row_count: int) -> np.ndarray:
    torch, tokenizer, model, device = backend
    predictions = np.zeros(row_count, dtype=float)
    filled = np.zeros(row_count, dtype=bool)
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            tensors, positions = encode_chunk(torch, tokenizer, chunk, device, args.max_length)
            read_logits, log_times = model(**tensors)
            values = combine_predictions(read_logits[0, positions].detach().cpu().numpy(), log_times[0, positions].detach().cpu().numpy())
            for row_index, value in zip(chunk.row_indices, values, strict=True):
                predictions[row_index] = value
                filled[row_index] = True
    if not np.all(filled):
        raise ValueError("each test row must receive a prediction")
    return predictions

def run_train_predict(args: argparse.Namespace) -> dict:
    train_rows = read_rows(args.train)
    test_rows = read_rows(args.test)
    train_chunks = chunk_rows(train_rows, True, args.chunk_words, args.max_train_chunks)
    test_chunks = chunk_rows(test_rows, False, args.chunk_words)
    backend = train_model(args, train_chunks)
    predictions = predict_chunks(args, backend, test_chunks, len(test_rows))
    write_submission(test_rows, predictions, args.output)
    report = {"mode": "bert_two_head_train_predict", "score_type": "local_estimate", "backend": args.backend,
              "model": args.model if args.backend == "hf" else "tiny", "train_chunks": len(train_chunks),
              "test_rows": len(test_rows), "promoted": False}
    write_json(args.report, report)
    return report

def run_smoke(args: argparse.Namespace) -> dict:
    rows = read_rows(args.train)
    chunks = chunk_rows(rows, True, args.chunk_words, args.max_train_chunks)
    compact = []
    offset = 0
    for chunk in chunks:
        row_indices = tuple(range(offset, offset + len(chunk.words)))
        compact.append(WordChunk(chunk.words, row_indices, chunk.targets, chunk.datapoint_ids))
        offset += len(chunk.words)
    backend = train_model(args, compact)
    predictions = predict_chunks(args, backend, compact, offset)
    finite = bool(np.all(np.isfinite(predictions)) and np.all(predictions >= 0.0))
    report = {"mode": "bert_two_head_smoke", "score_type": "local_estimate", "backend": args.backend,
              "row_count": int(len(predictions)), "finite_non_negative": finite, "promoted": False}
    write_json(args.report, report)
    return report

def write_json(path: Path | None, payload: dict) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--chunk-words", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-chunks", type=int)
    parser.add_argument("--skip-weight", type=float, default=0.5)
    parser.add_argument("--time-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a two-head Romanian BERT TRT experiment.")
    sub = parser.add_subparsers(dest="mode", required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--train", type=Path, required=True)
    add_common(smoke)
    predict = sub.add_parser("train-predict")
    predict.add_argument("--train", type=Path, required=True)
    predict.add_argument("--test", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    add_common(predict)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    result = run_smoke(args) if args.mode == "smoke" else run_train_predict(args)
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
