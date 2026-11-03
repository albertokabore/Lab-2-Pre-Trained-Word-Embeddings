"""Read-only checks of the supplied lab; does not train or invent predictions."""

import ast
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from docx import Document
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")
nb = json.loads((ROOT / "Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb").read_text(encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        compile("".join(cell["source"]), f"cell_{i}", "exec")
        assert not any(o.get("output_type") == "error" for o in cell.get("outputs", []))
print("PASS: all supplied code cells compile; no saved error outputs", flush=True)

# Stream the actual archive to avoid allocating the full embedding matrix.
vocab = set()
with zipfile.ZipFile(ROOT / "data/glove.6B.zip") as archive:
    with io.TextIOWrapper(archive.open("glove.6B.300d.txt"), encoding="utf-8") as stream:
        for line in stream:
            word, sep, rest = line.rstrip().partition(" ")
            assert sep and len(rest.split()) == 300
            vocab.add(word)
assert len(vocab) == 400_000
print("PASS: local GloVe member has 400,000 unique words and 300 values per row", flush=True)

scope = {"pd": pd}
clean_node = next(n for n in ast.parse("".join(nb["cells"][10]["source"])).body
                  if isinstance(n, ast.FunctionDef) and n.name == "clean_text")
exec(compile(ast.Module(body=[clean_node], type_ignores=[]), "clean_text", "exec"), scope)
clean_text = scope["clean_text"]
all_counts = Counter()
dataset_results = {}
error_rows = {1938, 348, 6203, 3354, 2883, 2574, 3315, 6278, 7505, 1509}
sample_articles = []
for split in ["train", "test"]:
    labels = Counter()
    rows = backslashes = remaining = max_length = empty = 0
    for frame in pd.read_csv(ROOT / f"data/ag_news_csv/{split}.csv", header=None,
                             names=["class index", "title", "description"], chunksize=4000):
        backslashes += int(frame["description"].str.contains("\\", regex=False).sum())
        labels.update(frame["class index"].tolist())
        frame = clean_text(frame)
        remaining += int(frame["text"].str.contains("\\", regex=False).sum())
        for idx, row in frame.iterrows():
            tokens = word_tokenize(row["text"])
            length = sum(tok in vocab for tok in tokens)
            max_length = max(max_length, length)
            empty += length == 0
            if split == "train":
                all_counts.update(tokens)
            elif idx in error_rows:
                sample_articles.append({"row": int(idx), "gold_index": int(row["class index"]),
                                        "title": row["title"], "description": row["description"]})
        rows += len(frame)
    dataset_results[split] = {"rows": rows, "class_counts": dict(labels),
                              "description_rows_with_backslash": backslashes,
                              "remaining_backslash_rows": remaining,
                              "max_known_token_length": max_length, "all_oov_documents": empty}
    assert remaining == 0
    print(f"PASS: {split} data checked: {dataset_results[split]}", flush=True)

coverage = {"types": len(all_counts), "tokens": sum(all_counts.values()),
            "known_types": sum(t in vocab for t in all_counts),
            "known_tokens": sum(c for t, c in all_counts.items() if t in vocab)}
assert coverage == {"types": 88122, "tokens": 5275725, "known_types": 64652, "known_tokens": 5211697}
assert dataset_results["train"]["rows"] == 120000
assert dataset_results["test"]["rows"] == 7600
assert dataset_results["train"]["description_rows_with_backslash"] == 12522
assert dataset_results["train"]["max_known_token_length"] == 212
assert dataset_results["test"]["max_known_token_length"] == 161
print(f"PASS: actual tokenization/GloVe coverage matches saved outputs: {coverage}", flush=True)

# Recalculate metrics from the submitted confusion table; this is a consistency
# check of saved results, not an independent recreation of model predictions.
doc = Document(ROOT / "Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx")
confusion_table = next(table for table in doc.tables
                       if [cell.text for cell in table.rows[0].cells]
                       == ["Gold / Predicted", "World", "Sports", "Business", "Sci/Tech"])
cm = np.array([[int(cell.text.replace(",", "")) for cell in row.cells[1:]]
               for row in confusion_table.rows[1:]], dtype=np.int64)
assert cm.shape == (4, 4)
assert np.all(cm.sum(axis=1) == 1900)
classes = (ROOT / "data/ag_news_csv/classes.txt").read_text().splitlines()
gold, predicted = [], []
for i in range(4):
    for j in range(4):
        gold.extend([i] * int(cm[i, j]))
        predicted.extend([j] * int(cm[i, j]))
report = classification_report(gold, predicted, target_names=classes, output_dict=True)
assert sum(cm.diagonal()) == 6832
assert round(report["accuracy"], 4) == 0.8989
assert round(report["macro avg"]["f1-score"], 4) == 0.8988
expected = [[.9171,.8847,.9006],[.9512,.9747,.9628],[.8523,.8626,.8574],[.8751,.8737,.8744]]
for name, row in zip(classes, expected):
    assert [round(report[name][m],4) for m in ["precision","recall","f1-score"]] == row
print("PASS: saved confusion counts reproduce every reported class metric, accuracy and macro-F1", flush=True)

del vocab, all_counts
import torch
from torch import nn

model_node = next(n for n in ast.parse("".join(nb["cells"][23]["source"])).body
                  if isinstance(n, ast.ClassDef))
model_scope = {"nn": nn, "torch": torch}
exec(compile(ast.Module(body=[model_node], type_ignores=[]), "model_definition", "exec"), model_scope)
vectors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 0.]], dtype=np.float32)
model = model_scope["FFNNTextClassifier"](vectors, 3, 5, 4, 0.3)
model.eval()
with torch.no_grad():
    ordinary = model(torch.tensor([[0, 1]], dtype=torch.long))
    padded = model(torch.tensor([[0, 1, 3, 3]], dtype=torch.long))
    empty = model(torch.tensor([[3, 3]], dtype=torch.long))
assert torch.allclose(ordinary, padded)
assert torch.isfinite(empty).all()
assert ordinary.shape == (1, 4)
model.train()
nn.CrossEntropyLoss()(model(torch.tensor([[0, 1, 3]])), torch.tensor([1])).backward()
assert model.embs.weight.grad is None
assert all(p.grad is not None for p in model.layers.parameters())
print("PASS: actual model class ignores padding, handles all-OOV input and freezes embeddings", flush=True)

result = {"scope": "Data and model checks plus consistency of saved metrics; no full retraining or prediction recreation.",
          "dataset": dataset_results, "coverage": coverage, "saved_confusion_matrix": cm.tolist(),
          "recomputed_saved_metrics": report, "sample_articles": sorted(sample_articles, key=lambda x: x["row"]),
          "code_compile": "passed", "model_smoke_checks": "passed", "full_training_rerun": False}
(ROOT / "review_assets/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved review_assets/verification.json", flush=True)
