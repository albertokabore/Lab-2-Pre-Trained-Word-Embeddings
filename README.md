# Lab 2: Pre-Trained Word Embeddings

Topic classification on the **AG News** dataset with a two-layer **feed-forward neural network** over averaged, frozen **GloVe 6B 300-d** word embeddings (PyTorch).

The reviewed submission by **Albert Kabore, PHD in AI.** is in [`Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb`](Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb):

- **Part A**: conceptual explanation (< 500 words)
- **Part B**: implementation, steps i–xi (load, clean, tokenize, GloVe, coverage, padding, arrays, model, training, test evaluation)
- **Part C**: explanation of every command and analysis of the test results
- **References**

The only Word report retained is [`Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx`](Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx). It includes the author credit and the saved run's 89.89% test accuracy; full training was not rerun during review. See [`REVIEW_NOTES.md`](REVIEW_NOTES.md) for the verification scope. Figures are in `report_assets/`.

## Running

```bash
pip install torch numpy pandas tqdm nltk scikit-learn matplotlib jupyter
jupyter notebook Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb   # then Run All
```

On the first run the notebook downloads AG News (~31 MB) and GloVe 6B (~862 MB, or 822 MiB, as a zip) into `data/`, which is git-ignored. `glove.6B.300d.txt` is read straight from the zip. With less than 1.5 GB of free disk, the code attempts to stream the remote zip using HTTP range requests; that fallback was not exercised during review. The embedding matrix alone uses about 480 MB, with additional memory needed for temporary arrays and batches. Runtime and total memory use depend on the environment. The saved run used a CPU.

To use files you already have:

```bash
LAB2_DATA_DIR=/path/to/ag_news_csv LAB2_GLOVE_PATH=/path/to/glove.6B.300d.txt jupyter notebook
```

Based on Chapter 9 of Surdeanu & Valenzuela-Escárcega, *Deep Learning for Natural Language Processing: A Gentle Introduction* ([notebook](https://github.com/clulab/gentlenlp/blob/main/notebooks/chap09_classification.ipynb)).
