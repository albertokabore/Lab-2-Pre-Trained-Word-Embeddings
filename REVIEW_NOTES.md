# Assignment review

The reviewed submission is available in either accepted format:

- [Reviewed Word report](Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx)
- [Reviewed Jupyter notebook](Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb)

The reviewed notebook and reviewed report are the only notebook and Word report retained. Both display the author credit **Albert Kabore, PHD in AI.** The model implementation, execution counts, and numerical output text are preserved. Display formatting uses black text; only presentation code, figure images, and a hidden notebook styling cell were adjusted. Learning curves were redrawn using the rounded saved epoch values, and confusion counts are unchanged. The Word report contains all implementation commands, the download helper, saved numerical outputs, and both figures. Some wide interactive table displays are condensed in Word; the notebook retains the saved displays.

## Alignment with the assignment

| Requirement | Reviewed submission |
|---|---|
| Part A: explain the implementation in fewer than 500 words | 319 words, excluding heading and word-count note (whitespace count). |
| B.i: initialize torch, NumPy, pandas, tqdm | Imports, progress registration, seeds, and device selection included. |
| B.ii: load and preview the AG News files | CSV reading, class names, preview, and class counts included. |
| B.iii: replace backslashes using pandas `replace()` | `DataFrame.replace(..., regex=True)` applied to both text columns. |
| B.iv: NLTK `word_tokenize()` for title and description | Both fields are combined before tokenization, as in the supplied reference notebook. |
| B.v: load GloVe embeddings | Reader, float32 matrix, and word-to-ID mapping included. |
| B.vi: estimate GloVe coverage | Both distinct-type and occurrence-weighted coverage included. |
| B.vii: add an all-zero padding embedding | New ID and zero row included. |
| B.viii: 2-D token-ID and 1-D gold-label arrays | Arrays, zero-based labels, and stratified development split included. |
| B.ix: two-layer feed-forward PyTorch model | Masked mean, two linear layers, ReLU, and dropout included. |
| B.x: training and development metrics each epoch | Training loop, loss, accuracy, five epochs, and checkpoint selection included. |
| B.xi: final test evaluation | Accuracy, classification report, confusion matrix, and error sample included. |
| Part C: explain commands and interpret results | All eleven steps explained, followed by evidence-based analysis. |
| Submission format and citations | Word and Jupyter formats supplied, with direct documentation and research links. |

## Changes to the explanation

- Replaced overly conversational or absolute claims with clear technical explanations.
- Distinguished selecting the best checkpoint after five epochs from stopping training early.
- Clarified that the network returns logits and that cross-entropy performs log-softmax internally.
- Explained why training and development accuracy use different measurement conditions.
- Removed claims that a small development–test gap proves representativeness or absence of overfitting.
- Explained that 98.79% token coverage does not establish that dropping unknown words is harmless.
- Treated questionable labels as annotation questions, without declaring them wrong or changing benchmark labels.
- Removed causal conclusions about embedding benefits or phrase features that were not tested here.
- Removed the optional cross-paper baseline comparison to keep the analysis focused on the assignment's own results.
- Removed unsupported promises about runtime, memory sufficiency, and exact cross-platform reproducibility.
- Clarified why training and test arrays can have different padding widths.
- Added direct source links and made saved-run provenance explicit.

## Verification performed

`python review_assets/verify_lab.py` completed successfully against the local data. It:

- Compiled all original code cells and checked that no saved cell contains an error output.
- Streamed the local GloVe 300-dimensional archive member and verified 400,000 unique words with 300 values per row.
- Reapplied the actual cleaning function and NLTK tokenization to all 120,000 training and 7,600 test articles.
- Confirmed 30,000 training and 1,900 test examples per class.
- Confirmed 12,522 training descriptions containing backslashes and zero remaining backslash rows after cleaning.
- Recomputed vocabulary coverage: 64,652 / 88,122 types and 5,211,697 / 5,275,725 token occurrences.
- Confirmed maximum known-token lengths of 212 for training and 161 for testing, with no all-unknown articles in either file.
- Recomputed accuracy, precision, recall, and F1 from the submitted confusion counts; all reported values agree after rounding.
- Checked the actual model class on small inputs: adding padding leaves predictions unchanged, all-padding inputs remain finite, embeddings receive no gradients, and linear-layer parameters receive gradients.

The confusion counts were also checked against the saved heatmap. Their diagonal sums to 6,832 correct predictions out of 7,600, giving 89.89% accuracy. This verifies numerical consistency of the saved results, not independently recreated predictions. Full training was not rerun, no trained checkpoint was supplied for independent inference, and remote download fallbacks were not exercised. The five-epoch loss and accuracy history remains evidence from the supplied saved outputs.

The generated submission was additionally checked for valid notebook structure, preservation of the model code and numerical output text, all eleven required implementation headings, inclusion of all implementation commands in Word, and both figures.

Machine-readable findings and the inspected sample article text are in [verification.json](review_assets/verification.json). The generation script is [build_reviewed.py](review_assets/build_reviewed.py).
