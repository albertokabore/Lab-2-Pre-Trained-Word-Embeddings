"""Build the submission from the retained notebook and its saved results."""

import base64
import copy
import io
import json
import re
from pathlib import Path

import nbformat
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from format_submission import format_notebooks, format_word, STYLE_ID


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Lab2_Pre_Trained_Word_Embeddings_Reviewed.ipynb"
REPORT_FINAL = ROOT / "Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx"
REPORT = ROOT / "review_assets/Lab2_Report.pending"
original = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
reviewed = copy.deepcopy(original)


def replace_cell(index, text):
    reviewed["cells"][index]["source"] = text.strip().splitlines(keepends=True)


INTRO = """
# Lab 2: Pre-Trained Word Embeddings

**Albert Kabore, PHD in AI.**

**AG News classification with averaged GloVe embeddings and a two-layer feed-forward network**

This submission explains the method in Part A, provides all eleven implementation steps in Part B, and interprets the results in Part C. It follows the workflow in the [Chapter 9 reference notebook](https://github.com/clulab/gentlenlp/blob/main/notebooks/chap09_classification.ipynb).

**Results provenance.** The numerical results and figures are retained from the supplied notebook's saved CPU run with seed 1234. This revision updates the explanations; it is not a new training run. The saved outputs report PyTorch 2.14.0+cpu, NumPy 2.3.4, and pandas 2.3.3. Data checks and recalculated metrics are documented separately in `REVIEW_NOTES.md`.

The code uses 300-dimensional GloVe 6B vectors, a 300-unit hidden layer, dropout of 0.3, an Adam learning rate of 0.001, batches of 500, and five epochs. An 80/20 stratified split gives 96,000 training and 24,000 development articles. The separate test file contains 7,600 articles. The hidden size and dropout differ from the reference notebook's 50 and 0.1; their effects have not been compared experimentally. This version also uses a custom GloVe loader, drops unknown tokens, and selects the checkpoint with the highest development accuracy.

**Running the notebook.** Install `torch numpy pandas tqdm nltk scikit-learn matplotlib jupyter`, then run the cells from top to bottom. The setup cell downloads missing AG News files and provides a GloVe reader. A local GloVe zip is used when available; an extracted file can be supplied through `LAB2_GLOVE_PATH`. `LAB2_DATA_DIR` can point to existing CSV files. The local zip is 862,182,613 bytes (about 822 MiB). The embedding matrix alone occupies about 480 MB before padding; temporary arrays and batches require additional memory. Runtime depends on the machine and download speed.
"""

PART_A_BODY = """
This model assigns an AG News article to World, Sports, Business, or Sci/Tech using its title and description. The main challenge is turning variable-length text into a fixed-size input for a feed-forward neural network.

Pandas loads the CSV files. The preprocessing replaces backslashes with spaces, lowercases both text fields, and joins them. NLTK's `word_tokenize()` then separates the combined text into words and punctuation. Lowercasing aligns the input with the uncased GloVe 6B vocabulary.

Each known token maps to a 300-dimensional pre-trained GloVe vector. GloVe learns word representations from corpus-wide co-occurrence statistics (Pennington et al., 2014). These vectors provide information learned outside the labeled news dataset. This implementation drops tokens missing from the vocabulary, so it cannot use their information directly.

The token IDs are stored in a rectangular NumPy array. Shorter sequences receive a special padding ID whose embedding is all zeros. During the forward pass, the model sums the word vectors and divides by the number of non-padding tokens. This masked average gives every article a 300-dimensional representation. It preserves information about which words occur while discarding their order.

The document vector passes through a linear layer with 300 hidden units, a ReLU activation, and a linear output layer with four scores. Dropout of 0.3 is applied before both linear layers during training. The scores are raw logits. PyTorch's `CrossEntropyLoss` handles the log-softmax calculation internally, so the training code does not apply softmax first.

Adam updates the network using mini-batches of 500 and a learning rate of 0.001. The GloVe embeddings remain frozen; only the two linear layers, containing 91,504 parameters, are trained. After each of five epochs, development loss and accuracy are measured with dropout disabled. The checkpoint with the highest development accuracy is restored for final test evaluation. Accuracy, per-class precision, recall, F1, and a confusion matrix describe performance. These results characterize the saved run; they do not establish how performance varies across seeds or datasets.
"""

PART_C = r"""
---
# Part C: Analysis

## C.1 How the commands in Part B work

**B.i Initialization.** NumPy handles arrays, pandas handles tables, PyTorch supplies tensors and neural-network operations, and tqdm displays progress. `tqdm.pandas()` registers `progress_map()` and related methods. Seeds control the random generators used for initialization, dropout, shuffling, and splitting. They support repeatability in a fixed environment, but do not guarantee identical results across hardware or software versions ([PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.14/notes/randomness.html)). `device` chooses CUDA when available; the saved run used the CPU.

**B.ii Loading and inspecting data.** `pd.read_csv(..., header=None, names=...)` reads the headerless CSV with explicit column names. `head()` previews articles, while `shape` and `value_counts()` check size and label balance. `classes.txt` supplies the class names in index order. The training file contains 120,000 rows, with 30,000 per class. This agrees with the dataset statistics in Zhang et al. (2015, Table 3). Class balance makes accuracy useful, while the per-class metrics reveal differences it can hide.

**B.iii Backslash replacement.** `df[['title', 'description']].replace(r'\\', ' ', regex=True)` replaces literal backslashes inside both text columns. The raw-string regular expression uses two backslashes to match one literal backslash. The saved example changes `dwindling\band` to `dwindling band`. There were 12,522 training descriptions containing a backslash; none remain in the cleaned combined text. `.str.lower()` and string concatenation then produce one lowercased `text` column ([pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html)).

**B.iv Tokenization.** `progress_map(word_tokenize)` applies NLTK's tokenizer to each article. Both required fields are covered because `text` contains the title and description; the supplied Chapter 9 notebook follows this approach too. NLTK combines sentence segmentation with an improved Treebank word tokenizer, separating punctuation and splitting contractions such as `don't` into `do` and `n't`. The setup checks for `punkt_tab` and downloads it if missing ([NLTK documentation](https://www.nltk.org/api/nltk.tokenize.html)).

**B.v GloVe loading.** `open_glove()` supplies a text stream from a local text file, a local zip, or a download. The loader reads each word and its 300 values, converts the values to `float32`, and stacks the rows. `word_to_id` connects each word to its row. The resulting matrix has shape `(400000, 300)`. The optional nearest-neighbor check uses cosine similarity to inspect the vectors; it is not a classifier evaluation. The original GloVe 6B release is uncased and was trained on Wikipedia 2014 and Gigaword 5 ([Stanford GloVe repository](https://github.com/stanfordnlp/GloVe)).

**B.vi Vocabulary coverage.** `Counter` counts token occurrences. Type coverage measures the fraction of distinct tokens present in GloVe: 64,652 / 88,122 = 73.37%. Token coverage weights by frequency: 5,211,697 / 5,275,725 = 98.79%. This means 64,028 token occurrences are unmatched. The contrast shows that unmatched types are relatively infrequent. The frequent unmatched items include HTML fragments, URLs, and source names. High coverage does not establish that dropping them has no effect; rare names or technical terms may still matter.

**B.vii Padding.** `np.vstack()` appends an all-zero vector and assigns it the new ID 400,000. The assertion checks that `<pad>` does not replace an existing entry. The model excludes padding from both the summed vectors and the token count. Its embedding layer also receives `padding_idx=pad_tok_id`. All embeddings are frozen in this implementation ([PyTorch embedding documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)).

**B.viii Arrays and development split.** `to_token_ids()` keeps known tokens and looks up their IDs. `np.full()` allocates an `int64` array filled with the padding ID, and each sequence is copied into its row. Training inputs have shape `(120000, 212)`; labels have shape `(120000,)`. Subtracting one converts labels 1-4 to the 0-3 range required by the loss function. `train_test_split(..., stratify=y, random_state=seed)` preserves class proportions in the 96,000/24,000 train/development split ([scikit-learn split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)).

**B.ix Two-layer network.** `nn.Embedding.from_pretrained(..., freeze=True)` performs the word-vector lookup. In `forward()`, the mask identifies non-padding positions; summing along the sequence and dividing by the masked count produces the document mean. `clamp(min=1)` prevents division by zero if no known tokens remain. The mean passes through Dropout, Linear(300, 300), ReLU, Dropout, and Linear(300, 4). The two linear layers contain 300 x 300 + 300 + 300 x 4 + 4 = 91,504 trainable parameters. The embedding table contains 120,000,300 frozen values after padding.

**B.x Training and development evaluation.** `TensorDataset` and `DataLoader` create batches. For each training batch, the model computes logits, cross-entropy measures the loss, `zero_grad()` clears earlier gradients, `backward()` computes gradients, and `optimizer.step()` updates the trainable weights. Multiplying each batch's mean loss by its size gives an example-weighted epoch average. Development evaluation disables dropout and gradient tracking. The code runs all five epochs and restores the checkpoint with the highest development accuracy. This is checkpoint selection; it does not stop training early ([PyTorch loss documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)).

**B.xi Testing.** The test file uses the same cleaning, tokenization, vocabulary, and masked averaging. Its array has shape `(7600, 161)`: padding width is computed separately for each file, which this model can handle because averaging removes the sequence dimension. Under `model.eval()` and `torch.no_grad()`, `argmax()` selects the largest logit. Precision measures how often a predicted class is correct; recall measures the fraction of that class recovered; F1 is their harmonic mean. `classification_report()` and the confusion matrix summarize these outcomes ([scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)).

## C.2 Interpretation of the test results

The results below describe the saved run. Recalculating metrics from the submitted confusion matrix confirms their arithmetic consistency; it does not independently reproduce the model's predictions.

**Overall performance.** The model correctly classified 6,832 of 7,600 test articles, giving 89.89% accuracy and a 10.11% error rate. Macro-F1 was 0.8988. An always-one-class prediction would obtain 25% accuracy on this balanced test set. The best development accuracy was 90.25% at epoch 5, approximately 0.36 percentage points above the test accuracy. The scores are close, but one run cannot establish that the development partition represents every future sample or that model selection introduced no overfitting.

**Learning behavior.** Development loss declined across the five epochs, while development accuracy improved overall with a small dip at epoch 4.

| Epoch | Training loss | Training accuracy | Development loss | Development accuracy |
|---|---|---|---|---|
| 1 | 0.5118 | 84.21% | 0.3339 | 89.40% |
| 2 | 0.3602 | 87.68% | 0.3092 | 89.80% |
| 3 | 0.3444 | 87.92% | 0.2973 | 90.11% |
| 4 | 0.3376 | 88.19% | 0.2923 | 90.09% |
| 5 | 0.3292 | 88.37% | 0.2877 | 90.25% |

Training accuracy is lower than development accuracy, but the two measurements use different conditions. Training includes dropout and averages predictions made while weights are changing. Development evaluation uses the end-of-epoch weights with dropout disabled. The development curve shows no rise in loss over these five epochs. It does not rule out overfitting, and it does not establish that additional epochs would improve test performance.

**Differences between classes.**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| World | 0.9171 | 0.8847 | 0.9006 | 1,900 |
| Sports | 0.9512 | 0.9747 | 0.9628 | 1,900 |
| Business | 0.8523 | 0.8626 | 0.8574 | 1,900 |
| Sci/Tech | 0.8751 | 0.8737 | 0.8744 | 1,900 |
| Macro average | 0.8989 | 0.8989 | 0.8988 | 7,600 |

Sports has the highest F1 in this run; Business has the lowest. Sports recall of 0.9747 means that 1,852 of its 1,900 articles were recovered. Business recall of 0.8626 corresponds to 1,639 of 1,900. Equal test support makes the macro and support-weighted averages identical before rounding, but it does not make performance equal across classes.

| Gold / Predicted | World | Sports | Business | Sci/Tech |
|---|---|---|---|---|
| World | 1,681 | 62 | 106 | 51 |
| Sports | 24 | 1,852 | 18 | 6 |
| Business | 63 | 18 | 1,639 | 180 |
| Sci/Tech | 65 | 15 | 160 | 1,660 |

Business was predicted as Sci/Tech 180 times, and Sci/Tech as Business 160 times. Together, these account for 340 of 768 errors, or 44.27%. This is the largest combined confusion between two classes. Stories about technology companies can contain both business and technology content, which is a plausible explanation for the overlap. The confusion matrix alone does not establish why the model made each decision.

**Inspecting individual mistakes.** The saved sample includes iTunes pricing and Oracle/PeopleSoft takeover stories labeled Sci/Tech but predicted Business. Their content spans the two categories. It also includes an Olympic sprint story labeled World and predicted Sports, and a gene-therapy story labeled World and predicted Sci/Tech. Those examples raise questions about category boundaries or individual annotations; they do not confirm incorrect gold labels. All benchmark labels remain unchanged. A sample of ten mistakes cannot estimate the prevalence of labeling errors across the dataset.

**Limits and next experiments.** This is one saved training run with one seed, five epochs, and fixed hyperparameters. No experiment here isolates the benefit of pre-training, measures the impact of dropping unknown tokens, or compares the chosen hidden size and dropout with the reference settings. Averaging discards word order, and some HTML artifacts remain. Useful follow-up experiments would compare HTML cleanup, an unknown-token representation, embedding fine-tuning, and a phrase-aware baseline using the same development protocol. Repeating training across seeds would measure run-to-run variability. These are proposals, not results.

---
## References

- clulab. (n.d.). *Chapter 9: Multiclass text classification with feed-forward neural networks and word embeddings* [Jupyter notebook]. [Reference notebook](https://github.com/clulab/gentlenlp/blob/main/notebooks/chap09_classification.ipynb).
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *Proceedings of EMNLP*, 1532-1543. [Paper](https://aclanthology.org/D14-1162/). Pre-trained release details: [Stanford GloVe repository](https://github.com/stanfordnlp/GloVe).
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems, 28*. [Paper; dataset statistics in Table 3](https://arxiv.org/abs/1509.01626). CSV copy used by the code: [CharCnn_Keras AG News files](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv).
- NLTK project. (n.d.). *nltk.tokenize package*. [Tokenizer documentation](https://www.nltk.org/api/nltk.tokenize.html).
- pandas development team. (n.d.). *pandas.read_csv* and *pandas.DataFrame.replace*. [CSV reader](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html); [replacement](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html).
- PyTorch contributors. (n.d.). *PyTorch 2.14 documentation*. [Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html); [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html); [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
- scikit-learn developers. (n.d.). *scikit-learn documentation*. [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html); [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
- Python Software Foundation. (n.d.). *Python 3 standard library documentation*. Download and archive helpers: [urllib.request](https://docs.python.org/3/library/urllib.request.html); [zipfile](https://docs.python.org/3/library/zipfile.html); [io](https://docs.python.org/3/library/io.html).
"""

replace_cell(0, INTRO)
part_a_count = len(PART_A_BODY.split())
assert part_a_count < 500
replace_cell(1, f"# Part A: Feed-Forward Neural Networks and Word Embeddings\n\n{PART_A_BODY.strip()}\n\n*Part A: {part_a_count} words (whitespace count).*\n")
replace_cell(4, """
### Setup: obtain the data

Run this cell after B.i and before B.ii. It downloads missing AG News files and defines `open_glove()`. Downloads use a temporary `.part` file that is renamed after completion. The GloVe reader first checks for an extracted text file, then a local zip. It can try three download locations or use HTTP range requests to read a remote archive. The remote path depends on the server supporting byte-range requests. The saved run used a local zip; the fallback paths were not re-tested for this revision.
""")
replace_cell(9, r"""
## B.iii Replace backslashes with spaces using pandas `replace()`

Backslashes appear inside some descriptions, including `dwindling\band`. Replacing them with spaces separates the surrounding text. The function cleans both text columns, lowercases them, and joins them into a single `text` field to match the uncased GloVe vocabulary.
""")
replace_cell(11, """
## B.iv Tokenize the title and description with NLTK's `word_tokenize()`

The combined `text` column contains both required fields. As in the supplied reference notebook, `word_tokenize()` produces one token list per article from that column. `progress_map()` adds a tqdm progress bar. The NLTK setup checks for the `punkt_tab` resource before tokenizing.
""")
replace_cell(16, """
## B.vi Estimate tokenization coverage against GloVe

Type coverage counts distinct tokens; token coverage counts all occurrences. Both are measured on the original training file, before its train/development split. The figures describe vocabulary overlap and do not directly measure the effect of unknown words on classification accuracy.
""")
replace_cell(20, """
## B.viii Build 2-D token-ID and 1-D gold-label NumPy arrays

Known tokens map to GloVe IDs; unknown tokens are omitted. Each sequence is right-padded to the longest known-token sequence in that file. Labels shift from 1-4 to 0-3. A stratified split reserves 20% of the original training file for development evaluation. The array width is computed before this split, but no development labels or fitted text statistics are used to create the embeddings.
""")
replace_cell(24, """
## B.x Train and evaluate development loss and accuracy after each epoch

Training uses shuffled mini-batches with dropout enabled. Development evaluation uses fixed end-of-epoch weights with dropout disabled and no gradient tracking. All five epochs run; the checkpoint with the highest development accuracy is restored afterward.
""")
replace_cell(27, """
## B.xi Evaluate on the test partition

The test file uses the same cleaning, tokenizer, GloVe vocabulary, and model. Its padding width is computed independently; masked averaging supports this different sequence length. The selected checkpoint produces test predictions, accuracy, per-class precision, recall, F1, and a confusion matrix.
""")
replace_cell(31, PART_C)

# Numerical evidence and executable commands must remain exactly as supplied.
for old, new in zip(original["cells"], reviewed["cells"]):
    if old["cell_type"] == "code":
        assert old == new
NOTEBOOK.write_text(json.dumps(reviewed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))


def add_link(paragraph, label, url):
    relationship = paragraph.part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "000000")
    props.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(underline)
    run.append(props)
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def inline(paragraph, text):
    pattern = r"(\[[^\]]+\]\(https?://[^\s)]+\)|`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)"
    for part in re.split(pattern, text):
        link = re.fullmatch(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", part)
        if link:
            add_link(paragraph, *link.groups())
        elif part.startswith("`") and part.endswith("`"):
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*"):
            paragraph.add_run(part[1:-1]).italic = True
        elif part:
            paragraph.add_run(part)


doc = Document()
section = doc.sections[0]
section.top_margin = section.bottom_margin = Inches(0.65)
section.left_margin = section.right_margin = Inches(0.65)
section.page_width, section.page_height = Inches(8.5), Inches(11)
normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Calibri", Pt(10)
normal.font.color.rgb = RGBColor.from_string("000000")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.06
for name in ["Heading 1", "Heading 2", "Heading 3"]:
    doc.styles[name].font.name = "Calibri"
    doc.styles[name].font.color.rgb = RGBColor.from_string("000000")
for name in ["Lab Code", "Lab Output"]:
    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.font.name = "Consolas"
    style.font.size = Pt(7.5)
    style.paragraph_format.space_after = Pt(0)
    style.paragraph_format.line_spacing = 1.0
    style.paragraph_format.widow_control = False
    if name == "Lab Output":
        style.font.color.rgb = RGBColor.from_string("000000")
footer = section.footer.paragraphs[0]
footer.add_run("Lab 2 | ")
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)


def markdown(text):
    lines = text.strip().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line == "---":
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                row = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in row):
                    rows.append(row)
                index += 1
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Light Shading Accent 1"
            for n, row in enumerate(rows):
                cells = table.add_row().cells
                for cell, value in zip(cells, row):
                    inline(cell.paragraphs[0], value)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(9)
                        if n == 0:
                            run.bold = True
            repeat = OxmlElement("w:tblHeader")
            table.rows[0]._tr.get_or_add_trPr().append(repeat)
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            paragraph = doc.add_heading(level=len(heading.group(1)))
            inline(paragraph, heading.group(2))
        elif line.startswith("- ") or line.startswith("* "):
            inline(doc.add_paragraph(style="List Bullet"), line[2:])
        else:
            inline(doc.add_paragraph(), line)
        index += 1


def code_block(text, output=False):
    for line in text.rstrip().splitlines():
        doc.add_paragraph(line or " ", style="Lab Output" if output else "Lab Code")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# The report follows notebook order. Long download plumbing is placed in an
# appendix, with its execution position identified in the main text.
for index, cell in enumerate(reviewed["cells"]):
    if cell.get("id") == STYLE_ID:
        continue
    if index in [4, 5]:
        continue
    if cell["cell_type"] == "markdown":
        if index == 2:
            doc.add_page_break()
        if index == 31:
            doc.add_page_break()
        markdown("".join(cell["source"]))
        continue
    code_block("".join(cell["source"]))
    if index == 3:
        markdown("**Setup before B.ii.** Execute the download helper in the Appendix after these imports. It defines `DATA_DIR`, `GLOVE_PATH`, and `open_glove()`.")
    stream_text = "".join("".join(o.get("text", [])) for o in cell.get("outputs", [])
                          if o.get("output_type") == "stream")
    if stream_text.strip():
        markdown("**Saved output**")
        code_block(stream_text, output=True)
    # Include the inspection table as a compact, readable excerpt.
    if index == 7:
        markdown("**Preview from `train_df.head()`** (titles are shown here; the notebook also displays descriptions).")
        markdown("| Class index | Title |\n|---|---|\n|3|Wall St. Bears Claw Back Into the Black (Reuters)|\n|3|Carlyle Looks Toward Commercial Aerospace (Reuters)|\n|3|Oil and Economy Cloud Stocks' Outlook (Reuters)|\n|3|Iraq Halts Oil Exports from Main Southern Pipeline (Reuters)|\n|3|Oil prices soar to all-time record, posing new menace to US economy (AFP)|")
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            png = data["image/png"]
            if isinstance(png, list):
                png = "".join(png)
            doc.add_picture(io.BytesIO(base64.b64decode(png)), width=Inches(6.6 if index == 26 else 5.4))
            caption = "Learning curves from the saved run." if index == 26 else "Test confusion matrix from the saved run; rows are gold labels."
            doc.add_paragraph(caption, style="Caption")

doc.add_page_break()
markdown("# Appendix: data download helper\n\nExecute this setup cell after B.i and before B.ii. The helper is unchanged from the supplied notebook. Its remote fallback paths were not exercised during this review.")
code_block("".join(reviewed["cells"][5]["source"]))
markdown("**Saved setup output**")
code_block("AG News: ['classes.txt', 'test.csv', 'train.csv']", output=True)
doc.core_properties.title = "Lab 2: Pre-Trained Word Embeddings"
doc.core_properties.subject = "Reviewed assignment with retained saved-run results"
doc.core_properties.author = "Albert Kabore"
doc.core_properties.keywords = "AG News, GloVe, PyTorch, text classification"
doc.save(REPORT)
format_word(REPORT)
format_notebooks()
try:
    REPORT.replace(REPORT_FINAL)
    report_status = REPORT_FINAL.name
except PermissionError:
    report_status = "Prepared in review_assets/Lab2_Report.pending; destination is open in Word."

print(f"Part A word count: {part_a_count}")
print(f"Notebook: {NOTEBOOK.name}")
print(f"Word report: {report_status}")
print("Numerical results and execution counts preserved; display formatting applied.")
