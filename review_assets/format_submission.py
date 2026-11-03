"""Apply consistent display formatting without changing numerical results."""

import base64
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lxml import etree
from matplotlib.text import Text
from sklearn.metrics import ConfusionMatrixDisplay


ROOT = Path(__file__).resolve().parents[1]
STYLE_ID = "lab2-display-style"
CSS = """<style>
.jp-Notebook, #notebook-container, .jp-Cell, .cell,
.jp-RenderedHTMLCommon, .jp-RenderedText, .text_cell_render,
.rendered_html, .input_area, .output_area, .highlight {
    color: #000000 !important;
    background-color: #ffffff !important;
}
.jp-Notebook *, #notebook-container *, .jp-Cell *, .cell *,
.jp-RenderedHTMLCommon *, .jp-RenderedText *, .text_cell_render *,
.rendered_html *, .input_area *, .output_area *, .highlight *,
.cm-editor .cm-content *, .cm-s-jupyter * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: none !important;
}
.jp-Notebook a, .rendered_html a, .text_cell_render a {
    color: #000000 !important;
    text-decoration: underline;
}
</style>"""
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def finish_figure(fig, destination):
    for label in fig.findobj(match=Text):
        label.set_color("black")
    fig.savefig(destination, format="png", dpi=100, facecolor="white")
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=100, facecolor="white")
    plt.close(fig)
    return base64.b64encode(output.getvalue()).decode("ascii")


def figure_outputs():
    # Values are the supplied saved outputs, not a new training run.
    history = pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "train_loss": [.5118, .3602, .3444, .3376, .3292],
        "dev_loss": [.3339, .3092, .2973, .2923, .2877],
        "train_acc": [.8421, .8768, .8792, .8819, .8837],
        "dev_acc": [.8940, .8980, .9011, .9009, .9025],
    }).set_index("epoch")
    fig, (loss, accuracy) = plt.subplots(1, 2, figsize=(11, 4))
    history[["train_loss", "dev_loss"]].plot(ax=loss, marker="o", title="Cross-entropy loss")
    history[["train_acc", "dev_acc"]].plot(ax=accuracy, marker="o", title="Accuracy")
    for axis in (loss, accuracy):
        axis.set_xlabel("epoch")
        axis.grid(alpha=.3)
    fig.tight_layout()
    curves = finish_figure(fig, ROOT / "report_assets/learning_curves.png")

    findings = json.loads((ROOT / "review_assets/verification.json").read_text(encoding="utf-8"))
    cm = np.array(findings["saved_confusion_matrix"], dtype=np.int64)
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    fig, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=axis, cmap="Blues", colorbar=False, im_kw={"vmin": 0, "vmax": 2 * cm.max()})
    for label in display.text_.ravel():
        label.set_color("black")
    axis.set_title("Test confusion matrix (rows = gold, columns = predicted)")
    fig.tight_layout()
    matrix = finish_figure(fig, ROOT / "report_assets/confusion_matrix.png")
    return curves, matrix


def format_notebooks():
    curves, matrix = figure_outputs()
    for path in ROOT.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        notebook["cells"] = [cell for cell in notebook["cells"] if cell.get("id") != STYLE_ID]
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if "ConfusionMatrixDisplay(cm," in source:
                source = source.replace(
                    "ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap='Blues', colorbar=False)",
                    "display = ConfusionMatrixDisplay(cm, display_labels=classes).plot(\n"
                    "    ax=ax, cmap='Blues', colorbar=False, im_kw={'vmin': 0, 'vmax': 2 * cm.max()})\n"
                    "for label in display.text_.ravel():\n"
                    "    label.set_color('black')")
                cell["source"] = source.splitlines(keepends=True)
            replacement = None
            if "history_df[['train_loss', 'dev_loss']].plot" in source:
                replacement = curves
            elif "ConfusionMatrixDisplay(cm," in source:
                replacement = matrix
            if replacement:
                for output in cell.get("outputs", []):
                    if "image/png" in output.get("data", {}):
                        output["data"]["image/png"] = replacement
        style_cell = {
            "cell_type": "code", "execution_count": None, "id": STYLE_ID,
            "metadata": {"jupyter": {"source_hidden": True}, "tags": ["remove_input"], "trusted": True},
            "source": ["from IPython.display import HTML, display\n", f"display(HTML({CSS!r}))"],
            "outputs": [{"output_type": "display_data", "metadata": {},
                         "data": {"text/html": CSS, "text/plain": [""]}}],
        }
        notebook["cells"].append(style_cell)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def format_word(path):
    with ZipFile(path) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    doc_ns = {"w": W, "v": "urn:schemas-microsoft-com:vml"}
    formatted = []
    for info, content in entries:
        if info.filename.startswith("word/") and info.filename.endswith(".xml"):
            root = etree.fromstring(content)
            for color in root.findall(f".//{{{W}}}color"):
                color.attrib.clear()
                color.set(f"{{{W}}}val", "000000")
            # Explicit colors override conditional table styles and link themes.
            for run in root.findall(f".//{{{W}}}r"):
                props = run.find(f"{{{W}}}rPr")
                if props is None:
                    props = etree.Element(f"{{{W}}}rPr")
                    run.insert(0, props)
                if props.find(f"{{{W}}}color") is None:
                    etree.SubElement(props, f"{{{W}}}color").set(f"{{{W}}}val", "000000")
            for shape in root.xpath("//v:shape", namespaces=doc_ns):
                marker = " ".join(shape.attrib.values()).lower()
                if "watermark" in marker or (info.filename.startswith("word/header") and shape.find("v:textpath", doc_ns) is not None):
                    shape.getparent().remove(shape)
            content = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        formatted.append((info, content))
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for info, content in formatted:
            archive.writestr(info, content)


if __name__ == "__main__":
    format_notebooks()
    format_word(ROOT / "Lab2_Pre_Trained_Word_Embeddings_Report_Reviewed.docx")
    print("Updated submission formatting.")
