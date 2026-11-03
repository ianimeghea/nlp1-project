import argparse
import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

W, H = 13.333, 7.5
NAVY = "#1f2d3d"
ACCENT = "#1f4e79"
GREY = "#5a6470"


def new_slide(pdf_title, number, footer):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.add_artist(plt.Rectangle((0, 0.9), 1, 0.1, transform=fig.transFigure, color=NAVY))
    fig.text(0.035, 0.95, pdf_title, color="white", fontsize=24, weight="bold", va="center")
    fig.text(0.965, 0.95, f"{number}/5", color="#c9d3dd", fontsize=13, va="center", ha="right")
    fig.text(0.035, 0.025, footer, color=GREY, fontsize=9)
    return fig


def bullets(fig, items, x=0.035, y=0.85, width=0.46, size=12.5, gap=0.012):
    for item in items:
        level = 0
        if item.startswith("- "):
            level, item = 1, item[2:]
        if item.startswith("## "):
            fig.text(x, y, item[3:], fontsize=size + 1, weight="bold", color=ACCENT, va="top")
            y -= 0.05
            continue
        marker = "•" if level == 0 else "–"
        indent = x + 0.02 * level
        chars = int(width * 150 * (12.5 / size)) - 3 * level
        lines = wrap(item, chars)
        fig.text(indent, y, marker, fontsize=size - level, va="top", color=ACCENT)
        for line in lines:
            fig.text(indent + 0.016, y, line, fontsize=size - level, va="top", color=NAVY)
            y -= 0.04 if level == 0 else 0.036
        y -= gap
    return y


def wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return lines


def image(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(mpimg.imread(path))
    ax.axis("off")
    return ax


def table(fig, rect, header, rows, col_widths=None, size=11):
    ax = fig.add_axes(rect)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(size)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#c9d3dd")
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--authors", default="")
    p.add_argument("--out", default="slides.pdf")
    args = p.parse_args()
    rep = json.load(open("results/report.json"))
    s = rep["data_stats"]
    footer = "NLP1 2026 · Project 1 · Emotion classification on ISEAR"
    if args.authors:
        footer += f" · {args.authors}"

    with PdfPages(args.out) as pdf:
        fig = new_slide("Data and metrics", 1, footer)
        bullets(fig, [
            "## ISEAR: 7-way emotion classification",
            "Self-reported descriptions of an emotional episode; labels: joy, fear, anger, "
            "sadness, disgust, shame, guilt",
            f"Cleaning: {s['raw']:,} raw → {s['kept']:,} texts (dropped {s['empty_or_placeholder']} "
            f"placeholders like “NO RESPONSE”, {s['duplicates']} duplicates, "
            f"{s['conflicting']} texts with conflicting labels)",
            f"Balanced (~1,050 per class, chance 14.3%); short texts: median "
            f"{rep['length_median']:.0f} tokens, 95% under {rep['length_p95']:.0f}",
            "## Evaluation",
            "Stratified 10-fold CV × 3 repeats = 30 scores per system; identical folds for all "
            "systems, so comparisons are paired",
            "Accuracy: classes are balanced, so it is close to macro-F1 (we log both; the ranking is the same)",
            "Bayesian correlated t-test (Benavoli et al.) on fold scores; correlation ρ = 1/10 "
            "accounts for overlapping training sets",
            "ROPE = ±1 accuracy point ≈ 7 of ~743 texts per fold, below the fold-to-fold std "
            "(≈1.5 points); decide when a region has P > 0.95",
        ], width=0.47)
        image(fig, "figures/data.png", [0.52, 0.3, 0.46, 0.55])
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_slide("Features", 2, footer)
        bullets(fig, [
            "## Segmentation: BPE, 8k vocabulary, lowercased",
            "ISEAR alone is small (~166k words), so BPE is trained on dair-ai/emotion "
            "(417k English tweets, first-person “i feel …” style), a similar register",
            "An external corpus also keeps the tokeniser from seeing the validation folds",
            "Punctuation is added to the initial alphabet: 0% [UNK] on ISEAR",
            "## Representations (V = 8,000 dims, sparse)",
            "Token indicator: h_v(x) = 1[v in x]",
            "Bag-of-words: h_v(x) = count(v, x)",
            "TF-IDF (ours): (1 + log tf) · idf, L2-normalised; idf is fitted on the training fold only",
            "- Down-weights words like “i”, “when”, “my” that occur in nearly every text "
            "and fixes the scale across text lengths",
            "86% of non-zero counts are 1, so indicators and counts carry almost the same information",
        ], width=0.5)
        image(fig, "figures/vocab.png", [0.56, 0.28, 0.42, 0.55])
        fig.text(0.77, 0.24, "Vocabulary size (TF-IDF, mean ± 2 s.e.): gains stop around 8k",
                 ha="center", fontsize=10, color=GREY)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_slide("Architectures", 3, footer)
        table(fig, [0.04, 0.45, 0.92, 0.41],
              ["", "Log-linear", "MLP-1 (1 hidden layer)", "MLP-2 (ours, 2 hidden)"],
              [["Structure", "softmax(Wh + b)", "V→256 → 7", "V→512→256 → 7"],
               ["Non-linearity", "–", "ReLU", "GELU"],
               ["Regularisation", "L2 (C = 1)", "dropout 0.5, weight decay 1e-4", "dropout 0.5, weight decay 1e-4"],
               ["Optimiser", "L-BFGS (sklearn)", "AdamW, lr 1e-3, batch 64", "AdamW, lr 1e-3, batch 64"],
               ["Parameters", "56k", "2.05M", "4.23M"]],
              col_widths=[0.16, 0.22, 0.31, 0.31], size=13)
        bullets(fig, [
            "Early stopping on cross-entropy over a 10% inner dev split of each training fold "
            "(patience 4, max 40 epochs); the validation fold is never used for model selection. "
            "Picking the best epoch on the validation fold, as in the tutorial, inflates the scores",
            "MLP-2 checks whether a deeper and wider network helps with 6.7k training texts",
            "Same seeds and same folds for every system; 270 models in total",
        ], y=0.5, width=0.93, size=13)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_slide("Results", 4, footer)
        image(fig, "figures/scores.png", [0.01, 0.38, 0.46, 0.5])
        image(fig, "figures/posteriors.png", [0.0, 0.06, 0.6, 0.3])
        bullets(fig, [
            "Best: log-linear + TF-IDF, 59.7% accuracy (macro-F1 0.594)",
            "Features: TF-IDF beats indicator and BoW for log-linear (+2.2 points, P = 0.997). "
            "Indicator ≡ BoW for every model (P(tie) > 0.97)",
            "For MLP-1 all three features are equivalent: the hidden layer learns its own reweighting",
            "Models: MLP-1 beats log-linear on indicator and BoW (+1.9 points, P ≈ 0.98) "
            "but is equivalent on TF-IDF (P(tie) = 0.96)",
            "MLP-2 never beats MLP-1; it is worse on TF-IDF (P = 0.93, undecided at 0.95)",
            "Our pick: log-linear + TF-IDF. It is equivalent to the best MLP with 37× fewer parameters, "
            "is deterministic and is interpretable",
        ], x=0.6, y=0.85, width=0.3, size=11.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig = new_slide("Extra: errors, shortcuts and ROPE sensitivity", 5, footer)
        image(fig, "figures/confusion.png", [0.0, 0.08, 0.36, 0.8])
        tops = rep["top_features"]
        rows = [[k, ", ".join(v[:5])] for k, v in tops.items()]
        table(fig, [0.37, 0.48, 0.3, 0.38], ["label", "top log-linear tokens"], rows,
              col_widths=[0.25, 0.75], size=10)
        bullets(fig, [
            "Errors: shame ↔ guilt (0.15 / 0.12) and anger → disgust/guilt; joy and fear are easiest",
            "Shortcut: the top weights are often the label word itself; 23% of texts name their own emotion",
            "ROPE sensitivity: at ±0.5 points MLP-2 + TF-IDF loses to every top system; "
            "at ±2 points no system is decisively better than any other",
            "So the claim that features and models matter holds only for differences of about 1–2 "
            "points",
            "Next: subword n-grams and pretrained embeddings (project 2)",
        ], x=0.69, y=0.85, width=0.23, size=11)
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
