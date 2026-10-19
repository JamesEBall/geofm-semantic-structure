"""Plot archived and corrected transfer results. Author: James Edward Ball.

Run with Python, NumPy and Matplotlib from any directory. Source JSON files
remain unchanged; plotted values and source hashes are exported alongside figures.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "full-20261018"
ARCHES = ("clay", "prithvi", "prithvi100")
NAMES = ("Clay", "Prithvi-300", "Prithvi-100")
REFS = ("qwen_instructed", "qwen_plain", "mpnet")
REF_NAMES = ("Qwen, instructed", "Qwen, plain", "MPNet")
CONTROL = "appearance_rff_metadata_rich"
BEFORE, AFTER = "#8B949D", "#245F80"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(data, route):
    rows = [x for x in data["estimates"] if x["route"] == route
            and x["pooling"] == "mean" and x["scheme"] == "family"
            and x["metric"] == "family.qwen_instructed.mean.ridge/rr"]
    assert len(rows) == 1
    return rows[0]


def gain(data, arch, ref="qwen_instructed"):
    rows = [x for x in data["comparisons"] if x["architecture"] == arch
            and x["ref"] == ref and x["control"] == "best-control-per-draw"]
    assert len(rows) == 1
    return rows[0]


def point(ax, row, y, colour, value, hollow=False):
    x, (lo, hi) = row[value], row["ci"]
    assert np.isfinite([x, lo, hi]).all() and lo <= x <= hi
    ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=colour,
                mfc="white" if hollow else colour, mec=colour, ms=5,
                mew=.9, elinewidth=.9, capsize=2.5, capthick=.8, zorder=3)


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#A8AFB5")
    ax.spines["bottom"].set_linewidth(.6)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", width=.6, color="#A8AFB5")
    ax.grid(axis="x", color="#E4E7EA", linewidth=.5, zorder=0)


def save(fig, stem):
    fig.savefig(HERE / (stem+".png"), dpi=190, facecolor="white")
    fig.savefig(HERE / (stem+".pdf"), metadata={"Author": "James Edward Ball"})
    plt.close(fig)


def main():
    old, new = read(OLD/"results.json"), read(HERE/"results.json")
    verdict = read(HERE/"qualify/verdict.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "normal", "axes.labelsize": 10,
                         "text.color": "#26323B", "axes.labelcolor": "#26323B",
                         "xtick.color": "#47545F", "ytick.color": "#26323B",
                         "pdf.fonttype": 42})
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.2), gridspec_kw={"wspace": .28})
    fig.subplots_adjust(left=.15, right=.97, top=.78, bottom=.24)
    routes = ["r1-"+a for a in ARCHES] + [CONTROL]
    labels = list(NAMES) + ["Appearance +\nmetadata control"]
    for version, data, colour, dy in [("Before", old, BEFORE, .12),
                                      ("Corrected", new, AFTER, -.12)]:
        for j, route in enumerate(routes):
            row = estimate(data, route)
            assert row["classes"] == 188 and row["pairs"] == (11495 if version=="Before" else 8755)
            point(axs[0], row, 3-j+dy, colour, "value", version=="Before")
            rows.append(dict(version=version, model=labels[j], reference="qwen_instructed",
                             metric="MRR", value=row["value"], lower=row["ci"][0], upper=row["ci"][1]))
        for j, arch in enumerate(ARCHES):
            row = gain(data, arch)
            point(axs[1], row, 3-j+dy, colour, "delta", version=="Before")
            rows.append(dict(version=version, model=NAMES[j], reference="qwen_instructed",
                             metric="gain", value=row["delta"], lower=row["ci"][0], upper=row["ci"][1]))
    for ax in axs:
        style(ax); ax.set_ylim(-.5, 3.5); ax.set_yticks(range(4))
    axs[0].set_yticklabels(labels[::-1]); axs[1].set_yticklabels([])
    axs[0].set_xlim(.15,.32); axs[0].set_xlabel("Mean reciprocal rank (higher is better)")
    axs[0].set_title("Transfer score", loc="left", pad=14)
    axs[1].axvline(0, color="#596670", linewidth=.8, zorder=1)
    axs[1].set_xlim(-.02,.065); axs[1].set_xlabel("MRR gain over the strongest control")
    axs[1].set_title("Advantage beyond the controls", loc="left", pad=14)
    axs[1].text(.004,0,"Control reference",va="center",color="#697680",fontsize=9)
    fig.suptitle("Clay and Prithvi-300 retain their transfer advantage", x=.15, ha="left", y=.97, fontsize=14)
    fig.legend(handles=[Line2D([],[],color=BEFORE,marker="o",mfc="white",lw=.9,label="Before correction"),
                        Line2D([],[],color=AFTER,marker="o",lw=.9,label="Corrected")],
               loc="upper left",bbox_to_anchor=(.143,.92),ncol=2,frameon=False,handlelength=2)
    fig.text(.15,.10,"Bars show 95% conditional bootstrap intervals. The two analyses use different cohorts and family exclusions.",fontsize=9)
    fig.text(.15,.055,"Primary test: held-out category families, mean pooling and instructed Qwen. Scores measure ranking performance.",fontsize=9)
    save(fig,"primary-transfer-comparison")

    fig, axs = plt.subplots(1,3,figsize=(11.6,4.25),sharex=True,sharey=True)
    fig.subplots_adjust(left=.16,right=.97,top=.77,bottom=.26,wspace=.15)
    for ax, arch, name in zip(axs,ARCHES,NAMES):
        style(ax); ax.axvline(0,color="#596670",lw=.8)
        for j,ref in enumerate(REFS):
            row=gain(new,arch,ref); point(ax,row,2-j,AFTER,"delta")
            rows.append(dict(version="Corrected",model=name,reference=ref,metric="gain",
                             value=row["delta"],lower=row["ci"][0],upper=row["ci"][1]))
        ax.set_title(name,loc="left",pad=12)
        ax.set_xlim(-.02,.065);ax.set_ylim(-.5,2.5);ax.set_yticks([0,1,2]);ax.set_yticklabels(REF_NAMES[::-1])
        ax.set_xlabel("MRR gain over controls")
    fig.suptitle("The corrected pattern holds across three language references",x=.16,ha="left",y=.96,fontsize=13)
    fig.text(.16,.12,"Each point compares the model with the strongest control per bootstrap draw; bars show 95% conditional intervals.",fontsize=9)
    fig.text(.16,.065,"Clay and Prithvi-300 pass the registered sensitivity checks for every reference. Prithvi-100 does not.",fontsize=9)
    save(fig,"language-reference-sensitivity")
    with (HERE/"plotted-values.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources=[OLD/"results.json",HERE/"results.json",HERE/"qualify/verdict.json"]
    provenance={"author":"James Edward Ball","sources":{str(p.relative_to(HERE.parent)):sha(p) for p in sources},
                "plot_script_sha256":sha(Path(__file__)),"plot_data_sha256":sha(HERE/"plotted-values.csv"),
                "limits":["Conditional on fitted maps, tuning, vocabulary and seed panel.",
                          "No confidence interval or significance test for the before/after change.",
                          "Geometry population interval qualification remains failed."]}
    (HERE/"figure-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__ == "__main__":
    main()
