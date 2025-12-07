import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   # for normal pdf

plt.xkcd()

ID_COL = "Fragebogen"  # questionnaire ID column

# Load data
df = pd.read_csv("christams_data.csv", sep=";")

# ============================================================
#  GROUP STYLING CONFIGURATION
# ============================================================

# Default marker styles for each respondent
# (facecolor + edgecolor + marker; can be overwritten by group definitions)
marker_styles = {}
for id_ in df[ID_COL].dropna().astype(str):
    marker_styles[id_] = {
        "facecolor": "white",
        "edgecolor": "black",
        "marker": "^",   # default: upward triangle
    }

# ------------------------------------------------------------
# GROUP 1: "the rouges"
# pipette moral crime: 1–5, rouges = <= 3
# ------------------------------------------------------------
rouge_mask = pd.to_numeric(df["q03_pipette_moral_crime"], errors="coerce") <= 3
rouge_ids = df.loc[rouge_mask, ID_COL].dropna().astype(str)

for rid in rouge_ids:
    # Overwrite ONLY the facecolor, keep other properties as-is
    marker_styles[rid]["facecolor"] = "black"

# ------------------------------------------------------------
# GROUP 2: "the braves"
# q09_punch_water_bath == 1 → yes
# ------------------------------------------------------------
brave_mask = pd.to_numeric(df["q09_punch_water_bath"], errors="coerce") == 1
brave_ids = df.loc[brave_mask, ID_COL].dropna().astype(str)

for bid in brave_ids:
    # Overwrite ONLY the edgecolor, keep other properties as-is
    marker_styles[bid]["edgecolor"] = "red"

# ------------------------------------------------------------
# GROUP 3: lids up vs down
# q14_lids_orientation: 1 = up, 2 = down
# ------------------------------------------------------------
orient14 = pd.to_numeric(df["q14_lids_orientation"], errors="coerce")

lid_up_ids = df.loc[orient14 == 1, ID_COL].dropna().astype(str)
lid_down_ids = df.loc[orient14 == 2, ID_COL].dropna().astype(str)

for lid in lid_down_ids:
    marker_styles[lid]["marker"] = "v"   # downward triangle


# ---------- TYPE 1: NUMERIC + NORMAL FIT + JITTER DOTS ----------

def plot_numeric_with_normal(df, col, xlabel, title, bin_width=0.5, filename=None):
    # Use only rows that have BOTH ID and this value
    sub = df[[ID_COL, col]].dropna()
    if sub.empty:
        print(f"No data for {col}")
        return

    ids = sub[ID_COL].astype(str).values
    values = sub[col].astype(float).values

    # Bins
    min_val = values.min()
    max_val = values.max()
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    # Fit normal
    mu = values.mean()
    sigma = values.std(ddof=1)

    counts, _ = np.histogram(values, bins=bins)

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, edgecolor="black", align="left",
             alpha=0.7, label="Data")

    # Only plot normal if we have some variance
    if sigma > 0:
        x = np.linspace(min_val, max_val, 200)
        pdf = norm.pdf(x, mu, sigma)
        pdf_scaled = pdf * (counts.max() / pdf.max())
        plt.plot(x, pdf_scaled, color="black", linewidth=2,
                 label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")

    # Overlay jittered dots for each individual
    max_count = counts.max() if len(counts) > 0 else 1
    y_j_min = 0.1 * max_count
    y_j_max = 0.9 * max_count

    x_jitter = values + np.random.uniform(-bin_width * 0.15,
                                          bin_width * 0.15,
                                          size=len(values))
    y_jitter = np.random.uniform(y_j_min, y_j_max, size=len(values))

    # Style per ID
    facecolors = np.array(
        [marker_styles.get(id_, {}).get("facecolor", "white") for id_ in ids]
    )
    edgecolors = np.array(
        [marker_styles.get(id_, {}).get("edgecolor", "black") for id_ in ids]
    )
    markers = np.array(
        [marker_styles.get(id_, {}).get("marker", "^") for id_ in ids]
    )

    # Split into upward and downward triangles
    up_mask_m = (markers == "^")
    down_mask_m = (markers == "v")

    if up_mask_m.any():
        plt.scatter(
            x_jitter[up_mask_m],
            y_jitter[up_mask_m],
            facecolors=facecolors[up_mask_m],
            edgecolors=edgecolors[up_mask_m],
            marker="^",
            s=180,
            zorder=3,
        )

    if down_mask_m.any():
        plt.scatter(
            x_jitter[down_mask_m],
            y_jitter[down_mask_m],
            facecolors=facecolors[down_mask_m],
            edgecolors=edgecolors[down_mask_m],
            marker="v",
            s=180,
            zorder=3,
        )

    # Label each dot with Fragebogen ID (for exploration)
    for x_val, y_val, id_ in zip(x_jitter, y_jitter, ids):
        plt.text(x_val, y_val, id_, fontsize=6, ha="center", va="center", zorder=4)

    plt.xticks(bins)
    plt.xlabel(xlabel)
    plt.ylabel("Number of people")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- TYPE 2: LIKERT (AGREEMENT SCALE) + JITTER DOTS ----------

LIKERT_LABELS = {
    1: "Strongly\ndisagree",
    2: "Disagree",
    3: "Indifferent",
    4: "Agree",
    5: "Strongly\nagree",
}

def plot_likert(df, col, title, filename=None, as_percent=True):
    sub = df[[ID_COL, col]].dropna()
    if sub.empty:
        print(f"No data for {col}")
        return

    ids = sub[ID_COL].astype(str).values
    values = sub[col].astype(int).values

    counts = pd.Series(values).value_counts().sort_index()
    index = pd.Index([1, 2, 3, 4, 5])
    counts = counts.reindex(index, fill_value=0)

    if as_percent:
        counts_plot = counts / counts.sum() * 100
        ylabel = "Percentage of people"
    else:
        counts_plot = counts
        ylabel = "Number of people"

    x = np.arange(1, 6)

    plt.figure(figsize=(7, 4))
    plt.bar(x, counts_plot.values, edgecolor="black", alpha=0.8)

    labels = [LIKERT_LABELS[i] for i in x]
    plt.xticks(x, labels)

    # Overlay jittered dots
    max_height = counts_plot.max() if len(counts_plot) > 0 else 1
    y_j_min = 0.1 * max_height
    y_j_max = 0.9 * max_height

    x_jitter = []
    y_jitter = []

    for v, id_ in zip(values, ids):
        x_val = v + np.random.uniform(-0.15, 0.15)
        y_val = np.random.uniform(y_j_min, y_j_max)
        x_jitter.append(x_val)
        y_jitter.append(y_val)

    x_jitter = np.array(x_jitter)
    y_jitter = np.array(y_jitter)

    facecolors = np.array(
        [marker_styles.get(id_, {}).get("facecolor", "white") for id_ in ids]
    )
    edgecolors = np.array(
        [marker_styles.get(id_, {}).get("edgecolor", "black") for id_ in ids]
    )
    markers = np.array(
        [marker_styles.get(id_, {}).get("marker", "^") for id_ in ids]
    )

    up_mask_m = (markers == "^")
    down_mask_m = (markers == "v")

    if up_mask_m.any():
        plt.scatter(
            x_jitter[up_mask_m],
            y_jitter[up_mask_m],
            facecolors=facecolors[up_mask_m],
            edgecolors=edgecolors[up_mask_m],
            marker="^",
            s=180,
            zorder=3,
        )

    if down_mask_m.any():
        plt.scatter(
            x_jitter[down_mask_m],
            y_jitter[down_mask_m],
            facecolors=facecolors[down_mask_m],
            edgecolors=edgecolors[down_mask_m],
            marker="v",
            s=180,
            zorder=3,
        )

    for x_val, y_val, id_ in zip(x_jitter, y_jitter, ids):
        plt.text(x_val, y_val, id_, fontsize=6, ha="center", va="center", zorder=4)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- TYPE 3: SINGLE CHOICE (1..N coded) + JITTER DOTS ----------

def plot_single_choice(df, col, title, option_labels, filename=None, as_percent=True):
    """
    df: dataframe
    col: column name in df (values encoded as 1..N integers)
    title: plot title
    option_labels: dict mapping int -> label
    """
    sub = df[[ID_COL, col]].dropna()
    if sub.empty:
        print(f"No data for {col}")
        return

    ids = sub[ID_COL].astype(str).values
    values = sub[col].astype(int).values

    categories = sorted(option_labels.keys())
    max_cat = max(categories)
    full_range = list(range(1, max_cat + 1))

    counts = pd.Series(values).value_counts()
    counts = counts.reindex(full_range, fill_value=0)

    total = counts.sum()
    if total == 0:
        print(f"No valid responses for {col}, saw values: {sorted(values.unique())}")
        return

    if as_percent:
        counts_plot = counts / total * 100
        ylabel = "Percentage of people"
    else:
        counts_plot = counts
        ylabel = "Number of people"

    x = np.arange(len(full_range))

    plt.figure(figsize=(7, 4))
    plt.bar(x, counts_plot.values, edgecolor="black", alpha=0.8)

    tick_labels = [option_labels.get(cat, str(cat)) for cat in full_range]
    plt.xticks(x, tick_labels)

    # Overlay jittered dots
    max_height = counts_plot.max() if len(counts_plot) > 0 else 1
    y_j_min = 0.1 * max_height
    y_j_max = 0.9 * max_height

    x_jitter = []
    y_jitter = []

    for v, id_ in zip(values, ids):
        x_center = (v - 1)
        x_val = x_center + np.random.uniform(-0.15, 0.15)
        y_val = np.random.uniform(y_j_min, y_j_max)
        x_jitter.append(x_val)
        y_jitter.append(y_val)

    x_jitter = np.array(x_jitter)
    y_jitter = np.array(y_jitter)

    facecolors = np.array(
        [marker_styles.get(id_, {}).get("facecolor", "white") for id_ in ids]
    )
    edgecolors = np.array(
        [marker_styles.get(id_, {}).get("edgecolor", "black") for id_ in ids]
    )
    markers = np.array(
        [marker_styles.get(id_, {}).get("marker", "^") for id_ in ids]
    )

    up_mask_m = (markers == "^")
    down_mask_m = (markers == "v")

    if up_mask_m.any():
        plt.scatter(
            x_jitter[up_mask_m],
            y_jitter[up_mask_m],
            facecolors=facecolors[up_mask_m],
            edgecolors=edgecolors[up_mask_m],
            marker="^",
            s=180,
            zorder=3,
        )

    if down_mask_m.any():
        plt.scatter(
            x_jitter[down_mask_m],
            y_jitter[down_mask_m],
            facecolors=facecolors[down_mask_m],
            edgecolors=edgecolors[down_mask_m],
            marker="v",
            s=180,
            zorder=3,
        )

    for x_val, y_val, id_ in zip(x_jitter, y_jitter, ids):
        plt.text(x_val, y_val, id_, fontsize=6, ha="center", va="center", zorder=4)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- Q14b: SPLIT TEXT REASONS + TRIBES ----------

def export_q14_reasons(df, orientation_col="q14_lids_orientation", reason_col="q14_why"):
    if orientation_col not in df.columns or reason_col not in df.columns:
        print("Q14 columns not found in dataframe.")
        return

    orient = pd.to_numeric(df[orientation_col], errors="coerce")

    # ----- original global outputs -----
    up_mask = orient == 1
    upwards_reasons = df.loc[up_mask, reason_col].dropna().astype(str).str.strip()
    upwards_reasons = upwards_reasons[upwards_reasons != ""]

    down_mask = orient == 2
    downwards_reasons = df.loc[down_mask, reason_col].dropna().astype(str).str.strip()
    downwards_reasons = downwards_reasons[downwards_reasons != ""]

    print("\n=== Q14b: Reasons for opening UPWARDS (orientation == 1) ===")
    for r in upwards_reasons:
        print("-", r)

    print("\n=== Q14b: Reasons for opening DOWNWARDS (orientation == 2) ===")
    for r in downwards_reasons:
        print("-", r)

    with open("q14_upwards_reasons.txt", "w", encoding="utf-8") as f:
        for r in upwards_reasons:
            f.write(r + "\n")

    with open("q14_downwards_reasons.txt", "w", encoding="utf-8") as f:
        for r in downwards_reasons:
            f.write(r + "\n")

    print("\nQ14b reasons exported to q14_upwards_reasons.txt and q14_downwards_reasons.txt")

    # ----- subgroup helper -----
    def export_subset(name, id_list, orientation_value, filename):
        if id_list is None or len(id_list) == 0:
            return
        id_mask = df[ID_COL].astype(str).isin(id_list)
        mask = id_mask & (orient == orientation_value)
        subset = df.loc[mask, reason_col].dropna().astype(str).str.strip()
        subset = subset[subset != ""]
        print(f"\n=== {name} ===")
        for r in subset:
            print("-", r)
        with open(filename, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(r + "\n")

    # rouges
    export_subset(
        "Q14b UPWARDS reasons – rouge pipetters",
        rouge_ids,
        1,
        "q14_upwards_reasons_rouge.txt",
    )
    export_subset(
        "Q14b DOWNWARDS reasons – rouge pipetters",
        rouge_ids,
        2,
        "q14_downwards_reasons_rouge.txt",
    )

    # braves
    export_subset(
        "Q14b UPWARDS reasons – punch drinkers",
        brave_ids,
        1,
        "q14_upwards_reasons_brave.txt",
    )
    export_subset(
        "Q14b DOWNWARDS reasons – punch drinkers",
        brave_ids,
        2,
        "q14_downwards_reasons_brave.txt",
    )


def plot_bathroom(df, col, title, filename=None):
    option_labels = {
        1: "*",
        2: "**",
        3: "***",
        4: "****",
        5: "*****",
    }

    plot_single_choice(
        df=df,
        col=col,
        title=title,
        option_labels=option_labels,
        filename=filename,
        as_percent=False,
    )


# ---------- RUN ANALYSES: PLOTS ----------

# Q1: coffees / teas (numeric)
plot_numeric_with_normal(
    df,
    col="q01_coffee",
    xlabel="Coffees / teas in the morning",
    title="Distribution of Morning Drinks (Q1)",
    bin_width=0.5,
    filename="q01_coffee.png",
)

# Q12: browser tabs (numeric)
plot_numeric_with_normal(
    df,
    col="q12_tabs_open",
    xlabel="Browser tabs open on average",
    title="Distribution of Open Tabs (Q12)",
    bin_width=1,
    filename="q12_tabs.png",
)

# Q2–Q6: agreement scale (Likert)
plot_likert(
    df,
    col="q02_whisper",
    title='I whisper "please work" to my experiment (Q2)',
    filename="q02_whisper.png",
)

plot_likert(
    df,
    col="q03_pipette_moral_crime",
    title="Taking a pipette tip from the middle is a moral crime (Q3)",
    filename="q03_pipette.png",
)

plot_likert(
    df,
    col="q04_suspicious_first_try",
    title="I get suspicious when an experiment works perfectly the first time (Q4)",
    filename="q04_suspicious.png",
)

plot_likert(
    df,
    col="q05_bottom_50_readers",
    title="I suspect I'm in the bottom 50% of paper readers (Q5)",
    filename="q05_readers.png",
)

plot_likert(
    df,
    col="q06_enjoy_labeling",
    title="I secretly enjoy labeling large numbers of aliquots (Q6)",
    filename="q06_labeling.png",
)

# SINGLE CHOICE
plot_single_choice(
    df,
    col="q07_crash_response",
    title="Your computer crashes mid-analysis (Q7)",
    option_labels={
        1: "Panic & pray",
        2: "Stare 30 s",
        3: "Reboot, lose faith",
        4: "Call “Uweeee!”",
    },
    filename="q07_crash.png",
)

plot_single_choice(
    df,
    col="q09_punch_water_bath",
    title="Would you drink punch heated in the water bath? (Q9)",
    option_labels={
        1: "Yes",
        2: "No",
    },
    filename="q09_punch.png",
)

plot_single_choice(
    df,
    col="q13_pretend_understand",
    title="Do you pretend to understand colleagues' methods? (Q13)",
    option_labels={
        1: "Yes",
        2: "No",
        3: "Always",
    },
    filename="q13_pretend.png",
)

plot_single_choice(
    df,
    col="q14_lids_orientation",
    title="Tube lids orientation (Q14a)",
    option_labels={
        1: "Opening upwards",
        2: "Opening downwards",
    },
    filename="q14_lids.png",
)

plot_single_choice(
    df,
    col="q16_reindeer_ethics",
    title="Would Santa's reindeer pass animal ethics approval? (Q16)",
    option_labels={
        1: "Yes",
        2: "No",
    },
    filename="q16_reindeer.png",
)

plot_single_choice(
    df,
    col="q18_true_center",
    title="Where is the true center of the institute? (Q18)",
    option_labels={
        1: "Lunch room",
        2: "Christian's office",
        3: "Cell culture lab",
        4: "Animal housing",
    },
    filename="q18_center.png",
)

plot_single_choice(
    df,
    col="q20_snacks_at_desk",
    title="Do you keep snacks at your desk? (Q20)",
    option_labels={
        1: "Yes",
        2: "No",
    },
    filename="q20_snacks.png",
)

plot_single_choice(
    df,
    col="q21_quick_meeting_expect",
    title='When your PI says "quick meeting"... (Q21)',
    option_labels={
        1: "5 minutes",
        2: "20 minutes",
        3: "1 hour",
        4: "Career\nreeevaluation",
    },
    filename="q21_meeting.png",
)

# Q14b text split (global + tribes)
export_q14_reasons(df, orientation_col="q14_lids_orientation", reason_col="q14_why")


# ---------- Q17: Bathroom star ratings (EXPORT TO TEXT + TRIBES) ----------

def export_bathroom_ratings(df, id_list=None, suffix=""):
    """
    If id_list is provided, restrict to those IDs.
    Suffix is added to the output file name.
    """
    bathroom_cols = [
        "q17_bathroom_back_3rd",
        "q17_bathroom_front_3rd",
        "q17_bathroom_old_2nd",
        "q17_bathroom_new_2nd",
        "q17_bathroom_1st",
    ]

    if id_list is not None:
        df_sub = df[df[ID_COL].astype(str).isin(id_list)].copy()
        label = suffix.strip("_")
        header_suffix = f" ({label})" if label else ""
    else:
        df_sub = df
        header_suffix = ""

    lines = []
    print(f"\n=== Bathroom Ratings Summary (Q17){header_suffix} ===")
    print("Bathroom, Average rating, Number of reviews")

    for col in bathroom_cols:
        if col not in df_sub.columns:
            print(f"{col}, NOT FOUND")
            lines.append(f"{col}, NOT FOUND\n")
            continue

        vals = pd.to_numeric(df_sub[col], errors="coerce").dropna()

        if vals.empty:
            print(f"{col}, no ratings, 0")
            lines.append(f"{col}, no ratings, 0\n")
            continue

        avg = vals.mean()
        count = len(vals)

        print(f"{col}, {avg:.2f}, {count}")
        lines.append(f"{col}, {avg:.2f}, {count}\n")

    outname = f"bathroom_ratings_summary{suffix}.txt"
    with open(outname, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)

    print(f"\nBathroom ratings exported to {outname}")


# all participants (original)
export_bathroom_ratings(df)

# rouges only
export_bathroom_ratings(df, id_list=list(rouge_ids), suffix="_rouge")

# punch drinkers only
export_bathroom_ratings(df, id_list=list(brave_ids), suffix="_brave")

# lids upwards only
export_bathroom_ratings(df, id_list=list(lid_up_ids), suffix="_lid_up")

# lids downwards only
export_bathroom_ratings(df, id_list=list(lid_down_ids), suffix="_lid_down")

# ---------- Q17 Bathrooms: plotted as single-choice 1–5 ----------

plot_bathroom(
    df,
    col="q17_bathroom_back_3rd",
    title="Bathroom rating: Back building, 3rd floor (Q17)",
    filename="q17_bathroom_back_3rd.png",
)

plot_bathroom(
    df,
    col="q17_bathroom_front_3rd",
    title="Bathroom rating: Front building, 3rd floor (Q17)",
    filename="q17_bathroom_front_3rd.png",
)

plot_bathroom(
    df,
    col="q17_bathroom_old_2nd",
    title="Bathroom rating: Old building, 2nd floor (Q17)",
    filename="q17_bathroom_old_2nd.png",
)

plot_bathroom(
    df,
    col="q17_bathroom_new_2nd",
    title="Bathroom rating: New building, 2nd floor (Q17)",
    filename="q17_bathroom_new_2nd.png",
)

plot_bathroom(
    df,
    col="q17_bathroom_1st",
    title="Bathroom rating: 1st floor (Q17)",
    filename="q17_bathroom_1st.png",
)
