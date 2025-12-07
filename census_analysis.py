import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm   # for normal pdf

plt.xkcd()

# Load data
df = pd.read_csv("christams_data.csv", sep=";")


# ---------- TYPE 1: NUMERIC + NORMAL FIT ----------

def plot_numeric_with_normal(df, col, xlabel, title, bin_width=0.5, filename=None):
    values = df[col].dropna().astype(float)
    if values.empty:
        print(f"No data for {col}")
        return

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

    plt.xticks(bins)
    plt.xlabel(xlabel)
    plt.ylabel("Number of people")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- TYPE 2: LIKERT (AGREEMENT SCALE) ----------

LIKERT_LABELS = {
    1: "Strongly\ndisagree",
    2: "Disagree",
    3: "Indifferent",
    4: "Agree",
    5: "Strongly\nagree",
}

def plot_likert(df, col, title, filename=None, as_percent=True):
    values = df[col].dropna().astype(int)
    if values.empty:
        print(f"No data for {col}")
        return

    # Count responses per category 1–5
    counts = values.value_counts().sort_index()
    index = pd.Index([1, 2, 3, 4, 5])
    counts = counts.reindex(index, fill_value=0)

    if as_percent:
        counts = counts / counts.sum() * 100
        ylabel = "Percentage of people"
    else:
        ylabel = "Number of people"

    x = np.arange(1, 6)

    plt.figure(figsize=(7, 4))
    plt.bar(x, counts.values, edgecolor="black", alpha=0.8)

    labels = [LIKERT_LABELS[i] for i in x]
    plt.xticks(x, labels)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- TYPE 3: SINGLE CHOICE (1..N coded) ----------

def plot_single_choice(df, col, title, option_labels, filename=None, as_percent=True):
    """
    df: dataframe
    col: column name in df (values encoded as 1..N integers)
    title: plot title
    option_labels: dict mapping int -> label
                   e.g. {1: "Panic & pray", 2: "Stare 30 s", ...}
                   keys define the category range and order.
    """
    values = df[col].dropna().astype(int)
    if values.empty:
        print(f"No data for {col}")
        return

    # Categories are 1..max_key, even if some weren't chosen
    categories = sorted(option_labels.keys())
    max_cat = max(categories)
    full_range = list(range(1, max_cat + 1))

    # Count occurrences and ensure all categories exist
    counts = values.value_counts()
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

    # Use labels from option_labels; if something is missing, fall back to the number
    tick_labels = [option_labels.get(cat, str(cat)) for cat in full_range]
    plt.xticks(x, tick_labels)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- Q14b: SPLIT TEXT REASONS BY ORIENTATION ----------

def export_q14_reasons(df, orientation_col="q14_lids_orientation", reason_col="q14_why"):
    """
    Splits Q14 'Why?' answers into two lists:
    - orientation == 1 -> upwards reasons
    - orientation == 2 -> downwards reasons

    Prints them to the console and also writes two text files:
    - q14_upwards_reasons.txt
    - q14_downwards_reasons.txt
    """
    if orientation_col not in df.columns or reason_col not in df.columns:
        print("Q14 columns not found in dataframe.")
        return

    # Make sure orientation is numeric
    orient = pd.to_numeric(df[orientation_col], errors="coerce")

    # Upwards = 1
    up_mask = orient == 1
    upwards_reasons = df.loc[up_mask, reason_col].dropna().astype(str).str.strip()
    upwards_reasons = upwards_reasons[upwards_reasons != ""]

    # Downwards = 2
    down_mask = orient == 2
    downwards_reasons = df.loc[down_mask, reason_col].dropna().astype(str).str.strip()
    downwards_reasons = downwards_reasons[downwards_reasons != ""]

    # Print to console for easy copy-paste
    print("\n=== Q14b: Reasons for opening UPWARDS (orientation == 1) ===")
    for r in upwards_reasons:
        print("-", r)

    print("\n=== Q14b: Reasons for opening DOWNWARDS (orientation == 2) ===")
    for r in downwards_reasons:
        print("-", r)

    # Also write to text files (optional but handy)
    with open("q14_upwards_reasons.txt", "w", encoding="utf-8") as f:
        for r in upwards_reasons:
            f.write(r + "\n")

    with open("q14_downwards_reasons.txt", "w", encoding="utf-8") as f:
        for r in downwards_reasons:
            f.write(r + "\n")

    print("\nQ14b reasons exported to q14_upwards_reasons.txt and q14_downwards_reasons.txt")


# ---------- RUN ANALYSES ----------

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

# ---------- SINGLE CHOICE (1..N) ----------

# Q7: crash response (1–4)
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

# Q9: Christmas punch (1–2)
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

# Q13: pretend to understand (1–3)
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

# Q14a: tube lids orientation (1–2)
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

# Q16: reindeer ethics (1–2)
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

# Q18: true center of the institute (1–4)
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

# Q20: snacks at desk (1–2)
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

# Q21: “quick meeting” expectations (1–4)
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

# ---------- Q14b: export reasons lists ----------

export_q14_reasons(df, orientation_col="q14_lids_orientation", reason_col="q14_why")

# ---------- Q17: Bathroom star ratings (EXPORT TO TEXT) ----------

def export_bathroom_ratings(df):

    bathroom_cols = [
        "q17_bathroom_back_3rd",
        "q17_bathroom_front_3rd",
        "q17_bathroom_old_2nd",
        "q17_bathroom_new_2nd",
        "q17_bathroom_1st",
    ]

    lines = []
    print("\n=== Bathroom Ratings Summary (Q17) ===")
    print("Bathroom, Average rating, Number of reviews")

    for col in bathroom_cols:
        if col not in df.columns:
            print(f"{col}, NOT FOUND")
            lines.append(f"{col}, NOT FOUND\n")
            continue

        vals = pd.to_numeric(df[col], errors="coerce").dropna()

        if vals.empty:
            print(f"{col}, no ratings, 0")
            lines.append(f"{col}, no ratings, 0\n")
            continue

        avg = vals.mean()
        count = len(vals)

        print(f"{col}, {avg:.2f}, {count}")
        lines.append(f"{col}, {avg:.2f}, {count}\n")

    # Write to file
    with open("bathroom_ratings_summary.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)

    print("\nBathroom ratings exported to bathroom_ratings_summary.txt")

export_bathroom_ratings(df)