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

# ---------- TYPE 3: SINGLE CHOICE ----------

def plot_single_choice(df, col, title, option_labels, filename=None, as_percent=True):
    """
    df: dataframe
    col: column name in df
    title: plot title
    option_labels: dict mapping codes -> labels
                   order of keys = order on x-axis!
                   e.g. {"A": "Panic & pray", "B": "Stare 30s", ...}
    """
    values = df[col].dropna().astype(str)
    if values.empty:
        print(f"No data for {col}")
        return

    # Count occurrences
    counts = values.value_counts()

    # Respect the order in option_labels
    expected_opts = list(option_labels.keys())
    counts = counts.reindex(expected_opts, fill_value=0)

    if as_percent:
        counts_plot = counts / counts.sum() * 100
        ylabel = "Percentage of people"
    else:
        counts_plot = counts
        ylabel = "Number of people"

    x = np.arange(len(expected_opts))

    plt.figure(figsize=(7, 4))
    plt.bar(x, counts_plot.values, edgecolor="black", alpha=0.8)

    tick_labels = [option_labels[o] for o in expected_opts]
    plt.xticks(x, tick_labels)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150)

    plt.show()


# ---------- RUN ANALYSES ----------

# Q1: coffees / teas
plot_numeric_with_normal(
    df,
    col="q01_coffee",
    xlabel="Coffees / teas in the morning",
    title="Distribution of Morning Drinks (Q1)",
    bin_width=0.5,
    filename="q01_coffee.png",
)

# Q12: browser tabs
plot_numeric_with_normal(
    df,
    col="q12_tabs_open",
    xlabel="Browser tabs open on average",
    title="Distribution of Open Tabs (Q12)",
    bin_width=1,
    filename="q12_tabs.png",
)

# Q2–Q6: agreement scale
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

plot_single_choice(
    df,
    col="q07_crash_response",
    title="Your computer crashes mid-analysis (Q7)",
    option_labels={
        "A": "Panic & pray",
        "B": "Stare 30 s",
        "C": "Reboot, lose faith",
        "D": "Call “Uweeee!”",
    },
    filename="q07_crash.png",
)

plot_single_choice(
    df,
    col="q09_punch_water_bath",
    title="Would you drink punch heated in the water bath? (Q9)",
    option_labels={
        "yes": "Yes",
        "no": "No",
    },
    filename="q09_punch.png",
)

plot_single_choice(
    df,
    col="q13_pretend_understand",
    title="Do you pretend to understand colleagues' methods? (Q13)",
    option_labels={
        "yes": "Yes",
        "no": "No",
        "always": "Always",
    },
    filename="q13_pretend.png",
)

plot_single_choice(
    df,
    col="q16_reindeer_ethics",
    title="Would Santa's reindeer pass animal ethics approval? (Q16)",
    option_labels={
        "yes": "Yes",
        "no": "No",
    },
    filename="q16_reindeer.png",
)

plot_single_choice(
    df,
    col="q18_true_center",
    title="Where is the true center of the institute? (Q18)",
    option_labels={
        "A": "Lunch room",
        "B": "Christian's office",
        "C": "Cell culture lab",
        "D": "Animal housing",
    },
    filename="q18_center.png",
)

plot_single_choice(
    df,
    col="q20_snacks_at_desk",
    title="Do you keep snacks at your desk? (Q20)",
    option_labels={
        "yes": "Yes",
        "no": "No",
    },
    filename="q20_snacks.png",
)

plot_single_choice(
    df,
    col="q21_quick_meeting_expect",
    title='When your PI says "quick meeting"... (Q21)',
    option_labels={
        "A": "5 minutes",
        "B": "20 minutes",
        "C": "1 hour",
        "D": "Career\nreevaluation",
    },
    filename="q21_meeting.png",
)
