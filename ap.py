import time
import pandas as pd
import streamlit as st
import anthropic
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="Prompt Evaluator", layout="wide")
st.title("Prompt Evaluation and Refinement Tool")
st.caption("Compare prompt versions on a labeled dataset using accuracy, F1, "
           "format compliance, and latency.")

# ---------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------
def get_secret_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

with st.sidebar:
    st.header("Settings")
    api_key = get_secret_key() or st.text_input("Anthropic API key", type="password")
    model = st.text_input("Model name", "claude-haiku-4-5")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    labels_text = st.text_input("Valid labels (comma separated)",
                                "Billing, Technical, Other")
    max_rows = st.number_input("Max rows to evaluate", 5, 200, 50)

VALID_LABELS = [l.strip() for l in labels_text.split(",") if l.strip()]

# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
DEFAULT_DATA = pd.DataFrame([
    ("I was charged twice this month", "Billing"),
    ("My invoice shows the wrong amount", "Billing"),
    ("How much does the premium plan cost?", "Billing"),
    ("I want a refund for last month", "Billing"),
    ("The app crashes when I upload a file", "Technical"),
    ("I cannot log in to my account", "Technical"),
    ("The page loads very slowly", "Technical"),
    ("I get error 500 when saving", "Technical"),
    ("I'd like a refund because the app crashes", "Billing"),
    ("Do you have an office in Berlin?", "Other"),
    ("Can I give feedback about your website design?", "Other"),
    ("Thanks for the great service!", "Other"),
], columns=["message", "label"])

st.subheader("1. Dataset")
file = st.file_uploader("Upload a CSV with columns: message, label (optional)",
                        type="csv")
if file:
    data = pd.read_csv(file)
    if not {"message", "label"}.issubset(data.columns):
        st.error("CSV must contain the columns 'message' and 'label'.")
        st.stop()
else:
    data = DEFAULT_DATA
    st.info("Using the built-in sample dataset.")
    st.download_button("Download sample CSV",
                       DEFAULT_DATA.to_csv(index=False), "sample.csv")

data = data.dropna(subset=["message", "label"]).head(int(max_rows)).reset_index(drop=True)
st.dataframe(data, height=200, use_container_width=True)

# ---------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------
st.subheader("2. Prompt versions")
st.write("Use `{message}` where the input text should go. Leave a box empty to skip it.")

DEFAULT_PROMPTS = [
    "Classify this message: {message}",
    ("You are a support-ticket classifier. Classify the message into exactly one "
     "category: Billing, Technical, or Other. Reply with only the category name.\n"
     "Message: {message}"),
    ("You are a support-ticket classifier. Classify the message into exactly one "
     "category: Billing, Technical, or Other. Reply with only the category name.\n"
     "Billing = payments, invoices, refunds, pricing. "
     "Technical = errors, bugs, performance, login problems. "
     "Other = everything else. If a message involves both, choose the category "
     "of the customer's main request.\n"
     "Examples:\n"
     "'I was charged twice' -> Billing\n"
     "'The app crashes when I upload a file' -> Technical\n"
     "'I'd like a refund because the app crashes' -> Billing\n"
     "Message: {message}"),
]

cols = st.columns(3)
prompts = {}
for i, col in enumerate(cols):
    name = f"Version {i + 1}"
    prompts[name] = col.text_area(name, DEFAULT_PROMPTS[i], height=300, key=name)

# ---------------------------------------------------------------
# Model call
# ---------------------------------------------------------------
def call_model(client, template, message):
    prompt = template.replace("{message}", str(message))
    start = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=100,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
    except anthropic.NotFoundError:
        st.error(f"Model '{model}' was not found. Change the model name in the sidebar.")
        st.stop()
    except anthropic.AuthenticationError:
        st.error("Invalid API key.")
        st.stop()
    except Exception as e:
        text = f"ERROR: {e}"
    return text, time.time() - start

# ---------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------
st.subheader("3. Evaluate")

if st.button("Run evaluation", type="primary"):
    if not api_key:
        st.error("Enter your API key in the sidebar.")
        st.stop()
    if not VALID_LABELS:
        st.error("Enter at least one valid label.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    active = {k: v for k, v in prompts.items() if v.strip()}
    summary, detail_tables = [], {}

    progress = st.progress(0.0)
    total_steps = len(active) * len(data)
    step = 0

    for name, template in active.items():
        outputs, latencies = [], []
        for msg in data["message"]:
            out, lat = call_model(client, template, msg)
            outputs.append(out)
            latencies.append(lat)
            step += 1
            progress.progress(step / total_steps)

        df = data.copy()
        df["output"] = outputs
        df["format_ok"] = df["output"].isin(VALID_LABELS)
        df["pred"] = df["output"].where(df["format_ok"], "INVALID")
        detail_tables[name] = df

        summary.append({
            "version": name,
            "accuracy": accuracy_score(df["label"], df["pred"]),
            "macro_f1": f1_score(df["label"], df["pred"], average="macro",
                                 labels=VALID_LABELS, zero_division=0),
            "format_compliance": df["format_ok"].mean(),
            "avg_latency_s": sum(latencies) / len(latencies),
        })

    progress.empty()
    summary_df = pd.DataFrame(summary)

    st.subheader("Results")
    st.dataframe(summary_df.round(3), use_container_width=True)
    st.bar_chart(summary_df.set_index("version")[["accuracy", "format_compliance"]])

    best = summary_df.sort_values(["accuracy", "format_compliance"],
                                  ascending=False).iloc[0]["version"]
    st.success(f"Best version on this dataset: {best}")

    st.download_button("Download prompt log (CSV)",
                       summary_df.to_csv(index=False), "prompt_log.csv")

    st.subheader("Error analysis")
    for name, df in detail_tables.items():
        with st.expander(f"{name}: details"):
            errors = df[df["label"] != df["pred"]]
            st.write(f"Misclassified: {len(errors)} of {len(df)}")
            if len(errors):
                st.dataframe(errors[["message", "label", "output"]],
                             use_container_width=True)
            cm_labels = VALID_LABELS + ["INVALID"]
            cm = confusion_matrix(df["label"], df["pred"], labels=cm_labels)
            st.write("Confusion matrix (rows = true, columns = predicted)")
            st.dataframe(pd.DataFrame(cm,
                                      index=["true_" + l for l in cm_labels],
                                      columns=["pred_" + l for l in cm_labels]))