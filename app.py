import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from difflib import get_close_matches
import matplotlib.pyplot as plt

st.title("💰 Intelligent Expense Tracker")

# ---------- TRAINING DATA ----------
train_data = pd.DataFrame({
    "Name": [
        "pizza","burger","sandwich","food","dinner","lunch","breakfast",
        "bus","uber","auto","train","travel","metro","cab","flight","petrol",
        "shirt","jeans","tshirt","shopping","mall","clothes","shoes",
        "book","pen","notebook","college fee","course",
        "medicine","doctor","hospital","clinic","tablet","checkup","syrup","injection",
        "movie","netflix","game","music","party"
    ],
    "Category": [
        "food","food","food","food","food","food","food",
        "travel","travel","travel","travel","travel","travel","travel","travel","travel",
        "shopping","shopping","shopping","shopping","shopping","shopping","shopping",
        "education","education","education","education","education",
        "health","health","health","health","health","health","health","health",
        "entertainment","entertainment","entertainment","entertainment","entertainment"
    ]
})

vocab = train_data["Name"].tolist()

# ---------- ML MODEL ----------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(train_data["Name"])

model = MultinomialNB()
model.fit(X, train_data["Category"])

# ---------- SESSION STORAGE ----------
if "data" not in st.session_state:
    st.session_state.data = []

# ---------- INPUT ----------
name = st.text_input("Enter expense name")
amount = st.number_input("Enter amount", min_value=0.0)

# ---------- ADD EXPENSE ----------
if st.button("Add Expense"):
    if name:
        word = name.lower()

        if word in vocab:
            pred = model.predict(vectorizer.transform([word]))[0]
            st.session_state.data.append([name, amount, pred])
            st.success(f"Predicted Category: {pred}")
        else:
            close = get_close_matches(word, vocab, n=1, cutoff=0.7)

            if close:
                st.warning(f"⚠ Spelling mistake! Did you mean '{close[0]}' ?")
                st.info("Expense NOT added. Please correct spelling.")
            else:
                st.error("❌ Unknown expense name! Expense NOT added.")
    else:
        st.error("Please enter expense name")

# ---------- SHOW DATA ----------
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data, columns=["Name", "Amount", "Category"])

    st.subheader("All Expenses")
    st.dataframe(df, use_container_width=True)

    # ---------- DELETE FEATURE ----------
    st.subheader("🗑 Delete Expense")

    delete_index = st.number_input(
        "Enter row number to delete (start from 0)",
        min_value=0,
        max_value=len(df)-1,
        step=1
    )

    if st.button("Delete Selected Expense"):
        st.session_state.data.pop(delete_index)
        st.success("Expense deleted successfully!")
        st.rerun()

    # ---------- GRAPH ----------
    summary = df.groupby("Category")["Amount"].sum()

    fig = plt.figure(figsize=(6, 4))
    plt.bar(summary.index, summary.values)
    plt.title("Expense by Category")
    plt.xlabel("Category")
    plt.ylabel("Amount")
    plt.tight_layout()

    st.pyplot(fig)

    # ---------- DOWNLOAD CSV ----------
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "expenses.csv", "text/csv")


