import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from difflib import get_close_matches
import matplotlib.pyplot as plt

st.title("💰 Smart Expense Tracker (AI Based)")

# ---------- TRAINING DATA ----------
train_data = pd.DataFrame({
    "Name": [
        # FOOD
        "pizza","burger","sandwich","food","dinner","lunch","breakfast",
        "rice","roti","dal","paneer","biryani","noodles","coffee","tea","milk","cake",

        # TRAVEL
        "bus","bus ticket","uber","auto","train","travel","metro","cab","flight","petrol","diesel","bike fuel","parking",

        # SHOPPING
        "shirt","jeans","tshirt","shopping","mall","clothes","shoes","watch","bag","mobile","laptop","headphones",

        # EDUCATION
        "book","notebook","pen","pencil","college fee","exam fee","tuition","course","stationery",

        # HEALTH
        "medicine","doctor","hospital","tablet","checkup","health insurance","vitamins",

        # ENTERTAINMENT
        "movie","cinema","netflix","game","concert","music","subscription","party"
    ],
    "Category": [
        # FOOD
        "food","food","food","food","food","food","food",
        "food","food","food","food","food","food","food","food","food","food",

        # TRAVEL
        "travel","travel","travel","travel","travel","travel","travel","travel","travel","travel","travel","travel","travel",

        # SHOPPING
        "shopping","shopping","shopping","shopping","shopping","shopping","shopping","shopping","shopping","shopping","shopping","shopping",

        # EDUCATION
        "education","education","education","education","education","education","education","education","education",

        # HEALTH
        "health","health","health","health","health","health","health",

        # ENTERTAINMENT
        "entertainment","entertainment","entertainment","entertainment","entertainment","entertainment","entertainment","entertainment"
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

        # CASE 1: correct spelling
        if word in vocab:
            pred = model.predict(vectorizer.transform([word]))[0]
            st.session_state.data.append([name, amount, pred])
            st.success(f"Predicted Category: {pred}")

        # CASE 2: spelling mistake
        else:
            close = get_close_matches(word, vocab, n=1, cutoff=0.7)

            if close:
                st.warning(f"⚠ Spelling mistake! Did you mean '{close[0]}' ?")
                st.info("Expense NOT added. Please correct spelling.")

            # CASE 3: unknown word
            else:
                st.error("❌ Unknown expense name! Expense NOT added.")
    else:
        st.error("Please enter expense name")

# ---------- SHOW DATA ----------
if st.session_state.data:
    df = pd.DataFrame(st.session_state.data, columns=["Name", "Amount", "Category"])

    st.subheader("All Expenses")
    st.dataframe(df, use_container_width=True)

    # ---------- GRAPH (FIXED SIZE) ----------
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
