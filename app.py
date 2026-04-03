import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from difflib import get_close_matches
import matplotlib.pyplot as plt

st.title("💰 Smart Expense Tracker (AI Based)")

# ---------- TRAINING DATA ----------
data = [
    # -------- FOOD --------
    ("pizza","food"),("burger","food"),("sandwich","food"),("food","food"),
    ("dinner","food"),("lunch","food"),("breakfast","food"),
    ("rice","food"),("roti","food"),("dal","food"),("paneer","food"),
    ("biryani","food"),("noodles","food"),("coffee","food"),("tea","food"),
    ("milk","food"),("cake","food"),("juice","food"),("snacks","food"),
    ("chips","food"),("ice cream","food"),("fruits","food"),("vegetables","food"),
    ("restaurant","food"),("cafe","food"),("tiffin","food"),
    ("swiggy","food"),("zomato","food"),("dominos","food"),
    ("kfc","food"),("mcdonalds","food"),("bakery","food"),("dessert","food"),

    # -------- TRAVEL --------
    ("bus","travel"),("bus ticket","travel"),("uber","travel"),("auto","travel"),
    ("train","travel"),("travel","travel"),("metro","travel"),("cab","travel"),
    ("flight","travel"),("petrol","travel"),("diesel","travel"),("bike fuel","travel"),
    ("parking","travel"),("toll","travel"),("ola","travel"),("rapido","travel"),
    ("taxi","travel"),("fuel","travel"),("airport","travel"),
    ("railway ticket","travel"),("uber ride","travel"),("ola ride","travel"),
    ("metro card","travel"),("fastag","travel"),

    # -------- SHOPPING --------
    ("shirt","shopping"),("jeans","shopping"),("tshirt","shopping"),
    ("shopping","shopping"),("mall","shopping"),("clothes","shopping"),
    ("shoes","shopping"),("watch","shopping"),("bag","shopping"),
    ("mobile","shopping"),("laptop","shopping"),("headphones","shopping"),
    ("keyboard","shopping"),("mouse","shopping"),("charger","shopping"),
    ("electronics","shopping"),("amazon","shopping"),("flipkart","shopping"),
    ("online shopping","shopping"),("grocery","shopping"),
    ("supermarket","shopping"),("dmart","shopping"),("reliance mart","shopping"),

    # -------- EDUCATION --------
    ("book","education"),("notebook","education"),("pen","education"),
    ("pencil","education"),("college fee","education"),("exam fee","education"),
    ("tuition","education"),("course","education"),("stationery","education"),
    ("online course","education"),("certification","education"),
    ("training","education"),("classes","education"),
    ("udemy","education"),("coursera","education"),
    ("byju","education"),("unacademy","education"),("skillshare","education"),

    # -------- HEALTH --------
    ("medicine","health"),("doctor","health"),("hospital","health"),
    ("tablet","health"),("checkup","health"),("health insurance","health"),
    ("vitamins","health"),("pharmacy","health"),("clinic","health"),
    ("syrup","health"),("lab test","health"),("scan","health"),
    ("xray","health"),("blood test","health"),
    ("apollo pharmacy","health"),("medical store","health"),

    # -------- ENTERTAINMENT --------
    ("movie","entertainment"),("cinema","entertainment"),
    ("netflix","entertainment"),("game","entertainment"),
    ("concert","entertainment"),("music","entertainment"),
    ("subscription","entertainment"),("party","entertainment"),
    ("ott","entertainment"),("amazon prime","entertainment"),
    ("hotstar","entertainment"),("youtube","entertainment"),
    ("spotify","entertainment"),("pubg","entertainment"),
    ("ipl ticket","entertainment"),

    # -------- BILLS --------
    ("electricity bill","bills"),("water bill","bills"),
    ("internet bill","bills"),("wifi","bills"),("broadband","bills"),
    ("recharge","bills"),("mobile recharge","bills"),
    ("gas bill","bills"),("rent","bills"),("maintenance","bills"),
    ("postpaid bill","bills"),("prepaid recharge","bills"),
    ("dish tv","bills"),("dth recharge","bills"),

    # -------- PERSONAL --------
    ("salon","personal"),("haircut","personal"),("spa","personal"),
    ("gym","personal"),("fitness","personal"),("cosmetics","personal"),
    ("makeup","personal"),("perfume","personal"),
    ("parlor","personal"),("skin care","personal"),
    ("self care","personal"),("grooming","personal"),

    # -------- FINANCE --------
    ("emi","finance"),("loan","finance"),("credit card bill","finance"),
    ("debit","finance"),("insurance premium","finance"),
    ("investment","finance"),("mutual fund","finance"),
    ("sip","finance"),("tax","finance"),("bank charges","finance"),

    # -------- MISC --------
    ("gift","misc"),("donation","misc"),("charity","misc"),
    ("repair","misc"),("service","misc"),("fine","misc"),
    ("misc","misc"),("other","misc"),("random","misc"),
    ("lost","misc"),("unexpected","misc"),("emergency","misc"),
    ("fees","misc"),("charges","misc")
]

train_data = pd.DataFrame(data, columns=["Name","Category"])

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
