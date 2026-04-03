import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# ---------- ML TRAINING ----------
train_data = pd.DataFrame({
    "Name": ["pizza", "burger", "bus ticket", "uber", "shirt", "jeans"],
    "Category": ["food", "food", "travel", "travel", "shopping", "shopping"]
})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_data["Name"])

model = MultinomialNB()
model.fit(X, train_data["Category"])

# ---------- DATA STORAGE ----------
data = []

# ---------- FUNCTIONS ----------
def add_expense():
    name = entry_name.get()
    amount = entry_amount.get()

    if name == "" or amount == "":
        messagebox.showerror("Error", "Please enter all fields")
        return

    amount = float(amount)

    pred = model.predict(vectorizer.transform([name]))[0]

    data.append([name, amount, pred])

    messagebox.showinfo("Added", f"Category: {pred}")

    entry_name.delete(0, tk.END)
    entry_amount.delete(0, tk.END)


def show_graph():
    if not data:
        messagebox.showerror("Error", "No data to show")
        return

    df = pd.DataFrame(data, columns=["Name", "Amount", "Category"])
    summary = df.groupby("Category")["Amount"].sum()

    summary.plot(kind="bar")
    plt.title("Expense by Category")
    plt.ylabel("Amount")
    plt.show()


def save_csv():
    if not data:
        messagebox.showerror("Error", "No data to save")
        return

    df = pd.DataFrame(data, columns=["Name", "Amount", "Category"])
    df.to_csv("expenses.csv", index=False)

    messagebox.showinfo("Saved", "Data saved to expenses.csv")


# ---------- GUI WINDOW ----------
root = tk.Tk()
root.title("Smart Expense Tracker")
root.geometry("300x250")

tk.Label(root, text="Expense Name").pack()
entry_name = tk.Entry(root)
entry_name.pack()

tk.Label(root, text="Amount").pack()
entry_amount = tk.Entry(root)
entry_amount.pack()

tk.Button(root, text="Add Expense", command=add_expense).pack(pady=5)
tk.Button(root, text="Show Graph", command=show_graph).pack(pady=5)
tk.Button(root, text="Save to CSV", command=save_csv).pack(pady=5)

root.mainloop()
