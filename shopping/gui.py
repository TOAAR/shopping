import tkinter as tk
from tkinter import messagebox
from model import ShoppingPredictorModel

class ShoppingPredictorApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Online Shopping Purchase Predictor")
        self.window.geometry("500x600")

        self.model = ShoppingPredictorModel()
        self.model.train_model('data.csv')

        # Input fields
        self.create_input_fields()

        # Predict button
        self.predict_button = tk.Button(self.window, text="Predict", command=self.predict)
        self.predict_button.pack(pady=20)

    def create_input_fields(self):
        self.entries = {}

        fields = [
            "Administrative Pages Visited", "Informational Pages Visited",
            "Product-Related Pages Visited", "Bounce Rate", "Exit Rate",
            "Page Values", "Special Day Closeness", "Month",
            "Visitor Type", "Weekend"
        ]

        for field in fields:
            label = tk.Label(self.window, text=field)
            label.pack()
            entry = tk.Entry(self.window)
            entry.pack(pady=5)
            self.entries[field] = entry

    def predict(self):
        try:
            input_data = {
                "Administrative": int(self.entries["Administrative Pages Visited"].get()),
                "Informational": int(self.entries["Informational Pages Visited"].get()),
                "ProductRelated": int(self.entries["Product-Related Pages Visited"].get()),
                "BounceRates": float(self.entries["Bounce Rate"].get()),
                "ExitRates": float(self.entries["Exit Rate"].get()),
                "PageValues": float(self.entries["Page Values"].get()),
                "SpecialDay": float(self.entries["Special Day Closeness"].get()),
                "Month": self.entries["Month"].get(),
                "VisitorType": self.entries["Visitor Type"].get(),
                "Weekend": int(self.entries["Weekend"].get())
            }

            # Get the prediction
            prediction = self.model.predict(input_data)
            messagebox.showinfo("Prediction Result", f"The customer {prediction}")

        except Exception as e:
            messagebox.showerror("Error", f"Invalid input. {str(e)}")

    def run(self):
        self.window.mainloop()
