import tkinter as tk
from tkinter import ttk
from Bert import summarize_with_bert
from T5 import summarize_with_t5
from Bart import summarize_with_bart

class TextSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Metin Özetleyici")

        # URL girişi için etiket ve giriş kutusu
        self.label_url = ttk.Label(self.root, text="İnternet Sitesi URL'si:")
        self.label_url.pack()
        self.entry_url = ttk.Entry(self.root, width=50)
        self.entry_url.pack(padx=10, pady=5)

        # Dil modeli seçimi için etiket ve seçim kutusu
        self.label_model = ttk.Label(self.root, text="Dil Modeli Seçimi:")
        self.label_model.pack()
        self.model_var = tk.StringVar()
        self.model_var.set("BERT")  # Varsayılan olarak BERT seçili
        self.model_combobox = ttk.Combobox(self.root, textvariable=self.model_var, values=["BERT", "T5", "BART"])
        self.model_combobox.pack(padx=10, pady=5)

        # Özetleme butonu
        self.button_summarize = ttk.Button(self.root, text="Metni Özetle", command=self.summarize_text)
        self.button_summarize.pack(padx=10, pady=10)

        # Özetlenmiş metin için etiket
        self.label_summary = ttk.Label(self.root, text="Özetlenmiş Metin:")
        self.label_summary.pack()

        # Metin kutusu ve scroll bar
        self.text_summary = tk.Text(self.root, width=80, height=20, font=("Times New Roman", 11))
        self.text_summary.pack(padx=10, pady=5)
        self.scrollbar = ttk.Scrollbar(self.root, orient=tk.VERTICAL, command=self.text_summary.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_summary.config(yscrollcommand=self.scrollbar.set)

    def summarize_text(self):
        url = self.entry_url.get()
        selected_model = self.model_var.get()

        if selected_model == "BERT":
            summary = summarize_with_bert(url)
        elif selected_model == "T5":
            summary = summarize_with_t5(url)
        elif selected_model == "BART":
            summary = summarize_with_bart(url)
        else:
            summary = "Hata: Geçersiz dil modeli seçimi"

        self.text_summary.delete(1.0, tk.END)  # Önceki içeriği temizle
        self.text_summary.insert(tk.END, summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()
