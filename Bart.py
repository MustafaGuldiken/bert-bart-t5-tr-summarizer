import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p") if not p.find(class_="advertisement")])
        return text
    except Exception as e:
        print("Hata:", e)
        return None

def summarize_with_bart(url):
    text = get_text_from_url(url)
    if text:
        tokenizer = AutoTokenizer.from_pretrained("mukayese/transformer-turkish-summarization")
        model = AutoModelForSeq2SeqLM.from_pretrained("mukayese/transformer-turkish-summarization")

        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=400, min_length=75, num_beams=6, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    else:
        return "Metin alınamadı."
