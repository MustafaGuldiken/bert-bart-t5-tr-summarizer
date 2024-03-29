import requests
import torch
from bs4 import BeautifulSoup
from transformers import BertTokenizerFast, EncoderDecoderModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-turkish-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

def generate_summary(text, min_length=30, max_length=1024):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, min_length=min_length, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def summarize_text_from_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            html_content = response.content.decode("utf-8")
            soup = BeautifulSoup(html_content, 'html.parser')
            paragraphs = soup.find_all('p')
            clean_text = '\n'.join(paragraph.text for paragraph in paragraphs if not paragraph.find('script') and not paragraph.find('style'))
            return clean_text.strip()
        else:
            print("Hata: İstek başarısız oldu.")
            return None
    except Exception as e:
        print("Hata:", e)
        return None

def summarize_with_bert(url):
    text = summarize_text_from_website(url)
    if text:
        summary = generate_summary(text)
        return summary
    else:
        return "Metin alınamadı."
