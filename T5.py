import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_text_from_website(url):
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

def summarize_with_t5(url):
    text = get_text_from_website(url)
    if text:
        model_name = "csebuetnlp/mT5_multilingual_XLSum"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        input_ids = tokenizer(
            [text],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )["input_ids"]

        output_ids = model.generate(
            input_ids=input_ids,
            max_length=1024,
            min_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            length_penalty=2.0,
            num_return_sequences=1
        )

        summary = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return summary
    else:
        return "Metin alınamadı."
