from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# Dictionary to map language pairs to model names
model_names = {
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
    # Add more language pairs as needed
}

def load_model(src_lang, tgt_lang):
    model_name = model_names.get((src_lang, tgt_lang))
    if not model_name:
        raise ValueError(f"Model for {src_lang} to {tgt_lang} not found.")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate(text, src_lang, tgt_lang):
    model, tokenizer = load_model(src_lang, tgt_lang)
    inputs = tokenizer.encode(text, return_tensors="pt")
    translated = model.generate(inputs)
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translated_text

# GET ROUTE
@app.route('/', methods=['GET'])
def hello():
    return jsonify({'message': "HELLO WORLD"}),200



# Translate a sentence to the tgt_lang
@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text')
    src_lang = data.get('src_lang')
    tgt_lang = data.get('tgt_lang')
    print(src_lang, tgt_lang, text)
    translated_text = translate(text, src_lang, tgt_lang)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True, port=8000)

# using venv-2