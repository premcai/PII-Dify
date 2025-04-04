from flask import Flask, request, jsonify
import os
import re
import random
import pymongo
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.predefined_recognizers import SpacyRecognizer
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
import PyPDF2

app = Flask(__name__)

class Config:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", database_name="pii_data", collection_name="pii_collection"):
        self.mongo_uri = mongo_uri
        self.database_name = database_name
        self.collection_name = collection_name

class MongoHandler:
    def __init__(self, config: Config):
        self.mongo_uri = config.mongo_uri
        self.database_name = config.database_name
        self.collection_name = config.collection_name

    def store_mapping(self, masked_value: str, original_value: str) -> None:
        with pymongo.MongoClient(self.mongo_uri) as client:
            db = client[self.database_name]
            collection = db[self.collection_name]
            collection.insert_one({"masked_value": masked_value, "original_value": original_value})

    def retrieve_mapping(self, masked_value: str) -> str:
        with pymongo.MongoClient(self.mongo_uri) as client:
            db = client[self.database_name]
            collection = db[self.collection_name]
            result = collection.find_one({"masked_value": masked_value})
            return result["original_value"] if result else None

config = Config()
mongo_handler = MongoHandler(config)

# Initialize the Presidio AnalyzerEngine and Flair model once at startup
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(SpacyRecognizer())
# Smaller Flair NER model
flair_model = SequenceTagger.load("ner")

@app.route("/mask", methods=["POST"])
def mask_data():
    try:
        data = request.json.get("data", "")
        masked_text = mask_pii(data)
        return jsonify({"masked_text": masked_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/unmask", methods=["POST"])
def unmask_data():
    try:
        data = request.json.get("data", "")
        unmasked_text = unmask_pii(data)
        return jsonify({"unmasked_text": unmasked_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        file = request.files["file"]
        file_location = f"./uploads/{file.filename}"
        file.save(file_location)
        if file.filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_location)
        else:
            with open(file_location, "r") as f:
                text = f.read()
        masked_text = mask_pii(text)
        return jsonify({"file_name": file.filename, "masked_text": masked_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({"status": "PII Masking API is running"})


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


def mask_pii(text: str) -> str:
    sentence = Sentence(text)
    flair_model.predict(sentence)
    results = analyzer.analyze(text=text, entities=["PERSON", "LOCATION"], language='en')
    masked_text = text
    if results:
        for result in results:
            original_value = text[result.start:result.end]
            masked_value = f"<MASKED_{result.entity_type}_{random.randint(1000, 9999)}>"
            mongo_handler.store_mapping(masked_value, original_value)
            masked_text = masked_text.replace(original_value, masked_value)
    return masked_text


def unmask_pii(text: str) -> str:
    masked_values = re.findall(r'<([^>]+)>', text)
    for masked_value in masked_values:
        original_value = mongo_handler.retrieve_mapping(masked_value)
        if original_value:
            text = text.replace(f"<{masked_value}>", original_value)
    return text


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)
