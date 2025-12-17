import os
from dotenv import load_dotenv

load_dotenv()

PDF_DIRECTORY = "data/pdfs"

ASTRA_DB_API_ENDPOINT = os.getenv("")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("")
ASTRA_DB_KEYSPACE = os.getenv("")

EMBEDDING_MODEL = os.getenv("BERT_MODEL")
GPT2_MODEL = os.getenv("GPT2_MODEL")

MAX_NEW_TOKENS = 256
