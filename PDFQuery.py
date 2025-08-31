# PDFQuery.py

from langchain_community.vectorstores import Cassandra
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import cassio

load_dotenv()

# Secure credentials (read from process env that we just set)
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- replace everything from "Load PDF content" down to add_texts(...) with this ---

from pathlib import Path
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_NAME = "Supervised-Learning.pdf"           # you've already renamed it to your ML PDF
ASTRA_DB_KEYSPACE = "default_keyspace"  # matches your setup
TABLE_NAME = "ml_supervised"

# Read PDF pages separately (so we can tag page numbers)
pdfreader = PdfReader(PDF_NAME)
pages = []
for i, page in enumerate(pdfreader.pages, start=1):
    txt = page.extract_text() or ""
    pages.append((i, txt))

# Split per page and build metadata
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)
all_chunks, all_meta = [], []
for page_no, txt in pages:
    if not txt.strip():
        continue
    chunks = splitter.split_text(txt)
    all_chunks.extend(chunks)
    all_meta.extend([{"source_pdf": PDF_NAME, "page": page_no}] * len(chunks))

print(f"Prepared {len(all_chunks)} chunks from {len(pages)} pages.")

# Connect via secure bundle (same as you already do)
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

BUNDLE_PATH = "secure-connect-ml-db.zip"
cloud_config = {"secure_connect_bundle": BUNDLE_PATH}
auth_provider = PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect(ASTRA_DB_KEYSPACE)

# Rebuild vector store handle
from langchain_community.vectorstores import Cassandra
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name=TABLE_NAME,
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
)

# Clean slate once (avoid duplicates), then ingest ALL chunks with metadata
session.execute(f"TRUNCATE {ASTRA_DB_KEYSPACE}.{TABLE_NAME};")  # run once at ingest time
astra_vector_store.add_texts(all_chunks, metadatas=all_meta)
print(f"Inserted {len(all_chunks)} text chunks with page metadata.")
