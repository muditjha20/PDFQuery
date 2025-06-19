from langchain_community.vectorstores import Cassandra
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import cassio
load_dotenv()

# Secure credentials 
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load PDF content
pdfreader = PdfReader("COMP2604-ch6.pdf")
raw_text = ""

for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Connect to Cassandra (DataStax Astra)
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Initialize OpenAI LLM and Embeddings
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4.1-nano")
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create vector store
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)

# Split the text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)

# Add text to the vector store
astra_vector_store.add_texts(texts[:50])
print(f"Inserted {len(texts[:50])} text chunks.")

# Create the vector index wrapper
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Q&A loop
first_question = True

while True:
    query_text = input("\nEnter your question (or type 'quit' to exit): " if first_question else "\nWhat's your next question (or type 'quit' to exit): ")
    
    if query_text.lower().strip() == "quit":
        break

    if not query_text.strip():
        continue

    first_question = False

    print(f"\nQUESTION: \"{query_text}\"")
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print(f"Answer: {answer}\n")

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print(f"     [{score:.4f}] {doc.page_content[:84]}...")
