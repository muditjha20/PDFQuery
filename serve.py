# serve.py — minimal API for your existing Cassandra vector store (with inline keys)
from langchain.prompts import PromptTemplate
import os, pathlib, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from langchain_community.vectorstores import Cassandra
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# === Config  ===
KEYSPACE      = (os.getenv("ASTRA_DB_KEYSPACE") or "default_keyspace").strip()
TABLE_NAME    = (os.getenv("CASSANDRA_TABLE") or "ml_supervised").strip()
BUNDLE_PATH   = (os.getenv("ASTRA_SECURE_BUNDLE") or "secure-connect-ml-db.zip").strip()
OPENAI_KEY    = (os.getenv("OPENAI_API_KEY") or "").strip()
ASTRA_TOKEN   = (os.getenv("ASTRA_DB_APPLICATION_TOKEN") or "").strip()
ASTRA_DB_ID   = (os.getenv("ASTRA_DB_ID") or "").strip()
MODEL_NAME    = (os.getenv("OPENAI_MODEL") or "gpt-4.1-nano").strip()
ALLOWED_ORIGS = (os.getenv("ALLOWED_ORIGINS") or "*").strip()

ALLOWED_ORIGS = "*"

assert OPENAI_KEY.startswith("sk-"), "OPENAI_API_KEY missing/invalid"
assert ASTRA_TOKEN.startswith("AstraCS:"), "ASTRA token missing/invalid"
assert len(ASTRA_DB_ID) > 10, "ASTRA DB ID missing/invalid"

# auto-download secure bundle on boot if missing — handy locally & on Render
def ensure_bundle():
    if pathlib.Path(BUNDLE_PATH).exists():
        return
    url = f"https://api.astra.datastax.com/v2/databases/{ASTRA_DB_ID}/secureBundleURL"
    r = requests.post(url, headers={
        "Authorization": f"Bearer {ASTRA_TOKEN}",
        "Content-Type": "application/json"
    }, json={})
    r.raise_for_status()
    dl = r.json().get("downloadURL")
    if not dl:
        raise RuntimeError("Could not get secure bundle URL from Astra.")
    with requests.get(dl, stream=True) as resp:
        resp.raise_for_status()
        with open(BUNDLE_PATH, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)

ensure_bundle()

# === Connect to Astra via secure bundle ===
cloud_config = {"secure_connect_bundle": BUNDLE_PATH}
auth_provider = PlainTextAuthProvider("token", ASTRA_TOKEN)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect(KEYSPACE)

# === Vector store + LLM (must match ingest) ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
vs = Cassandra(embedding=embeddings, table_name=TABLE_NAME, session=session, keyspace=KEYSPACE)
index = VectorStoreIndexWrapper(vectorstore=vs)
llm = ChatOpenAI(openai_api_key=OPENAI_KEY, model_name=MODEL_NAME, temperature=0.2)

# === FastAPI app ===
app = FastAPI(title="RAG ML Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGS == "*" else [o.strip() for o in ALLOWED_ORIGS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    top_k: int = 4

class AskResponse(BaseModel):
    answer: str
    sources: list

@app.get("/health")
def health():
    return {"status": "ok"}

from langchain.prompts import PromptTemplate

CONTEXT_PROMPT = PromptTemplate.from_template("""
You are an ML assistant. Answer ONLY using the context below. If the answer is not present, say:
"I don't know based on the indexed PDFs."

Question: {question}

Context:
{context}

Give a concise answer (3–6 sentences).
""")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        return AskResponse(answer="Please provide a question.", sources=[])

    # Stronger retrieval: prefer diverse MMR; otherwise use larger k
    if hasattr(vs, "max_marginal_relevance_search_with_score"):
        docs_scores = vs.max_marginal_relevance_search_with_score(q, k=max(6, req.top_k), fetch_k=24)
    else:
        docs_scores = vs.similarity_search_with_score(q, k=max(8, req.top_k))

    # Build compact context block
    context_chunks = []
    sources = []
    for doc, score in docs_scores:
        text = (doc.page_content or "").strip().replace("\n", " ")
        if text:
            context_chunks.append(text[:1200])  # cap per chunk to keep prompt lean
        meta = doc.metadata or {}
        sources.append({
            "score": float(score),
            "snippet": (doc.page_content or "")[:220],
            "source_pdf": meta.get("source_pdf"),
            "page": meta.get("page"),
        })
    context_str = "\n\n".join(context_chunks)

    # Low-temp generation with explicit grounding prompt
    prompt = CONTEXT_PROMPT.format(question=q, context=context_str)
    ans = llm.invoke(prompt).content.strip()

    return AskResponse(answer=ans, sources=sources)
