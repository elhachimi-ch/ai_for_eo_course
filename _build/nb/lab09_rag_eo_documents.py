# %% [markdown]
# # Lab 10 — A cited RAG assistant for Earth-observation documents
# **Deep Learning for Earth Observation** · Part X
#
# Retrieval-augmented generation (Lewis et al., 2020): **ingest → chunk → embed → index → retrieve → generate with citations**.
# Documents: open-access papers used in this course (arXiv). Replace them with your own handbooks, reports or theses.
#
# LLM options (choose one in the config cell):
# * **Local & free**: [Ollama](https://ollama.com) with a model such as `qwen2.5:7b` (works offline — good for confidential documents).
# * **Hosted**: any provider supported by `langchain.chat_models.init_chat_model` (set the API key as an environment variable).

# %%
!pip -q install langchain langchain-community langchain-text-splitters langchain-huggingface sentence-transformers faiss-cpu pypdf
# for a local model:  !pip -q install langchain-ollama   (and run `ollama pull qwen2.5:7b` on the machine)

# %%
import os, urllib.request
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model

MODEL = "ollama:qwen2.5:7b"          # e.g. "openai:gpt-4o-mini", "anthropic:<model-name>", "google_genai:<model-name>"

# %% [markdown]
# ## 1 · Ingest

# %%
PAPERS = {"zhu_earth_fm.pdf": "https://arxiv.org/pdf/2405.04285",          # Zhu et al., foundations of Earth FMs
          "prithvi_eo2.pdf": "https://arxiv.org/pdf/2412.02732",           # Prithvi-EO-2.0
          "alphaearth.pdf": "https://arxiv.org/pdf/2507.22291"}            # AlphaEarth Foundations
os.makedirs("docs", exist_ok=True)
for name, url in PAPERS.items():
    path = f"docs/{name}"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

docs = []
for name in PAPERS:
    docs += PyPDFLoader(f"docs/{name}").load()                       # one Document per page
print(len(docs), "pages")

# %% [markdown]
# ## 2 · Chunk, embed, index

# %%
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
db = FAISS.from_documents(chunks, emb)
retriever = db.as_retriever(search_kwargs={"k": 4})
print(len(chunks), "chunks indexed")

# %% [markdown]
# ## 3 · Retrieve + generate with citations

# %%
llm = init_chat_model(MODEL, temperature=0)

SYSTEM = """You are an Earth-observation analyst.
Use ONLY the provided context. If the answer is not in the context, say you don't know.
Return: (1) a short answer, (2) the evidence with [source, page], (3) caveats."""

def ask(q, show_chunks=False):
    ctx = retriever.invoke(q)
    context = "\n\n".join(f"[{os.path.basename(d.metadata['source'])} p.{d.metadata.get('page', 0) + 1}] {d.page_content}"
                          for d in ctx)
    if show_chunks:
        print(context[:3000], "\n" + "-" * 80)
    msg = [("system", SYSTEM), ("user", f"Context:\n{context}\n\nQuestion: {q}")]
    return llm.invoke(msg).content

print(ask("What are the must-have features of an ideal Earth foundation model?"))

# %%
print(ask("How many parameters does Prithvi-EO-2.0 have and what data was it trained on?"))

# %%
print(ask("What is the population of Casablanca?"))     # not in the documents: the assistant should refuse

# %% [markdown]
# ## 4 · Evaluate retrieval separately
# A wrong answer can come from bad retrieval *or* bad generation. Check whether the right page is retrieved.

# %%
tests = [("11 features ideal Earth foundation model", "zhu_earth_fm.pdf"),
         ("4.2 million time series HLS pre-training", "prithvi_eo2.pdf"),
         ("64-dimensional embedding field 10 m annual", "alphaearth.pdf")]
for q, expected in tests:
    hits = [os.path.basename(d.metadata["source"]) for d in retriever.invoke(q)]
    print(f"{'OK ' if expected in hits else 'MISS'}  {q!r:55} -> {hits}")

# %% [markdown]
# ## Exercises
# 1. Change `chunk_size` (300, 1000, 2000) and `k`. How do answers and retrieval hits change?
# 2. Add French or Arabic documents and switch to a multilingual embedder (`BAAI/bge-m3`).
# 3. Re-implement the pipeline with LlamaIndex (as in Rodriguez, 2024, Ch. 7) and compare.
