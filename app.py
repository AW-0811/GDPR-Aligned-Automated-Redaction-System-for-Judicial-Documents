import os, re, fitz, faiss, numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-02-15-preview"
)

DEPLOYMENT_CHAT  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
DEPLOYMENT_EMBED = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

def chunk_text(text, max_words=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def embed_texts(texts, batch_size=5):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=DEPLOYMENT_EMBED,
            input=batch
        )
        for d in resp.data:
            embeddings.append(np.array(d.embedding, dtype="float32"))
    return embeddings

SYSTEM_PROMPT = """
You are a Government AI Assistant used for judicial document processing,
operating under federal confidentiality standards (Privacy Act, FOIA Ex. 6).

PRIVACY RULES:
1. NEVER output or reconstruct PII: names, emails, phone numbers, addresses,
   account numbers, SSNs, or DOBs.
2. If PII appears in retrieved context, automatically replace it with [REDACTED].
3. If the user directly requests PII, respond ONLY with:
   "I’m sorry, but I can’t share that information."
4. Summaries must be high-level and non-specific.
"""

PII_REFUSAL_PATTERN = re.compile(
    r"(?i)(what is|give me|tell me|show me|provide|list|reveal).*"
    r"(ssn|social security|email|phone|address|dob|date of birth|name)"
)

def ask(query, chunks, index):

    # 1. Refusal mode
    if PII_REFUSAL_PATTERN.search(query):
        return "I’m sorry, but I can’t share that information."

    # 2. Retrieve
    q_emb = embed_texts([query])[0].reshape(1, -1)
    D, I = index.search(q_emb, 3)
    retrieved = "\n---\n".join(chunks[i] for i in I[0])

    # 3. LLM response
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{retrieved}\n\nQuestion: {query}"}
    ]

    resp = client.chat.completions.create(
        model=DEPLOYMENT_CHAT,
        messages=messages,
        temperature=0
    )
    return resp.choices[0].message.content.strip()

st.title("🔒 GDPR Compliant Judicial Document Redaction RAG System")
st.caption("Upload a court transcript PDF and ask privacy-safe questions.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    st.success("PDF uploaded successfully. Processing…")

    # Read PDF
    doc = fitz.open(stream=uploaded.read(), filetype="pdf")
    text = "\n".join(page.get_text("text") for page in doc)

    # Chunk + embed
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    # Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Generating response…"):
            answer = ask(query, chunks, index)

        st.markdown("### Response")
        st.write(answer)

        # Optional debug display
        if st.checkbox("Show Retrieved Chunks (for debugging)"):
            q_emb = embed_texts([query])[0].reshape(1, -1)
            D, I = index.search(q_emb, 3)
            for i in I[0]:
                st.markdown(f"**Chunk {i}:**\n\n{chunks[i]}")
