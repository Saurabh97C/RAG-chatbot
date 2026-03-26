
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load PDFs
nec_loader = PyPDFLoader("NEC.pdf")
nec_docs = nec_loader.load()

wattmonk_loader = PyPDFLoader("Wattmonk Information (1).pdf")
wattmonk_docs = wattmonk_loader.load()

all_docs = nec_docs + wattmonk_docs

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")

# Vector DB
db = FAISS.from_documents(chunks, embeddings)

# LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def classify_query(query):
    query = query.lower()
    if "wattmonk" in query:
        return "wattmonk"
    elif "nec" in query or "grounding" in query:
        return "nec"
    else:
        return "general"

def get_answer(query):
    intent = classify_query(query)

    if intent == "general":
        return "Hello! How can I help you?"

    if "full form of nec" in query.lower():
        return "NEC stands for National Electrical Code."

    results = db.similarity_search(query, k=4)
    context = " ".join([r.page_content for r in results])

    prompt = f"""
    You are an expert assistant.

    Answer the question clearly and completely using the context.
    If full form is asked, give full form properly.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=120)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
