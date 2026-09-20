import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="University Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Academic & Student Knowledge Assistant")
st.caption("Enterprise RAG Application powered by FAISS, Groq, and LangChain (LCEL)")

# ------------------------------------------------------------------------------
# 2. LOAD PRE-COMPUTED FAISS VECTORSTORE
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Knowledge Base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(
        folder_path="faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

try:
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
except Exception as e:
    st.error(f"Error loading FAISS vector database: {e}")
    st.info("Make sure the 'faiss_index' folder with 'index.faiss' and 'index.pkl' exists in your repository root.")
    st.stop()

# ------------------------------------------------------------------------------
# 3. GROQ LLM INITIALIZATION
# ------------------------------------------------------------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.warning("⚠️ GROQ_API_KEY missing! Please add GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-120b",
    temperature=0.2
)

# ------------------------------------------------------------------------------
# 4. RAG PROMPT & LCEL PIPELINE SETUP
# ------------------------------------------------------------------------------
system_prompt = (
    "You are an official University Academic and Student Knowledge Assistant.\n"
    "Answer the user's question accurately using ONLY the provided context below.\n"
    "If the answer cannot be found in the context, explicitly say: "
    "'I could not find relevant information in the university policy documents.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------------------------------------------------------------------
# 5. CHAT INTERFACE & SOURCE TRACEABILITY
# ------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Source Traceability"):
                for src in msg["sources"]:
                    st.write(f"• **Document:** `{src['file']}` | **Page:** {src['page']}")

# Handle new user query
if user_input := st.chat_input("Ask about scholarship policies, fee structure, exam rules, or academic schedule..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching official university documents..."):
            # Fetch retrieved documents for source tracing
            retrieved_docs = retriever.invoke(user_input)
            
            # Generate answer using LCEL chain
            answer = rag_chain.invoke(user_input)

            st.markdown(answer)

            # Trace metadata source context
            sources = []
            if retrieved_docs:
                with st.expander("📚 Source Traceability"):
                    for doc in retrieved_docs:
                        file_name = doc.metadata.get("file_name", doc.metadata.get("source", "Unknown"))
                        page_num = doc.metadata.get("page", "N/A")
                        st.write(f"• **Document:** `{file_name}` | **Page:** {page_num}")
                        sources.append({"file": file_name, "page": page_num})

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
