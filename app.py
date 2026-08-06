"""
CyberGuardian AI - An Agentic AI Based Cyber Security Incident Response Assistant
Capstone Project - Generative AI + Agentic AI Training

Everything lives in this single file on purpose (no utils/, no separate
modules) to keep the project simple and beginner-friendly, as required
for this capstone.
"""

#========LOAD MODULES====================
import os
import tempfile
from datetime import datetime
from io import BytesIO

import streamlit as st

# ---- Modern LangChain building blocks (no deprecated APIs) ----
# We use plain LCEL chains: prompt | llm | StrOutputParser()
# instead of RetrievalQA / LLMChain / ConversationChain.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# LLM + Embeddings providers
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# ReportLab for the downloadable PDF report
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


#========PAGE CONFIG====================
st.set_page_config(
    page_title="CyberGuardian AI",
    page_icon="🛡️",
    layout="wide",
)


#========SESSION STATE (APP MEMORY)====================
defaults = {
    "vector_store": None,        # FAISS vector store built from uploaded docs
    "retriever": None,           # retriever built from the vector store
    "raw_analysis_text": None,   # Stage A: deep technical analysis (free text)
    "analysis_result": None,     # Stage B: parsed dict for the result cards
    "qa_history": [],            # list of (question, answer) tuples
    "uploaded_file_names": [],   # names of processed knowledge base files
    "incident_description": "",  # last incident description used
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


#========SIDEBAR====================
with st.sidebar:

    # ---- Logo + Project Name ----
    st.markdown("## 🛡️ CyberGuardian AI")
    st.caption("Agentic AI Cyber Security Incident Response Assistant")

    st.divider()

    # ---- LLM Provider Selection ----
    st.subheader("⚙️ LLM Settings")

    llm_provider = st.selectbox(
        "Choose your LLM Provider",
        options=["Gemini", "Groq"],
        help="Select which AI provider you want to use for generating answers.",
    )

    # ---- API Key Input(s) ----
    # IMPORTANT: No API key is ever hardcoded. Keys are typed in by
    # the user and only live in this session's memory.
    #
    # NOTE ON EMBEDDINGS: Groq does not provide an embeddings API,
    # so even when "Groq" is selected for chat answers, we still
    # need a Google Gemini API key to create the document embeddings
    # used for search (FAISS).
    if llm_provider == "Gemini":
        gemini_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Paste your Gemini API key here",
            help="Used for both chat answers and document embeddings.",
        )
        groq_key_input = None
        embedding_key = gemini_key_input
        chat_key = gemini_key_input
    else:  # Groq selected
        groq_key_input = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Paste your Groq API key here",
            help="Used for generating chat answers.",
        )
        st.info(
            "ℹ️ Groq doesn't offer an embeddings API. A Google Gemini "
            "key is also needed just to search your document."
        )
        gemini_key_input = st.text_input(
            "Google Gemini API Key (for embeddings)",
            type="password",
            placeholder="Paste your Gemini API key here",
        )
        embedding_key = gemini_key_input
        chat_key = groq_key_input

    if chat_key and embedding_key:
        st.success("API Key(s) received ✅")
    else:
        st.warning("Please enter the required API key(s) to use the app.")

    st.divider()

    # ---- Advanced RAG Settings ----
    with st.expander("🛠️ Advanced RAG Settings"):
        model_name = st.text_input(
            "Model Name",
            value="gemini-3.5-flash" if llm_provider == "Gemini" else "llama-3.3-70b-versatile",
            help="The exact model ID sent to the provider. Change if you know a different model name.",
        )
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150, step=50)
        top_k = st.slider("Chunks to Retrieve (top-k)", min_value=2, max_value=10, value=4, step=1)

    st.divider()

    # ---- Reset Session ----
    if st.button("🔄 Reset Session", use_container_width=True):
        for key in defaults:
            st.session_state[key] = defaults[key]
        st.rerun()

    st.divider()

    # ---- About Section ----
    with st.expander("ℹ️ About This Project"):
        st.write(
            """
            **CyberGuardian AI** is a capstone project built using
            Generative AI + Agentic AI concepts.

            It analyzes uploaded security reports/logs (your "knowledge
            base") using a simple RAG (Retrieval Augmented Generation)
            pipeline, then runs a two-stage LLM analysis:

            1. A deep **technical analysis** (attack mechanics, IOCs,
               business impact, containment/eradication/recovery).
            2. A **structured summary** with:
               Incident Summary, Attack Type, Severity, Root Cause,
               Explanation, Mitigation, Best Practices, and a Disclaimer.

            Built with: Python, Streamlit, LangChain, FAISS,
            Google Gemini API, Groq API, and ReportLab.
            """
        )

    st.caption("🎓 Final Capstone Project | Generative AI + Agentic AI Training")


#========RAG PIPELINE FUNCTIONS====================
# Document(s) -> Chunking -> Embeddings -> FAISS -> Retriever

def load_single_document(uploaded_file):
    """
    Saves one uploaded file to a temporary location and loads it
    using the correct LangChain loader based on file type (PDF/TXT).
    Returns a list of LangChain Document objects.
    """
    file_suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if file_suffix.lower() == ".pdf":
        loader = PyPDFLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path, encoding="utf-8")

    documents = loader.load()
    os.remove(tmp_file_path)
    return documents


def load_all_documents(uploaded_files):
    """
    Loads every uploaded file (the user's "knowledge base") and
    combines them into a single list of Document objects.
    """
    all_documents = []
    for uploaded_file in uploaded_files:
        all_documents.extend(load_single_document(uploaded_file))
    return all_documents


def split_documents(documents, chunk_size, chunk_overlap):
    """Splits documents into smaller overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_vector_store(chunks, embedding_key):
    """Embeds the chunks and stores them in a FAISS vector store."""
    # NOTE: Google retired the old "models/embedding-001" and
    # "models/text-embedding-004" endpoints (Jan 2026). The current
    # replacement is "models/gemini-embedding-001".
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=embedding_key,
    )
    return FAISS.from_documents(chunks, embeddings)


def get_retriever(vector_store, top_k):
    """Returns a retriever that fetches the top-k relevant chunks."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def get_context(retriever, query):
    """
    Manually retrieves relevant chunks for a query and joins them
    into a single context string. This is the "simple RAG" style:
    similarity search -> plain string -> fed into an LCEL chain.
    """
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


#========LLM FUNCTION====================

def get_llm(provider, api_key, model_name):
    """Creates and returns a chat LLM object based on the chosen provider."""
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
    else:
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.3,
        )


#========STAGE A: TECHNICAL INCIDENT ANALYSIS (LCEL CHAIN)====================
# This mirrors "incident_analyzer.py" - a deep technical pass over
# the incident using the retrieved knowledge base context.

analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert Cyber Security Incident Response Analyst.

Your task is to analyze the cyber incident using the provided knowledge
base context (retrieved from the user's uploaded document(s)).

Knowledge Base Context:
{context}

Incident:
{incident}

Perform the following:
1. Identify the type of attack.
2. Explain how the attack works.
3. Determine severity (Low/Medium/High/Critical).
4. List Indicators of Compromise (IOCs), if any are present in the context.
5. Explain the possible business impact.
6. Recommend immediate containment actions.
7. Recommend eradication steps.
8. Recommend recovery actions.
9. Suggest future prevention measures.

If the context does not contain enough information for a point, say so
briefly instead of making facts up. Provide a detailed technical analysis.
""")


def analyze_incident(incident, context, llm):
    """
    Stage A: Runs a simple LCEL chain (prompt | llm | parser) to produce
    a detailed technical analysis of the incident. Returns plain text.
    """
    chain = analysis_prompt | llm | StrOutputParser()
    return chain.invoke({"incident": incident, "context": context})


#========STAGE B: STRUCTURED REPORT (LCEL CHAIN)====================
# This mirrors "report_generator.py" - it takes the technical analysis
# from Stage A and reformats it into the exact sections this project
# requires, so we can display them as result cards and export a PDF.

report_prompt = ChatPromptTemplate.from_template("""
You are an expert Cyber Security Report Writer.

Using the incident description and the technical analysis below, write
a clear, structured incident report for a Security Operations Center.

Incident:
{incident}

Technical Analysis:
{analysis}

Respond STRICTLY in the following format. Use these exact section
headers, each on its own line starting with '### ', with nothing else
outside this format:

### Incident Summary
(A short summary of what happened)

### Attack Type
(The type of attack, e.g. Phishing, DDoS, Malware, Brute Force, etc.)

### Severity
(Low, Medium, High, or Critical - with a one-line reason)

### Root Cause
(What likely caused this incident, including Indicators of Compromise
if available)

### Explanation
(A simple explanation a non-expert could understand, including
possible business impact)

### Mitigation
(Immediate containment, eradication, and recovery steps)

### Best Practices
(Best practices to prevent this in the future)

### Disclaimer
(A short note that this is an AI-generated analysis and should be
reviewed by a qualified security professional before taking action)
""")


def generate_structured_report(incident, analysis, llm):
    """
    Stage B: Runs a simple LCEL chain to turn the technical analysis
    into our required structured sections. Returns plain text.
    """
    chain = report_prompt | llm | StrOutputParser()
    return chain.invoke({"incident": incident, "analysis": analysis})


def parse_analysis_output(text):
    """
    Splits the '### Header' formatted report into a dictionary so we
    can display each part as its own result card.
    """
    sections = {}
    parts = text.split("### ")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        header = lines[0].strip()
        content = lines[1].strip() if len(lines) > 1 else ""
        sections[header] = content
    return sections


#========CUSTOM Q&A CHAIN====================
qa_prompt = ChatPromptTemplate.from_template("""
You are CyberGuardian AI, a helpful cyber security assistant. Answer the
question using ONLY the context below, which comes from the user's
uploaded document(s). If the answer isn't in the context, say you don't
have enough information in the document. Be clear and concise.

Context:
{context}

Question:
{question}
""")


def answer_question(question, context, llm):
    """Simple LCEL chain used for the custom Q&A chat section."""
    chain = qa_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


#========PDF REPORT GENERATION (REPORTLAB)====================

def build_pdf_report(sections, incident_description, file_names):
    """
    Builds a downloadable PDF incident report using ReportLab.
    Includes: Project Name, Date, Summary, Findings, Recommendations,
    and a Footer, as required by the project spec.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.8 * inch, bottomMargin=0.8 * inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1a1a2e")
    )
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], textColor=colors.HexColor("#0f4c81"),
        spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["BodyText"], leading=16,
    )
    footer_style = ParagraphStyle(
        "FooterStyle", parent=styles["Normal"], fontSize=8, textColor=colors.grey,
    )

    story = []

    # ---- Project Name + Date ----
    story.append(Paragraph("CyberGuardian AI - Incident Response Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d %B %Y, %H:%M')}", body_style))
    if file_names:
        story.append(Paragraph(f"Source Document(s): {', '.join(file_names)}", body_style))
    if incident_description:
        story.append(Paragraph(f"Incident Description: {incident_description}", body_style))
    story.append(Spacer(1, 12))

    # ---- Summary ----
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(sections.get("Incident Summary", "Not available."), body_style))

    # ---- Findings ----
    story.append(Paragraph("Findings", heading_style))
    findings_map = ["Attack Type", "Severity", "Root Cause", "Explanation"]
    for label in findings_map:
        story.append(Paragraph(f"<b>{label}:</b> {sections.get(label, 'Not available.')}", body_style))
        story.append(Spacer(1, 4))

    # ---- Recommendations ----
    story.append(Paragraph("Recommendations", heading_style))
    reco_map = ["Mitigation", "Best Practices"]
    for label in reco_map:
        story.append(Paragraph(f"<b>{label}:</b> {sections.get(label, 'Not available.')}", body_style))
        story.append(Spacer(1, 4))

    # ---- Disclaimer / Footer ----
    story.append(Spacer(1, 16))
    story.append(Paragraph("Disclaimer", heading_style))
    story.append(Paragraph(sections.get("Disclaimer", "This is an AI-generated analysis."), body_style))

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Generated by CyberGuardian AI - Final Capstone Project | "
        "Generative AI + Agentic AI Training",
        footer_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


#========MAIN PAGE: TITLE + DESCRIPTION====================
st.title("🛡️ CyberGuardian AI")
st.write(
    "An **Agentic AI based Cyber Security Incident Response Assistant** "
    "that reads your uploaded security reports/logs and helps you "
    "understand and respond to incidents faster."
)

st.divider()

#========MAIN PAGE: UPLOAD SECTION (KNOWLEDGE BASE)====================
st.subheader("📂 Upload Security Report(s) / Log File(s)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF or TXT files as your knowledge base",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"Files selected: **{', '.join(f.name for f in uploaded_files)}**")

#========MAIN PAGE: INCIDENT DESCRIPTION (OPTIONAL)====================
st.subheader("📝 Describe the Incident (optional)")
incident_description = st.text_area(
    "If you already suspect what happened, describe it here. Otherwise, "
    "leave it blank and CyberGuardian AI will analyze the document as a whole.",
    placeholder="e.g. Multiple failed login attempts followed by a successful "
    "login from an unfamiliar IP address...",
)

analyze_clicked = st.button("🔍 Analyze Incident", type="primary", use_container_width=True)

#========MAIN PAGE: ANALYZE BUTTON LOGIC====================
if analyze_clicked:
    if not uploaded_files:
        st.error("Please upload at least one PDF or TXT file first.")
    elif not chat_key or not embedding_key:
        st.error("Please enter the required API key(s) in the sidebar first.")
    else:
        with st.spinner("Reading document(s) and building the search index..."):
            documents = load_all_documents(uploaded_files)
            chunks = split_documents(documents, chunk_size, chunk_overlap)
            vector_store = build_vector_store(chunks, embedding_key)
            retriever = get_retriever(vector_store, top_k)

            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.uploaded_file_names = [f.name for f in uploaded_files]

        # Use the user's incident description if given, otherwise fall
        # back to a generic query so the whole document gets analyzed.
        query = incident_description.strip() or (
            "Analyze this document as a whole for any security incidents, "
            "attacks, or suspicious activity described in it."
        )
        st.session_state.incident_description = query

        with st.spinner("Running technical analysis..."):
            llm = get_llm(llm_provider, chat_key, model_name)
            context = get_context(retriever, query)
            raw_analysis = analyze_incident(query, context, llm)
            st.session_state.raw_analysis_text = raw_analysis

        with st.spinner("Structuring the final report..."):
            structured_report = generate_structured_report(query, raw_analysis, llm)
            st.session_state.analysis_result = parse_analysis_output(structured_report)

        st.success("Analysis complete! Scroll down to see the results. ⬇️")


#========MAIN PAGE: RESULTS CARDS====================
if st.session_state.analysis_result:
    st.divider()
    st.subheader("📊 Incident Analysis Results")

    result = st.session_state.analysis_result

    section_icons = {
        "Incident Summary": "📝",
        "Attack Type": "🎯",
        "Severity": "🚨",
        "Root Cause": "🔍",
        "Explanation": "💡",
        "Mitigation": "🛠️",
        "Best Practices": "✅",
        "Disclaimer": "⚠️",
    }

    section_names = list(section_icons.keys())
    col1, col2 = st.columns(2)

    for i, section in enumerate(section_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            with st.container(border=True):
                icon = section_icons.get(section, "📌")
                st.markdown(f"**{icon} {section}**")
                st.write(result.get(section, "_Not available in the analysis._"))

    # Optional deep-dive into the Stage A technical analysis
    if st.session_state.raw_analysis_text:
        with st.expander("🔬 View Full Technical Analysis"):
            st.write(st.session_state.raw_analysis_text)

    #========DOWNLOAD PDF BUTTON====================
    pdf_bytes = build_pdf_report(
        result,
        st.session_state.incident_description,
        st.session_state.uploaded_file_names,
    )
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name="cyberguardian_incident_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


#========MAIN PAGE: CUSTOM QUESTION (CHAT) SECTION====================
if st.session_state.retriever:
    st.divider()
    st.subheader("💬 Ask a Question About the Document")

    question = st.text_input(
        "Ask anything about the uploaded document(s)",
        placeholder="e.g. Which IP addresses are mentioned in the log?",
    )
    ask_clicked = st.button("🤖 Ask", use_container_width=True)

    if ask_clicked:
        if not question.strip():
            st.error("Please type a question first.")
        elif not chat_key:
            st.error("Please enter the required API key in the sidebar first.")
        else:
            with st.spinner("Thinking..."):
                llm = get_llm(llm_provider, chat_key, model_name)
                context = get_context(st.session_state.retriever, question)
                answer = answer_question(question, context, llm)
                st.session_state.qa_history.append((question, answer))

    # ---- Chat-style response area ----
    for past_question, past_answer in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(past_question)
        with st.chat_message("assistant"):
            st.write(past_answer)
