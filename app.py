"""
CyberGuardian AI - An Agentic AI Based Cyber Security Incident Response Assistant
Capstone Project - Generative AI + Agentic AI Training

Everything lives in this single file on purpose (no utils/, no separate
modules) to keep the project simple and beginner-friendly, as required
for this capstone.
"""

#========LOAD MODULES====================
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import pandas as pd
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
    initial_sidebar_state="expanded",
)


#========THREAT LEVEL DEFINITIONS====================
# Central place that maps a parsed "Severity" string to a visual
# threat level used across the dashboard, status bar and badges.
THREAT_LEVELS = {
    "CRITICAL":  {"emoji": "🔴", "label": "CRITICAL THREAT",   "css": "tl-critical",  "order": 5},
    "HIGH":      {"emoji": "🟠", "label": "HIGH THREAT",       "css": "tl-high",      "order": 4},
    "MEDIUM":    {"emoji": "🟡", "label": "MEDIUM THREAT",     "css": "tl-medium",    "order": 3},
    "LOW":       {"emoji": "🔵", "label": "LOW THREAT",        "css": "tl-low",       "order": 2},
    "SAFE":      {"emoji": "🟢", "label": "SYSTEM SAFE",       "css": "tl-safe",      "order": 1},
    "UNSCANNED": {"emoji": "⚪", "label": "NOT YET SCANNED",   "css": "tl-unscanned", "order": 0},
}


def detect_threat_level(severity_text: str) -> str:
    """
    Maps free-text severity (from the LLM report) to a THREAT_LEVELS key.
    Never silently claims "SAFE" - that label is only used when the LLM
    text explicitly says so (e.g. "no incident", "no risk"). Anything the
    parser can't confidently classify falls back to MEDIUM so it gets a
    human's attention instead of being hidden as green/safe.
    """
    if not severity_text or not severity_text.strip():
        return "UNSCANNED"
    text = severity_text.lower()
    if "critical" in text:
        return "CRITICAL"
    if "high" in text:
        return "HIGH"
    if "medium" in text or "moderate" in text:
        return "MEDIUM"
    if "low" in text:
        return "LOW"
    if any(phrase in text for phrase in ("no incident", "no risk", "no threat", "not malicious", "benign")):
        return "SAFE"
    # Ambiguous / unrecognized wording - flag for manual review rather than
    # defaulting to "safe".
    return "MEDIUM"


#========KNOWLEDGE BASE EXPORT (DUMMY / METADATA DOWNLOAD)====================
def build_kb_summary_text(file_names, chunk_count, last_updated, iocs) -> bytes:
    """
    Builds a plain-text 'Knowledge Base Summary' export.

    NOTE: the original uploaded files are only ever written to a temp path
    for parsing and then deleted (see load_single_document) - the app never
    retains raw source bytes. So instead of pretending to re-package files
    we no longer have, this generates a real, useful export of what the
    knowledge base actually contains: indexed file names, chunk count, and
    the indicators that were extracted from it.
    """
    lines = []
    lines.append("CYBERGUARDIAN AI - KNOWLEDGE BASE SUMMARY")
    lines.append("=" * 45)
    lines.append(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}")
    lines.append(f"Last built/updated: {last_updated.strftime('%d %B %Y, %H:%M') if last_updated else 'N/A'}")
    lines.append("")
    lines.append(f"Indexed Documents ({len(file_names)}):")
    if file_names:
        for name in file_names:
            lines.append(f"  - {name}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Chunks Indexed: {chunk_count}")
    lines.append("")
    if iocs:
        lines.append("Indicators of Compromise Detected:")
        lines.append(f"  - Unique IPs: {len(iocs.get('ip_counter', {}))}")
        lines.append(f"  - External IPs: {len(iocs.get('external_ips', {}))}")
        lines.append(f"  - File Hashes: {len(iocs.get('hash_counter', {}))}")
        lines.append(f"  - CVEs Referenced: {len(iocs.get('cve_counter', {}))}")
        lines.append(f"  - Domains / URLs: {len(iocs.get('url_counter', {}))}")
        lines.append("")
    lines.append("-" * 45)
    lines.append("Generated by CyberGuardian AI | Capstone Project")
    return "\n".join(lines).encode("utf-8")


#========SESSION STATE (APP MEMORY)====================
defaults = {
    "vector_store": None,        # FAISS vector store built from the knowledge base
    "retriever": None,           # retriever built from the vector store
    "raw_analysis_text": None,   # Stage A: deep technical analysis (free text)
    "analysis_result": None,     # Stage B: parsed dict for the result cards
    "qa_history": [],            # list of (question, answer) tuples - Chat with Expert
    "uploaded_file_names": [],   # names of processed knowledge base files
    "incident_description": "",  # last incident description used
    "analysis_history": [],      # list of dicts: {time, attack_type, severity, summary}
    "kb_last_updated": None,     # datetime the knowledge base was last (re)built
    "kb_chunk_count": 0,         # number of chunks currently indexed
    "threat_level": "UNSCANNED", # current overall THREAT_LEVELS key
    "iocs": {},                  # live indicators extracted from the knowledge base text
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


#========CUSTOM STYLING====================

def inject_custom_css():
    """Lightweight CSS-only theming layer - no extra libraries required."""
    st.markdown(
        """
        <style>
        .cg-hero {
            padding: 1.3rem 1.6rem;
            border-radius: 16px;
            background: linear-gradient(120deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 28px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        .cg-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 85% -10%, rgba(76,201,240,0.22), transparent 55%);
            pointer-events: none;
        }
        .cg-hero h1 {
            margin: 0; color: #eafaff; font-size: 1.7rem; font-weight: 750;
            letter-spacing: 0.01em;
        }
        .cg-hero p { margin: 0.3rem 0 0 0; color: #b9d7e0; font-size: 0.95rem; }

        /* ---- Global polish ---- */
        div[data-testid="stTabs"] button[role="tab"] {
            font-weight: 600;
            border-radius: 8px 8px 0 0;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 9px;
            font-weight: 600;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(76,201,240,0.18);
        }
        div[data-testid="stExpander"] {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        div[data-testid="stFileUploaderDropzone"] {
            border-radius: 12px;
        }

        .cg-file-card {
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.55rem 0.85rem;
            border-radius: 10px;
            background: rgba(76,201,240,0.06);
            border: 1px solid rgba(76,201,240,0.18);
            margin-bottom: 0.4rem;
        }
        .cg-file-card .cg-file-icon { font-size: 1.1rem; }
        .cg-file-card .cg-file-name { font-size: 0.9rem; color: #e6edf3; font-weight: 500; word-break: break-word; }

        .cg-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.55rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            border: 1px solid rgba(255,255,255,0.12);
            width: 100%;
            box-sizing: border-box;
        }
        .tl-unscanned{ background: rgba(255,255,255,0.06); color: #b8c2cc; }
        .tl-safe     { background: rgba(31,143,90,0.18);  color: #7CE0B0; }
        .tl-low      { background: rgba(45,116,182,0.20); color: #8ecae6; }
        .tl-medium   { background: rgba(196,138,20,0.22); color: #ffd166; }
        .tl-high     { background: rgba(196,74,20,0.25);  color: #ffab73; }
        .tl-critical { background: rgba(196,20,40,0.28);  color: #ff8a9a; animation: cg-pulse 1.6s infinite; }

        @keyframes cg-pulse {
            0%   { box-shadow: 0 0 0 0 rgba(255,80,90,0.45); }
            70%  { box-shadow: 0 0 0 8px rgba(255,80,90,0); }
            100% { box-shadow: 0 0 0 0 rgba(255,80,90,0); }
        }

        .cg-metric {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(255,255,255,0.03);
            padding: 0.6rem 0.9rem;
            text-align: center;
        }
        .cg-metric .cg-metric-value { font-size: 1.35rem; font-weight: 700; }
        .cg-metric .cg-metric-label { font-size: 0.75rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.03em; }

        .cg-timeline { position: relative; margin-left: 0.4rem; padding-left: 1.2rem; border-left: 2px solid rgba(255,255,255,0.15); }
        .cg-tl-item { position: relative; padding-bottom: 0.9rem; }
        .cg-tl-item::before {
            content: ""; position: absolute; left: -1.53rem; top: 0.15rem;
            width: 10px; height: 10px; border-radius: 50%;
            background: #4cc9f0; border: 2px solid rgba(255,255,255,0.4);
        }
        .cg-tl-time { font-size: 0.75rem; font-weight: 700; color: #8ecae6; letter-spacing: 0.02em; }
        .cg-tl-event { font-size: 0.9rem; color: #e6edf3; margin-top: 0.05rem; }

        .cg-chip {
            display: inline-block; padding: 0.3rem 0.65rem; border-radius: 999px;
            background: rgba(76,201,240,0.14); color: #8ecae6; font-size: 0.8rem;
            margin: 0.15rem 0.3rem 0.15rem 0; border: 1px solid rgba(76,201,240,0.3);
        }

        .cg-snapshot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.6rem; }
        .cg-snapshot-cell { border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 0.5rem 0.7rem; }
        .cg-snapshot-cell .cg-sc-label { font-size: 0.7rem; opacity: 0.65; text-transform: uppercase; letter-spacing: 0.03em; }
        .cg-snapshot-cell .cg-sc-value { font-size: 0.92rem; font-weight: 600; color: #eafaff; word-break: break-word; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_badge(level_key: str, size: str = "normal"):
    """Renders the colored threat-level pill used in the header bar and dashboard."""
    level = THREAT_LEVELS.get(level_key, THREAT_LEVELS["SAFE"])
    font_size = "1.05rem" if size == "large" else "0.95rem"
    st.markdown(
        f"""
        <div class="cg-badge {level['css']}" style="font-size:{font_size};">
            <span style="font-size:1.2em;">{level['emoji']}</span> {level['label']}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_box(label: str, value):
    st.markdown(
        f"""
        <div class="cg-metric">
            <div class="cg-metric-value">{value}</div>
            <div class="cg-metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_incident_snapshot(kv_fields: dict):
    """Renders the known important fields pulled out of a structured SOC report."""
    cells = []
    for label, aliases in INTERESTING_FIELDS:
        value = find_field(kv_fields, aliases)
        if value:
            cells.append((label, value))
    if not cells:
        return False
    html_cells = "".join(
        f"""<div class="cg-snapshot-cell">
                <div class="cg-sc-label">{label}</div>
                <div class="cg-sc-value">{value}</div>
            </div>"""
        for label, value in cells
    )
    st.markdown(f'<div class="cg-snapshot-grid">{html_cells}</div>', unsafe_allow_html=True)
    return True


def render_mitre_chips(mitre_counter):
    if not mitre_counter:
        return False
    chips = "".join(
        f'<span class="cg-chip">🎯 {tid} — {MITRE_TECHNIQUE_NAMES.get(tid, "Technique")} ({count}x)</span>'
        for tid, count in mitre_counter.most_common(15)
    )
    st.markdown(chips, unsafe_allow_html=True)
    return True


def render_attack_timeline(timeline_events):
    if not timeline_events:
        return False
    items = "".join(
        f"""<div class="cg-tl-item">
                <div class="cg-tl-time">{t}</div>
                <div class="cg-tl-event">{event}</div>
            </div>"""
        for t, event in timeline_events
    )
    st.markdown(f'<div class="cg-timeline">{items}</div>', unsafe_allow_html=True)
    return True


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

    # ---- Quick System Snapshot ----
    st.subheader("📟 System Snapshot")
    render_status_badge(st.session_state.threat_level)
    st.caption(
        f"📂 KB docs: {len(st.session_state.uploaded_file_names)}  |  "
        f"🧩 Chunks: {st.session_state.kb_chunk_count}"
    )
    if st.session_state.kb_last_updated:
        st.caption(f"🕒 KB last updated: {st.session_state.kb_last_updated.strftime('%d %b, %H:%M')}")

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

            It analyzes your uploaded security reports/logs (your "knowledge
            base") using a simple RAG (Retrieval Augmented Generation)
            pipeline, then runs a two-stage LLM analysis:

            1. A deep **technical analysis** (attack mechanics, IOCs,
               business impact, containment/eradication/recovery).
            2. A **structured summary** with:
               Incident Summary, Attack Type, Severity, Root Cause,
               Explanation, Mitigation, Best Practices, and a Disclaimer.

            A live **Dashboard** tracks overall threat status, and a
            **Chat with Expert** tab lets you ask free-form questions
            grounded in the same knowledge base.

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
        model="gemini-embedding-001",
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


#========LIVE THREAT INDICATOR EXTRACTION (REGEX, NO API CALLS)====================
# Pulls real, analyst-relevant indicators of compromise straight out of the
# uploaded knowledge base text - IPs, domains/URLs, emails, file hashes,
# CVE IDs and referenced ports. This runs locally (no LLM call, instant,
# free) every time the knowledge base is (re)built, so the dashboard
# reflects the *actual* data the analyst just loaded rather than a static
# document count.

_IP_REGEX = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")
_URL_REGEX = re.compile(r"https?://[^\s\"'<>\)\]]+")
_EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_HASH_REGEX = re.compile(r"\b[a-fA-F0-9]{64}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{32}\b")
_CVE_REGEX = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
_PORT_REGEX = re.compile(r"\bport[:\s]+(\d{1,5})\b", re.IGNORECASE)
_MITRE_REGEX = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_TIME_DASH_LINE = re.compile(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–—]\s*(.+)$")
_TIME_ONLY_LINE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$")

# A short lookup of common MITRE ATT&CK technique IDs -> readable names, so
# the dashboard can show "T1566 - Phishing" instead of a bare code. Falls
# back to a generic label for anything not in this list.
MITRE_TECHNIQUE_NAMES = {
    "T1566": "Phishing", "T1566.001": "Spearphishing Attachment",
    "T1059": "Command and Scripting Interpreter", "T1059.001": "PowerShell",
    "T1003": "OS Credential Dumping", "T1041": "Exfiltration Over C2 Channel",
    "T1547": "Boot or Logon Autostart Execution", "T1547.001": "Registry Run Keys",
    "T1071": "Application Layer Protocol", "T1071.001": "Web Protocols",
    "T1105": "Ingress Tool Transfer", "T1053": "Scheduled Task/Job",
    "T1027": "Obfuscated Files or Information", "T1486": "Data Encrypted for Impact",
    "T1078": "Valid Accounts", "T1190": "Exploit Public-Facing Application",
    "T1110": "Brute Force", "T1021": "Remote Services", "T1082": "System Information Discovery",
    "T1204": "User Execution", "T1055": "Process Injection",
}

# Fields worth surfacing as an "Incident Snapshot" card when a structured
# report (Label:\nValue style, like a typical SOC incident report) is
# uploaded. Matching is case-insensitive substring against the parsed key.
INTERESTING_FIELDS = [
    ("Incident ID", ["incident id"]),
    ("Organization", ["organization", "organisation"]),
    ("Classification", ["classification"]),
    ("Overall Severity", ["overall severity"]),
    ("Hostname", ["hostname"]),
    ("Department", ["department"]),
    ("Username", ["username", "user name"]),
    ("Source IP", ["source ip"]),
    ("Destination IP", ["destination ip"]),
    ("Protocol", ["protocol"]),
    ("Destination Port", ["destination port"]),
    ("Country", ["country"]),
    ("Detection Name", ["detection name"]),
    ("Multi-Factor Authentication", ["multi-factor authentication", "mfa"]),
    ("Threat Actor", ["threat actor"]),
    ("Bytes Uploaded", ["bytes uploaded", "bytes exfiltrated"]),
]


def is_private_ip(ip: str) -> bool:
    """Flags RFC1918 / loopback ranges as internal; everything else is external."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        o = [int(p) for p in parts]
    except ValueError:
        return False
    if o[0] == 10:
        return True
    if o[0] == 172 and 16 <= o[1] <= 31:
        return True
    if o[0] == 192 and o[1] == 168:
        return True
    if o[0] == 127:
        return True
    return False


def extract_key_value_fields(text: str) -> dict:
    """
    Parses "Label:\\nValue" style lines - the common pattern in structured
    SOC / incident report templates (e.g. "Hostname:\\nFINANCE-PC-07") -
    into a flat dict so the dashboard can show a real Incident Snapshot
    instead of just a document count.
    """
    lines = [l.strip() for l in text.splitlines()]
    fields = {}
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        if line and line.endswith(":") and 2 <= len(line) <= 60 and not line.startswith("="):
            key = line[:-1].strip()
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if j < len(lines):
                value = lines[j]
                if value and not value.startswith("=") and not value.endswith(":") and len(value) <= 120:
                    fields.setdefault(key, value)
            i = j
        else:
            i += 1
    return fields


def find_field(kv: dict, aliases: list):
    """Case-insensitive substring lookup of a canonical field in the parsed kv dict."""
    for k, v in kv.items():
        kl = k.lower()
        for alias in aliases:
            if alias == kl or alias in kl:
                return v
    return None


def extract_timeline(text: str) -> list:
    """
    Extracts chronological (time, event) pairs from two common log styles:
      "08:51:12 - Failed Login"          (same line)
      "08:40\\nPhishing email delivered"  (time alone, description on next line)
    Returns a list of (time_str, event_text) in the order found.
    """
    lines = [l.strip() for l in text.splitlines()]
    events = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        dash_match = _TIME_DASH_LINE.match(line)
        if dash_match:
            events.append((dash_match.group(1), dash_match.group(2).strip()))
            i += 1
            continue
        if _TIME_ONLY_LINE.match(line):
            j = i + 1
            while j < len(lines) and lines[j] == "":
                j += 1
            if (
                j < len(lines) and lines[j]
                and not lines[j].startswith("=")
                and not _TIME_ONLY_LINE.match(lines[j])
                and not _TIME_DASH_LINE.match(lines[j])
                and len(lines[j]) <= 120
            ):
                events.append((line, lines[j]))
                i = j + 1
                continue
        i += 1
    return events[:40]


def extract_iocs(text: str) -> dict:
    """Scans raw knowledge-base text and returns Counters/lists of real indicators found."""
    ip_counter = Counter(_IP_REGEX.findall(text))
    external_ips = {ip for ip in ip_counter if not is_private_ip(ip)}
    internal_ips = {ip for ip in ip_counter if is_private_ip(ip)}

    return {
        "ip_counter": ip_counter,
        "external_ips": external_ips,
        "internal_ips": internal_ips,
        "url_counter": Counter(_URL_REGEX.findall(text)),
        "email_counter": Counter(_EMAIL_REGEX.findall(text)),
        "hash_counter": Counter(_HASH_REGEX.findall(text)),
        "cve_counter": Counter(m.upper() for m in _CVE_REGEX.findall(text)),
        "port_counter": Counter(_PORT_REGEX.findall(text)),
        "mitre_counter": Counter(_MITRE_REGEX.findall(text)),
        "kv_fields": extract_key_value_fields(text),
        "timeline": extract_timeline(text),
    }


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


#========CUSTOM Q&A CHAIN (CHAT WITH EXPERT)====================
qa_prompt = ChatPromptTemplate.from_template("""
You are CyberGuardian AI, a senior cyber security expert assistant.
Answer the question using ONLY the context below, which comes from the
company's live knowledge base (uploaded reports/logs). If the answer
isn't in the context, say you don't have enough information in the
knowledge base rather than guessing. Be clear, concise, and speak like
a trusted security analyst briefing a colleague.

Context:
{context}

Question:
{question}
""")


def answer_question(question, context, llm):
    """Simple LCEL chain used for the Chat with Expert section."""
    chain = qa_prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context})


#========PDF REPORT GENERATION (REPORTLAB)====================
from reportlab.platypus import Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart


def build_ioc_bar_chart(iocs: dict) -> Drawing:
    """Native ReportLab vector bar chart (no matplotlib needed) summarizing IOC counts."""
    labels = ["IPs", "Ext. IPs", "Domains", "Hashes", "CVEs", "MITRE"]
    values = [
        len(iocs.get("ip_counter", {})),
        len(iocs.get("external_ips", [])),
        len(iocs.get("url_counter", {})),
        len(iocs.get("hash_counter", {})),
        len(iocs.get("cve_counter", {})),
        len(iocs.get("mitre_counter", {})),
    ]
    drawing = Drawing(430, 170)
    chart = VerticalBarChart()
    chart.x, chart.y = 40, 25
    chart.width, chart.height = 370, 120
    chart.data = [values]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = colors.HexColor("#0f4c81")
    chart.barWidth = 12
    drawing.add(chart)
    return drawing


def build_ioc_table(iocs: dict, styles) -> Table:
    """Structured Type/Value/Count table of the top indicators - a real visual, not prose."""
    cell_style = ParagraphStyle("IOCCell", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK")
    header = ["Type", "Value", "Mentions"]
    rows = [header]

    def add_rows(kind, counter, limit=5):
        for value, count in counter.most_common(limit):
            rows.append([kind, Paragraph(str(value), cell_style), str(count)])

    add_rows("IP Address", iocs.get("ip_counter", Counter()))
    add_rows("CVE", iocs.get("cve_counter", Counter()))
    add_rows("File Hash", iocs.get("hash_counter", Counter()), limit=3)
    add_rows("Domain/URL", iocs.get("url_counter", Counter()), limit=3)
    add_rows("MITRE Technique", iocs.get("mitre_counter", Counter()), limit=5)

    if len(rows) == 1:
        return None

    table = Table(rows, colWidths=[100, 260, 70], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d6e2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_timeline_table(timeline_events, styles) -> Table:
    """Chronological attack timeline as a colored table - a real visual for the PDF."""
    if not timeline_events:
        return None
    cell_style = ParagraphStyle("TLCell", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK")
    rows = [["Time", "Event"]] + [[t, Paragraph(e, cell_style)] for t, e in timeline_events[:20]]
    table = Table(rows, colWidths=[70, 360], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f4c81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d6e2")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_pdf_report(sections, incident_description, file_names, iocs=None):
    """
    Builds a downloadable PDF incident report using ReportLab.
    Includes: Project Name, Date, Summary, Findings, Recommendations,
    an Incident Snapshot table, an IOC table, a native bar chart, an
    attack timeline table, and a Footer.
    """
    iocs = iocs or {}
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

    # ---- Incident Snapshot (from parsed report fields) ----
    kv_fields = iocs.get("kv_fields", {})
    snapshot_rows = []
    snapshot_cell_style = ParagraphStyle("SnapCell", parent=styles["BodyText"], fontSize=8, leading=10, wordWrap="CJK")
    for label, aliases in INTERESTING_FIELDS:
        value = find_field(kv_fields, aliases)
        if value:
            snapshot_rows.append([label, Paragraph(value, snapshot_cell_style)])
    if snapshot_rows:
        story.append(Paragraph("Incident Snapshot", heading_style))
        snap_table = Table(snapshot_rows, colWidths=[150, 280])
        snap_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d6e2")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(snap_table)
        story.append(Spacer(1, 10))

    # ---- Summary ----
    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(sections.get("Incident Summary", "Not available."), body_style))

    # ---- Findings ----
    story.append(Paragraph("Findings", heading_style))
    findings_map = ["Attack Type", "Severity", "Root Cause", "Explanation"]
    for label in findings_map:
        story.append(Paragraph(f"<b>{label}:</b> {sections.get(label, 'Not available.')}", body_style))
        story.append(Spacer(1, 4))

    # ---- Indicators of Compromise (table visual) ----
    ioc_table = build_ioc_table(iocs, styles)
    if ioc_table:
        story.append(Paragraph("Indicators of Compromise", heading_style))
        story.append(ioc_table)
        story.append(Spacer(1, 10))
        story.append(Paragraph("IOC Category Counts", heading_style))
        story.append(build_ioc_bar_chart(iocs))
        story.append(Spacer(1, 6))

    # ---- Attack Timeline (table visual) ----
    timeline_table = build_timeline_table(iocs.get("timeline", []), styles)
    if timeline_table:
        story.append(Paragraph("Attack Timeline", heading_style))
        story.append(timeline_table)
        story.append(Spacer(1, 10))

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


#========HERO HEADER + LIVE STATUS BAR====================
inject_custom_css()

st.markdown(
    """
    <div class="cg-hero">
        <h1>🛡️ CyberGuardian AI</h1>
        <p>Agentic AI based Cyber Security Incident Response Assistant — reads your company's
        live security reports/logs and helps you detect, understand and respond to incidents faster.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3, status_col4 = st.columns([1.4, 1, 1, 1])
with status_col1:
    render_status_badge(st.session_state.threat_level, size="large")
with status_col2:
    render_metric_box("KB Documents", len(st.session_state.uploaded_file_names))
with status_col3:
    render_metric_box("Incidents Analyzed", len(st.session_state.analysis_history))
with status_col4:
    render_metric_box("Expert Q&A", len(st.session_state.qa_history))

st.write("")

#========MODE SELECTION (TABS)====================
tab_dashboard, tab_analysis, tab_chat, tab_kb = st.tabs(
    ["📊 Dashboard", "🔍 Incident Analysis", "💬 Chat with Expert", "📂 Knowledge Base"]
)


#========TAB: DASHBOARD====================
with tab_dashboard:
    st.subheader("📊 Security Operations Overview")

    if not st.session_state.uploaded_file_names:
        st.info(
            "No knowledge base loaded yet. Head to the **📂 Knowledge Base** tab and upload "
            "your company's security reports/logs to activate live monitoring."
        )
    else:
        top_left, top_right = st.columns([1.3, 1])

        with top_left:
            st.markdown("**Current Threat Status**")
            render_status_badge(st.session_state.threat_level, size="large")
            if st.session_state.analysis_result:
                st.write("")
                st.markdown(f"**Last Attack Type:** {st.session_state.analysis_result.get('Attack Type', 'N/A')}")
                st.markdown(f"**Last Severity:** {st.session_state.analysis_result.get('Severity', 'N/A')}")
                st.markdown(f"**Summary:** {st.session_state.analysis_result.get('Incident Summary', 'N/A')}")
            else:
                st.caption("No incidents analyzed yet — knowledge base is loaded and monitored, no active findings.")

        with top_right:
            st.markdown("**Knowledge Base Status**")
            render_metric_box("Files Loaded", len(st.session_state.uploaded_file_names))
            st.write("")
            render_metric_box("Chunks Indexed", st.session_state.kb_chunk_count)
            st.write("")
            if st.session_state.kb_last_updated:
                st.caption(f"🕒 Last updated: {st.session_state.kb_last_updated.strftime('%d %b %Y, %H:%M')}")

    st.divider()
    st.markdown("**🧾 Incident Snapshot** _(key fields parsed from your uploaded report)_")
    snapshot_shown = render_incident_snapshot(st.session_state.iocs.get("kv_fields", {})) if st.session_state.iocs else False
    if not snapshot_shown:
        st.caption("Upload a structured report (with labeled fields like Hostname, Source IP, Country...) to populate this.")

    st.divider()
    st.markdown("**🛰️ Live Threat Telemetry** _(extracted directly from your knowledge base — no extra API calls)_")

    iocs = st.session_state.iocs
    if not iocs or not iocs.get("ip_counter") and not iocs.get("cve_counter") and not iocs.get("hash_counter"):
        st.caption("No network indicators (IPs, CVEs, hashes, domains) detected in the current knowledge base yet.")
    else:
        t1, t2, t3, t4, t5 = st.columns(5)
        with t1:
            render_metric_box("Unique IPs", len(iocs["ip_counter"]))
        with t2:
            render_metric_box("External IPs", len(iocs["external_ips"]))
        with t3:
            render_metric_box("File Hashes", len(iocs["hash_counter"]))
        with t4:
            render_metric_box("CVEs Referenced", len(iocs["cve_counter"]))
        with t5:
            render_metric_box("Domains / URLs", len(iocs["url_counter"]))

        st.write("")
        ip_col, ioc_col = st.columns([1.3, 1])

        with ip_col:
            st.markdown("**Top IP Addresses (by mentions in the knowledge base)**")
            top_ips = iocs["ip_counter"].most_common(10)
            if top_ips:
                df_ips = pd.DataFrame(top_ips, columns=["IP Address", "Mentions"])
                df_ips["Scope"] = df_ips["IP Address"].apply(
                    lambda ip: "🌐 External" if ip in iocs["external_ips"] else "🏠 Internal"
                )
                st.dataframe(df_ips, hide_index=True, use_container_width=True)
                st.bar_chart(df_ips.set_index("IP Address")["Mentions"])
            else:
                st.caption("No IP addresses found in the knowledge base text.")

        with ioc_col:
            st.markdown("**CVEs Referenced**")
            if iocs["cve_counter"]:
                for cve, count in iocs["cve_counter"].most_common(8):
                    st.markdown(f"- 🧬 `{cve}` — {count}x")
            else:
                st.caption("No CVE identifiers found.")

            st.markdown("**File Hashes (potential IOCs)**")
            if iocs["hash_counter"]:
                for h, _ in list(iocs["hash_counter"].items())[:5]:
                    st.code(h, language=None)
            else:
                st.caption("No file hashes found.")

            if iocs["port_counter"]:
                st.markdown("**Ports Referenced**")
                st.write(", ".join(f"`{p}`" for p, _ in iocs["port_counter"].most_common(10)))

        if iocs.get("mitre_counter"):
            st.write("")
            st.markdown("**🎯 MITRE ATT&CK Techniques Observed**")
            render_mitre_chips(iocs["mitre_counter"])

        if iocs.get("timeline"):
            st.write("")
            st.markdown("**🕓 Attack Timeline** _(chronological events found in the knowledge base)_")
            with st.container(border=True):
                render_attack_timeline(iocs["timeline"])

    st.divider()
    st.markdown("**🕘 Recent Incident Analyses**")
    if not st.session_state.analysis_history:
        st.caption("Nothing analyzed yet. Run an analysis from the 🔍 Incident Analysis tab.")
    else:
        for entry in reversed(st.session_state.analysis_history[-8:]):
            level = THREAT_LEVELS.get(entry["level"], THREAT_LEVELS["UNSCANNED"])
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.markdown(f"**{entry['attack_type']}** — {entry['summary']}")
                    st.caption(f"🕒 {entry['time'].strftime('%d %b %Y, %H:%M')}")
                with c2:
                    st.markdown(f"{level['emoji']} {level['label']}")


#========TAB: INCIDENT ANALYSIS====================
with tab_analysis:
    st.subheader("🔍 Run Incident Analysis")

    if not st.session_state.retriever:
        st.warning("⚠️ No knowledge base loaded. Upload your reports/logs in the **📂 Knowledge Base** tab first.")
    else:
        st.caption(f"Analyzing against {len(st.session_state.uploaded_file_names)} loaded document(s).")

        incident_description = st.text_area(
            "Describe the incident (optional)",
            placeholder="e.g. Multiple failed login attempts followed by a successful "
            "login from an unfamiliar IP address...",
            help="Leave blank to let CyberGuardian AI scan the whole knowledge base for incidents.",
        )

        analyze_clicked = st.button("🔍 Analyze Incident", type="primary", use_container_width=True)

        if analyze_clicked:
            if not chat_key or not embedding_key:
                st.error("Please enter the required API key(s) in the sidebar first.")
            else:
                query = incident_description.strip() or (
                    "Analyze this document as a whole for any security incidents, "
                    "attacks, or suspicious activity described in it."
                )
                st.session_state.incident_description = query

                with st.spinner("Running technical analysis..."):
                    llm = get_llm(llm_provider, chat_key, model_name)
                    context = get_context(st.session_state.retriever, query)
                    raw_analysis = analyze_incident(query, context, llm)
                    st.session_state.raw_analysis_text = raw_analysis

                with st.spinner("Structuring the final report..."):
                    structured_report = generate_structured_report(query, raw_analysis, llm)
                    sections = parse_analysis_output(structured_report)
                    st.session_state.analysis_result = sections

                # ---- Update dashboard / threat-level state ----
                level_key = detect_threat_level(sections.get("Severity", ""))
                st.session_state.threat_level = level_key
                st.session_state.analysis_history.append({
                    "time": datetime.now(),
                    "attack_type": sections.get("Attack Type", "Unknown"),
                    "severity": sections.get("Severity", "Unknown"),
                    "summary": sections.get("Incident Summary", ""),
                    "level": level_key,
                })

                st.success("Analysis complete! Scroll down to see the results. ⬇️")

        # ---- Results Cards ----
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

            # ---- Download PDF Button ----
            pdf_bytes = build_pdf_report(
                result,
                st.session_state.incident_description,
                st.session_state.uploaded_file_names,
                st.session_state.iocs,
            )
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="cyberguardian_incident_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


#========TAB: CHAT WITH EXPERT====================
with tab_chat:
    st.subheader("💬 Chat with Expert")
    st.caption("Ask free-form questions grounded in your company's live knowledge base.")

    if not st.session_state.retriever:
        st.warning("⚠️ No knowledge base loaded. Upload your reports/logs in the **📂 Knowledge Base** tab first.")
    else:
        # ---- Chat-style history ----
        for past_question, past_answer in st.session_state.qa_history:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(past_question)
            with st.chat_message("assistant", avatar="🛡️"):
                st.write(past_answer)

        question = st.chat_input("Ask the expert anything about your knowledge base...")

        if question:
            if not chat_key:
                st.error("Please enter the required API key in the sidebar first.")
            else:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.write(question)
                with st.chat_message("assistant", avatar="🛡️"):
                    with st.spinner("Thinking..."):
                        llm = get_llm(llm_provider, chat_key, model_name)
                        context = get_context(st.session_state.retriever, question)
                        answer = answer_question(question, context, llm)
                        st.write(answer)
                st.session_state.qa_history.append((question, answer))


#========TAB: KNOWLEDGE BASE====================
with tab_kb:
    st.subheader("📂 Knowledge Base (Company Live Data)")
    st.caption(
        "Upload your organization's security reports, SIEM exports, or log files. "
        "CyberGuardian AI indexes them so the Dashboard, Incident Analysis and "
        "Chat with Expert tabs all stay grounded in this data."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    build_clicked = st.button("⚙️ Build / Update Knowledge Base", type="primary", use_container_width=True)

    if build_clicked:
        if not uploaded_files:
            st.error("Please select at least one PDF or TXT file first.")
        elif not embedding_key:
            st.error("Please enter the required API key(s) in the sidebar first.")
        else:
            with st.spinner("Reading document(s) and building the search index..."):
                documents = load_all_documents(uploaded_files)
                chunks = split_documents(documents, chunk_size, chunk_overlap)
                vector_store = build_vector_store(chunks, embedding_key)
                retriever = get_retriever(vector_store, top_k)

                # Extract real IOCs (IPs, domains, hashes, CVEs...) locally -
                # no LLM call needed, so this is instant and free.
                full_text = "\n".join(doc.page_content for doc in documents)
                iocs = extract_iocs(full_text)

                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
                st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                st.session_state.kb_chunk_count = len(chunks)
                st.session_state.kb_last_updated = datetime.now()
                st.session_state.iocs = iocs

            st.success(
                f"✅ Knowledge base built from {len(uploaded_files)} file(s) — {len(chunks)} chunks indexed. "
                f"Detected {len(iocs['ip_counter'])} unique IP(s), {len(iocs['cve_counter'])} CVE(s), "
                f"{len(iocs['hash_counter'])} file hash(es)."
            )

    st.divider()
    st.markdown("**Current Knowledge Base**")
    if st.session_state.uploaded_file_names:
        for name in st.session_state.uploaded_file_names:
            st.markdown(
                f"""<div class="cg-file-card">
                        <span class="cg-file-icon">📄</span>
                        <span class="cg-file-name">{name}</span>
                    </div>""",
                unsafe_allow_html=True,
            )

        kb_m1, kb_m2 = st.columns(2)
        with kb_m1:
            render_metric_box("Chunks Indexed", st.session_state.kb_chunk_count)
        with kb_m2:
            render_metric_box("Documents Loaded", len(st.session_state.uploaded_file_names))

        st.write("")
        kb_summary_bytes = build_kb_summary_text(
            st.session_state.uploaded_file_names,
            st.session_state.kb_chunk_count,
            st.session_state.kb_last_updated,
            st.session_state.iocs,
        )
        st.download_button(
            label="📥 Download Knowledge Base Summary",
            data=kb_summary_bytes,
            file_name="cyberguardian_kb_summary.txt",
            mime="text/plain",
            use_container_width=True,
            help="Exports the indexed file list, chunk count, and detected indicators as a text file.",
        )
    else:
        st.caption("No knowledge base loaded yet.")
