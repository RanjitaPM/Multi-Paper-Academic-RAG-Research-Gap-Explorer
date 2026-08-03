import streamlit as st

class PaperAnalysisRAG:
    def __init__(self, groq_api_key):
        self.groq_api_key = groq_api_key
        self.parsed_papers = {}

    def ingest_papers(self, parsed_all):
        self.parsed_papers = parsed_all




from rag_system import PaperAnalysisRAG
import os
import streamlit as st

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & SESSION STATE INITIALIZATION
# ==============================================================================
st.set_page_config(
    page_title="PragyanAI Academic Paper RAG Engine",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-Paper Academic RAG & Research Gap Explorer")

# Persistent state management across reruns
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "paper_list" not in st.session_state:
    st.session_state.paper_list = []

if "status_msg" not in st.session_state:
    st.session_state.status_msg = "Upload PDF files and click 'Ingest Papers' to start."

# Placeholder default key if defined globally in your project
DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# ==============================================================================
# 2. LOGIC & HANDLER FUNCTIONS
# ==============================================================================
def initialize_rag(files, api_key):

    if not api_key:
        st.session_state.status_msg = "⚠️ Please enter a valid Groq API Key."
        return

    if not files:
        st.session_state.status_msg = "⚠️ Please upload at least one PDF paper."
        return

    try:
        # Initialize backend RAG system
        rag_system = PaperAnalysisRAG(
            groq_api_key=api_key
        )

        parsed_all = {}

        for file in files:
            paper_name = file.name

            # Extract sections from uploaded PDF
            sections = extract_structured_sections(file)

            parsed_all[paper_name] = sections

        # Ingest papers into RAG
        rag_system.ingest_papers(parsed_all)

        # Save in Streamlit session
        st.session_state.rag_system = rag_system
        st.session_state.paper_list = list(parsed_all.keys())

        st.session_state.status_msg = (
            f"✅ Processed {len(files)} paper(s). "
            "Vector store ready!"
        )

    except Exception as e:
        st.session_state.status_msg = (
            f"❌ RAG initialization failed: "
            f"{type(e).__name__}: {str(e)}"
        )

def extract_sections_ui(paper_name, sections):
    if not st.session_state.rag_system:
        return "⚠️ Please initialize the system first."
    return st.session_state.rag_system.extract_specific_sections(paper_name, sections)

def run_comparison(aspect):
    if not st.session_state.rag_system:
        return "⚠️ Please initialize the system first."
    return st.session_state.rag_system.generate_comparative_matrix(aspect)

def run_gap_analysis():
    if not st.session_state.rag_system:
        return "⚠️ Please initialize the system first."
    return st.session_state.rag_system.identify_research_gaps()

def answer_query(query, section_filter):
    if not st.session_state.rag_system:
        return "⚠️ Please initialize the system first."
    return st.session_state.rag_system.query_rag(query, section_filter)

def explore_external_papers(topic):
    if not topic:
        return "⚠️ Please enter a search topic."
    
    arxiv_res = search_arxiv_papers(topic)
    web_res = search_similar_online_papers(topic)

    out = "### Relevant arXiv Papers\n"
    for p in arxiv_res:
        out += f"- **[{p['title']}]({p['url']})** ({p['published']})\n  *{p['summary']}*\n\n"

    out += "\n### Related Web & OpenReview Papers\n"
    for w in web_res:
        out += f"- **[{w['title']}]({w['link']})**\n  {w['snippet']}\n\n"
    return out

# ==============================================================================
# 3. TOP CONTROLS & INGESTION
# ==============================================================================
col1, col2 = st.columns([1, 2])

with col1:
    api_key_input = st.text_input(
        label="Groq API Key",
        type="password",
        value=DEFAULT_GROQ_KEY,
        placeholder="gsk_..."
    )

with col2:
    file_uploader = st.file_uploader(
        label="Upload PDF Papers",
        type=["pdf"],
        accept_multiple_files=True
    )

if st.button("🚀 Ingest Papers", type="primary", use_container_width=True):
    with st.spinner("Ingesting and vectorizing papers..."):
        initialize_rag(file_uploader, api_key_input)

# Display System Status
if "✅" in st.session_state.status_msg:
    st.success(st.session_state.status_msg)
elif "⚠️" in st.session_state.status_msg:
    st.warning(st.session_state.status_msg)
else:
    st.info(st.session_state.status_msg)

st.markdown("---")

# ==============================================================================
# 4. STREAMLIT TABS UI
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Extract Sections",
    "2. Comparative Analysis",
    "3. Research Gap Finder",
    "4. Deep Q&A (RAG)",
    "5. Discover Similar Papers"
])

# ------------------------------------------------------------------------------
# TAB 1: Extract Sections
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Extract Specific Paper Sections")
    col_p, col_s = st.columns([1, 2])
    
    with col_p:
        paper_dropdown = st.selectbox(
            label="Select Paper",
            options=st.session_state.paper_list,
            disabled=len(st.session_state.paper_list) == 0
        )
    with col_s:
        section_selector = st.multiselect(
            label="Select Sections to Inspect",
            options=["Abstract", "Introduction", "Related Work", "Methodology", "Results", "Discussion & Gaps", "Conclusion"],
            default=["Abstract", "Methodology", "Results"]
        )

    if st.button("Extract Selected Sections"):
        if paper_dropdown:
            with st.spinner("Extracting..."):
                result = extract_sections_ui(paper_dropdown, section_selector)
                st.markdown(result)
        else:
            st.warning("Please upload and ingest papers first.")

# ------------------------------------------------------------------------------
# TAB 2: Comparative Analysis
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Cross-Paper Comparative Analysis")
    aspect_input = st.text_input(
        label="Comparison Focus",
        value="Methodology, Datasets, Models, and Results"
    )
    if st.button("Generate Comparison Table", type="primary"):
        with st.spinner("Generating matrix..."):
            comparison_res = run_comparison(aspect_input)
            st.markdown(comparison_res)

# ------------------------------------------------------------------------------
# TAB 3: Research Gap Finder
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Research Gap & Future Work Identifier")
    if st.button("Analyze Gaps & Future Work", type="primary"):
        with st.spinner("Identifying research gaps..."):
            gap_res = run_gap_analysis()
            st.markdown(gap_res)

# ------------------------------------------------------------------------------
# TAB 4: Deep Q&A (RAG)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Context-Aware RAG Engine")
    query_input = st.text_input(
        label="Enter Question",
        placeholder="What are the main architectural limitations mentioned?"
    )
    section_filter = st.selectbox(
        label="Filter Context by Section",
        options=["All", "Abstract", "Introduction", "Methodology", "Results", "Discussion & Gaps"],
        index=0
    )
    if st.button("Ask RAG Engine"):
        if query_input:
            with st.spinner("Querying system..."):
                qa_res = answer_query(query_input, section_filter)
                st.markdown(qa_res)
        else:
            st.warning("Please enter a question.")

# ------------------------------------------------------------------------------
# TAB 5: Discover Similar Papers
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("External Literature Discovery")
    topic_input = st.text_input(
        label="Search Query / Research Topic",
        placeholder="Multi-agent systems for PCB design"
    )
    if st.button("Find Similar Papers (arXiv + Web)"):
        with st.spinner("Searching arXiv & Web..."):
            discovery_res = explore_external_papers(topic_input)
            st.markdown(discovery_res)
