import os
import tempfile
import streamlit as st

# ============================================================
# LangChain & Vector Store Imports
# ============================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PragyanAI Academic Paper RAG Engine",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# DEFAULT GROQ KEY
# ============================================================

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")


# ============================================================
# SESSION STATE
# ============================================================

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "paper_list" not in st.session_state:
    st.session_state.paper_list = []

if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def extract_structured_sections(file_path):
    """
    Parses a PDF file and extracts content by basic academic sections.
    """

    sections = {
        "Abstract": "",
        "Introduction": "",
        "Related Work": "",
        "Methodology": "",
        "Results": "",
        "Discussion & Gaps": "",
        "Conclusion": "",
        "Full Text": ""
    }

    if not os.path.exists(file_path):
        return sections

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    full_text = "\n".join(
        [doc.page_content for doc in docs]
    )

    sections["Full Text"] = full_text

    # Basic chunk segmentation
    sections["Abstract"] = full_text[:1500]

    sections["Methodology"] = (
        full_text[1500:5000]
        if len(full_text) > 5000
        else full_text
    )

    sections["Results"] = (
        full_text[5000:9000]
        if len(full_text) > 9000
        else full_text
    )

    sections["Discussion & Gaps"] = (
        full_text[9000:]
        if len(full_text) > 9000
        else full_text
    )

    return sections


# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def search_arxiv_papers(topic):
    """
    Placeholder arXiv search.
    """

    return [
        {
            "title": f"Advances in {topic.title()}: A Comprehensive Survey",
            "url": "https://arxiv.org",
            "published": "2024",
            "summary": (
                f"Recent developments and benchmark analysis "
                f"in {topic}."
            )
        },
        {
            "title": f"Scalable Models for {topic.title()}",
            "url": "https://arxiv.org",
            "published": "2025",
            "summary": (
                f"A novel framework for high-throughput "
                f"execution in {topic}."
            )
        }
    ]


def search_similar_online_papers(topic):
    """
    Placeholder Web/OpenReview search.
    """

    return [
        {
            "title": f"Benchmarking {topic.title()} in Open Environments",
            "link": "https://openreview.net",
            "snippet": (
                f"Empirical studies comparing state-of-the-art "
                f"implementations of {topic}."
            )
        }
    ]


# ============================================================
# 2. PAPER ANALYSIS RAG CLASS
# ============================================================

class PaperAnalysisRAG:
    """
    Core RAG engine managing embeddings,
    FAISS vector store and Groq LLM.
    """

    def __init__(self, groq_api_key: str):

        self.groq_api_key = groq_api_key

        # Groq LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )

        # HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = None

        self.parsed_papers = {}


    # --------------------------------------------------------
    # INGEST PAPERS
    # --------------------------------------------------------

    def ingest_papers(self, parsed_all: dict):

        self.parsed_papers = parsed_all

        documents = []

        for paper_name, sections in parsed_all.items():

            for sec_name, sec_content in sections.items():

                if sec_content and sec_name != "Full Text":

                    documents.append(
                        Document(
                            page_content=sec_content,
                            metadata={
                                "paper": paper_name,
                                "section": sec_name
                            }
                        )
                    )

        if documents:

            self.vectorstore = FAISS.from_documents(
                documents,
                self.embeddings
            )


    # --------------------------------------------------------
    # EXTRACT SPECIFIC SECTIONS
    # --------------------------------------------------------

    def extract_specific_sections(
        self,
        paper_name: str,
        sections: list
    ) -> str:

        if paper_name not in self.parsed_papers:
            return "Paper not found."

        paper_data = self.parsed_papers[paper_name]

        output = f"## 📄 {paper_name}\n\n"

        for sec in sections:

            content = paper_data.get(
                sec,
                "Section not explicitly extracted."
            )

            output += (
                f"### {sec}\n"
                f"{content[:1200]}\n\n"
                f"---\n"
            )

        return output


    # --------------------------------------------------------
    # COMPARATIVE MATRIX
    # --------------------------------------------------------

    def generate_comparative_matrix(
        self,
        aspect: str
    ) -> str:

        if not self.parsed_papers:
            return "No papers ingested."

        context = ""

        for paper, sections in self.parsed_papers.items():

            context += (
                f"\n--- PAPER: {paper} ---\n"
                f"{sections['Full Text'][:3000]}\n"
            )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are an expert academic research assistant.

                Create a comparative Markdown matrix
                evaluating the given research papers
                based on the requested aspects.
                """
            ),
            (
                "human",
                """
                Aspects to compare:
                {aspect}

                Papers Context:
                {context}
                """
            )
        ])

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({
            "aspect": aspect,
            "context": context
        })


    # --------------------------------------------------------
    # RESEARCH GAP ANALYSIS
    # --------------------------------------------------------

    def identify_research_gaps(self) -> str:

        if not self.parsed_papers:
            return "No papers ingested."

        context = ""

        for paper, sections in self.parsed_papers.items():

            context += (
                f"\n--- PAPER: {paper} ---\n"
                f"{sections.get('Discussion & Gaps', '')}\n"
            )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                You are a senior academic reviewer.

                Identify major research gaps,
                limitations, and future directions
                from these papers.
                """
            ),
            (
                "human",
                """
                Papers Context:

                {context}
                """
            )
        ])

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({
            "context": context
        })


    # --------------------------------------------------------
    # RAG QUESTION ANSWERING
    # --------------------------------------------------------

    def query_rag(
        self,
        query: str,
        section_filter: str
    ) -> str:

        if not self.vectorstore:
            return "Vector store is not initialized."

        # Retrieve documents
        docs = self.vectorstore.similarity_search(
            query,
            k=4
        )

        # Apply section filtering manually
        if section_filter != "All":

            docs = [
                doc for doc in docs
                if doc.metadata.get("section") == section_filter
            ]

        if not docs:
            return (
                "No relevant information found "
                "for the selected section."
            )

        retrieved_text = "\n\n".join(
            [
                (
                    f"[{d.metadata.get('paper')} - "
                    f"{d.metadata.get('section')}]\n"
                    f"{d.page_content}"
                )
                for d in docs
            ]
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Answer the user's question accurately
                using ONLY the retrieved paper context.

                Cite the paper name where applicable.

                If the answer is not available in the
                retrieved context, say so clearly.
                """
            ),
            (
                "human",
                """
                Context:

                {context}

                Question:
                {question}
                """
            )
        ])

        chain = (
            prompt
            | self.llm
            | StrOutputParser()
        )

        return chain.invoke({
            "context": retrieved_text,
            "question": query
        })


# ============================================================
# 3. INITIALIZE RAG
# ============================================================

def initialize_rag(files, api_key):

    if not api_key:

        st.session_state.status_msg = (
            "⚠️ Please enter a valid Groq API Key."
        )

        return

    if not files:

        st.session_state.status_msg = (
            "⚠️ Please upload at least one PDF paper."
        )

        return

    try:

        # Create RAG system
        rag_system = PaperAnalysisRAG(
            groq_api_key=api_key
        )

        parsed_all = {}

        progress = st.progress(0)

        for i, file in enumerate(files):

            paper_name = file.name

            # Save uploaded Streamlit file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(file.getbuffer())
                temp_path = tmp.name

            try:

                # Extract PDF sections
                sections = extract_structured_sections(
                    temp_path
                )

                parsed_all[paper_name] = sections

            finally:

                # Remove temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            progress.progress(
                (i + 1) / len(files)
            )

        # Ingest papers into FAISS
        rag_system.ingest_papers(parsed_all)

        # Save to session
        st.session_state.rag_system = rag_system

        st.session_state.paper_list = list(
            parsed_all.keys()
        )

        st.session_state.status_msg = (
            f"✅ Processed {len(files)} paper(s). "
            "Vector store ready!"
        )

    except Exception as e:

        st.session_state.status_msg = (
            f"❌ RAG initialization failed: "
            f"{type(e).__name__}: {str(e)}"
        )


# ============================================================
# 4. PAGE HEADER
# ============================================================

st.title(
    "📚 Multi-Paper Academic RAG & "
    "Research Gap Explorer"
)

st.caption(
    "Powered by Groq + LangChain + FAISS + "
    "HuggingFace Embeddings"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=DEFAULT_GROQ_KEY,
        placeholder="gsk_..."
    )

    st.markdown("---")

    st.info(
        """
        Upload one or more research papers
        and initialize the RAG engine.

        The system can then:

        • Extract paper sections
        • Compare papers
        • Find research gaps
        • Answer questions using RAG
        • Discover similar papers
        """
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.subheader("📄 Upload Research Papers")

uploaded_files = st.file_uploader(
    "Upload PDF Papers",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# INGEST BUTTON
# ============================================================

if st.button(
    "🚀 Ingest Papers",
    type="primary",
    use_container_width=True
):

    initialize_rag(
        uploaded_files,
        api_key_input
    )


# ============================================================
# STATUS
# ============================================================

if st.session_state.status_msg:

    st.info(
        st.session_state.status_msg
    )


# ============================================================
# CHECK RAG STATUS
# ============================================================

rag_system = st.session_state.rag_system


if rag_system:

    st.success(
        f"RAG system ready with "
        f"{len(st.session_state.paper_list)} paper(s)."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Extract Sections",
    "2. Comparative Analysis",
    "3. Research Gap Finder",
    "4. Deep Q&A (RAG)",
    "5. Discover Similar Papers"
])


# ============================================================
# TAB 1 - EXTRACT SECTIONS
# ============================================================

with tab1:

    st.header("📑 Extract Paper Sections")

    if not rag_system:

        st.warning(
            "Please upload papers and click "
            "**Ingest Papers** first."
        )

    else:

        selected_paper = st.selectbox(
            "Select Paper",
            st.session_state.paper_list
        )

        selected_sections = st.multiselect(
            "Select Sections to Inspect",
            [
                "Abstract",
                "Introduction",
                "Related Work",
                "Methodology",
                "Results",
                "Discussion & Gaps",
                "Conclusion"
            ],
            default=[
                "Abstract",
                "Methodology",
                "Results"
            ]
        )

        if st.button(
            "🔍 Extract Selected Sections",
            key="extract_sections"
        ):

            if not selected_sections:

                st.warning(
                    "Please select at least one section."
                )

            else:

                result = rag_system.extract_specific_sections(
                    selected_paper,
                    selected_sections
                )

                st.markdown(result)


# ============================================================
# TAB 2 - COMPARATIVE ANALYSIS
# ============================================================

with tab2:

    st.header("📊 Comparative Analysis")

    if not rag_system:

        st.warning(
            "Please initialize the RAG system first."
        )

    else:

        aspect = st.text_input(
            "Comparison Focus",
            value=(
                "Methodology, Datasets, "
                "Models, and Results"
            )
        )

        if st.button(
            "📊 Generate Comparison Table",
            type="primary",
            key="comparison"
        ):

            with st.spinner(
                "Analyzing research papers..."
            ):

                result = (
                    rag_system
                    .generate_comparative_matrix(
                        aspect
                    )
                )

            st.markdown(result)


# ============================================================
# TAB 3 - RESEARCH GAP FINDER
# ============================================================

with tab3:

    st.header("🔎 Research Gap Finder")

    st.write(
        "Identify limitations, research gaps "
        "and future research directions."
    )

    if not rag_system:

        st.warning(
            "Please initialize the RAG system first."
        )

    else:

        if st.button(
            "🧠 Analyze Gaps & Future Work",
            type="primary",
            key="gap_analysis"
        ):

            with st.spinner(
                "Analyzing research gaps..."
            ):

                result = (
                    rag_system
                    .identify_research_gaps()
                )

            st.markdown(result)


# ============================================================
# TAB 4 - DEEP Q&A
# ============================================================

with tab4:

    st.header("🤖 Deep Q&A (RAG)")

    if not rag_system:

        st.warning(
            "Please initialize the RAG system first."
        )

    else:

        query = st.text_area(
            "Enter Question",
            placeholder=(
                "What are the main architectural "
                "limitations mentioned?"
            )
        )

        section_filter = st.selectbox(
            "Filter Context by Section",
            [
                "All",
                "Abstract",
                "Introduction",
                "Methodology",
                "Results",
                "Discussion & Gaps"
            ]
        )

        if st.button(
            "💬 Ask RAG Engine",
            type="primary",
            key="rag_question"
        ):

            if not query.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching papers and generating answer..."
                ):

                    answer = rag_system.query_rag(
                        query,
                        section_filter
                    )

                st.markdown("### Answer")

                st.markdown(answer)


# ============================================================
# TAB 5 - DISCOVER SIMILAR PAPERS
# ============================================================

with tab5:

    st.header("🌐 Discover Similar Papers")

    topic = st.text_input(
        "Search Query / Research Topic",
        placeholder=(
            "Multi-agent systems for PCB design"
        )
    )

    if st.button(
        "🔎 Find Similar Papers",
        type="primary",
        key="discover"
    ):

        if not topic.strip():

            st.warning(
                "Please enter a research topic."
            )

        else:

            with st.spinner(
                "Searching for related papers..."
            ):

                arxiv_res = search_arxiv_papers(
                    topic
                )

                web_res = search_similar_online_papers(
                    topic
                )

            st.subheader(
                "📚 Relevant arXiv Papers"
            )

            for p in arxiv_res:

                st.markdown(
                    f"""
                    **[{p['title']}]({p['url']})**
                    ({p['published']})

                    *{p['summary']}*
                    """
                )

            st.subheader(
                "🌐 Related Web & OpenReview Papers"
            )

            for w in web_res:

                st.markdown(
                    f"""
                    **[{w['title']}]({w['link']})**

                    {w['snippet']}
                    """
                )
