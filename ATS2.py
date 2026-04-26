"""
app.py - Streamlit Frontend for ATS Resume Parser

A modern, interactive web interface for the ATS Resume Parser.
Allows users to upload resumes, toggle semantic matching, and 
visualize the extracted structured data.

Run the app:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json

# Import our custom parser
from parser import ResumeParser

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ATS Resume Parser AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make skill tags look like badges
st.markdown("""
<style>
    .skill-badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 0.85em;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        background-color: #e0f2fe;
        color: #0369a1;
        margin: 2px;
    }
    .soft-skill-badge {
        background-color: #f3e8ff;
        color: #6b21a8;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STATE & RESOURCE MANAGEMENT
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_parser(use_semantic: bool) -> ResumeParser:
    """
    Load and cache the ResumeParser instance.
    This prevents the SentenceTransformer model from reloading on every interaction.
    """
    return ResumeParser(use_semantic=use_semantic)

# FIX 1: Initialize session state keys so results persist across reruns
if "parse_result" not in st.session_state:
    st.session_state.parse_result = None
if "parsed_filename" not in st.session_state:
    st.session_state.parsed_filename = None


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    use_semantic = st.toggle(
        "Enable Semantic Matching", 
        value=True,
        help="Uses AI (Sentence Transformers) to find skills contextually. Turn off for faster processing."
    )
    
    st.divider()
    st.markdown("### About")
    st.markdown(
        "This parser uses a combination of **Regex**, **spaCy NER**, "
        "and **Sentence Transformers** to extract highly structured data "
        "from unstructured PDF and DOCX files."
    )


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.title("📄 Intelligent ATS Resume Parser")
st.write("Upload a resume in PDF or DOCX format to instantly extract and structure its contents.")

# Load the parser based on sidebar toggle
with st.spinner("Initializing AI Models..."):
    parser = load_parser(use_semantic)

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    # Action Button
    if st.button("Extract Data", type="primary"):

        # FIX 2: Run parsing logic outside the spinner context before calling st.stop()
        # so the spinner doesn't hang on error
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name

        with st.spinner("Parsing document..."):
            result = parser.parse_file(
                file_bytes=file_bytes,
                filename=filename
            )

        # FIX 2 (cont): Check for errors AFTER the spinner context has closed
        if "error" in result.get("metadata", {}):
            st.error(f"Failed to parse resume: {result['metadata']['error']}")
            st.stop()

        # FIX 1 (cont): Store result in session state so it survives reruns
        st.session_state.parse_result = result
        st.session_state.parsed_filename = filename
        st.success("Parsing complete!")


# ─────────────────────────────────────────────────────────────
# DISPLAY RESULTS (read from session state, not from button scope)
# ─────────────────────────────────────────────────────────────
# FIX 1 (cont): Render results from session_state so they persist
# across reruns (e.g. toggling sidebar, scrolling, etc.)
if st.session_state.parse_result is not None:
    result = st.session_state.parse_result
    filename = st.session_state.parsed_filename

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "💼 Experience", "🎓 Education", "🛠️ Skills", "⚙️ Raw JSON"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Candidate Information")
            cand = result.get("candidate", {})
            st.markdown(f"**Name:** {cand.get('name') or '*Not found*'}")

            contact = cand.get("contact", {})
            st.markdown(f"**Email:** {', '.join(contact.get('emails', [])) or '*Not found*'}")
            st.markdown(f"**Phone:** {', '.join(contact.get('phones', [])) or '*Not found*'}")

            if contact.get('linkedin'):
                st.markdown(f"**LinkedIn:** [{contact['linkedin']}]({contact['linkedin']})")
            if contact.get('github'):
                st.markdown(f"**GitHub:** [{contact['github']}]({contact['github']})")

        with col2:
            score = result.get("ats_score", 0)
            st.metric(
                label="ATS Compatibility Score",
                value=f"{score}/100",
                help="Score based on formatting, skill count, and section completeness."
            )
            st.progress(score / 100)

        st.divider()
        st.subheader("Professional Summary")
        st.write(result.get("summary") or "*No summary section detected.*")

    # --- TAB 2: EXPERIENCE ---
    with tab2:
        experiences = result.get("experience", [])
        if not experiences:
            st.info("No work experience found.")
        else:
            for exp in experiences:
                with st.container():
                    col_title, col_date = st.columns([3, 1])
                    with col_title:
                        st.markdown(f"#### {exp.get('title', 'Unknown Role')}")
                        st.markdown(f"**{exp.get('company', 'Unknown Company')}**")
                    with col_date:
                        st.markdown(f"*{exp.get('dates', '')}*")

                    st.markdown(exp.get('description', ''))
                    st.divider()

    # --- TAB 3: EDUCATION ---
    with tab3:
        education = result.get("education", [])
        if not education:
            st.info("No education found.")
        else:
            for edu in education:
                st.markdown(f"#### {edu.get('institution', 'Unknown Institution')}")
                st.markdown(f"**Degree:** {edu.get('degree', 'Not specified')}")

                cols = st.columns(2)
                with cols[0]:
                    st.markdown(f"**Dates:** {edu.get('dates', 'Not specified')}")
                with cols[1]:
                    st.markdown(f"**GPA:** {edu.get('gpa', 'Not specified')}")
                st.divider()

    # --- TAB 4: SKILLS ---
    with tab4:
        skills_data = result.get("skills", {})

        st.subheader("Technical Skills")
        tech_skills = skills_data.get("technical", [])
        if tech_skills:
            html_tech = "".join([f'<span class="skill-badge">{s}</span>' for s in tech_skills])
            st.markdown(html_tech, unsafe_allow_html=True)
        else:
            st.write("None detected.")

        st.write("")  # Spacer

        st.subheader("Soft Skills")
        soft_skills = skills_data.get("soft", [])
        if soft_skills:
            html_soft = "".join([f'<span class="skill-badge soft-skill-badge">{s}</span>' for s in soft_skills])
            st.markdown(html_soft, unsafe_allow_html=True)
        else:
            st.write("None detected.")

    # --- TAB 5: RAW JSON ---
    with tab5:
        st.subheader("Parser Payload")
        st.json(result)

        # FIX 3: Clean filename for download (strip original extension before appending .json)
        base_name = filename.rsplit(".", 1)[0]  # e.g. "resume.pdf" → "resume"
        json_string = json.dumps(result, indent=2)
        st.download_button(
            label="Download JSON",
            file_name=f"{base_name}_parsed.json",   # e.g. "resume_parsed.json"
            mime="application/json",
            data=json_string
        )