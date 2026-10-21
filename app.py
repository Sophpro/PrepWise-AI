from datetime import datetime
from io import BytesIO
import re
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import streamlit as st
from google import genai
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from xml.sax.saxutils import escape

from ai_service import (
    analyse_concept_understanding,
    analyse_job_description,
    assess_understanding_checkpoint,
    evaluate_student_response,
    fallback_hint,
    fallback_interview_question,
    generate_depth_question,
    generate_hint,
    generate_interview_question,
    is_help_seeking_response,
    reconcile_initial_checkpoint,
)


st.set_page_config(
    page_title="PrepWise AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Responsive styling
# --------------------------------------------------
# Streamlit already stacks st.columns vertically below ~640px. The CSS below
# limits line length on large screens, enlarges touch targets and tightens
# spacing on phones. Only static markup is injected; AI and learner text is
# always rendered through st.markdown without unsafe HTML.
st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: #f4f6fc;
}
[data-testid="stHeader"] {
    background: rgba(244, 246, 252, .88);
    backdrop-filter: blur(10px);
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e7e9f2;
}
[data-testid="stSidebarContent"] {
    padding-top: 1.1rem;
}
.block-container {
    max-width: 1120px;
    /* Keep the first heading below Streamlit's fixed top toolbar. */
    padding-top: 4.5rem;
    padding-bottom: 7rem;
}

/* Keep the fixed chat composer aligned with the readable content column. */
[data-testid="stBottom"] > div {
    width: calc(100% - 2rem);
    max-width: 1120px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 1rem;
    padding-right: 1rem;
}
[data-testid="stChatInput"] {
    width: 100%;
    max-width: 1088px;
    margin-left: auto;
    margin-right: auto;
    border: 1px solid #dfe3f0;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(58, 61, 89, .10);
    background: #ffffff;
}

/* Streamlit stretches a selectbox label across the full widget width and puts
   its help icon at the far edge. Keep the icon beside the label text so the
   relationship is visually clear. */
div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] {
    width: fit-content !important;
    justify-content: flex-start !important;
    gap: .32rem !important;
}
div[data-testid="stSelectbox"] [data-testid="stWidgetLabel"]
    [data-testid="stTooltipIcon"] {
    margin-left: 0 !important;
}

/* Application identity in the left navigation. */
.app-brand {
    display: flex;
    align-items: center;
    gap: .72rem;
    padding: .45rem .15rem 1rem .15rem;
}
.brand-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 12px;
    color: #ffffff;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    box-shadow: 0 8px 18px rgba(99, 102, 241, .28);
}
.brand-copy strong {
    display: block;
    color: #20243a;
    font-size: 1rem;
}
.brand-copy span {
    color: #858aa0;
    font-size: .76rem;
}
.side-label {
    margin: .25rem 0 .55rem 0;
    color: #989caf;
    font-size: .7rem;
    font-weight: 750;
    letter-spacing: .09em;
    text-transform: uppercase;
}

/* White dashboard surfaces inspired by the supplied reference layouts. */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e7e9f2 !important;
    border-radius: 18px !important;
    background: rgba(255, 255, 255, .94);
    box-shadow: 0 8px 28px rgba(48, 52, 80, .055);
}
[data-testid="stChatMessage"] {
    border: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
    margin-bottom: .65rem;
    padding: .55rem .2rem;
    align-items: flex-start;
}
/* Streamlit gives Markdown headings a large top margin. Inside a chat row this
   pushes the title below the avatar. Remove that margin only when the heading
   is the first element, preserving spacing for later feedback sections. */
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    padding-top: 0 !important;
}
/* In some Streamlit releases the content wrapper itself sits lower than the
   avatar even after heading margins are removed. Shift assistant content by
   half a rem so its first text line shares the avatar's visual centre. */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"][aria-label*="assistant" i]
    [data-testid="stChatMessageContent"] {
    transform: translateY(-0.5rem);
    margin-bottom: -0.5rem;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > h1:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > h2:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > h3:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] > h4:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
    line-height: 2.25rem;
}
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
    margin-top: 0 !important;
}
[data-testid="stChatMessage"][aria-label*="user" i],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(99, 102, 241, .07);
    border: 1px solid #dfe3ff;
    border-radius: 14px;
    padding: .7rem .8rem;
}
.conversation-title {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: .85rem 0 .55rem 0;
}
.conversation-title strong {
    color: #252941;
    font-size: 1.02rem;
}
.conversation-title span {
    color: #8c91a5;
    font-size: .78rem;
}
.workspace-heading {
    margin-bottom: .8rem;
}
.workspace-heading h1 {
    margin: 0;
    padding: 0;
    color: #20243a;
    font-size: 1.85rem;
}
.workspace-heading p {
    margin: .28rem 0 0 0;
    color: #7f8499;
}
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: .65rem;
    margin: 1rem 0 1.4rem 0;
}
.metric-card {
    padding: .7rem .8rem;
    border: 1px solid #e8eaf3;
    border-radius: 13px;
    background: #fafbff;
}
.metric-card span {
    display: block;
    color: #9296a8;
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .055em;
}
.metric-card strong {
    display: block;
    margin-top: .18rem;
    color: #30344d;
    font-size: .9rem;
    overflow-wrap: anywhere;
}

/* Vertical interview cycle in the left navigation. */
.cycle-item {
    display: flex;
    align-items: center;
    gap: .65rem;
    border-radius: 10px;
    padding: .62rem .7rem;
    margin: .3rem 0;
    background: rgba(128, 128, 128, .11);
    border: 1px solid rgba(128, 128, 128, .14);
    font-size: .9rem;
    font-weight: 600;
}
.cycle-item.done {
    color: #15803d;
    background: rgba(34, 197, 94, .14);
    border-color: rgba(34, 197, 94, .30);
}
.cycle-item.current {
    color: #ffffff;
    background: linear-gradient(135deg, #6366f1, #7c3aed);
    border-color: #6366f1;
    box-shadow: 0 7px 16px rgba(99, 102, 241, .20);
}
.cycle-dot {
    display: inline-flex;
    width: 1.55rem;
    height: 1.55rem;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    background: rgba(128, 128, 128, .15);
    flex: 0 0 auto;
}
.cycle-item.done .cycle-dot { background: rgba(34, 197, 94, .20); }
.cycle-item.current .cycle-dot { background: rgba(255, 255, 255, .20); }

/* Comfortable touch targets */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    min-height: 2.75rem;
    font-weight: 600;
    border-radius: 12px;
}
button[kind="primary"],
.stFormSubmitButton > button {
    color: #ffffff !important;
    border: 0 !important;
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    box-shadow: 0 7px 16px rgba(99, 102, 241, .18);
}
[data-baseweb="tab-list"] {
    gap: .35rem;
    padding: .25rem;
    border: 1px solid #e8eaf3;
    border-radius: 14px;
    background: #ffffff;
}
[data-baseweb="tab"] {
    border-radius: 10px;
    padding-left: 1rem;
    padding-right: 1rem;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #eef0ff;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.hero h1 {
    color: #ffffff;
    font-size: 2rem;
    line-height: 1.2;
    margin: 0 0 .35rem 0;
    padding: 0;
}
.hero p {
    margin: 0;
    opacity: .92;
    font-size: 1.02rem;
}
.hero .chips {
    margin-top: .9rem;
    display: flex;
    flex-wrap: wrap;
    gap: .4rem;
}
.hero .chip {
    background: rgba(255, 255, 255, .18);
    border-radius: 999px;
    padding: .2rem .7rem;
    font-size: .82rem;
}

/* Stage stepper */
.stepper {
    display: flex;
    gap: .35rem;
    margin: .25rem 0 .15rem 0;
}
.step {
    flex: 1 1 0;
    min-width: 0;
    text-align: center;
    font-size: .8rem;
    font-weight: 600;
    padding: .45rem .25rem;
    border-radius: 999px;
    background: rgba(128, 128, 128, .14);
    opacity: .75;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.step.done {
    background: rgba(34, 197, 94, .18);
    opacity: 1;
}
.step.current {
    background: #4f46e5;
    color: #ffffff;
    opacity: 1;
}

/* Phones */
@media (max-width: 640px) {
    .block-container {
        padding-left: .9rem;
        padding-right: .9rem;
        padding-top: 4rem;
    }
    [data-testid="stBottom"] > div {
        padding-left: .9rem;
        padding-right: .9rem;
    }
    .hero {
        padding: 1.1rem 1.15rem;
        border-radius: 14px;
    }
    .hero h1 { font-size: 1.45rem; }
    .hero p { font-size: .93rem; }
    .step { font-size: .7rem; padding: .4rem .1rem; }
    .step .num { display: none; }
    .metric-strip { grid-template-columns: 1fr; }
    .workspace-heading h1 { font-size: 1.45rem; }
    .conversation-title span { display: none; }
    /* Prevent iOS from zooming into text fields */
    textarea, input { font-size: 16px !important; }
    [data-testid="stChatMessage"] { padding: .6rem .4rem; }
}

/* Contrast fix: align Streamlit's native widgets with the light dashboard.
   Keep custom hero/chips, badges and primary-button contrast intact. */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
    color: #172033;
}
[data-testid="stAppViewContainer"] .stMarkdown h1,
[data-testid="stAppViewContainer"] .stMarkdown h2,
[data-testid="stAppViewContainer"] .stMarkdown h3,
[data-testid="stAppViewContainer"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: #20243a;
}
[data-testid="stAppViewContainer"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stAppViewContainer"] .stCaptionContainer,
[data-testid="stSidebar"] .stCaptionContainer,
[data-testid="stAppViewContainer"] .stWidgetLabel,
[data-testid="stSidebar"] .stWidgetLabel {
    color: #475569;
}
/* Restore intentionally light text inside the purple hero. */
[data-testid="stAppViewContainer"] .stMarkdown .hero,
[data-testid="stAppViewContainer"] .stMarkdown .hero h1,
[data-testid="stAppViewContainer"] .stMarkdown .hero p,
[data-testid="stAppViewContainer"] .stMarkdown .hero .chip {
    color: #ffffff;
}
/* Tabs: inactive labels must remain visible, too. */
button[data-baseweb="tab"],
button[data-baseweb="tab"] * {
    color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"][aria-selected="true"] * {
    color: #5b21b6 !important;
}
/* Input fields and text areas should not retain dark-theme surfaces. */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    caret-color: #172033;
}
[data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    background-color: #ffffff !important;
    border-color: #dfe3f0;
}
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder,
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
    -webkit-text-fill-color: #64748b !important;
}
/* Native non-primary buttons use dark text on a light background. */
.stButton > button:not([kind="primary"]),
.stDownloadButton > button:not([kind="primary"]) {
    color: #172033;
}
/* Preserve white text in selected cycle stages and brand logo. */
[data-testid="stSidebar"] .cycle-item.current,
[data-testid="stSidebar"] .brand-mark {
    color: #ffffff;
}


/* Contrast patch v2: prevent browser dark-mode preference / forced widget
   colours from producing pale tab labels or dark secondary buttons. */
[data-testid="stTabs"] [role="tab"],
[data-testid="stTabs"] [role="tab"] span,
[data-testid="stTabs"] [role="tab"] p,
[data-testid="stTabs"] [role="tab"] div {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [role="tab"][aria-selected="true"] span,
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p,
[data-testid="stTabs"] [role="tab"][aria-selected="true"] div {
    color: #5b21b6 !important;
    -webkit-text-fill-color: #5b21b6 !important;
}
/* Secondary buttons have their own light surface and clear dark labels. */
[data-testid="stButton"] button[kind="secondary"],
[data-testid="stButton"] button[kind="secondaryFormSubmit"],
[data-testid="stDownloadButton"] button[kind="secondary"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}
[data-testid="stButton"] button[kind="secondary"] *,
[data-testid="stButton"] button[kind="secondaryFormSubmit"] *,
[data-testid="stDownloadButton"] button[kind="secondary"] * {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    opacity: 1 !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #f3f0ff !important;
    border-color: #7c3aed !important;
}
/* Keep the purple CTA readable even if text colour is inherited. */
[data-testid="stButton"] button[kind="primary"] *,
[data-testid="stFormSubmitButton"] button[kind="primary"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}


/* Unified light surfaces and visible text for all native Streamlit widgets. */
:root, [data-testid="stApp"], [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], [data-testid="stBottom"] {
    color-scheme: light !important;
}
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {
    background: #f4f6fc !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] [data-baseweb="textarea"] {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
}
[data-testid="stChatInput"] button {
    background-color: #ede9fe !important;
    color: #5b21b6 !important;
}
[data-testid="stChatInput"] button:disabled { opacity: .55; }
/* Progress dashboard: the numeric value and label must be legible. */
[data-testid="stMetric"], [data-testid="stMetric"] *,
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #20243a !important;
    -webkit-text-fill-color: #20243a !important;
    opacity: 1 !important;
}
/* Expanders: dark-theme backgrounds previously hid header text. */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e1e5ef !important;
    border-radius: 12px;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    background-color: #ffffff !important;
    color: #26324a !important;
    -webkit-text-fill-color: #26324a !important;
    opacity: 1 !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #ffffff !important;
    color: #334155 !important;
}
/* Labels, including radio/segment selectors and help text. */
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
.stWidgetLabel, .stWidgetLabel * {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    opacity: 1 !important;
}
/* Select box: normalize the dark native input and dropdown menu. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] input {
    background: #ffffff !important;
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #172033 !important;
}
[role="listbox"], [role="listbox"] [role="option"] {
    background: #ffffff !important;
    color: #172033 !important;
}
[role="listbox"] [role="option"]:hover {
    background: #ede9fe !important;
}
/* Segmented controls use dark backgrounds with light text in browser dark mode. */
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] label,
[data-testid="stSegmentedControl"] [role="radiogroup"] label {
    background: #ffffff !important;
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    border-color: #cbd5e1 !important;
}
[data-testid="stSegmentedControl"] button *,
[data-testid="stSegmentedControl"] label * {
    color: inherit !important;
    -webkit-text-fill-color: currentColor !important;
}
[data-testid="stSegmentedControl"] [aria-checked="true"],
[data-testid="stSegmentedControl"] [data-checked="true"],
[data-testid="stSegmentedControl"] label:has(input:checked) {
    background: #ede9fe !important;
    color: #5b21b6 !important;
    border-color: #7c3aed !important;
}
/* Status badges: the default neutral badge was white on pale gray. */
[data-testid="stBadge"], [data-testid="stBadge"] * {
    color: #334155;
}
/* Use light background on scroll containers and custom dashboard cards. */
[data-testid="stVerticalBlockBorderWrapper"] {
    color: #172033 !important;
}


/* Final light-control palette: white surfaces, purple selected options. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 1px solid #d8dced !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover {
    border-color: #a78bfa !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"],
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    color: #20243a !important;
    fill: #475569 !important;
}
/* The segment group itself can retain a dark surface even if its buttons are white. */
[data-testid="stSegmentedControl"] [data-baseweb="button-group"],
[data-testid="stSegmentedControl"] [role="radiogroup"] {
    background: #ffffff !important;
    border-radius: 10px !important;
}
[data-testid="stSegmentedControl"] button,
[data-testid="stSegmentedControl"] [role="radio"] {
    background: #ffffff !important;
    color: #30344d !important;
    -webkit-text-fill-color: #30344d !important;
    border: 1px solid #dde2ef !important;
}
[data-testid="stSegmentedControl"] button[aria-pressed="true"],
[data-testid="stSegmentedControl"] button[aria-checked="true"],
[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"],
[data-testid="stSegmentedControl"] label:has(input:checked) {
    background: #f0e7ff !important;
    color: #6d28d9 !important;
    -webkit-text-fill-color: #6d28d9 !important;
    border-color: #a78bfa !important;
}
/* Topic pathway progress only (sidebar): a white empty track and purple fill.
   At 0 completed concepts, the fill is intentionally not visible. */
[data-testid="stSidebar"] [data-testid="stProgressBar"] [role="progressbar"],
[data-testid="stSidebar"] [data-testid="stProgressBar"] [data-baseweb="progress-bar"],
[data-testid="stSidebar"] [data-testid="stProgressBar"] [data-baseweb="progress-bar"] > div {
    background-color: #ffffff !important;
    border-radius: 999px !important;
}
[data-testid="stSidebar"] [data-testid="stProgressBar"] [role="progressbar"] > div,
[data-testid="stSidebar"] [data-testid="stProgressBar"] [data-baseweb="progress-bar"] > div > div,
[data-testid="stSidebar"] [data-testid="stProgressBar"] [aria-valuenow] > div {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    border-radius: 999px !important;
}

</style>
""",
    unsafe_allow_html=True,
)


if "GEMINI_API_KEY" not in st.secrets:
    st.error("Add GEMINI_API_KEY to .streamlit/secrets.toml.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


SAMPLE_JOB_DESCRIPTIONS = {
    "Software — Junior Developer": """
Junior Software Engineer

Responsibilities:
- Develop and maintain applications using Java or Python.
- Build REST APIs and write SQL queries for relational databases.
- Create unit and integration tests.
- Participate in code reviews, debugging and technical design discussions.

Requirements:
- Understanding of object-oriented programming and data structures.
- Basic knowledge of SQL and relational databases.
- Good problem-solving and communication skills.
""",
    "Finance — Financial Analyst": """
Junior Financial Analyst

Responsibilities:
- Analyse financial performance, budgets and monthly variances.
- Build financial models and prepare management reports.
- Explain findings to business stakeholders and support forecasting.
- Maintain accurate documentation and follow internal controls.

Requirements:
- Understanding of financial statements and key performance indicators.
- Proficiency in Excel and data analysis.
- Strong analytical thinking, attention to detail and communication skills.
""",
    "Healthcare — Clinical Operations Executive": """
Clinical Operations Executive

Responsibilities:
- Coordinate clinic workflows and support safe, timely patient services.
- Maintain accurate operational records and protect confidential information.
- Investigate service issues and support quality-improvement activities.
- Communicate with clinicians, patients and administrative stakeholders.

Requirements:
- Understanding of healthcare operations and patient-safety principles.
- Strong organisation, escalation judgement and communication skills.
- Familiarity with data reporting and process improvement.
""",
    "General — Operations Executive": """
Operations Executive

Responsibilities:
- Coordinate daily operations and monitor service performance.
- Investigate process problems and recommend practical improvements.
- Prepare reports and communicate with internal and external stakeholders.
- Manage competing priorities while following organisational policies.

Requirements:
- Strong problem-solving, organisation and communication skills.
- Ability to use spreadsheets and interpret operational data.
- Sound judgement and attention to detail.
""",
}

SAMPLE_JOB_DESCRIPTION = SAMPLE_JOB_DESCRIPTIONS["Software — Junior Developer"]

MAX_JOB_DESCRIPTION_LENGTH = 6000
MAX_PROJECT_CONTEXT_LENGTH = 2500
MAX_PROFILE_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DOCX_XML_BYTES = 2 * 1024 * 1024
MAX_STUDENT_RESPONSE_LENGTH = 4000
MAX_CODE_RESPONSE_LENGTH = 2600
MAX_CODE_REASONING_LENGTH = 1000

CODING_LANGUAGES = [
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "C++",
    "C#",
    "Go",
    "Kotlin",
    "Swift",
    "Pseudocode",
]

LANGUAGE_FENCES = {
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "C++": "cpp",
    "C#": "csharp",
    "Go": "go",
    "Kotlin": "kotlin",
    "Swift": "swift",
    "Pseudocode": "text",
}

QUESTION_TYPES = [
    "Experience / STAR",
    "Role knowledge",
    "Scenario / case",
    "Coding / DSA",
]

DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]

MAJOR_CONCEPTS = [
    "Role-specific Knowledge",
    "Applied Problem Solving",
    "Communication and Collaboration",
    "Tools and Methods",
    "Risk, Ethics and Compliance",
]

# This is a deliberately bounded prototype curriculum. Completing these items
# means completing the app's coverage, not mastering the entire discipline.
TOPIC_CONCEPTS = {
    "Role-specific Knowledge": [
        "Core concepts and terminology",
        "Role-specific procedures and standards",
        "Application to realistic work",
        "Limitations and trade-offs",
        "Current learning priorities",
    ],
    "Applied Problem Solving": [
        "Problem framing and clarification",
        "Evidence and assumptions",
        "Prioritisation and decision making",
        "Alternatives and trade-offs",
        "Validation and reflection",
    ],
    "Communication and Collaboration": [
        "Audience-aware explanation",
        "Stakeholder communication",
        "Team contribution and ownership",
        "Conflict and feedback",
        "Clear outcome reporting",
    ],
    "Tools and Methods": [
        "Tool selection and purpose",
        "Workflow and implementation",
        "Quality checks and testing",
        "Efficiency and maintainability",
        "Limitations and alternatives",
    ],
    "Risk, Ethics and Compliance": [
        "Risk identification",
        "Privacy and confidentiality",
        "Professional boundaries",
        "Escalation and accountability",
        "Monitoring and improvement",
    ],
}

CONCEPT_ICONS = {
    "Role-specific Knowledge": "📘",
    "Applied Problem Solving": "🧠",
    "Communication and Collaboration": "🤝",
    "Tools and Methods": "🛠️",
    "Risk, Ethics and Compliance": "🛡️",
}

READINESS_ORDER = {
    "Not practised": 0,
    "Needs development": 1,
    "Developing": 2,
    "Interview-ready": 3,
    "Strong evidence": 4,
}

READINESS_COLOURS = {
    "Not practised": "gray",
    "Needs development": "red",
    "Developing": "orange",
    "Interview-ready": "blue",
    "Strong evidence": "green",
}

PRIORITY_COLOURS = {"High": "red", "Medium": "orange", "Low": "gray"}


def empty_progress():
    return {
        concept: {
            "completed_cycles": 0,
            "latest_readiness": "Not practised",
            "strongest_readiness": "Not practised",
            "question_types": [],
            "competencies": [],
            "latest_focus": "",
            "last_practised": "",
            "history": [],
        }
        for concept in MAJOR_CONCEPTS
    }

STAGE_ORDER = [
    "initial_answer",
    "follow_up",
    "synthesis_reflection",
    "final_rewrite",
    "completed",
]

STAGE_NAMES = {
    "initial_answer": "Initial answer",
    "reflection": "Follow-up",
    "revision": "Follow-up",
    "follow_up": "Follow-up",
    "synthesis_reflection": "Reflection",
    "final_rewrite": "Complete rewrite",
    "completed": "Final analysis",
}

STAGE_SHORT_NAMES = {
    "initial_answer": "Initial answer",
    "reflection": "Follow-up",
    "revision": "Follow-up",
    "follow_up": "Follow-up",
    "synthesis_reflection": "Reflect",
    "final_rewrite": "Rewrite",
    "completed": "Analysis",
}


DEFAULT_STATE = {
    "setup_complete": False,
    "job_description_input": "",
    "candidate_background_input": "",
    "project_context_input": "",
    "candidate_upload_signature": "",
    "project_upload_signature": "",
    "job_analysis": None,
    "active_role": "",
    "active_domain": "Other",
    "active_competency": None,
    "active_difficulty": "Beginner",
    "active_question_type": "Role knowledge",
    "active_question": None,
    "follow_up_question": None,
    "stage": "initial_answer",
    "messages": [],
    "initial_answer": "",
    "reflection": "",
    "revised_answer": "",
    "follow_up_answer": "",
    "synthesis_reflection": "",
    "final_rewrite": "",
    "initial_evaluation": None,
    "revised_evaluation": None,
    "follow_up_evaluation": None,
    "final_rewrite_evaluation": None,
    "depth_questions": [],
    "depth_answers": [],
    "depth_evaluations": [],
    "final_analysis": None,
    "understanding_checkpoint": None,
    "probe_completion_reason": "",
    "pending_response": "",
    "stage_hint_levels": {
        "initial_answer": 0,
        "reflection": 0,
        "revision": 0,
        "final_rewrite": 0,
    },
    "hint_level": 0,
    "follow_up_hint_level": 0,
    "last_error": None,
    "last_error_details": None,
    "service_notice": None,
    "pending_retry": None,
    "support_mode_active": False,
    "plan_b_mode": False,
    "privacy_consent": False,
    "concept_progress": empty_progress(),
    "topic_plan": [],
    "topic_plan_index": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Add newly supported competency families without deleting progress from an
# earlier app version in the same browser session.
for concept, value in empty_progress().items():
    st.session_state.concept_progress.setdefault(concept, value)


# --------------------------------------------------
# Small rendering helpers
# --------------------------------------------------
def _badge_text(text):
    """Keep Markdown badge syntax intact when text contains brackets."""
    return str(text).replace("[", "(").replace("]", ")")


def badge(text, colour):
    return f":{colour}-badge[{_badge_text(text)}]"


def readiness_badge(readiness):
    return badge(
        progress_readiness_label(readiness),
        READINESS_COLOURS.get(readiness, "gray"),
    )


def progress_readiness_label(readiness):
    """Use learner-friendly wording without changing stored progress values."""
    if readiness == "Not practised":
        return "No completed practice yet"
    return readiness


def question_type_label(question_type):
    """Translate internal prompt modes into simple learner-facing labels."""
    labels = {
        "Experience / STAR": "Project",
        "Coding / DSA": "Coding",
        "Role knowledge": "Role knowledge",
        "Scenario / case": "Scenario",
    }
    return labels.get(question_type, question_type)


def extract_uploaded_summary(uploaded_file):
    """Extract safe plain text from TXT, Markdown or modern Word files."""
    if uploaded_file.size > MAX_PROFILE_UPLOAD_BYTES:
        raise ValueError("The file is larger than the 5 MB upload limit.")

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith((".txt", ".md")):
        text = file_bytes.decode("utf-8", errors="replace").strip()
    elif file_name.endswith(".docx"):
        try:
            with ZipFile(BytesIO(file_bytes)) as document_archive:
                document_info = document_archive.getinfo("word/document.xml")
                if document_info.file_size > MAX_DOCX_XML_BYTES:
                    raise ValueError(
                        "The Word document contains too much text to process safely."
                    )
                document_xml = document_archive.read(document_info)
        except (BadZipFile, KeyError) as error:
            raise ValueError(
                "This is not a readable .docx file. Open it in Word and save it "
                "again as a Word Document (.docx)."
            ) from error

        # DOCX files are ZIP archives containing XML. Reject custom entity
        # declarations before parsing and read only the main document text.
        upper_xml = document_xml.upper()
        if b"<!DOCTYPE" in upper_xml or b"<!ENTITY" in upper_xml:
            raise ValueError("The Word document contains unsupported XML content.")

        try:
            root = ElementTree.fromstring(document_xml)
        except ElementTree.ParseError as error:
            raise ValueError("The Word document text could not be parsed.") from error

        word_namespace = (
            "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        )
        paragraph_tag = f"{{{word_namespace}}}p"
        text_tag = f"{{{word_namespace}}}t"
        tab_tag = f"{{{word_namespace}}}tab"
        break_tags = {
            f"{{{word_namespace}}}br",
            f"{{{word_namespace}}}cr",
        }
        paragraphs = []
        for paragraph in root.iter(paragraph_tag):
            parts = []
            for element in paragraph.iter():
                if element.tag == text_tag and element.text:
                    parts.append(element.text)
                elif element.tag == tab_tag:
                    parts.append("\t")
                elif element.tag in break_tags:
                    parts.append("\n")
            paragraph_text = "".join(parts).strip()
            if paragraph_text:
                paragraphs.append(paragraph_text)
        text = "\n".join(paragraphs).strip()
    else:
        raise ValueError("Upload a .txt, .md or .docx file.")

    if not text:
        raise ValueError("No readable text was found in the uploaded file.")

    return text


def render_candidate_context_input(question_category):
    """Render context that is proportionate to the selected question mode."""
    is_project = question_category == "Project"

    if is_project:
        container = st.container()
        heading = "##### Project experience to discuss (optional)"
        description = (
            "Describe an anonymised project, placement or work experience that "
            "you are comfortable discussing. Include the situation, your "
            "responsibility, actions, technical decisions and result. Leave it "
            "blank if you prefer to choose an experience after seeing the question."
        )
        upload_label = "Upload an anonymised project summary"
        text_label = "Project experience summary"
        placeholder = (
            "Example: In a team project, I developed a Java booking application. "
            "I was responsible for the order workflow, wrote SQL queries, tested "
            "edge cases and helped reduce duplicate booking errors."
        )
        context_state_key = "project_context_input"
        upload_widget_key = "project_context_file"
        upload_signature_key = "project_upload_signature"
    else:
        container = st.expander(
            "Personalise using my background (optional)",
            expanded=False,
            icon="🧩",
        )
        heading = None
        description = (
            "Add an anonymised summary of your education, skills, tools or "
            "experience. PrepWise AI may use it to make the selected question "
            "more relevant. This is optional; the job description is sufficient."
        )
        upload_label = "Upload an anonymised background summary"
        text_label = "Candidate background summary"
        placeholder = (
            "Example: I am a computing student with experience in Java, Python, "
            "SQL, Git and software testing."
        )
        context_state_key = "candidate_background_input"
        upload_widget_key = "candidate_context_file"
        upload_signature_key = "candidate_upload_signature"

    with container:
        if heading:
            st.markdown(heading)
        st.caption(description)
        st.caption(
            "Do not include names, contact details, confidential company "
            "information or proprietary source code."
        )

        uploaded_context = st.file_uploader(
            upload_label,
            type=["txt", "md", "docx"],
            key=upload_widget_key,
            help=(
                "Optional: upload a TXT, Markdown or Word (.docx) summary of up "
                "to 5 MB. Its text will be copied into the editable box below."
            ),
        )

        if uploaded_context is None:
            st.session_state[upload_signature_key] = ""
        else:
            upload_signature = f"{uploaded_context.name}:{uploaded_context.size}"
            if upload_signature != st.session_state[upload_signature_key]:
                try:
                    uploaded_text = extract_uploaded_summary(uploaded_context)
                    if len(uploaded_text) > MAX_PROJECT_CONTEXT_LENGTH:
                        st.error(
                            "The extracted summary is too long. Shorten it to "
                            f"{MAX_PROJECT_CONTEXT_LENGTH:,} characters or fewer."
                        )
                    else:
                        st.session_state[context_state_key] = uploaded_text
                        st.session_state[upload_signature_key] = upload_signature
                        st.success(
                            "The uploaded text was extracted successfully. You "
                            "can review and edit it below."
                        )
                except ValueError as error:
                    st.error(f"Could not read the uploaded file: {error}")

        candidate_context = st.text_area(
            text_label,
            key=context_state_key,
            height=170 if is_project else 130,
            placeholder=placeholder,
            help=(
                "Include only information that you are authorised and "
                "comfortable to discuss in an interview."
            ),
        )

        context_length = len(candidate_context)
        if context_length > MAX_PROJECT_CONTEXT_LENGTH:
            st.error(
                "Shorten the summary to "
                f"{MAX_PROJECT_CONTEXT_LENGTH:,} characters or fewer."
            )
        elif candidate_context:
            st.caption(
                f"{context_length:,}/{MAX_PROJECT_CONTEXT_LENGTH:,} characters ✓"
            )

    return candidate_context, context_length


def bullet_list(items, empty_text="_None noted._"):
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def weaker_dimension_actions(evaluation):
    """Return actionable gaps from non-ready rubric dimensions."""
    actions = []
    for item in evaluation.get("dimensions", []):
        judgement = str(item.get("judgement", "")).lower()
        if judgement in {"developing", "needs development"}:
            action = str(item.get("next_step", "")).strip()
            if action and action not in actions:
                actions.append(action)
    return actions


def coaching_focus_items(evaluation, checkpoint):
    """Never let an empty checkpoint hide visible rubric weaknesses."""
    checkpoint_gaps = [
        str(item).strip()
        for item in checkpoint.get("remaining_gaps", [])
        if str(item).strip()
    ]
    if checkpoint_gaps:
        return checkpoint_gaps

    dimension_actions = weaker_dimension_actions(evaluation)
    if dimension_actions:
        return dimension_actions

    return [
        str(item).strip()
        for item in evaluation.get("gaps", [])
        if str(item).strip()
    ]


def _meaningful_feedback(items):
    """Discard empty / placeholder feedback rather than showing it to learners."""
    placeholders = (
        "none", "n/a", "not applicable", "no gap", "no gaps",
        "no issues", "no improvement", "fully addresses", "nothing to improve",
        "already complete", "not identified",
    )
    cleaned = []
    for item in items or []:
        value = str(item or "").strip().lstrip("-• ").strip()
        if not value or any(value.lower().startswith(x) for x in placeholders):
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def reflection_guidance(evaluation, checkpoint):
    """Choose genuine gaps, grounded refinements or strength-based reflection."""
    evaluation = evaluation or {}
    checkpoint = checkpoint or {}
    met = bool(checkpoint.get("expectation_met"))
    weak_actions = _meaningful_feedback(weaker_dimension_actions(evaluation))
    required_gaps = _meaningful_feedback(checkpoint.get("remaining_gaps", []))

    if not met:
        focus = _meaningful_feedback(required_gaps + weak_actions + evaluation.get("gaps", []))
        if not focus:
            focus = [
                "Identify a concept or decision from the conversation that still "
                "needs stronger evidence in your complete answer."
            ]
        return {
            "mode": "gap",
            "focus": focus[:2],
            "prompt": (
                "Choose the most important learning gap above. What will you "
                "add or change in your complete rewrite, and why?"
            ),
        }

    # The expected understanding is met. Weak non-required dimensions may
    # still provide an honest optional refinement, but do not call it a gap.
    refinements = weak_actions
    if not refinements:
        for dimension in evaluation.get("dimensions", []):
            judgement = str(dimension.get("judgement", "")).strip().lower()
            if judgement == "interview-ready":
                action = _meaningful_feedback([dimension.get("next_step", "")])
                for suggestion in action:
                    if suggestion not in refinements:
                        refinements.append(suggestion)

    if refinements:
        return {
            "mode": "refinement",
            "focus": refinements[:2],
            "prompt": (
                "Your understanding has been demonstrated. Which optional "
                "refinement will you use in your complete answer, and why?"
            ),
        }

    strengths = _meaningful_feedback(evaluation.get("strengths", []))
    if not strengths:
        strengths = _meaningful_feedback(checkpoint.get("demonstrated_evidence", []))
    if not strengths:
        strengths = ["Your responses demonstrated the expected understanding."]
    return {
        "mode": "strength",
        "focus": strengths[:2],
        "prompt": (
            "Which strength above made your explanation effective? How will you "
            "preserve that strength when you deliver one concise complete answer?"
        ),
    }


def build_reflection_message(evaluation, checkpoint, completion_reason):
    """Render one coherent evidence-based reflection task for every outcome."""
    guidance = reflection_guidance(evaluation, checkpoint)
    labels = {
        "gap": "Learning gaps to address",
        "refinement": "Optional ways to strengthen your answer",
        "strength": "Strengths to carry into your rewrite",
    }
    introduction = {
        "gap": "Focus on the most important evidence that is still missing.",
        "refinement": (
            "You have demonstrated the expected understanding. These suggestions "
            "are optional; you do not need to invent a weakness."
        ),
        "strength": (
            "You have demonstrated the expected understanding, and no meaningful "
            "additional correction was identified. Reflect on what worked well."
        ),
    }
    return f"""
### Reflection before your complete rewrite

The adaptive questioning stage ended because {completion_reason}.

{introduction[guidance['mode']]}

**{labels[guidance['mode']]}:**

{bullet_list(guidance['focus'])}

**Reflection prompt:** {guidance['prompt']}

Write two or three sentences in your own words. The next step will ask you to
combine your learning into one complete interview response.
    """


def technical_error_details(error):
    """Return the service's redacted diagnostic details, when available."""
    return getattr(error, "technical_details", "") or ""


def render_ai_error(title, error):
    """Show an actionable message and optional diagnostics for development."""
    st.error(f"{title}: {error}")
    details = technical_error_details(error)
    if details:
        with st.expander("Technical details for troubleshooting", icon="🛠️"):
            st.code(details, language="text")
            st.caption("Do not share an API key when asking someone for help.")


def render_hero(title, subtitle, chips=()):
    chip_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips)
    st.markdown(
        f"""
<div class="hero">
  <h1>{title}</h1>
  <p>{subtitle}</p>
  {'<div class="chips">' + chip_html + '</div>' if chips else ''}
</div>
""",
        unsafe_allow_html=True,
    )


def render_app_brand():
    """Render a compact product identity in the persistent navigation."""
    st.markdown(
        """
<div class="app-brand">
  <span class="brand-mark">PW</span>
  <div class="brand-copy">
    <strong>PrepWise AI</strong>
    <span>Adaptive AI Interview Coach</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_workspace_heading():
    """Render the active-practice heading and compact session metadata."""
    active_concept = st.session_state.active_competency["mapped_topic"]
    active_progress = st.session_state.concept_progress[active_concept]
    latest_result = progress_readiness_label(
        active_progress["latest_readiness"]
    )
    current_position = min(
        st.session_state.topic_plan_index + 1,
        max(len(st.session_state.topic_plan), 1),
    )
    total_concepts = max(len(st.session_state.topic_plan), 1)

    st.markdown(
        """
<div class="workspace-heading">
  <h1>Interview practice</h1>
  <p>Build a stronger answer through evidence-based feedback and adaptive questions.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="metric-strip">
  <div class="metric-card">
    <span>Current stage</span>
    <strong>{STAGE_NAMES[st.session_state.stage]}</strong>
  </div>
  <div class="metric-card">
    <span>Topic pathway</span>
    <strong>Concept {current_position} of {total_concepts}</strong>
  </div>
  <div class="metric-card">
    <span>Latest completed result</span>
    <strong>{latest_result}</strong>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_stepper(current_stage):
    current_display_stage = display_stage(current_stage)
    steps = []
    for index, stage in enumerate(STAGE_ORDER):
        if stage_is_completed(stage):
            css_class, prefix = "done", "✓"
        elif stage == current_display_stage:
            css_class, prefix = "current", f"{index + 1}"
        else:
            css_class, prefix = "", f"{index + 1}"
        steps.append(
            f'<div class="step {css_class}"><span class="num">{prefix} · </span>'
            f"{STAGE_SHORT_NAMES[stage]}</div>"
        )
    st.markdown(
        f'<div class="stepper">{"".join(steps)}</div>',
        unsafe_allow_html=True,
    )


def stage_is_completed(stage):
    """Return true only for steps the learner actually completed."""
    if stage == "initial_answer":
        return st.session_state.initial_evaluation is not None
    if stage == "follow_up":
        return st.session_state.stage in {
            "synthesis_reflection",
            "final_rewrite",
            "completed",
        }
    if stage == "synthesis_reflection":
        return bool(st.session_state.synthesis_reflection.strip())
    if stage == "final_rewrite":
        return st.session_state.final_rewrite_evaluation is not None
    return stage == "completed" and st.session_state.final_analysis is not None


def render_sidebar_cycle(current_stage):
    """Render a persistent vertical cycle with completed stages in green."""
    current_display_stage = display_stage(current_stage)
    st.markdown("### Learning cycle")
    for index, stage in enumerate(STAGE_ORDER):
        if stage_is_completed(stage):
            css_class = "done"
            marker = "✓"
        elif stage == current_display_stage:
            css_class = "current"
            marker = str(index + 1)
        else:
            css_class = ""
            marker = str(index + 1)

        st.markdown(
            f'<div class="cycle-item {css_class}">'
            f'<span class="cycle-dot">{marker}</span>'
            f'<span>{STAGE_SHORT_NAMES[stage]}</span></div>',
            unsafe_allow_html=True,
        )


def display_stage(stage):
    """Map internal adaptive turns to one clear learner-facing stage."""
    if stage in {"reflection", "revision"}:
        return "follow_up"
    return stage


def add_message(role, content="", kind="text", data=None):
    st.session_state.messages.append(
        {"role": role, "content": content, "kind": kind, "data": data}
    )


def build_topic_plan(skill):
    """Create a finite concept pathway beginning with the job-linked skill."""
    topic = skill["mapped_topic"]
    candidates = [
        (skill["skill"], "job_focus"),
        *((concept, "core_concept") for concept in TOPIC_CONCEPTS.get(topic, [])),
    ]
    seen = set()
    concepts = []
    for concept, category in candidates:
        key = "".join(character.lower() for character in concept if character.isalnum())
        if key and key not in seen:
            seen.add(key)
            concepts.append((concept, category))
    return [
        {
            "concept": concept,
            "category": category,
            "status": "current" if index == 0 else "pending",
        }
        for index, (concept, category) in enumerate(concepts)
    ]


def mark_current_topic_concept_complete():
    """Mark exactly one pathway item complete after its full learning cycle."""
    plan = st.session_state.topic_plan
    index = st.session_state.topic_plan_index
    if plan and 0 <= index < len(plan):
        plan[index]["status"] = "completed"
        st.session_state.topic_plan = plan


def next_topic_concept_index():
    for index, item in enumerate(st.session_state.topic_plan):
        if item["status"] == "pending":
            return index
    return None


def normalised_label(value):
    """Create a comparison key for labels with different punctuation/casing."""
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def render_topic_pathway():
    """Show the broad topic, job-linked focus and core concepts hierarchically."""
    plan = st.session_state.topic_plan
    if not plan:
        return
    completed = sum(item["status"] == "completed" for item in plan)
    topic = st.session_state.active_competency["mapped_topic"]
    topic_icon = CONCEPT_ICONS.get(topic, "📘")
    st.markdown(f"### {topic_icon} {topic} pathway")
    st.caption(
        f"{completed} of {len(plan)} practice areas completed"
    )
    st.progress(
        completed / len(plan),
        text=None,
    )

    previous_category = None
    for item in plan:
        category = item.get("category", "core_concept")
        item_label = item["concept"]
        if (
            category == "job_focus"
            and normalised_label(item_label) == normalised_label(topic)
        ):
            item_label = "Topic overview"

        if category != previous_category:
            if category == "job_focus":
                st.caption("JOB-LINKED STARTING POINT")
            else:
                st.caption("CORE CONCEPTS")
            previous_category = category

        if item["status"] == "completed":
            st.success(item_label, icon="✅")
        elif item["status"] == "current":
            st.info(f"Current · {item_label}", icon="🎯")
        else:
            st.caption(f"○ {item_label}")


def render_practice_context():
    """Present session metadata before the pathway that it determines."""
    st.markdown("### Practice context")
    with st.container(border=True):
        st.caption("TARGET ROLE")
        st.write(st.session_state.active_role)

        st.caption("FIELD")
        st.write(st.session_state.active_domain)

        question_col, difficulty_col = st.columns(2)
        with question_col:
            st.caption("QUESTION TYPE")
            st.write(question_type_label(st.session_state.active_question_type))
        with difficulty_col:
            st.caption("DIFFICULTY")
            st.write(st.session_state.active_difficulty)


def start_interview(
    question_data,
    skill,
    role_title,
    difficulty,
    preserve_topic_plan=False,
):
    # Remove drafts left by a previous coding question while preserving all
    # unrelated session state and the learner's recorded progress.
    for key in list(st.session_state.keys()):
        if key.startswith("coding_composer_"):
            del st.session_state[key]

    st.session_state.active_question = question_data
    st.session_state.follow_up_question = None
    st.session_state.active_role = role_title
    st.session_state.active_competency = skill
    st.session_state.active_difficulty = difficulty
    st.session_state.active_question_type = question_data["question_type"]
    st.session_state.stage = "initial_answer"
    st.session_state.initial_answer = ""
    st.session_state.reflection = ""
    st.session_state.revised_answer = ""
    st.session_state.follow_up_answer = ""
    st.session_state.synthesis_reflection = ""
    st.session_state.final_rewrite = ""
    st.session_state.initial_evaluation = None
    st.session_state.revised_evaluation = None
    st.session_state.follow_up_evaluation = None
    st.session_state.final_rewrite_evaluation = None
    st.session_state.depth_questions = []
    st.session_state.depth_answers = []
    st.session_state.depth_evaluations = []
    st.session_state.final_analysis = None
    st.session_state.understanding_checkpoint = None
    st.session_state.probe_completion_reason = ""
    st.session_state.pending_response = ""
    st.session_state.stage_hint_levels = {
        "initial_answer": 0,
        "reflection": 0,
        "revision": 0,
        "final_rewrite": 0,
    }
    st.session_state.hint_level = 0
    st.session_state.follow_up_hint_level = 0
    st.session_state.last_error = None
    st.session_state.last_error_details = None
    st.session_state.pending_retry = None
    st.session_state.support_mode_active = False
    st.session_state.plan_b_mode = False

    if not preserve_topic_plan:
        st.session_state.topic_plan = build_topic_plan(skill)
        st.session_state.topic_plan_index = 0

    st.session_state.messages = []
    add_message(
        "assistant",
        kind="question",
        data={
            "follow_up": False,
            "show_welcome": not preserve_topic_plan,
            **question_data,
        },
    )


def reset_to_setup():
    for key in list(st.session_state.keys()):
        if key.startswith("coding_composer_"):
            del st.session_state[key]

    st.session_state.setup_complete = False
    st.session_state.job_analysis = None
    st.session_state.active_question = None
    st.session_state.active_domain = "Other"
    st.session_state.follow_up_question = None
    st.session_state.messages = []
    st.session_state.last_error = None
    st.session_state.last_error_details = None
    st.session_state.pending_retry = None
    st.session_state.support_mode_active = False
    st.session_state.depth_questions = []
    st.session_state.depth_answers = []
    st.session_state.depth_evaluations = []
    st.session_state.final_analysis = None
    st.session_state.understanding_checkpoint = None
    st.session_state.synthesis_reflection = ""
    st.session_state.final_rewrite = ""
    st.session_state.final_rewrite_evaluation = None
    st.session_state.probe_completion_reason = ""
    st.session_state.pending_response = ""
    st.session_state.stage_hint_levels = {
        "initial_answer": 0,
        "reflection": 0,
        "revision": 0,
        "final_rewrite": 0,
    }
    st.session_state.plan_b_mode = False
    st.session_state.topic_plan = []
    st.session_state.topic_plan_index = 0


def record_completed_cycle(final_analysis):
    """Record one completed interview cycle under its major concept."""
    concept = st.session_state.active_competency["mapped_topic"]
    progress = st.session_state.concept_progress

    if concept not in progress:
        progress[concept] = {
            "completed_cycles": 0,
            "latest_readiness": "Not practised",
            "strongest_readiness": "Not practised",
            "question_types": [],
            "competencies": [],
            "latest_focus": "",
            "last_practised": "",
            "history": [],
        }

    concept_data = progress[concept]
    final_readiness = final_analysis["overall_readiness"]
    current_best = concept_data["strongest_readiness"]

    if READINESS_ORDER[final_readiness] > READINESS_ORDER[current_best]:
        concept_data["strongest_readiness"] = final_readiness

    question_type = st.session_state.active_question_type
    competency = st.session_state.active_competency["skill"]

    if question_type not in concept_data["question_types"]:
        concept_data["question_types"].append(question_type)

    if competency not in concept_data["competencies"]:
        concept_data["competencies"].append(competency)

    concept_data["completed_cycles"] += 1
    concept_data["latest_readiness"] = final_readiness
    concept_data["latest_focus"] = (
        final_analysis["knowledge_gaps"][0]
        if final_analysis["knowledge_gaps"]
        else "Continue practising independent explanation and application."
    )
    concept_data["last_practised"] = datetime.now().strftime("%d %b %Y, %H:%M")
    concept_data["history"].append(
        {
            "date": concept_data["last_practised"],
            "competency": competency,
            "question_type": question_type,
            "difficulty": st.session_state.active_difficulty,
            "final_readiness": final_readiness,
            "depth_questions_answered": len(st.session_state.depth_answers),
            "hints_used": dict(st.session_state.stage_hint_levels),
            "next_focus": concept_data["latest_focus"],
        }
    )

    st.session_state.concept_progress = progress
    mark_current_topic_concept_complete()


def render_progress_dashboard():
    progress = st.session_state.concept_progress

    completed_total = sum(
        item["completed_cycles"] for item in progress.values()
    )
    practised_total = sum(
        1 for item in progress.values() if item["completed_cycles"] > 0
    )

    col1, col2 = st.columns(2)
    with col1.container(border=True):
        st.metric("Completed practice cycles", completed_total)
    with col2.container(border=True):
        st.metric("Major concepts practised", f"{practised_total}/{len(MAJOR_CONCEPTS)}")

    # Build rows explicitly so cards keep their order when columns stack on phones.
    for row_start in range(0, len(MAJOR_CONCEPTS), 3):
        columns = st.columns(3)
        for column, concept in zip(columns, MAJOR_CONCEPTS[row_start:row_start + 3]):
            item = progress[concept]
            with column.container(border=True):
                st.markdown(f"**{CONCEPT_ICONS[concept]} {concept}**")
                st.markdown(readiness_badge(item["latest_readiness"]))
                cycles = item["completed_cycles"]
                st.caption(
                    f"{cycles} cycle{'s' if cycles != 1 else ''} · "
                    f"best: {item['strongest_readiness']}"
                )
                if item["question_types"]:
                    st.caption(
                        "Types: "
                        + ", ".join(
                            question_type_label(value)
                            for value in item["question_types"]
                        )
                    )
                st.caption(
                    "Next focus: "
                    + (item["latest_focus"] or "Complete a practice cycle")
                )

    practised_concepts = [
        concept
        for concept in MAJOR_CONCEPTS
        if progress[concept]["history"]
    ]

    if practised_concepts:
        with st.expander("Practice history", icon="🗂️"):
            selected_history_concept = st.selectbox(
                "Concept",
                options=practised_concepts,
                key="progress_history_concept",
            )
            st.dataframe(
                progress[selected_history_concept]["history"],
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        "Progress uses readiness evidence rather than an average score. "
        "For this prototype, progress remains available during the current browser session."
    )


def build_practice_report():
    """Create an anonymous, downloadable record of learning evidence."""
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "role": st.session_state.active_role,
        "industry_domain": st.session_state.active_domain,
        "competency": st.session_state.active_competency["skill"],
        "competency_family": st.session_state.active_competency["mapped_topic"],
        "question_type": question_type_label(st.session_state.active_question_type),
        "question_mode_internal": st.session_state.active_question_type,
        "difficulty": st.session_state.active_difficulty,
        "question": st.session_state.active_question["question"],
        "topic_pathway": st.session_state.topic_plan,
        "topic_pathway_position": st.session_state.topic_plan_index + 1,
        "initial_response": st.session_state.initial_answer,
        "depth_questions": [
            question["question"] for question in st.session_state.depth_questions
        ],
        "depth_responses": st.session_state.depth_answers,
        "reflection_before_rewrite": st.session_state.synthesis_reflection,
        "final_rewrite": st.session_state.final_rewrite,
        "learning_evidence": {
            "initial_evaluation": st.session_state.initial_evaluation,
            "final_rewrite_evaluation": st.session_state.final_rewrite_evaluation,
            "final_analysis": st.session_state.final_analysis,
            "understanding_checkpoint": st.session_state.understanding_checkpoint,
            "hints_used": dict(st.session_state.stage_hint_levels),
        },
        "notice": (
            "Formative AI guidance only; this is not an official employer assessment."
        ),
    }


def _report_label(value):
    """Turn internal snake-case keys into reader-friendly labels."""
    return str(value).replace("_", " ").strip().title()


def _clean_report_text(value):
    """Convert AI Markdown into readable plain text for exported reports."""
    if value is None:
        return "Not recorded"
    text = str(value).replace("\r\n", "\n")
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or "Not recorded"


def _append_structured_blocks(blocks, title, value, level=2):
    """Flatten nested evaluation evidence into headings and readable text."""
    if value in (None, "", [], {}):
        return

    blocks.append(("heading", level, _report_label(title)))
    if isinstance(value, dict):
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            if isinstance(item, (dict, list)):
                _append_structured_blocks(blocks, key, item, min(level + 1, 4))
            else:
                blocks.append(("label", _report_label(key), _clean_report_text(item)))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key, child in item.items():
                    if child not in (None, "", [], {}):
                        blocks.append(
                            ("label", _report_label(key), _clean_report_text(child))
                        )
                blocks.append(("space",))
            else:
                blocks.append(("bullet", _clean_report_text(item)))
    else:
        blocks.append(("body", _clean_report_text(value)))


def build_report_blocks(report):
    """Create one shared, human-readable report structure for Word and PDF."""
    blocks = [
        ("title", "PrepWise AI Practice Report"),
        ("body", "Formative learning evidence from one interview-practice cycle."),
        ("heading", 1, "Session overview"),
        ("label", "Generated", report["generated_at"]),
        ("label", "Target role", report["role"]),
        ("label", "Industry", report["industry_domain"]),
        ("label", "Competency", report["competency"]),
        ("label", "Competency family", report["competency_family"]),
        ("label", "Question type", report["question_type"]),
        ("label", "Difficulty", report["difficulty"]),
        ("heading", 1, "Interview question"),
        ("body", _clean_report_text(report["question"])),
        ("heading", 1, "Initial response"),
        ("body", _clean_report_text(report["initial_response"])),
    ]

    _append_structured_blocks(
        blocks,
        "Initial coaching evaluation",
        report["learning_evidence"].get("initial_evaluation"),
        1,
    )

    if report["depth_questions"]:
        blocks.append(("heading", 1, "Adaptive follow-up conversation"))
        for index, question in enumerate(report["depth_questions"], start=1):
            blocks.append(("heading", 2, f"Follow-up question {index}"))
            blocks.append(("body", _clean_report_text(question)))
            response = (
                report["depth_responses"][index - 1]
                if index <= len(report["depth_responses"])
                else "Not answered"
            )
            blocks.append(("label", "Learner response", _clean_report_text(response)))

    blocks.extend(
        [
            ("heading", 1, "Reflection before final rewrite"),
            ("body", _clean_report_text(report["reflection_before_rewrite"])),
            ("heading", 1, "Final rewritten response"),
            ("body", _clean_report_text(report["final_rewrite"])),
        ]
    )

    _append_structured_blocks(
        blocks,
        "Final rewrite evaluation",
        report["learning_evidence"].get("final_rewrite_evaluation"),
        1,
    )
    _append_structured_blocks(
        blocks,
        "Final concept analysis",
        report["learning_evidence"].get("final_analysis"),
        1,
    )
    _append_structured_blocks(
        blocks,
        "Understanding checkpoint",
        report["learning_evidence"].get("understanding_checkpoint"),
        1,
    )
    _append_structured_blocks(
        blocks,
        "Hints used",
        report["learning_evidence"].get("hints_used"),
        1,
    )
    blocks.extend(
        [
            ("heading", 1, "Important notice"),
            ("body", report["notice"]),
        ]
    )
    return blocks


def create_pdf_report(report):
    """Return the completed practice report as an in-memory PDF file."""
    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="PrepWise AI Practice Report",
        author="PrepWise AI",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PrepTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#4F46E5"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    for level, size, colour in (
        (1, 15, "#20243A"),
        (2, 12, "#343A5E"),
        (3, 10.5, "#4F46E5"),
        (4, 10, "#4F46E5"),
    ):
        styles.add(
            ParagraphStyle(
                name=f"PrepHeading{level}",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=size,
                leading=size + 3,
                textColor=colors.HexColor(colour),
                spaceBefore=10,
                spaceAfter=5,
            )
        )
    body_style = ParagraphStyle(
        name="PrepBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2A2E42"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        name="PrepBullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-7,
        bulletIndent=3,
    )

    story = []
    for kind, *content in build_report_blocks(report):
        if kind == "title":
            story.append(Paragraph(escape(content[0]), styles["PrepTitle"]))
        elif kind == "heading":
            level, text = content
            story.append(Paragraph(escape(text), styles[f"PrepHeading{min(level, 4)}"]))
        elif kind == "label":
            label, text = content
            safe_text = escape(text).replace("\n", "<br/>")
            story.append(
                Paragraph(f"<b>{escape(label)}:</b> {safe_text}", body_style)
            )
        elif kind == "bullet":
            story.append(Paragraph(escape(content[0]), bullet_style, bulletText="•"))
        elif kind == "body":
            for paragraph_text in content[0].split("\n\n"):
                story.append(
                    Paragraph(escape(paragraph_text).replace("\n", "<br/>"), body_style)
                )
        elif kind == "space":
            story.append(Spacer(1, 4))

    def add_page_number(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#74788F"))
        canvas.drawCentredString(A4[0] / 2, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    return output.getvalue()


# --------------------------------------------------
# Chat message renderers
# --------------------------------------------------
STAR_GUIDE = """
- **Situation:** Briefly explain the project context and problem.
- **Task:** State your responsibility and objective.
- **Action:** Explain your own decisions and actions and how you applied the selected competency.
- **Result:** Describe the outcome, evidence and what you learned.

The coach evaluates whether these elements are present, even if you do not use
the four labels explicitly.
"""


def render_question(data):
    if data.get("depth_number"):
        introduction = "Here is a follow-up question based on your response:"
    elif data.get("show_welcome", True):
        introduction = (
            "Hello! I am your PrepWise AI Interview Coach. 👋 Here is your question."
        )
    else:
        introduction = "Let’s continue with the next question in your pathway:"

    # Keep the question at the same text size as the conversational introduction.
    st.markdown(f"{introduction}\n\n{data['question']}")

    if data["question_type"] == "Experience / STAR" and not data.get("depth_number"):
        with st.expander("Suggested structure: STAR", icon="⭐"):
            st.markdown(STAR_GUIDE)

    if data.get("depth_number"):
        st.caption("Answer in your own words. You may request progressive hints if needed.")
    else:
        if data["question_type"] == "Coding / DSA":
            st.caption(
                "Use the coding response editor below. Select a language, write code "
                "or pseudocode, and explain your approach and complexity in your own words."
            )
        else:
            st.caption(
                "Answer as though you were speaking to an interviewer. Use "
                "the hint button if you need support — the coach will not write the "
                "answer for you."
            )


def render_evaluation(evaluation, title):
    st.markdown(f"**{title}**")
    st.markdown(f"Readiness signal: {readiness_badge(evaluation['overall_readiness'])}")
    st.markdown(evaluation["summary"])

    strengths_col, gaps_col = st.columns(2)
    with strengths_col.container(border=True):
        st.markdown("**✅ Strengths**")
        st.markdown(bullet_list(evaluation["strengths"]))
    with gaps_col.container(border=True):
        st.markdown("**🎯 Priority gaps**")
        st.markdown(bullet_list(evaluation["gaps"]))

    if evaluation["dimensions"]:
        with st.expander(
            f"Evidence by criterion ({len(evaluation['dimensions'])})",
            icon="🔍",
        ):
            for index, item in enumerate(evaluation["dimensions"]):
                if index:
                    st.divider()
                st.markdown(
                    f"**{item['criterion']}** {readiness_badge(item['judgement'])}"
                )
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Next step:** {item['next_step']}")

    st.caption("Formative guidance, not an official company hiring score.")


def render_coaching_feedback(data):
    """Show one concise feedback card before the next learner action."""
    evaluation = data["evaluation"]
    checkpoint = data.get("checkpoint") or {}
    focus_items = coaching_focus_items(evaluation, checkpoint)
    response_label = data.get("response_label", "Answer")
    st.markdown(f"### {response_label} feedback")
    st.markdown(
        f"**Current evidence:** {readiness_badge(evaluation['overall_readiness'])}"
    )
    st.markdown(evaluation["summary"])

    strengths_col, focus_col = st.columns(2)
    with strengths_col.container(border=True):
        st.markdown("**✅ What you demonstrated**")
        st.markdown(
            bullet_list(
                checkpoint.get("demonstrated_evidence", evaluation["strengths"])
            )
        )
    with focus_col.container(border=True):
        if data.get("cycle_complete"):
            st.markdown("**🎯 Optional answer refinements**")
            if focus_items:
                st.markdown(bullet_list(focus_items))
            else:
                st.markdown(
                    "No required conceptual gap remains. You can still improve "
                    "precision, conciseness or interview delivery."
                )
        else:
            st.markdown(
                "**🎯 Remaining areas to improve**"
                if data.get("cycle_complete")
                else "**🎯 Focus for your next question**"
            )
            st.markdown(bullet_list(focus_items))

    if evaluation.get("dimensions"):
        with st.expander(
            f"View interview rubric evidence ({len(evaluation['dimensions'])} criteria)",
            icon="🔍",
        ):
            for index, item in enumerate(evaluation["dimensions"]):
                if index:
                    st.divider()
                st.markdown(
                    f"**{item['criterion']}** {readiness_badge(item['judgement'])}"
                )
                st.markdown(f"**Evidence:** {item['evidence']}")
                st.markdown(f"**Next step:** {item['next_step']}")

    if data.get("cycle_complete"):
        st.caption(
            "The follow-up stage is complete. Reflect on the feedback, then rewrite "
            "your complete answer."
        )
    else:
        st.caption(
            "Next, answer the highlighted follow-up question below. Your final "
            "analysis will be provided after this learning cycle."
        )


def render_final_analysis(analysis):
    st.markdown("### Final concept analysis")
    if analysis.get("completion_reason"):
        st.caption(f"Cycle ended because: {analysis['completion_reason']}")
    st.markdown(f"Readiness signal: {readiness_badge(analysis['overall_readiness'])}")
    st.markdown(analysis["summary"])

    strengths_col, gaps_col = st.columns(2)
    with strengths_col.container(border=True):
        st.markdown("**✅ Demonstrated understanding**")
        st.markdown(bullet_list(analysis["demonstrated_strengths"]))
    with gaps_col.container(border=True):
        st.markdown("**🎯 Knowledge gaps**")
        st.markdown(bullet_list(analysis["knowledge_gaps"]))

    with st.expander("Evidence from your responses", expanded=True, icon="🔍"):
        st.markdown(bullet_list(analysis["evidence"]))

    with st.container(border=True):
        st.markdown("**📈 Improvement from your initial answer**")
        st.markdown(bullet_list(analysis["improvements_from_initial"]))

    if analysis["recommended_topics"]:
        st.warning(
            "**Recommended reading and practice**\n\n"
            + bullet_list(analysis["recommended_topics"]),
            icon="📚",
        )

    with st.container(border=True):
        st.markdown("**Key points an ideal answer should cover**")
        st.markdown(bullet_list(analysis["ideal_answer_key_points"]))
        st.caption(
            "These are key points, not a full model answer. Use them to construct "
            "your own explanation."
        )

    st.info(f"**Next practice step:** {analysis['next_practice_step']}", icon="➡️")
    st.caption("Formative guidance, not an official company hiring decision.")


def render_hint(hint):
    st.markdown(f"**💡 Hint {hint['level']} of 3**")
    st.markdown(hint["guidance_question"])
    if hint["key_points"]:
        st.markdown("Consider these key points:")
        st.markdown(bullet_list(hint["key_points"]))
    st.caption("Use these prompts to construct the answer in your own words.")


def render_guided_support(data):
    """Render support without implying that the learner was evaluated."""
    hint = data["hint"]
    st.markdown("**Let’s build your answer one small step at a time**")
    st.caption(
        "This help request has not been evaluated and no readiness signal has "
        "been assigned."
    )
    with st.container(border=True):
        st.markdown(f"💭 **Start with this question:** {hint['guidance_question']}")
        if hint["key_points"]:
            st.markdown("You may use these key points:")
            st.markdown(bullet_list(hint["key_points"]))
    st.markdown(
        "Reply with your own sentence or example. It does not need to be perfect. "
        "The coach will evaluate only after you make a genuine attempt."
    )


def render_summary(data):
    st.markdown("**Practice summary ✅**")
    if data.get("readiness"):
        columns = st.columns(3)
        for column, (label, value) in zip(
            columns,
            [
                ("First response", data["readiness"][0]),
                ("Revised response", data["readiness"][1]),
                ("Follow-up", data["readiness"][2]),
            ],
        ):
            with column.container(border=True):
                st.caption(label)
                st.markdown(readiness_badge(value))
    st.markdown(data["text"])


def render_message(message):
    kind = message.get("kind", "text")
    data = message.get("data")
    if kind == "question":
        render_question(data)
    elif kind == "evaluation":
        render_evaluation(data["evaluation"], data["title"])
    elif kind in {"coaching_feedback", "coaching_review"}:
        render_coaching_feedback(data)
    elif kind == "final_analysis":
        render_final_analysis(data)
    elif kind == "hint":
        render_hint(data)
    elif kind == "guided_support":
        render_guided_support(data)
    elif kind == "summary":
        render_summary(data)
    else:
        st.markdown(message["content"])


def question_for_stage(stage):
    if stage == "initial_answer":
        return st.session_state.active_question
    depth_index = {"reflection": 0, "revision": 1}.get(stage)
    if depth_index is not None and len(st.session_state.depth_questions) > depth_index:
        return st.session_state.depth_questions[depth_index]
    return st.session_state.active_question


def provide_guided_support(stage):
    """Respond to a help request without evaluating or advancing the stage."""
    question_data = question_for_stage(stage)
    current_level = st.session_state.stage_hint_levels.get(stage, 0)
    next_level = min(current_level + 1, 3)

    progress = st.progress(10, text="Socratic Evaluator AI is preparing a guided hint...")
    try:
        progress.progress(35, text="Selecting helpful concepts without giving the answer...")
        hint = generate_hint(client, question_data, next_level)
        progress.progress(100, text="Guidance ready.")
    except Exception as error:
        hint = fallback_hint(question_data, next_level)
        st.session_state.service_notice = (
            f"{error} Curated guidance was used instead."
        )
    finally:
        progress.empty()

    add_message(
        "assistant",
        kind="guided_support",
        data={"hint": hint, "stage": stage},
    )
    st.session_state.stage_hint_levels[stage] = max(current_level, next_level)
    st.session_state.support_mode_active = True
    if stage == "initial_answer":
        st.session_state.hint_level = st.session_state.stage_hint_levels[stage]


def run_evaluation(stage):
    """Evaluate one answer and either probe one gap or begin consolidation."""
    stage_to_depth_index = {"reflection": 0, "revision": 1}
    response_labels = {
        "initial_answer": "Initial answer",
        "reflection": "Follow-up answer",
        "revision": "Follow-up answer",
    }

    if stage not in response_labels:
        raise ValueError("There is no interview response to process at this stage.")

    response = st.session_state.pending_response
    question_data = question_for_stage(stage)

    dialogue_turns = [
        {
            "question_number": 0,
            "question": st.session_state.active_question["question"],
            "answer": st.session_state.initial_answer,
        }
    ]
    for index, (question, answer) in enumerate(
        zip(st.session_state.depth_questions, st.session_state.depth_answers),
        start=1,
    ):
        dialogue_turns.append(
            {
                "question_number": index,
                "question": question["question"],
                "answer": answer,
            }
        )

    if stage in stage_to_depth_index:
        current_index = stage_to_depth_index[stage]
        if len(st.session_state.depth_answers) <= current_index:
            dialogue_turns.append(
                {
                    "question_number": current_index + 1,
                    "question": question_data["question"],
                    "answer": response,
                }
            )

    progress = st.progress(5, text="Socratic Evaluator AI is preparing your review...")
    try:
        progress.progress(15, text="Socratic Evaluator AI is applying the interview rubric...")
        evaluation = evaluate_student_response(
            client=client,
            question_data=question_data,
            student_response=response,
            difficulty=st.session_state.active_difficulty,
        )

        progress.progress(45, text="Checking your cumulative understanding...")
        cumulative_evaluations = st.session_state.depth_evaluations + [evaluation]
        checkpoint = assess_understanding_checkpoint(
            client=client,
            original_question=st.session_state.active_question,
            competency=st.session_state.active_competency["skill"],
            difficulty=st.session_state.active_difficulty,
            dialogue_turns=dialogue_turns,
            response_evaluations=cumulative_evaluations,
        )

        checkpoint = reconcile_initial_checkpoint(
            checkpoint=checkpoint,
            evaluation=evaluation,
            question_data=question_data,
            difficulty=st.session_state.active_difficulty,
        )

        follow_ups_answered = {
            "initial_answer": 0,
            "reflection": 1,
            "revision": 2,
        }[stage]
        cycle_complete = checkpoint["expectation_met"] or follow_ups_answered >= 2

        next_question = None
        if not cycle_complete:
            progress.progress(70, text="Question Generator AI is creating one targeted follow-up...")
            depth_number = follow_ups_answered + 1
            next_question = generate_depth_question(
                client=client,
                original_question=st.session_state.active_question,
                competency=st.session_state.active_competency["skill"],
                difficulty=st.session_state.active_difficulty,
                question_number=depth_number,
                dialogue_turns=dialogue_turns,
                latest_evaluation=evaluation,
                checkpoint=checkpoint,
            )
        else:
            progress.progress(80, text="Preparing your reflection and rewrite stage...")
            st.session_state.probe_completion_reason = (
                "the expected understanding was demonstrated"
                if checkpoint["expectation_met"]
                else "the maximum of two follow-up questions was reached"
            )
        progress.progress(100, text="Coaching review ready.")
    finally:
        progress.empty()

    if stage == "initial_answer":
        st.session_state.initial_evaluation = evaluation
    else:
        depth_index = stage_to_depth_index[stage]
        if len(st.session_state.depth_answers) <= depth_index:
            st.session_state.depth_answers.append(response)
        else:
            st.session_state.depth_answers[depth_index] = response

    st.session_state.depth_evaluations.append(evaluation)
    st.session_state.understanding_checkpoint = checkpoint
    add_message(
        "assistant",
        kind="coaching_feedback",
        data={
            "evaluation": evaluation,
            "checkpoint": checkpoint,
            "response_label": response_labels[stage],
            "cycle_complete": cycle_complete,
        },
    )

    if next_question:
        depth_number = len(st.session_state.depth_questions) + 1
        st.session_state.depth_questions.append(next_question)
        st.session_state.follow_up_question = next_question
        add_message(
            "assistant",
            kind="question",
            data={"depth_number": depth_number, **next_question},
        )
        st.session_state.stage = {
            "initial_answer": "reflection",
            "reflection": "revision",
        }[stage]
    else:
        st.session_state.follow_up_evaluation = evaluation
        reflection_message = build_reflection_message(
            evaluation=evaluation,
            checkpoint=checkpoint,
            completion_reason=st.session_state.probe_completion_reason,
        )
        add_message(
            "assistant",
            reflection_message,
        )
        st.session_state.stage = "synthesis_reflection"

    st.session_state.pending_retry = None
    st.session_state.pending_response = ""
    st.session_state.support_mode_active = False
    st.session_state.last_error = None
    st.session_state.last_error_details = None


def run_final_rewrite_evaluation():
    """Evaluate the consolidated answer and produce the final comparison."""
    response = st.session_state.pending_response
    dialogue_turns = [
        {
            "turn_type": "initial_answer",
            "question": st.session_state.active_question["question"],
            "answer": st.session_state.initial_answer,
        }
    ]
    for index, (question, answer) in enumerate(
        zip(st.session_state.depth_questions, st.session_state.depth_answers),
        start=1,
    ):
        dialogue_turns.append(
            {
                "turn_type": f"adaptive_follow_up_{index}",
                "question": question["question"],
                "answer": answer,
            }
        )
    dialogue_turns.extend(
        [
            {
                "turn_type": "learner_reflection",
                "reflection": st.session_state.synthesis_reflection,
            },
            {
                "turn_type": "final_rewrite",
                "question": st.session_state.active_question["question"],
                "answer": response,
            },
        ]
    )

    progress = st.progress(5, text="Socratic Evaluator AI is reading your complete rewrite...")
    try:
        progress.progress(20, text="Applying the interview rubric to the rewritten answer...")
        final_evaluation = evaluate_student_response(
            client=client,
            question_data=st.session_state.active_question,
            student_response=response,
            difficulty=st.session_state.active_difficulty,
            previous_response=st.session_state.initial_answer,
        )

        cumulative_evaluations = st.session_state.depth_evaluations + [
            final_evaluation
        ]
        progress.progress(50, text="Checking whether the rewritten answer demonstrates understanding...")
        final_checkpoint = assess_understanding_checkpoint(
            client=client,
            original_question=st.session_state.active_question,
            competency=st.session_state.active_competency["skill"],
            difficulty=st.session_state.active_difficulty,
            dialogue_turns=dialogue_turns,
            response_evaluations=cumulative_evaluations,
        )

        progress.progress(75, text="Comparing your first answer with your complete rewrite...")
        final_analysis = analyse_concept_understanding(
            client=client,
            original_question=st.session_state.active_question,
            competency=st.session_state.active_competency["skill"],
            difficulty=st.session_state.active_difficulty,
            dialogue_turns=dialogue_turns,
            response_evaluations=cumulative_evaluations,
            checkpoint=final_checkpoint,
        )
        final_analysis["completion_reason"] = (
            f"{st.session_state.probe_completion_reason}, followed by reflection "
            "and a complete rewrite"
        )
        if not final_checkpoint["expectation_met"] and not final_analysis[
            "recommended_topics"
        ]:
            final_analysis["recommended_topics"] = final_checkpoint[
                "remaining_gaps"
            ]
        progress.progress(100, text="Final learning analysis ready.")
    finally:
        progress.empty()

    st.session_state.final_rewrite = response
    st.session_state.final_rewrite_evaluation = final_evaluation
    st.session_state.understanding_checkpoint = final_checkpoint
    st.session_state.final_analysis = final_analysis
    add_message(
        "assistant",
        kind="evaluation",
        data={
            "evaluation": final_evaluation,
            "title": "Complete-rewrite interview rubric",
        },
    )
    add_message("assistant", kind="final_analysis", data=final_analysis)
    record_completed_cycle(final_analysis)
    st.session_state.stage = "completed"
    st.session_state.pending_retry = None
    st.session_state.pending_response = ""
    st.session_state.support_mode_active = False
    st.session_state.last_error = None
    st.session_state.last_error_details = None


def remember_failed_evaluation(stage, error):
    """Keep the learner's saved answer and expose an explicit retry action."""
    st.session_state.pending_retry = {"stage": stage}
    st.session_state.last_error = (
        f"{error} Your response has been preserved. Select ‘Retry current step’ "
        "when you are ready."
    )
    st.session_state.last_error_details = technical_error_details(error)


def _demo_evaluation(readiness, summary, gaps):
    """Create compact static evidence for the pre-generated Plan B example."""
    return {
        "overall_readiness": readiness,
        "summary": summary,
        "dimensions": [],
        "strengths": ["Uses accurate technical language and stays relevant."],
        "gaps": gaps,
    }


def load_plan_b_demo():
    """Load a complete offline example without making any Gemini API calls."""
    question = {
        "question_type": "Role knowledge",
        "question": (
            "Explain the difference between inheritance and composition, and "
            "describe when you would prefer one over the other."
        ),
        "target_competency": "Object-oriented design",
        "interviewer_intent": "Assess conceptual understanding and design judgement.",
        "reference_points": [
            "Inheritance models an is-a relationship",
            "Composition models a has-a relationship",
            "Composition can reduce coupling",
            "Choice depends on substitutability and change",
        ],
        "hint_keywords": ["is-a", "has-a", "coupling", "flexibility"],
    }
    follow_up = {
        "question_type": "Role knowledge",
        "question": (
            "Why can composition make a game character system easier to test "
            "and change than a deep inheritance hierarchy?"
        ),
        "target_competency": "Object-oriented design",
        "interviewer_intent": "Check transfer to a practical design scenario.",
        "reference_points": ["replaceable dependencies", "shallow hierarchy", "testing"],
        "hint_keywords": ["mock", "replace", "coupling", "unit test"],
    }
    second_depth = {
        **follow_up,
        "question": (
            "What trade-off would you consider before replacing inheritance with "
            "composition in an existing codebase?"
        ),
    }
    initial = _demo_evaluation(
        "Developing",
        "The response identifies both mechanisms but does not yet justify the choice.",
        ["Add a practical scenario and explain the coupling trade-off."],
    )
    revised = _demo_evaluation(
        "Interview-ready",
        "The revision adds a relevant game example and explains flexibility.",
        ["Add how the design affects testing."],
    )
    final = _demo_evaluation(
        "Interview-ready",
        "The follow-up connects replaceable components with isolated testing.",
        ["Continue practising concise discussion of trade-offs."],
    )
    rewrite_evaluation = _demo_evaluation(
        "Strong evidence",
        "The complete rewrite integrates the definition, application, testing and trade-offs.",
        ["Keep the same depth while making the spoken response slightly shorter."],
    )
    final_analysis = {
        "overall_readiness": "Interview-ready",
        "summary": (
            "Across the initial response and two follow-up questions, the learner "
            "explains the relationship, flexibility and testing implications clearly."
        ),
        "demonstrated_strengths": [
            "Distinguishes is-a and has-a relationships",
            "Connects composition with reduced coupling",
            "Applies the concept to testing",
        ],
        "evidence": [
            "Explained that Weapon and Armor can change independently",
            "Connected replaceable dependencies with isolated unit tests",
        ],
        "improvements_from_initial": [
            "Added a concrete game-character example",
            "Explained how composition reduces coupling and supports testing",
            "Qualified the choice by acknowledging composition's added indirection",
        ],
        "knowledge_gaps": ["Could discuss when inheritance remains the simpler choice"],
        "recommended_topics": ["Liskov substitution principle", "dependency injection"],
        "ideal_answer_key_points": [
            "is-a versus has-a",
            "substitutability",
            "coupling and flexibility",
            "testing implications",
            "context-dependent trade-offs",
        ],
        "next_practice_step": (
            "Practise one example where inheritance is preferable and justify why."
        ),
        "completion_reason": (
            "the maximum of two follow-up questions was reached, followed by "
            "reflection and a complete rewrite"
        ),
    }

    st.session_state.active_role = "Junior Software Engineer (Plan B demo)"
    st.session_state.active_domain = "Software and Technology"
    st.session_state.active_competency = {
        "skill": "Object-oriented design",
        "evidence": "Understanding of object-oriented programming",
        "priority": "High",
        "mapped_topic": "Tools and Methods",
    }
    st.session_state.topic_plan = [
        {"concept": "Inheritance and composition", "status": "completed"}
    ]
    st.session_state.topic_plan_index = 0
    st.session_state.active_difficulty = "Intermediate"
    st.session_state.active_question_type = "Role knowledge"
    st.session_state.active_question = question
    st.session_state.follow_up_question = follow_up
    st.session_state.initial_answer = (
        "Inheritance creates a parent-child relationship, while composition puts "
        "objects inside another object. Composition is usually more flexible."
    )
    st.session_state.reflection = (
        "Composition makes Weapon and Armor replaceable, which reduces coupling "
        "and makes the Player easier to test independently."
    )
    st.session_state.revised_answer = (
        "Composition adds indirection and more objects, so I would balance its "
        "flexibility against the simplicity of a stable inheritance relationship."
    )
    st.session_state.synthesis_reflection = (
        "I will add a concrete example, testing implications and a balanced trade-off "
        "because my first answer only stated the basic distinction."
    )
    st.session_state.final_rewrite = (
        "Inheritance represents an is-a relationship and works when a subtype can "
        "safely replace its parent. Composition represents a has-a relationship. "
        "For a game character, I would compose Weapon and Armor objects because "
        "they can be changed independently without creating many subclasses. This "
        "reduces coupling, supports isolated testing and makes new combinations "
        "easier to add. However, composition introduces additional objects and "
        "indirection, so I would still use inheritance for a stable, genuine is-a "
        "relationship that satisfies substitutability."
    )
    st.session_state.follow_up_answer = (
        "Composition lets a test replace Weapon or Armor with a small test double, "
        "so the Player can be tested independently. It also avoids behaviour being "
        "spread across a deep hierarchy, reducing unexpected effects from base-class changes."
    )
    st.session_state.initial_evaluation = initial
    st.session_state.revised_evaluation = revised
    st.session_state.follow_up_evaluation = final
    st.session_state.final_rewrite_evaluation = rewrite_evaluation
    st.session_state.depth_questions = [follow_up, second_depth]
    st.session_state.depth_answers = [
        st.session_state.reflection,
        st.session_state.revised_answer,
    ]
    st.session_state.depth_evaluations = [initial, revised, final]
    st.session_state.understanding_checkpoint = {
        "expectation_met": True,
        "rationale": "The learner explained the mechanism, application and trade-off.",
        "demonstrated_evidence": [
            "Explained replaceable components and reduced coupling",
            "Identified composition's added indirection as a trade-off",
        ],
        "remaining_gaps": [],
        "next_probe_focus": "",
    }
    st.session_state.final_analysis = final_analysis
    st.session_state.probe_completion_reason = (
        "the maximum of two follow-up questions was reached"
    )

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "",
            "kind": "question",
            "data": {"follow_up": False, **question},
        },
        {"role": "user", "content": st.session_state.initial_answer},
        {
            "role": "assistant",
            "content": "",
            "kind": "coaching_feedback",
            "data": {
                "evaluation": initial,
                "response_label": "Initial answer",
                "cycle_complete": False,
            },
        },
        {
            "role": "assistant",
            "content": "",
            "kind": "question",
            "data": {"depth_number": 1, **follow_up},
        },
        {"role": "user", "content": st.session_state.reflection},
        {
            "role": "assistant",
            "content": "",
            "kind": "coaching_feedback",
            "data": {
                "evaluation": revised,
                "response_label": "Follow-up answer",
                "cycle_complete": False,
            },
        },
        {
            "role": "assistant",
            "content": "",
            "kind": "question",
            "data": {"depth_number": 2, **second_depth},
        },
        {"role": "user", "content": st.session_state.depth_answers[1]},
        {
            "role": "assistant",
            "content": "",
            "kind": "coaching_feedback",
            "data": {
                "evaluation": final,
                "response_label": "Follow-up answer",
                "cycle_complete": True,
            },
        },
        {
            "role": "assistant",
            "content": """
### Reflection before your complete rewrite

Identify the most important change you will make to your original response and
explain why that change matters.
            """,
            "kind": "text",
            "data": None,
        },
        {"role": "user", "content": st.session_state.synthesis_reflection},
        {
            "role": "assistant",
            "content": f"""
### Rewrite the complete answer

Return to the original interview question:

**{question['question']}**

Combine the relevant ideas from the coaching conversation into one coherent
answer in your own words.
            """,
            "kind": "text",
            "data": None,
        },
        {"role": "user", "content": st.session_state.final_rewrite},
        {
            "role": "assistant",
            "content": "",
            "kind": "evaluation",
            "data": {
                "evaluation": rewrite_evaluation,
                "title": "Complete-rewrite interview rubric",
            },
        },
        {
            "role": "assistant",
            "content": "",
            "kind": "final_analysis",
            "data": final_analysis,
        },
        {
            "role": "assistant",
            "content": "",
            "kind": "summary",
            "data": {
                "readiness": ["Developing", "Interview-ready", "Interview-ready"],
                "text": (
                    "This is a pre-generated completed example for demonstration "
                    "when the live AI service is unavailable."
                ),
            },
        },
    ]
    st.session_state.stage = "completed"
    st.session_state.setup_complete = True
    st.session_state.pending_retry = None
    st.session_state.support_mode_active = False
    st.session_state.last_error = None
    st.session_state.last_error_details = None
    st.session_state.plan_b_mode = True
    st.session_state.stage_hint_levels = {
        "initial_answer": 0,
        "reflection": 0,
        "revision": 0,
        "final_rewrite": 0,
    }


def continue_topic_pathway():
    """Generate the next question and begin a fresh cycle for its concept."""
    next_index = next_topic_concept_index()
    if next_index is None:
        return

    plan = st.session_state.topic_plan
    plan[next_index]["status"] = "current"
    next_concept = plan[next_index]["concept"]
    next_skill = {
        "skill": next_concept,
        "evidence": (
            f"Core concept within {st.session_state.active_competency['mapped_topic']}"
        ),
        "priority": st.session_state.active_competency.get("priority", "Medium"),
        "mapped_topic": st.session_state.active_competency["mapped_topic"],
    }

    progress = st.progress(
        10,
        text=f"Question Generator AI is preparing the next concept: {next_concept}...",
    )
    try:
        progress.progress(
            40,
            text="Matching the next concept to the same role and difficulty...",
        )
        question_data = generate_interview_question(
            client=client,
            question_type=st.session_state.active_question_type,
            difficulty=st.session_state.active_difficulty,
            role_title=st.session_state.active_role,
            competency=next_concept,
            competency_evidence=next_skill["evidence"],
            job_description=(
                st.session_state.job_description_input or SAMPLE_JOB_DESCRIPTION
            ),
            project_context=(
                st.session_state.project_context_input
                if st.session_state.active_question_type == "Experience / STAR"
                else st.session_state.candidate_background_input
            ),
        )
        progress.progress(100, text="Next interview question ready.")
    except Exception as error:
        question_data = fallback_interview_question(
            question_type=st.session_state.active_question_type,
            difficulty=st.session_state.active_difficulty,
            competency=next_concept,
        )
        st.session_state.service_notice = (
            f"{error} A curated backup question was used for the next concept."
        )
    finally:
        progress.empty()

    st.session_state.topic_plan = plan
    st.session_state.topic_plan_index = next_index
    start_interview(
        question_data=question_data,
        skill=next_skill,
        role_title=st.session_state.active_role,
        difficulty=st.session_state.active_difficulty,
        preserve_topic_plan=True,
    )


# --------------------------------------------------
# Setup: job description and Question Generator AI
# --------------------------------------------------
if not st.session_state.setup_complete:
    with st.sidebar:
        render_app_brand()
        st.markdown('<div class="side-label">Workspace</div>', unsafe_allow_html=True)
        st.info(
            "Set up one targeted interview using an anonymised job description "
            "and optional candidate background.",
            icon="✨",
        )
        st.markdown('<div class="side-label">Learning design</div>', unsafe_allow_html=True)
        st.caption(
            "Attempt → feedback → adaptive follow-up → reflection → rewrite → analysis"
        )
        st.divider()
        st.caption("Your progress is stored only for this browser session.")

    render_hero(
        "PrepWise AI",
        "An adaptive AI interview coach for role-specific practice across technology, finance, healthcare and other fields.",
        chips=("🧠 Question Generator AI", "💭 Socratic Evaluator AI", "🚫 No hiring scores"),
    )

    practice_tab, progress_tab, about_tab = st.tabs(
        ["🎯 Practice", "📈 My progress", "ℹ️ How it works"]
    )

    with progress_tab:
        render_progress_dashboard()

    with about_tab:
        with st.container(border=True):
            st.markdown("**How to use this coach**")
            st.markdown(
                """
1. Read and acknowledge the AI use and privacy notice.
2. Paste an anonymised job description from your target field.
3. Optionally add an anonymised background and project summary so questions can connect to your experience.
4. Choose a competency, question type and difficulty.
5. Give a first response without copying an AI-generated answer.
6. Respond to up to two adaptive follow-up questions based on your own explanation.
7. Reflect on the most important improvement and rewrite one complete answer.
8. Review the final analysis and continue with the next concept.

The purpose is to practise independent reasoning and communication—not to
memorise a model response.
                """
            )
        with st.container(border=True):
            st.markdown("**Two AI roles**")
            st.markdown(
                "**Question Generator AI** creates targeted questions. "
                "**Socratic Evaluator AI** provides evidence-based coaching and "
                "progressive hints."
            )
        with st.container(border=True):
            st.markdown("**Presentation Plan B**")
            st.caption(
                "If Gemini is temporarily unavailable, open a complete pre-generated "
                "interview to demonstrate the learning cycle without an API call."
            )
            if st.button(
                "Open pre-generated completed interview",
                use_container_width=True,
            ):
                load_plan_b_demo()
                st.rerun()

    with practice_tab:
        # Step 1 — job description
        with st.container(border=True):
            st.markdown("#### 1 · Paste a job description")
            sample_choice_col, sample_button_col = st.columns(
                [3, 1], vertical_alignment="bottom"
            )
            sample_choice = sample_choice_col.selectbox(
                "Optional sample role",
                options=list(SAMPLE_JOB_DESCRIPTIONS.keys()),
                help="Choose a sample only if you do not have a suitable job description.",
            )
            if sample_button_col.button("✨ Use sample", use_container_width=True):
                st.session_state.job_description_input = SAMPLE_JOB_DESCRIPTIONS[
                    sample_choice
                ]
                st.session_state.job_analysis = None
                # The button click has already started a full Streamlit rerun.
                # Continue this run so the text area is populated without
                # immediately forcing a second page refresh.

            job_description = st.text_area(
                "Job description",
                key="job_description_input",
                height=240,
                placeholder="Paste the responsibilities and requirements here...",
                label_visibility="collapsed",
            )

            description_length = len(job_description)
            if description_length > MAX_JOB_DESCRIPTION_LENGTH:
                st.error(
                    f"Shorten the job description to {MAX_JOB_DESCRIPTION_LENGTH:,} "
                    "characters or fewer."
                )
            elif description_length < 50:
                st.caption(
                    f"{description_length:,}/{MAX_JOB_DESCRIPTION_LENGTH:,} characters · "
                    "enter at least 50 characters"
                )
            else:
                st.caption(
                    f"{description_length:,}/{MAX_JOB_DESCRIPTION_LENGTH:,} characters ✓"
                )

            # AI use and privacy acknowledgement
            with st.expander("Read AI Use & Privacy Notice"):
                st.markdown(
                    """
This is an AI-powered interview practice tool. AI is used to generate
interview questions, provide feedback, and offer hints to support learning.

Please do not enter personal, sensitive, private, or confidential information.
This includes names, contact details, identification numbers, passwords,
confidential company information, unpublished source code, proprietary project
details, or information that you are not authorised to share.

Job descriptions, optional candidate background or project summaries, and
interview responses may be sent to the Gemini API for question generation and
feedback. AI-generated feedback may contain errors or inaccuracies and should
be reviewed critically.
                    """
                )

            st.checkbox(
                "I have read and acknowledge the AI Use & Privacy Notice.",
                key="privacy_consent",
                help=(
                    "You must acknowledge the notice before the job description "
                    "can be analysed or an interview question can be generated."
                ),
            )

            if st.button(
                "Analyse job description",
                type="primary",
                use_container_width=True,
                disabled=(
                    not st.session_state.privacy_consent
                    or description_length < 50
                    or description_length > MAX_JOB_DESCRIPTION_LENGTH
                ),
            ):
                progress = st.progress(
                    10,
                    text="Job-description Analysis AI is reading the role requirements...",
                )
                try:
                    progress.progress(
                        35,
                        text="Identifying competencies and matching supported topics...",
                    )
                    st.session_state.job_analysis = analyse_job_description(
                        client,
                        job_description,
                    )
                    progress.progress(100, text="Job-description analysis ready.")
                    # The analysis section is rendered later in this same run,
                    # so an additional st.rerun() would only create a second
                    # visible refresh.
                except Exception as error:
                    render_ai_error("Job-description analysis failed", error)
                finally:
                    progress.empty()

            if not st.session_state.privacy_consent:
                st.caption("Acknowledge the privacy notice to continue.")

        analysis = st.session_state.job_analysis

        if analysis:
            # Step 2 — role analysis
            with st.container(border=True):
                st.markdown("#### 2 · Role analysis")
                st.markdown(
                    f"**{analysis['role_title']}** "
                    f"{badge(analysis['industry_domain'], 'gray')} "
                    f"{badge(analysis['seniority'], 'violet')} "
                    f"{badge('Suggested: ' + analysis['recommended_difficulty'], 'blue')}"
                )

                with st.expander(
                    f"Extracted skills ({len(analysis['skills'])})",
                    expanded=True,
                    icon="🧾",
                ):
                    for index, skill in enumerate(analysis["skills"]):
                        if index:
                            st.divider()
                        topic_label = (
                            "Outside prototype scope"
                            if skill["mapped_topic"] == "Unsupported"
                            else skill["mapped_topic"]
                        )
                        st.markdown(
                            f"**{skill['skill']}** "
                            f"{badge(skill['priority'], PRIORITY_COLOURS[skill['priority']])} "
                            f"{badge(topic_label, 'gray')}"
                        )
                        st.caption(f"“{skill['evidence']}”")

            supported = [
                skill for skill in analysis["skills"]
                if skill["mapped_topic"] != "Unsupported"
            ]

            if not supported:
                st.warning("No interview-practice competency could be extracted from this description.")
            else:
                # Step 3 — configure the interview
                with st.container(border=True):
                    st.markdown("#### 3 · Configure the interview")

                    skill_index = st.selectbox(
                        "Target competency",
                        options=range(len(supported)),
                        format_func=lambda index: (
                            f"{supported[index]['skill']} — {supported[index]['priority']} priority"
                        ),
                    )
                    selected_skill = supported[skill_index]

                    visible_question_types = [
                        "Project",
                        "Role knowledge",
                        "Scenario",
                    ]
                    if analysis["industry_domain"] == "Software and Technology":
                        visible_question_types.append("Coding")

                    question_category = st.segmented_control(
                        "Question type",
                        options=visible_question_types,
                        default="Role knowledge",
                        help=(
                            "Project uses STAR; Role knowledge checks what you "
                            "understand; Scenario checks workplace judgement; "
                            "Coding provides a code editor for software and "
                            "technology roles."
                        ),
                    ) or "Role knowledge"

                    question_type_descriptions = {
                        "Project": (
                            "Discuss a real experience using STAR: Situation, "
                            "Task, Action and Result, including your own contribution."
                        ),
                        "Role knowledge": (
                            "Explain role-relevant knowledge accurately and show "
                            "how you would apply it."
                        ),
                        "Scenario": (
                            "Respond to a hypothetical workplace situation by "
                            "explaining priorities, actions, risks and judgement."
                        ),
                        "Coding": (
                            "Solve an original coding, SQL or data-structures task "
                            "and explain correctness, efficiency and testing."
                        ),
                    }
                    st.caption(
                        f"**{question_category}:** "
                        f"{question_type_descriptions[question_category]}"
                    )

                    if question_category == "Project":
                        question_type = "Experience / STAR"
                    elif question_category == "Coding":
                        question_type = "Coding / DSA"
                        st.caption(
                            "Choose Coding when the target role or competency "
                            "requires programming, algorithms, SQL or a similar "
                            "code-based task."
                        )
                    elif question_category == "Scenario":
                        question_type = "Scenario / case"
                    else:
                        question_type = "Role knowledge"

                    suggested = analysis["recommended_difficulty"]
                    difficulty = st.segmented_control(
                        "Difficulty",
                        options=DIFFICULTY_LEVELS,
                        default=suggested,
                        help=f"Suggested for this role: {suggested}",
                    ) or suggested

                    candidate_context, candidate_context_length = (
                        render_candidate_context_input(question_category)
                    )

                    if st.button(
                        "Generate interview question",
                        type="primary",
                        use_container_width=True,
                        disabled=(
                            not st.session_state.privacy_consent
                            or candidate_context_length > MAX_PROJECT_CONTEXT_LENGTH
                        ),
                    ):
                        progress = st.progress(
                            10,
                            text="Question Generator AI is preparing your interview question...",
                        )
                        try:
                            progress.progress(
                                40,
                                text="Matching the role, competency and selected difficulty...",
                            )
                            question_data = generate_interview_question(
                                client=client,
                                question_type=question_type,
                                difficulty=difficulty,
                                role_title=analysis["role_title"],
                                competency=selected_skill["skill"],
                                competency_evidence=selected_skill["evidence"],
                                job_description=job_description,
                                project_context=candidate_context,
                            )
                            progress.progress(100, text="Interview question ready.")
                        except Exception as error:
                            question_data = fallback_interview_question(
                                question_type=question_type,
                                difficulty=difficulty,
                                competency=selected_skill["skill"],
                            )
                            st.session_state.service_notice = (
                                f"{error} A curated backup question was used instead."
                            )
                        finally:
                            progress.empty()

                        start_interview(
                            question_data,
                            selected_skill,
                            analysis["role_title"],
                            difficulty,
                        )
                        st.session_state.active_domain = analysis["industry_domain"]
                        st.session_state.setup_complete = True
                        st.rerun()

    st.caption(
        "Do not paste confidential information. Coding questions are original "
        "DSA practice questions and are not claimed to come from LeetCode or any employer."
    )
    st.stop()


# --------------------------------------------------
# Active interview
# --------------------------------------------------
active_concept = st.session_state.active_competency["mapped_topic"]
active_progress = st.session_state.concept_progress[active_concept]
current_question = question_for_stage(st.session_state.stage)

with st.sidebar:
    render_app_brand()
    render_sidebar_cycle(st.session_state.stage)
    st.divider()
    render_practice_context()
    st.divider()
    render_topic_pathway()
    if st.session_state.support_mode_active:
        st.info(
            "Guided-support mode is active. No readiness judgement will be made "
            "until you submit your own attempt.",
            icon="💡",
        )
    st.divider()

    if st.session_state.pending_retry:
        st.warning("Your response is preserved.")
        if st.button(
            "🔁 Retry current step",
            type="primary",
            use_container_width=True,
        ):
            failed_stage = st.session_state.pending_retry["stage"]
            try:
                if failed_stage == "final_rewrite":
                    run_final_rewrite_evaluation()
                else:
                    run_evaluation(failed_stage)
            except Exception as error:
                remember_failed_evaluation(failed_stage, error)
            st.rerun()

    if st.button("↺ New interview", use_container_width=True):
        reset_to_setup()
        st.rerun()

render_workspace_heading()

# The session header and coaching history share one visual workspace. Keeping
# one parent container avoids competing cards and makes the next learner action
# easier to locate.
practice_panel = st.container(border=True)
with practice_panel:
    title_col, action_col = st.columns([3, 1], vertical_alignment="center")
    with title_col:
        st.markdown("#### 💬 Interview practice")
        st.markdown(
            " ".join(
                [
                    badge(st.session_state.active_role, "violet"),
                    badge(st.session_state.active_competency["skill"], "blue"),
                    badge(
                        question_type_label(st.session_state.active_question_type),
                        "gray",
                    ),
                    badge(st.session_state.active_difficulty, "orange"),
                ]
            )
        )
    with action_col:
        st.markdown(readiness_badge(active_progress["latest_readiness"]))

    st.caption(
        f"Stage: **{STAGE_NAMES[st.session_state.stage]}** · "
        f"{CONCEPT_ICONS.get(active_concept, '')} {active_concept}: "
        f"{active_progress['completed_cycles']} completed cycle(s), "
        f"latest completed result: "
        f"{progress_readiness_label(active_progress['latest_readiness'])}"
    )

    # Keeps the question one tap away on small screens after long feedback.
    if st.session_state.stage != "completed":
        with st.expander("Current question", icon="📌"):
            st.markdown(f"**{current_question['question']}**")

if st.session_state.plan_b_mode:
    st.warning(
        "Plan B demonstration: this completed example is pre-generated and did "
        "not call Gemini. Use ‘New interview’ to return to live practice.",
        icon="🎬",
    )

if st.session_state.service_notice:
    st.toast(st.session_state.service_notice, icon="⚠️")
    st.session_state.service_notice = None

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.last_error_details:
        with st.expander("Technical details for troubleshooting", icon="🛠️"):
            st.code(st.session_state.last_error_details, language="text")
            st.caption("Do not share an API key when asking someone for help.")

with practice_panel:
    st.divider()
    st.markdown(
        """
<div class="conversation-title">
  <strong>💬 Coaching conversation</strong>
  <span>Respond in your own words · use hints only when needed</span>
</div>
""",
        unsafe_allow_html=True,
    )
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            render_message(message)


# Progressive help is available for the initial question and every depth
# question. Each question has its own three-level hint allowance.
if st.session_state.stage in {
    "initial_answer",
    "reflection",
    "revision",
    "final_rewrite",
}:
    question_for_hint = question_for_stage(st.session_state.stage)
    current_level = st.session_state.stage_hint_levels.get(
        st.session_state.stage,
        0,
    )
    hint_col, hint_info_col = st.columns([1, 2], vertical_alignment="center")
    if current_level < 3:
        if hint_col.button(
            f"💡 I need a hint ({current_level + 1}/3)",
            key=f"hint_{st.session_state.stage}_{current_level}",
            use_container_width=True,
        ):
            next_level = current_level + 1
            progress = st.progress(
                10,
                text="Socratic Evaluator AI is preparing a progressive hint...",
            )
            try:
                progress.progress(
                    40,
                    text="Selecting key points while preserving your own thinking...",
                )
                hint = generate_hint(client, question_for_hint, next_level)
                progress.progress(100, text="Hint ready.")
            except Exception as error:
                hint = fallback_hint(question_for_hint, next_level)
                st.session_state.service_notice = (
                    f"{error} A curated hint was used instead."
                )
            finally:
                progress.empty()
            add_message("assistant", kind="hint", data=hint)
            st.session_state.stage_hint_levels[st.session_state.stage] = next_level
            st.session_state.support_mode_active = True
            if st.session_state.stage == "initial_answer":
                st.session_state.hint_level = next_level
            st.rerun()
        hint_info_col.caption(
            "Hints reveal key points progressively — never a full answer."
        )
    else:
        st.caption("All three hint levels have been used. Form the response in your own words.")


# A failed API call leaves the current stage and saved answer unchanged. The
# learner retries the evaluation itself rather than resubmitting the answer.
if st.session_state.pending_retry:
    failed_stage = st.session_state.pending_retry["stage"]

    with st.container(border=True):
        st.warning(
            "Your response is safe in this browser session. You do not need to type "
            "or paste it again.",
            icon="🛟",
        )
        with st.expander("View preserved response", expanded=False):
            st.write(st.session_state.pending_response)
        st.caption("Use **Retry current step** in the left navigation when ready.")
        if st.button("Open Plan B example", use_container_width=True):
            load_plan_b_demo()
            st.rerun()


placeholders = {
    "initial_answer": "Type your first interview response...",
    "reflection": "Answer the follow-up question in your own words...",
    "revision": "Answer the follow-up question in your own words...",
    "synthesis_reflection": "What will you improve or preserve in your final answer, and why?",
    "final_rewrite": "Rewrite one complete interview answer in your own words...",
    "completed": "Practice completed",
}

student_input = None
learner_attempt_text = None
use_coding_composer = False
if st.session_state.stage != "completed":
    use_coding_composer = (
        st.session_state.active_question_type == "Coding / DSA"
        and st.session_state.stage != "synthesis_reflection"
    )

    if use_coding_composer:
        # A regular chat input is convenient for prose but too small for an
        # algorithm. This native Streamlit form needs no extra package, works on
        # mobile, and sends both implementation and reasoning to the evaluator.
        composer_id = (
            f"coding_composer_{st.session_state.topic_plan_index}_"
            f"{st.session_state.stage}_{len(st.session_state.depth_questions)}"
        )
        with st.container(border=True):
            st.markdown("#### ⌨️ Coding response")
            st.caption(
                "Code or pseudocode is accepted. Explain the approach so the coach "
                "can evaluate your reasoning, not only the final implementation."
            )
            with st.form(composer_id, clear_on_submit=True):
                selected_language = st.selectbox(
                    "Programming language",
                    CODING_LANGUAGES,
                    key=f"{composer_id}_language",
                )
                code_answer = st.text_area(
                    "Code or pseudocode",
                    height=300,
                    max_chars=MAX_CODE_RESPONSE_LENGTH,
                    placeholder=(
                        "Write your solution here. You can begin with a function, "
                        "class, or clear pseudocode."
                    ),
                    key=f"{composer_id}_code",
                    disabled=st.session_state.pending_retry is not None,
                )
                reasoning_answer = st.text_area(
                    "Approach, assumptions and complexity",
                    height=150,
                    max_chars=MAX_CODE_REASONING_LENGTH,
                    placeholder=(
                        "Explain your approach, important edge cases, and expected "
                        "time and space complexity."
                    ),
                    key=f"{composer_id}_reasoning",
                    disabled=st.session_state.pending_retry is not None,
                )
                coding_submitted = st.form_submit_button(
                    "Submit coding response",
                    type="primary",
                    use_container_width=True,
                    disabled=st.session_state.pending_retry is not None,
                )

            if coding_submitted:
                code_answer = code_answer.strip()
                reasoning_answer = reasoning_answer.strip()
                if not code_answer and not reasoning_answer:
                    st.warning(
                        "Write some code, pseudocode, or an explanation before submitting."
                    )
                else:
                    learner_attempt_text = "\n".join(
                        part for part in [code_answer, reasoning_answer] if part
                    )
                    fence = LANGUAGE_FENCES[selected_language]
                    code_section = (
                        f"```{fence}\n{code_answer}\n```"
                        if code_answer
                        else "_No code submitted._"
                    )
                    reasoning_section = (
                        reasoning_answer
                        if reasoning_answer
                        else "No separate explanation submitted."
                    )
                    student_input = (
                        f"**Language:** {selected_language}\n\n"
                        f"**Code or pseudocode:**\n\n{code_section}\n\n"
                        f"**Approach, assumptions and complexity:**\n\n"
                        f"{reasoning_section}"
                    )
    else:
        student_input = st.chat_input(
            placeholders[st.session_state.stage],
            max_chars=MAX_STUDENT_RESPONSE_LENGTH,
            disabled=st.session_state.pending_retry is not None,
        )
        learner_attempt_text = student_input


if student_input:
    add_message("user", student_input)
    stage = st.session_state.stage
    st.session_state.last_error = None
    st.session_state.last_error_details = None

    # A short request for help is not evidence of interview readiness. Keep the
    # learner at the same stage and provide a Socratic prompt instead of calling
    # the evaluator or storing the request as their attempted answer.
    if is_help_seeking_response(learner_attempt_text or student_input):
        if stage == "synthesis_reflection":
            add_message(
                "assistant",
                """
You can reflect on a meaningful improvement **or** a strength worth keeping.

- If feedback identified a gap, explain what you will address and why.
- If feedback suggested an optional refinement, choose one that matters to you.
- If no meaningful weakness was identified, explain which part of your reasoning
  worked well and how you will preserve it in a concise interview answer.

Try: **“I will improve/preserve ___ because ___.”**
                """,
            )
            st.session_state.support_mode_active = True
        else:
            provide_guided_support(stage)
        st.rerun()

    if not use_coding_composer and len(student_input.strip()) < 10:
        add_message(
            "assistant",
            "Please add one complete sentence or a concrete example before I "
            "review your understanding. If you are unsure, select **I need a hint**.",
        )
        st.rerun()

    st.session_state.support_mode_active = False

    if stage == "synthesis_reflection":
        st.session_state.synthesis_reflection = student_input
        add_message(
            "assistant",
            f"""
### Rewrite the complete answer

Return to the original interview question:

**{st.session_state.active_question['question']}**

Write one complete, coherent answer as though you were speaking to the
interviewer. Combine the relevant ideas you developed during the follow-up
questions. Use your own words; the coach will compare this rewrite with your
first answer.
            """,
        )
        st.session_state.stage = "final_rewrite"
        st.rerun()

    st.session_state.pending_response = student_input

    if stage == "initial_answer":
        st.session_state.initial_answer = student_input
    elif stage == "reflection":
        st.session_state.reflection = student_input
    elif stage == "revision":
        st.session_state.revised_answer = student_input
    elif stage == "final_rewrite":
        st.session_state.final_rewrite = student_input

    try:
        if stage == "final_rewrite":
            run_final_rewrite_evaluation()
        else:
            run_evaluation(stage)
    except Exception as error:
        remember_failed_evaluation(stage, error)

    st.rerun()


if st.session_state.stage == "completed":
    with st.container(border=True):
        st.markdown("#### 🎓 Your learning evidence")
        report = build_practice_report()
        pdf_report = create_pdf_report(report)
        next_concept_index = next_topic_concept_index()
        can_continue_pathway = (
            next_concept_index is not None and not st.session_state.plan_b_mode
        )
        if can_continue_pathway:
            pdf_col, continue_col, restart_col = st.columns(3)
        else:
            pdf_col, restart_col = st.columns(2)
        pdf_col.download_button(
            "Download PDF report",
            data=pdf_report,
            file_name="PrepWise_AI_practice_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            icon="📕",
        )
        if can_continue_pathway:
            next_concept = st.session_state.topic_plan[next_concept_index]["concept"]
            if continue_col.button(
                "Continue pathway",
                type="primary",
                use_container_width=True,
                icon="➡️",
                help=f"Start a new full interview cycle for {next_concept}.",
            ):
                continue_topic_pathway()
                st.rerun()
            st.caption(f"Next concept: **{next_concept}**")
        elif not st.session_state.plan_b_mode:
            st.success(
                "You completed every concept in this prototype's pathway for "
                f"{st.session_state.active_competency['mapped_topic']}."
            )
        if restart_col.button(
            "Choose another topic",
            use_container_width=True,
            icon="🔄",
        ):
            reset_to_setup()
            st.rerun()

st.divider()
st.caption(
    "Session-only prototype: progress may disappear when the app restarts. "
    "Do not submit personal, confidential or proprietary information. "
    "This prototype does not reproduce any employer's confidential or official "
    "hiring rubric. AI feedback may be wrong and should be checked against "
    "reliable domain sources and relevant professional guidance."
)
