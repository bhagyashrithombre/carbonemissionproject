import streamlit as st

COLORS = {
    "background": "#F6F5F1",
    "surface": "#FFFFFF",
    "primary": "#0E3B36",
    "primary_light": "#174D46",
    "accent": "#C7963E",
    "text": "#1E2A28",
    "muted": "#AAB9B5",
    "border": "#E4E0D6",
    "success": "#2E7D32",
    "danger": "#B23A48"
}


def apply_theme():

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
        }}

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(
                180deg,
                {COLORS["primary"]} 0%,
                #092D29 100%
            );
            border-right: 1px solid rgba(255,255,255,0.08);
        }}

        section[data-testid="stSidebar"] > div {{
            padding-top: 1.5rem;
        }}

        section[data-testid="stSidebar"] * {{
            color: white;
        }}

        .sidebar-brand {{
            padding: 5px 15px 20px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.12);
            margin-bottom: 20px;
        }}

        .sidebar-brand-title {{
            font-size: 22px;
            font-weight: 700;
            color: white;
            margin-bottom: 4px;
        }}

        .sidebar-brand-subtitle {{
            font-size: 12px;
            color: {COLORS["muted"]} !important;
        }}

        .navigation-title {{
            padding: 0 15px 8px 15px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            color: {COLORS["muted"]} !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] {{
            gap: 6px;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            width: 100%;
            padding: 11px 14px;
            border-radius: 10px;
            margin: 0;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
            background-color: {COLORS["primary_light"]};
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
            font-size: 14px;
            font-weight: 500;
            color: #E8EFED !important;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
            display: none;
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(
            input:checked
        ) {{
            background-color: {COLORS["accent"]};
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(
            input:checked
        ) p {{
            color: #FFFFFF !important;
            font-weight: 700;
        }}

        .sidebar-footer {{
            margin-top: 30px;
            padding: 15px;
            border-top: 1px solid rgba(255,255,255,0.12);
            font-size: 11px;
            color: {COLORS["muted"]} !important;
            line-height: 1.6;
        }}

        .main-title {{
            color: {COLORS["primary"]};
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .subtitle {{
            color: #68736F;
            font-size: 16px;
            margin-bottom: 25px;
        }}

        .metric-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        }}

        .metric-title {{
            font-size: 14px;
            color: #68736F;
        }}

        .metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: {COLORS["primary"]};
        }}

        .decision-card {{
            background: {COLORS["surface"]};
            border-left: 5px solid {COLORS["accent"]};
            border-radius: 10px;
            padding: 18px;
            margin: 10px 0;
        }}

        .success-card {{
            background: #EEF7EF;
            border-left: 5px solid {COLORS["success"]};
            border-radius: 10px;
            padding: 18px;
            margin: 10px 0;
        }}

        .warning-card {{
            background: #FFF6E5;
            border-left: 5px solid {COLORS["accent"]};
            border-radius: 10px;
            padding: 18px;
            margin: 10px 0;
        }}

        .section-title {{
            color: {COLORS["primary"]};
            font-size: 24px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 15px;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid {COLORS["border"]};
            padding: 15px;
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


def metric_card(title, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                {title}
            </div>
            <div class="metric-value">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


def sidebar_navigation(pages):

    st.sidebar.markdown(
        """
        <div class="navigation-title">
            Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    selected_page = st.sidebar.radio(
        "Navigation",
        pages,
        label_visibility="collapsed"
    )



    return selected_page