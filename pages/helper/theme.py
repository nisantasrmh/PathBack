"""
Modern Dark Glassmorphism Design System & Theme Engine for PathBack.
Provides sleek UI cards, animated badges, glassmorphic metric blocks, and responsive layouts.
"""

import streamlit as st


def inject_custom_theme():
    """Injects high-end glassmorphism styling, custom typography, and animations into Streamlit."""
    theme_css = """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

      :root {
        --pb-bg-dark: #070b13;
        --pb-bg-card: rgba(15, 23, 42, 0.85);
        --pb-card-border: rgba(255, 255, 255, 0.08);
        --pb-primary: #3b82f6;
        --pb-primary-gradient: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #6366f1 100%);
        --pb-accent-gradient: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --pb-danger-gradient: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
        --pb-text-main: #f8fafc;
        --pb-text-muted: #94a3b8;
      }

      /* Global typography */
      html, body, [class*="css"], .stMarkdown, .stText, p, span, h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
      }

      code, pre, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
      }

      /* Background gradient aura */
      .stApp {
        background-color: var(--pb-bg-dark) !important;
        background-image: 
          radial-gradient(circle at 10% 8%, rgba(56, 189, 248, 0.08) 0%, transparent 35%),
          radial-gradient(circle at 90% 18%, rgba(99, 102, 241, 0.09) 0%, transparent 40%),
          radial-gradient(circle at 50% 85%, rgba(16, 185, 129, 0.05) 0%, transparent 45%) !important;
        background-attachment: fixed !important;
      }

      /* Sleek Metric Cards */
      [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 1.15rem 1.4rem !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
      }
      [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
      }
      [data-testid="stMetricLabel"] {
        color: var(--pb-text-muted) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
      }
      [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 1.85rem !important;
      }

      /* Custom Banner Cards */
      .pb-hero-banner {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 18px !important;
        padding: 1.75rem 2rem !important;
        margin-bottom: 2rem !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
      }
      .pb-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(59, 130, 246, 0.15);
        color: #93c5fd;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid rgba(59, 130, 246, 0.3);
        margin-bottom: 0.75rem;
      }
      .pb-pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #3b82f6;
        border-radius: 50%;
        box-shadow: 0 0 8px #3b82f6;
        animation: pbPulse 2s infinite;
      }
      @keyframes pbPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.8); }
      }

      /* Action Card Buttons */
      .pb-action-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        backdrop-filter: blur(14px);
        transition: all 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
      }
      .pb-action-card:hover {
        transform: translateY(-3px);
        border-color: rgba(59, 130, 246, 0.45);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
      }
      .pb-action-icon {
        font-size: 2.2rem;
        margin-bottom: 0.75rem;
        display: inline-block;
      }
      .pb-action-card h3 {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        color: #f8fafc;
      }
      .pb-action-card p {
        font-size: 0.85rem;
        color: var(--pb-text-muted);
        line-height: 1.5;
        margin-bottom: 1rem;
      }

      /* Streamlit Buttons Styling */
      .stButton > button[kind="primary"] {
        background: var(--pb-primary-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35) !important;
        transition: all 0.2s ease !important;
      }
      .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 22px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-2px) !important;
      }

      .stButton > button[kind="secondary"], .stButton > button:not([kind="primary"]) {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
      }
      .stButton > button[kind="secondary"]:hover, .stButton > button:not([kind="primary"]):hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-1px) !important;
      }

      /* Streamlit Form Input Fields */
      .stTextInput > div > div > input,
      .stNumberInput > div > div > input,
      .stSelectbox > div > div > div,
      .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.85) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
      }
      .stTextInput > div > div > input:focus,
      .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
      }

      /* Sidebar polish */
      [data-testid="stSidebar"] {
        background: rgba(9, 13, 22, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(20px) !important;
      }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
