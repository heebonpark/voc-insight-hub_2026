import streamlit as st
import os

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_glass_card(content_html: str):
    st.markdown(f'<div class="glass-card">{content_html}</div>', unsafe_allow_html=True)

def render_metric(label: str, value: str):
    html = f"""
    <div class="metric-container">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
