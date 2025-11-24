import streamlit as st
import os
import sys
from pathlib import Path
import base64

from dotenv import load_dotenv
load_dotenv()

# FIXED: Use environment variable for path
NOBOT_PATH = os.getenv('NOBOT_DATA_PATH', str(Path(__file__).parent))
sys.path.insert(0, NOBOT_PATH)

from unifyllm import UnifiedRetriever, RAGPipeline

# Page config
st.set_page_config(
    page_title="noBOT - RNS Institute of Technology", 
    page_icon="🎓", 
    layout="centered"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = ""
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'thinking' not in st.session_state:
    st.session_state.thinking = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'chat_ended' not in st.session_state:
    st.session_state.chat_ended = False

# Function to check if user wants to exit
def is_exit_command(text):
    """Check if the user input is an exit command"""
    exit_keywords = ['exit', 'quit', 'bye', 'goodbye', 'close', 'end chat', 'stop']
    return text.lower().strip() in exit_keywords

# Function to get exit message
def get_exit_message():
    """Return a friendly exit message"""
    return """Thank you for using noBOT! 🎓

I hope I was able to help you today. Feel free to come back anytime you have questions about:
• Campus locations and facilities
• Events and programs
• Class schedules
• Student clubs and activities
• Dining services
• And much more!

Have a great day! 👋

*Click "Clear Chat History" in the sidebar to start a new conversation.*"""

# Function to load and encode image
def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Dynamic CSS based on dark mode
def get_css():
    if st.session_state.dark_mode:
        # Dark Mode Colors
        bg_gradient = "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)"
        header_bg = "linear-gradient(135deg, #0f3460 0%, #16213e 100%)"
        header_border = "#e94560"
        text_primary = "#ffffff"
        text_secondary = "#c0c0c0"
        bot_title_color = "#e94560"
        input_bg = "#16213e"
        input_border = "#e94560"
        input_text = "#eaeaea"
        input_label_color = "#ffffff"
        chat_bg = "#0f3460"
        chat_border = "#16213e"
        empty_text = "#c0c0c0"
        bot_msg_bg = "linear-gradient(135deg, #16213e 0%, #1a1a2e 100%)"
        bot_msg_border = "#0f3460"
        bot_msg_text = "#eaeaea"
        user_msg_bg = "linear-gradient(135deg, #e94560 0%, #c43f56 100%)"
        sidebar_bg = "#0f3460"
        info_bg = "#16213e"
        info_border = "#e94560"
        info_text = "#eaeaea"
        button_bg = "linear-gradient(135deg, #e94560 0%, #c43f56 100%)"
    else:
        # Light Mode Colors
        bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        header_bg = "linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%)"
        header_border = "#667eea"
        text_primary = "#1a202c"
        text_secondary = "#4a5568"
        bot_title_color = "#667eea"
        input_bg = "white"
        input_border = "#667eea"
        input_text = "#1a202c"
        input_label_color = "#ffffff"
        chat_bg = "white"
        chat_border = "#e2e8f0"
        empty_text = "#718096"
        bot_msg_bg = "linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%)"
        bot_msg_border = "#e2e8f0"
        bot_msg_text = "#1a202c"
        user_msg_bg = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        sidebar_bg = "#f7fafc"
        info_bg = "white"
        info_border = "#667eea"
        info_text = "#1a202c"
        button_bg = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    
    return f"""
<style>
    /* Force background on main container */
    .main {{
        background: {bg_gradient} !important;
        padding: 20px;
    }}
    
    /* Fix Streamlit default backgrounds */
    .stApp {{
        background: {bg_gradient} !important;
    }}
    
    .block-container {{
        background: transparent !important;
    }}
    
    .header-container {{
        background: {header_bg};
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 3px solid {header_border};
    }}
    
    .college-logo {{
        width: 120px;
        height: 120px;
        object-fit: contain;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .college-name {{
        font-size: 34px;
        font-weight: 800;
        color: {text_primary} !important;
        margin: 0;
        letter-spacing: 1px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }}
    
    .tagline {{
        font-size: 16px;
        color: {text_secondary} !important;
        margin-top: 8px;
        font-weight: 600;
    }}
    
    .bot-title {{
        color: {bot_title_color} !important;
        margin: 12px 0 0 0;
        font-size: 26px;
        font-weight: 700;
    }}
    
    .stTextInput>div>div>input {{
        background-color: {input_bg} !important;
        border-radius: 30px;
        padding: 16px 24px;
        border: 3px solid {input_border} !important;
        font-size: 17px;
        font-weight: 500;
        color: {input_text} !important;
        transition: all 0.3s ease;
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {header_border} !important;
        box-shadow: 0 0 0 4px rgba(102,126,234,0.2) !important;
    }}
    
    .stTextInput>div>div>input::placeholder {{
        color: {text_secondary} !important;
        font-weight: 500;
    }}
    
    .stTextInput>div>div>input:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
    }}
    
    .chat-container {{
        background: {chat_bg};
        border-radius: 20px;
        padding: 25px;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        border: 2px solid {chat_border};
    }}
    
    .chat-message {{
        padding: 14px 20px;
        border-radius: 20px;
        margin: 12px 0;
        animation: fadeIn 0.4s ease;
        max-width: 85%;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: 15px;
        font-weight: 500;
    }}
    
    @keyframes fadeIn {{
        from {{opacity: 0; transform: translateY(15px);}}
        to {{opacity: 1; transform: translateY(0);}}
    }}
    
    .user-message {{
        background: {user_msg_bg};
        color: white !important;
        margin-left: auto;
        text-align: right;
        float: right;
        clear: both;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }}
    
    .bot-message {{
        background: {bot_msg_bg};
        color: {bot_msg_text} !important;
        margin-right: auto;
        border: 2px solid {bot_msg_border};
        float: left;
        clear: both;
    }}
    
    .exit-message {{
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white !important;
        margin-right: auto;
        border: 2px solid #2f855a;
        float: left;
        clear: both;
        max-width: 90%;
    }}
    
    .thinking-container {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 20px;
        border-radius: 20px;
        margin: 12px 0;
        max-width: 75%;
        background: {bot_msg_bg};
        border: 2px solid {bot_msg_border};
        float: left;
        clear: both;
    }}
    
    .thinking-dots {{
        display: flex;
        gap: 5px;
    }}
    
    .thinking-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {bot_title_color};
        animation: bounce 1.4s infinite ease-in-out both;
    }}
    
    .thinking-dot:nth-child(1) {{ animation-delay: -0.32s; }}
    .thinking-dot:nth-child(2) {{ animation-delay: -0.16s; }}
    
    @keyframes bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    
    .stButton>button {{
        background: {button_bg} !important;
        color: white !important;
        border-radius: 30px;
        border: none;
        padding: 14px 32px;
        font-weight: 700 !important;
        width: 100%;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }}
    
    .stButton>button:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
    }}
    
    section[data-testid="stSidebar"] > div {{
        background: {sidebar_bg} !important;
    }}
    
    .info-section {{
        background: {info_bg} !important;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 4px solid {info_border};
        color: {info_text} !important;
    }}
    
    .info-section h3 {{
        color: {info_text} !important;
        font-weight: 700 !important;
        margin-bottom: 10px;
    }}
    
    .info-section p {{
        color: {info_text} !important;
    }}
    
    .sidebar .stButton>button {{
        background: {info_bg} !important;
        color: {bot_title_color} !important;
        border: 2px solid {info_border} !important;
        padding: 10px 16px;
        font-size: 14px;
        font-weight: 600 !important;
        margin: 6px 0;
        border-radius: 15px;
    }}
    
    .sidebar .stButton>button:hover {{
        background: {button_bg} !important;
        color: white !important;
    }}
    
    .error-banner {{
        background: #ff6b6b;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: 600;
    }}
    
    /* Input label styling */
    .input-label {{
        color: {input_label_color} !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        margin-bottom: 10px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }}
    
    /* Empty state styling */
    .empty-state h2 {{
        color: {text_primary} !important;
    }}
    
    .empty-state p {{
        color: {text_secondary} !important;
    }}
    
    /* Dark mode specific overrides */
    .stMarkdown {{
        color: {text_primary} !important;
    }}
</style>
"""

st.markdown(get_css(), unsafe_allow_html=True)

# Initialize system (cached)
@st.cache_resource
def load_system():
    """Load the RAG system with error handling."""
    try:
        # Check for API key
        api_key = os.getenv('AIzaSyBnBn1uq_x61qE93OKcs9JO-ECifKTE6ik')
        if not api_key:
            st.warning("⚠️ GEMINI_API_KEY not found. Running in retrieval-only mode.")
        
        # Initialize retriever
        retriever = UnifiedRetriever(NOBOT_PATH)
        
        # Initialize RAG pipeline
        rag = RAGPipeline(retriever, api_key=api_key)
        
        return rag
    except Exception as e:
        st.error(f"❌ Failed to initialize noBOT: {str(e)}")
        st.stop()

# Header with logo support
logo_path = os.path.join(NOBOT_PATH, "logo.png")
logo_base64 = get_base64_image(logo_path)

if logo_base64:
    # Header WITH logo
    st.markdown(f"""
    <div class="header-container">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <img src="data:image/png;base64,{logo_base64}" class="college-logo" alt="College Logo">
            <div>
                <p class="college-name">RNS INSTITUTE OF TECHNOLOGY</p>
                <p class="tagline">Channasandra, Bangalore - 560098</p>
            </div>
        </div>
        <div style="margin-top: 18px; padding-top: 18px; border-top: 3px solid #e2e8f0;">
            <h2 class="bot-title">🤖 noBOT - Your Intelligent Campus Assistant</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Header WITHOUT logo (fallback)
    st.markdown("""
    <div class="header-container">
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
            <div style="font-size: 56px;">🎓</div>
            <div>
                <p class="college-name">RNS INSTITUTE OF TECHNOLOGY</p>
                <p class="tagline">Channasandra, Bangalore - 560098</p>
            </div>
        </div>
        <div style="margin-top: 18px; padding-top: 18px; border-top: 3px solid #e2e8f0;">
            <h2 class="bot-title">🤖 noBOT - Your Intelligent Campus Assistant</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Show error banner if exists
if st.session_state.error_message:
    st.markdown(f'<div class="error-banner">⚠️ {st.session_state.error_message}</div>', unsafe_allow_html=True)

# Dark Mode Toggle in Sidebar
with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🌓 Dark Mode")
    with col2:
        if st.button("🔄"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

# Chat display
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <div style="font-size: 80px; margin-bottom: 20px;">🎓</div>
        <h2 class="empty-state">Welcome to noBOT!</h2>
        <p style="font-size: 16px; margin-top: 10px; max-width: 400px; margin-left: auto; margin-right: auto;">
            I'm your intelligent campus assistant. Ask me about campus locations, 
            events, schedules, clubs, dining options, and more!
        </p>
        <p style="font-size: 14px; margin-top: 15px; opacity: 0.8;">
            💡 Try asking a question or click a quick question from the sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        role, content = message["role"], message["content"]
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 {content}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
        else:
            # Check if this is an exit message
            if message.get("is_exit", False):
                st.markdown(f'<div class="chat-message exit-message">🤖 {content}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message">🤖 {content}</div><div style="clear:both;"></div>', unsafe_allow_html=True)
    
    if st.session_state.thinking:
        st.markdown('''
        <div class="thinking-container">
            <span style="font-weight: 600;">🤖 noBOT is thinking</span>
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        </div>
        <div style="clear:both;"></div>
        ''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input area - disable if chat has ended
input_disabled = st.session_state.chat_ended
placeholder_text = "Chat ended. Clear history to start new conversation." if input_disabled else "Type your question here..."

st.markdown('<p class="input-label">💬 Ask me anything about campus</p>', unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input("", placeholder=placeholder_text, key="user_input", label_visibility="collapsed", disabled=input_disabled)
with col2:
    send_button = st.button("Send →", disabled=input_disabled)

# Handle input
if send_button and user_input and user_input != st.session_state.last_processed and not input_disabled:
    st.session_state.last_processed = user_input
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Check if it's an exit command
    if is_exit_command(user_input):
        # Immediately show exit message without calling RAG
        exit_msg = get_exit_message()
        st.session_state.messages.append({
            "role": "assistant", 
            "content": exit_msg,
            "is_exit": True
        })
        st.session_state.chat_ended = True
        st.rerun()
    else:
        st.session_state.thinking = True
        st.session_state.error_message = None
        st.rerun()

# Process if thinking
if st.session_state.thinking:
    try:
        response = load_system().ask(st.session_state.messages[-1]["content"])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.error_message = None
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.session_state.error_message = error_msg
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
    finally:
        st.session_state.thinking = False
        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown('<div class="info-section"><h3>🎓 RNS Institute of Technology</h3><p><strong>noBOT</strong> - AI-Powered Campus Assistant</p></div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="info-section"><h3>ℹ️ What can I help with?</h3><p>📍 Campus Locations<br>📅 Events & Programs<br>🗓️ Class Schedules<br>🎯 Student Clubs<br>🍽️ Dining Services<br>📚 Academic Info</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Questions")
    
    questions = [
        ("📍", "Where is the principal's office?"),
        ("📅", "What events are happening this week?"),
        ("🗓️", "Show me Monday's schedule"),
        ("🎯", "What clubs can I join?"),
        ("🍽️", "Where is the cafeteria located?")
    ]
    
    for icon, question in questions:
        if st.button(f"{icon} {question}", key=question, disabled=input_disabled):
            if question != st.session_state.last_processed:
                st.session_state.last_processed = question
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.thinking = True
                st.rerun()
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_processed = ""
        st.session_state.thinking = False
        st.session_state.error_message = None
        st.session_state.chat_ended = False
        st.rerun()
    
    st.markdown("---")
    st.markdown('<div class="info-section"><strong>📊 Powered by</strong><br>🧠 RAG + Gemini AI<br>🔍 Semantic Search<br>⚡ Real-time Responses</div>', unsafe_allow_html=True)
    
    # Show exit hint
    if not st.session_state.chat_ended:
        st.markdown("---")
        st.markdown('<div class="info-section"><strong>💡 Tip</strong><br>Type "exit", "quit", or "bye" to end the conversation</div>', unsafe_allow_html=True)
        