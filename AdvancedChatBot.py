import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import random

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(page_title="Multi-Model AI Chatbot", page_icon="🤖", layout="wide")

# ----------------------------------------------------------------------
# CUSTOM CSS - colours, bubbles, slideshow
# ----------------------------------------------------------------------
st.markdown("""
<style>
/* User chat bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #DCF8C6;
    border-radius: 14px;
    padding: 10px;
    border: 1px solid #b6e6a3;
}

/* Assistant chat bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #E8E6F8;
    border-radius: 14px;
    padding: 10px;
    border: 1px solid #c7c1f0;
}

/* Title gradient */
.gradient-title {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4em;
    font-weight: 800;
}

.tagline {
    font-size: 1.05em;
    color: #555;
    font-style: italic;
}

/* CSS-only slideshow */
.slideshow-container {
    position: relative;
    max-width: 100%;
    height: 180px;
    margin: 10px 0 20px 0;
    border-radius: 12px;
    overflow: hidden;
}
.slideshow-container img {
    position: absolute;
    width: 100%;
    height: 100%;
    object-fit: cover;
    opacity: 0;
    animation: fade 16s infinite;
    border-radius: 12px;
}
.slideshow-container img:nth-child(1) { animation-delay: 0s; }
.slideshow-container img:nth-child(2) { animation-delay: 4s; }
.slideshow-container img:nth-child(3) { animation-delay: 8s; }
.slideshow-container img:nth-child(4) { animation-delay: 12s; }

@keyframes fade {
    0%   { opacity: 0; }
    5%   { opacity: 1; }
    25%  { opacity: 1; }
    30%  { opacity: 0; }
    100% { opacity: 0; }
}

/* History items */
.history-item {
    padding: 6px 10px;
    margin-bottom: 4px;
    border-radius: 8px;
    background-color: #043eb3;
    font-size: 0.85em;
    cursor: default;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HEADER + TAGLINE + SLIDESHOW
# ----------------------------------------------------------------------
TAGLINES = [
    "⚡ Lightning-fast answers, powered by your favourite LLM.",
    "🧠 One chatbot. Three brains. Infinite curiosity.",
    "🚀 Groq, Gemini, or GPT — your call, your conversation.",
    "💬 Ask anything. Switch models like switching channels.",
    "✨ Where speed meets intelligence — pick your engine and go!",
]

if "tagline" not in st.session_state:
    st.session_state.tagline = random.choice(TAGLINES)

st.markdown('<div class="gradient-title">🤖 Multi-Model AI Chatbot</div>', unsafe_allow_html=True)
st.markdown(f'<div class="tagline">{st.session_state.tagline}</div>', unsafe_allow_html=True)

# Pure CSS slideshow (replace these URLs with your own banner images if you like)
st.markdown("""
<div class="slideshow-container">
    <img src="https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200&q=80" />
    <img src="https://images.unsplash.com/photo-1531746790731-6c087fecd65a?w=1200&q=80" />
    <img src="https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=1200&q=80" />
    <img src="https://images.unsplash.com/photo-1555255707-c07966088b7b?w=1200&q=80" />
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# AVATARS
# ----------------------------------------------------------------------
USER_AVATAR = "🧑‍💻"
BOT_AVATAR = "🤖"

# ----------------------------------------------------------------------
# PROVIDER / MODEL CONFIG
# ----------------------------------------------------------------------
PROVIDER_MODELS = {
    "Groq": ["llama-3.1-8b-instant", "openai/gpt-oss-120b", "llama-3.3-70b-versatile", "gemma2-9b-it"],
    "Google Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
}

PROVIDER_HELP = {
    "Groq": "Get a free key at https://console.groq.com",
    "Google Gemini": "Get a key at https://aistudio.google.com/app/apikey",
    "OpenAI": "Get a key at https://platform.openai.com/api-keys",
}

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0)

    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        help=PROVIDER_HELP[provider],
    )

    model_name = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # ---------------- HISTORY SECTION ----------------
    st.subheader("🕘 History")
    if "history" not in st.session_state:
        st.session_state.history = []

    if st.session_state.history:
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        for topic in reversed(st.session_state.history[-15:]):
            st.markdown(f'<div class="history-item">💭 {topic}</div>', unsafe_allow_html=True)
    else:
        st.caption("No topics searched yet.")

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# CHAIN BUILDER (cached per provider/key/model)
# ----------------------------------------------------------------------
@st.cache_resource
def get_chain(provider, api_key, model_name):
    if not api_key:
        return None

    if provider == "Groq":
        llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.7, streaming=True)
    elif provider == "Google Gemini":
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name, temperature=0.7)
    elif provider == "OpenAI":
        llm = ChatOpenAI(api_key=api_key, model=model_name, temperature=0.7, streaming=True)
    else:
        return None

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful, friendly assistant. Answer clearly and concisely."),
        ("user", "{question}")
    ])

    return prompt | llm | StrOutputParser()

chain = get_chain(provider, api_key, model_name)

# ----------------------------------------------------------------------
# MAIN CHAT AREA
# ----------------------------------------------------------------------
if not chain:
    st.warning(f"👆 Please enter your **{provider}** API key in the sidebar to start chatting!")
else:
    # Display chat history
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.history.append(question)

        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(question)

        with st.chat_message("assistant", avatar=BOT_AVATAR):
            message_placeholder = st.empty()
            full_response = ""

            try:
                for chunk in chain.stream({"question": question}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + " ▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ----------------------------------------------------------------------
# EXAMPLES
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 Try these examples:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")
with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using LangChain · Groq · Gemini · OpenAI 🫡")