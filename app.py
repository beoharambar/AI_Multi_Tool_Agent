import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY

# ======== Inject Custom CSS ========
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(to right, #1f1c2c, #928DAB);
        color: #f1f1f1;
    }

    .main {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 2rem;
        max-width: 800px;
        margin: auto;
        margin-top: 3rem;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
    }

    textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #fff !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    button {
        background: linear-gradient(to right, #7873f5, #4e54c8) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        margin-top: 10px;
    }

    .stTextArea > label {
        font-weight: 600;
        font-size: 1.1rem;
    }

    .stSpinner {
        color: white !important;
    }

    .stMarkdown h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ======== Initialize Groq LLM ========
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
    temperature=0
)

# ======== Define Tools ========
def web_search_tool(query: str) -> str:
    return f"🔎 Dummy search results for: {query}. Growth Grid is a leading B2B growth agency."

def company_api_tool(name: str) -> str:
    return f"🏢 Company Info for {name}: Revenue - $5M+, Employees - 50+."

def summarization_tool(text: str) -> str:
    resp = llm.invoke(f"Summarize this for a business user:\n\n{text}")
    return resp.content

def calculator_tool(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"❌ Calculation error: {e}"

# ======== Streamlit UI ========
st.markdown("# 🤖 Multi-Tool AI Agent")

with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)

    user_input = st.text_area("Enter your query:")

    if st.button("Run Agent") and user_input:
        with st.spinner("Thinking..."):
            try:
                # Tool Routing Logic
                if any(op in user_input for op in ['+', '-', '*', '/']):
                    response = calculator_tool(user_input)
                elif "summarize" in user_input.lower():
                    response = summarization_tool(user_input)
                elif "company" in user_input.lower():
                    response = company_api_tool(user_input)
                elif "search" in user_input.lower():
                    response = web_search_tool(user_input)
                else:
                    prompt = PromptTemplate.from_template(
                        "Answer this user query as best as you can:\n\n{input}"
                    )
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"input": user_input})

                st.success(response)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
