# main.py

import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config import GROQ_API_KEY  # Make sure this exists: GROQ_API_KEY = "your-api-key"

# === Initialize Groq LLM ===
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
    temperature=0
)

# === Tool Logic ===
def calculator_tool(expr: str) -> str:
    try:
        return str(eval(expr))
    except Exception as e:
        return f"❌ Calculation error: {e}"

def summarization_tool(text: str) -> str:
    resp = llm.invoke(f"Summarize this for a business user:\n\n{text}")
    return resp.content

def company_api_tool(name: str) -> str:
    return f"🏢 Company Info for {name}: Revenue - $5M+, Employees - 50+"

def web_search_tool(query: str) -> str:
    return f"🔎 Dummy search results for: {query}"

# === Shared Logic ===
def run_agent_logic(user_input: str) -> str:
    user_input = user_input.strip()
    if any(op in user_input for op in ['+', '-', '*', '/']):
        return calculator_tool(user_input)
    elif "summarize" in user_input.lower():
        return summarization_tool(user_input)
    elif "company" in user_input.lower():
        return company_api_tool(user_input)
    elif "search" in user_input.lower():
        return web_search_tool(user_input)
    else:
        prompt = PromptTemplate.from_template("Answer this user query as best as you can:\n\n{input}")
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"input": user_input})

# ====================== FASTAPI BACKEND ======================
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use your domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    try:
        result = run_agent_logic(request.user_input)
        return {"response": result}
    except Exception as e:
        return {"response": f"⚠️ Error: {str(e)}"}

# ====================== STREAMLIT FRONTEND ======================
def run_streamlit_ui():
    st.set_page_config(page_title="AI Agent", page_icon="🤖")
    st.title("🤖 Multi-Tool AI Agent")
    
    user_input = st.text_area("Enter your query:")
    
    if st.button("Run Agent") and user_input:
        with st.spinner("Thinking..."):
            try:
                response = run_agent_logic(user_input)
                st.success(response)
            except Exception as e:
                st.error(f"Error: {e}")

# ====================== MAIN ENTRY ======================
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run FastAPI in a separate thread
    threading.Thread(target=start_fastapi, daemon=True).start()

    # Run Streamlit UI
    run_streamlit_ui()
