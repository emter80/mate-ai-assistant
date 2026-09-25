import os
import time

from dotenv import load_dotenv
import streamlit as st
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled

load_dotenv()
set_tracing_disabled(True)

DISPLAY_NAME = "Mariusz"
CV_PATH = "cv.md"
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "4000"))
MAX_RESPONSE_CHARS = int(os.getenv("MAX_RESPONSE_CHARS", "0"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "12000"))


def load_cv_text(cv_path=CV_PATH):
    """Load the generated Markdown CV without extra anonymization or masking."""
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"CV file not found: '{cv_path}'. Generate it first or restore the file before starting the app.")

    try:
        with open(cv_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            raise ValueError(f"CV file is empty: '{cv_path}'")

        print(f"✅ Loaded CV from '{cv_path}'. Using the Markdown file as the public source of truth.")
        return text
    except Exception as e:
        raise RuntimeError(f"Could not read the CV file '{cv_path}': {str(e)}") from e


cv_content = load_cv_text(CV_PATH)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
gemini_url = os.getenv("GEMINI_BASE_URL") or os.getenv("BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
model_name = os.getenv("GEMINI_MODEL") or os.getenv("MODEL") or "gemini-2.5-flash"

client = None
model = None
agent = None

if api_key and gemini_url:
    client = AsyncOpenAI(api_key=api_key, base_url=gemini_url)
    model = OpenAIChatCompletionsModel(model=model_name, openai_client=client)
    agent = Agent(
        name="cv_recruitment_agent",
        model=model,
        instructions=(
            f"You are a professional yet approachable AI assistant representing {DISPLAY_NAME} in a recruitment context. "
            "This profile is based on the public CV stored in cv.md and should be answered as-is without additional anonymization.\n\n"
            f"--- {DISPLAY_NAME.upper()}'S PUBLIC CV PROFILE ---\n{cv_content}\n"
            "------------------------------------------\n\n"
            "Behavioral rules:\n"
            f"1. Answer accurately, politely, and specifically about {DISPLAY_NAME}'s experience, skills, employment history, and projects based on the profile above.\n"
            "2. If a user asks for contact information, answer only from the CV if it is present. Otherwise, say it is not included in the public CV.\n"
            "3. If a user pastes a job description and asks to check the fit, analyze it thoroughly against the public CV profile.\n"
            "4. If someone asks about private or missing information, answer professionally and say it is not included in the public CV.\n"
            "5. Use the same language as the user input.\n"
            "6. Keep the answer concise, professional, and recruiter-friendly.\n"
            "7. Never ever invent facts or claim private details that are not in the CV.\n"
            "8. Chat only about job related topics and avoid personal or unrelated discussions."
        ),
    )


def build_transcript(history, message):
    transcript = ""
    for item in history:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = item.get("content", "")
            transcript += f"{role.title()}: {content}\n"
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            human, assistant = item
            if human:
                transcript += f"User: {human}\n"
            if assistant:
                transcript += f"Assistant: {assistant}\n"

    transcript += f"User: {message}\n"
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        transcript = transcript[-MAX_TRANSCRIPT_CHARS:]
    return transcript


def get_agent_reply(message, history):
    if agent is None:
        return "The demo is not configured yet. Please add a valid Gemini API key and base URL in your .env file."

    if len(str(message)) > MAX_MESSAGE_CHARS:
        return f"Your message is too long. Please keep it under {MAX_MESSAGE_CHARS} characters."

    start_build = time.perf_counter()
    transcript = build_transcript(history, message)
    build_time = time.perf_counter() - start_build
    print(f"[perf] transcript build: {build_time:.3f}s")

    try:
        start_run = time.perf_counter()
        result = Runner.run_sync(agent, transcript)
        run_time = time.perf_counter() - start_run
        print(f"[perf] agent run: {run_time:.3f}s")
        answer = result.final_output or ""
        if MAX_RESPONSE_CHARS > 0:
            answer = answer[:MAX_RESPONSE_CHARS]
        return answer
    except Exception as e:
        return f"A technical error occurred while communicating with the model: {str(e)}. Please try again in a moment!"


def main():
    st.set_page_config(page_title=f"{DISPLAY_NAME}'s AI assistant", page_icon="👤", layout="wide")
    st.title(f"{DISPLAY_NAME}'s AI assistant")
    st.caption("Ask about experience, skills, projects, and role fit based on the public CV.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if agent is None:
        st.warning("The demo is not configured yet. Please add a valid Gemini API key and base URL in your .env file.")
        st.stop()

    with st.sidebar:
        st.subheader("Chat controls")
        if st.button("Clear conversation"):
            st.session_state.messages = []

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])

    prompt = st.chat_input(f"Ask about {DISPLAY_NAME}'s experience, skills, or role fit...")
    if prompt is not None:
        if len(str(prompt)) > MAX_MESSAGE_CHARS:
            st.warning(f"Your message is too long. Please keep it under {MAX_MESSAGE_CHARS} characters.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]
        ]

        with st.spinner("Thinking..."):
            answer = get_agent_reply(prompt, history)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()