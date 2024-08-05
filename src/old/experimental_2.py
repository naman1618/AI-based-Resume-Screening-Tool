import gradio as gr
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
import os
import boto3
import warnings


# Suppress warnings
warnings.filterwarnings("ignore")

load_dotenv()
s3 = boto3.client("s3")

# Set up llama_index settings
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
llm = OpenAI(temperature=0.9, model="gpt-4o")


def download_resumes_from_s3(bucket_name, local_dir, prefix=""):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".pdf", ".docx", ".doc", ".txt")):
                file_path = os.path.join(local_dir, key)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                if os.path.exists(file_path):
                    print(f"Skipped: {key}")
                else:
                    s3.download_file(bucket_name, key, file_path)
                    print(f"Downloaded: {key}")


def create_advanced_rag_system(documents):
    # Create a VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)

    # Create a base query engine
    base_query_engine = index.as_query_engine(
        similarity_top_k=100000,
    )

    # Create query engine tools
    tools = [
        QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="resumes",
                description="Useful for answering questions about the content of the resumes and documents",
            ),
        )
    ]

    # Create the LLM
    llm = OpenAI(temperature=0.9, model="gpt-4o")

    # Create the RAG agent
    rag_agent = ReActAgent.from_tools(
        tools, llm=llm, verbose=True, react_chat_kwargs={"max_iterations": 5}
    )

    return rag_agent


def load_or_create_advanced_rag_system(persist_dir="resumes"):
    download_resumes_from_s3(os.environ["S3_BUCKET_NAME"], persist_dir)

    documents = SimpleDirectoryReader(
        input_files=[f"resumes/{file}" for file in os.listdir("resumes")]
    ).load_data()

    # Create the advanced RAG system
    advanced_rag = create_advanced_rag_system(documents)

    return advanced_rag


# Initialize the advanced RAG system once at the start
advanced_rag = load_or_create_advanced_rag_system()


def add_text(history, text):
    if not text:
        raise gr.Error("Enter text")
    history.append((text, ""))
    return history, text


def generate_response(history, query):
    global advanced_rag

    chat_history_str = []
    for entry in history:
        if isinstance(entry, tuple) and len(entry) == 2:
            chat_history_str.append(entry)
        elif isinstance(entry, list) and len(entry) == 2:
            chat_history_str.append(tuple(entry))

    # Use the RAG agent
    rag_response = advanced_rag.chat(query)

    # Get the response
    response = f"RAG Agent: {rag_response}"

    history[-1] = (query, response)
    return history, ""


# Gradio application setup
with gr.Blocks(
    css="""
#chatbot {
  height: 600px;
  overflow-y: auto;
}
"""
) as demo:
    with gr.Column():
        chatbot = gr.Chatbot(value=[], elem_id="chatbot")
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
        submit_btn = gr.Button("Submit")

    def submit_interaction(chatbot_value, txt_value):
        chatbot_value, query = add_text(chatbot_value, txt_value)
        chatbot_value, token_info = generate_response(chatbot_value, query)
        return chatbot_value, "", token_info

    submit_btn.click(
        fn=submit_interaction,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt, gr.Textbox()],
    )

    txt.submit(
        fn=submit_interaction,
        inputs=[chatbot, txt],
        outputs=[chatbot, txt, gr.Textbox()],
    )

demo.queue()
if __name__ == "__main__":
    demo.launch(share=True)
