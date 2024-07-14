from typing import TypedDict
import gradio as gr
from llama_index.core import (
    Document as LlamaDocument,
    VectorStoreIndex,
)
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    StorageContext,
    SimpleKeywordTableIndex,
    get_response_synthesizer,
)
from dotenv import load_dotenv
import os
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from agentic_splitter import AgenticSplitter
from llama_index.core import Settings
from file_loader import FileLoader
from hybrid_retriever import HybridRetriever
from llama_index.core.agent import ReActAgent

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
s3 = boto3.client("s3")

# Set up llama_index settings
embed_model = OpenAIEmbedding(model="text-embedding-3-large")


class ProcessedFile(TypedDict):
    documents: list[LlamaDocument]
    candidate_name: str
    file_path: str


def process_file(path: str) -> ProcessedFile:
    loader = FileLoader(verbose=True)
    splitter = AgenticSplitter(verbose=True)

    propositions, md_hash, name = loader.load(file_path=path)
    try:
        splitter.load_chunks(md_hash)
    except FileNotFoundError as e:
        splitter.add_propositions(propositions=propositions, candidate_name=name)
        splitter.persist(md_hash=md_hash)

    return {
        "documents": splitter.to_documents(candidate_name=name),
        "candidate_name": name,
        "file_path": path,
    }


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


def preprocess_and_store_documents() -> list[ProcessedFile]:
    download_resumes_from_s3(os.environ["S3_BUCKET_NAME"], "resumes")

    documents = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(process_file, os.path.join(root, file))
            for root, _, files in os.walk("resumes")
            for file in files
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                documents.append(result)

    return documents


def create_query_engine(docs: list[LlamaDocument]):
    nodes = Settings.node_parser.get_nodes_from_documents(docs)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(docs)

    vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
    keyword_index = SimpleKeywordTableIndex(nodes, storage_context=storage_context)

    vector_retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=20, verbose=True
    )

    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index, verbose=True)
    custom_retriever = HybridRetriever(vector_retriever, keyword_retriever)

    response_synthesizer = get_response_synthesizer()
    custom_query_engine = RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=response_synthesizer,
    )

    return custom_query_engine


def create_advanced_rag_system():
    list_of_docs = preprocess_and_store_documents()

    tools = [
        # QueryEngineTool(
        #     query_engine=create_query_engine(
        #         [y for x in list_of_docs for y in x['candidate_name']]
        #     ),
        #     metadata=ToolMetadata(
        #         name="resumes_metadata",
        #         description="Contains information about the resume documents",
        #     ),
        # )
    ]

    for docs in list_of_docs:
        tools.append(
            QueryEngineTool(
                query_engine=create_query_engine(docs["documents"]),
                metadata=ToolMetadata(
                    name=f"{docs['candidate_name']}",
                    description=f"Contains the resume contents of {docs['candidate_name']}",
                ),
            )
        )

    # Create the LLM
    llm = OpenAI(temperature=0.5, model="gpt-4o", max_tokens=None)

    subquestion_agent = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools, llm=llm
    )

    # Create the RAG agent
    rag_agent = ReActAgent.from_tools(
        [
            FunctionTool.from_defaults(
                subquestion_agent.query,
                name="Resume",
                description='Provides resume information about a candidate. This takes a string which is the query you want to find about a candidate. An example would be "What is John Smith\'s job experience?" or "Return everything about the individuals with a computer science degree"',
            )
        ],
        llm=llm,
        verbose=True,
        react_chat_kwargs={"max_iterations": 5},
        context="You are a job recruier. You are an expert in the finding, screening and attracting of applicants for open positions. You have been given a list of resumes, and it is your job to identify and examine prospects of each candidate and determine which candidates are best for a job position.",
    )
    # rag_agent.update_prompts(
    #     {"agent_worker:system_prompt": PromptTemplate(RAG_AGENT_PROMPT)}
    # )

    return rag_agent


# Initialize the advanced RAG system once at the start
advanced_rag = create_advanced_rag_system()


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
    response = f"RAG Agent:\n{rag_response}"

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
