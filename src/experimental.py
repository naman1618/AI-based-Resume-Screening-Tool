from typing import TypedDict
import gradio as gr
from llama_index.core import (
    Document as LlamaDocument,
    VectorStoreIndex,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.postprocessor import LongContextReorder
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    StorageContext,
    SimpleKeywordTableIndex,
    get_response_synthesizer,
)
from llama_index.core.node_parser import MarkdownNodeParser
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
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

NPROC = os.cpu_count()


class ProcessedFile(TypedDict):
    documents: list[LlamaDocument]
    candidate_name: str
    file_path: str


class EngineTool(TypedDict):
    candidate_name: str
    resume_contents: list[str]
    engine: RetrieverQueryEngine


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
        # "documents": SimpleDirectoryReader(
        #     input_files=[f".cache/{md_hash}.md"]
        # ).load_data(show_progress=True),
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
    with ThreadPoolExecutor(max_workers=NPROC) as executor:
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
        index=vector_index,
        similarity_top_k=20,
        verbose=True,
        node_postprocessors=[LongContextReorder()],
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

    def resumes_metadata():
        """
        Returns information about the resume database
        """
        return {
            "resume_count": len(list_of_docs),
            "candidates": [*set([x["candidate_name"] for x in list_of_docs])],
        }

    vector_indexes = {
        x["candidate_name"]: EngineTool(
            candidate_name=x["candidate_name"],
            engine=create_query_engine(x["documents"]),
            resume_contents=[y.get_text() for y in x["documents"]],
        )
        for x in list_of_docs
    }

    def get_response_txt(response):
        return str(response) if str(response) != None else None

    def get_resume_data(candidates: list[str], str_or_query_bundle: str | None):
        results: dict[str, RetrieverQueryEngine] = {}
        with ThreadPoolExecutor(max_workers=NPROC) as executor:
            futures = [
                executor.submit(
                    lambda: (
                        {
                            "candidate_name": candidate,
                            "result": (
                                get_response_txt(
                                    vector_indexes[candidate]["engine"].query(
                                        str_or_query_bundle
                                    )
                                )
                                if str_or_query_bundle
                                else " ".join(
                                    vector_indexes[candidate]["resume_contents"]
                                )
                            ),
                        }
                    )
                )
                for candidate in candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                results[result["candidate_name"]] = result["result"]
        return results

    # Create the RAG agent
    rag_agent = ReActAgent.from_tools(
        [
            # QueryEngineTool(
            #     query_engine=create_query_engine(
            #         [y for x in list_of_docs for y in x["documents"]]
            #     ),
            #     metadata=ToolMetadata(
            #         name="vector_index",
            #         description="Useful for getting information about the resumes",
            #     ),
            # )
            # FunctionTool.from_defaults(
            #     subquestion_agent.query,
            #     name="query_resumes",
            #     description='Gets an aggregation of candidates\' information in their resumes. This takes a string which is the query you want to find about a candidate or multiple candidates. If the query is an aggregation of candidates, please specify their names. An example would be "What is John Smith\'s job experience?" or "Determine if each individual has a degree in Computer Science" or "Determine if Joseph Marbella and Parth Bhambure have any experience with the Rust programming language". This uses a sub question generator, so make sure your query is precise, specific, and descriptive.',
            # ),
            FunctionTool.from_defaults(
                get_resume_data,
                name="resume_data",
                description="Before executing, you MUST know the exact names of all the candidates. To get the names of all candidates, please invoke `resumes_info` tool. This tool provides information about candidate(s). The first argument is a list of candidate names that you want to get information from, and the second argument is vector query about those candidates. If provided a blank string, everything about the candidate is returned. A non-blank string is always preferred to reduce costs.",
            ),
            FunctionTool.from_defaults(
                resumes_metadata,
                name="resumes_info",
                description="Provides information about the resume/CV database including all candidates and the number of entries",
            ),
            # FunctionTool.from_defaults(
            #     get_resume,
            #     name="get_resume_of_candidate",
            #     description="Returns all the information about a candidate. The argument is the name of the candidate. Multiple candidates may share the same name. This tool returns the names of the candidates that match the argument. If a name returns nothing, please refer to the `candidates_found` property in `resumes-info` tool.",
            # ),
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
