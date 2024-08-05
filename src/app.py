from architecture.resume_agent import ResumeAgent
from util.console import console
import gradio as gr
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import os
from architecture.resume_agent import ResumeAgent
from concurrent.futures import ThreadPoolExecutor
from util.text_extractor import TextExtractor
from llama_index.llms.ollama import Ollama
from transformers import pipeline
import torch
from faster_whisper import WhisperModel
import numpy as np

model = WhisperModel("large-v3", compute_type="int8", device="cpu")

# transcriber = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-base.en",
#     torch_dtype=torch.float32,
# )

# extractor = TextExtractor(use_linux_support=True, persist_file="./cache.json")
# resume_paths = [
#     os.path.join(root, f) for root, __, files in os.walk("./src/resumes") for f in files
# ][:100]
# with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
#     results = [y for x in executor.map(extractor.extract, resume_paths) for y in x]
#     results.sort(key=lambda x: x.candidate_name)
#     results = [x for x in results if x.candidate_name != "None"]
#     print("Done.")

llm = OpenAI(model="gpt-4o-mini", temperature=0.0)
# llm = Ollama(model="gemma2", request_timeout=60.0, temperature=0.0)

# resume_agents = [ResumeAgent(document=doc) for doc in results]

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.0)

action_input = "{'prompt_template': 'Does {candidate_name} have at least 2 years of door-to-door sales experience?', 'results_per_page': 25, 'page': 0}"
action_input_last_page = "{'prompt_template': 'Does {candidate_name} have at least 2 years of door-to-door sales experience?', 'results_per_page': 25, 'page': 32}"

context = f"""
You are a job recruiter bot called "Joyful Jobs". You have all the resumes and information about candidates. Always use pagination for `prompt_many` and end the answer on the page. Whenever you use pagination, please notify the user that the results shown have been limited to reduce API spending, and if they want to show more, they can change the value of the `page` argument. There are {len(ResumeAgent.get_candidate_names())} candidates and {len(ResumeAgent.agents)} resumes belonging to those candidates--which include duplicate resumes and CVs.

# Example
User: Can you list all candidates who have at least 2 years of door-to-door sales experience?

Thought: I need to use the `prompt_many` tool to query all candidates about their door-to-door sales experience, and I will paginate the results to manage the data efficiently. I will provide the first page to the user and then proceed to ask them what page do they want to navigate into.
Action: prompt_many
Action Input: {action_input}
Thought: I have the first page now. I will tell the user that only a limited amount of results are shown, and I will prompt them on what page to navigate to from 0 to 32, which is the max number of pages. I will also keep track of the page the user is on until they decide to leave page navigation. When the user change's the page, I will use `prompt_many` with the `page` argument set with the user's page navigation request.

User: Go to the last page.

Thought: The user wants to navigate to the last page. I need to use the `prompt_many` tool with the `page` argument set to 32 to get the candidates for that page.
Action: prompt_many
Action Input: {action_input_last_page}
Thought: I have the last page now. I will tell the user that only a limited amount of results are shown, and I will prompt them on what page to navigate to from 0 to 32, which is the max number of pages. I will also keep track of the page the user is on until they decide to leave page navigation. When the user change's the page, I will use `prompt_many` with the `page` argument set with the user's page navigation request.
"""

react = ReActAgent.from_tools(
    [
        # FunctionTool.from_defaults(ResumeAgent.get_candidate_names),
        FunctionTool.from_defaults(ResumeAgent.get_total_pages),
        FunctionTool.from_defaults(ResumeAgent.prompt_one),
        FunctionTool.from_defaults(ResumeAgent.prompt_many),
        FunctionTool.from_defaults(ResumeAgent.get_resume_of_many),
        FunctionTool.from_defaults(ResumeAgent.get_resume_of_one),
    ],
    llm=llm,
    verbose=True,
    context=context,
    max_iterations=20,
)


def read_logs():
    with open("output.html", "r") as f:
        return f.read()


def chat(msg: str, history: list):
    # response = react.chat(message=msg)
    # return response.response
    response = react.stream_chat(message=msg)
    bot_msg = ""
    for text in response.response_gen:
        bot_msg += text
        yield bot_msg


def transcribe(audio):
    sr, y = audio
    y = y.astype(np.int8)
    # y /= np.max(np.abs(y))

    segments, _ = model.transcribe(y, beam_size=5)["text"]

    segments = [s.text for s in segments]

    print(segments)

    return ""


with gr.Blocks(
    css="""
#component-2 { height: 60vh }
#component-19 { background-color: black; height: 60vh; overflow: scroll !important; padding: 1rem; }
"""
) as demo:
    with gr.Row():
        with gr.Column():
            gr.ChatInterface(chat)
        logs = gr.HTML(label="Logs")
    gr.Interface(transcribe, gr.Audio(sources=["microphone"]), "text")

    demo.load(read_logs, None, logs, every=1)
demo.queue()
demo.launch(share=True)
