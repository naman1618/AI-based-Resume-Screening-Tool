from concurrent.futures import ThreadPoolExecutor
from math import ceil
import os
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
import threading
import nanoid
from constants import NANOID_TEMPLATE, OPENAI_TPM
from util.extracted_document import ExtractedDocument
from util.token_counter import num_of_tokens
from llama_index.core.tools import FunctionTool, ToolMetadata
import time
from typing import Self
import re
from functools import cache

__SYSTEM_PROMPT__ = """### Instructions for Resume/CV Analysis

As an expert job recruiter with extensive experience in resume/CV analysis, when provided with a resume or curriculum vitae, you will:

1. Offer a comprehensive and in-depth analysis of the document without suggesting improvements.
2. Focus on aspects such as experience relevance, skill alignment, career progression, and any other notable features.
3. When a job role and its description are provided, relate the resume's content to the job requirements and responsibilities to assess fit.

### Context and purpose

You are to offer detailed responses based on the provided resume/CV and job description to help users understand how well the document positions the candidate for the role in question.

### Example:

Prompt: How well does this resume align with the job description for a Senior Project Manager role at XYZ Corporation?

Example Response: The resume demonstrates a strong background in project management with over 10 years of relevant experience. Notably, the candidate has led multiple large-scale projects to successful completion, which aligns well with the key responsibilities listed in the job description for the Senior Project Manager role at XYZ Corporation. Additionally, the listed skills in stakeholder management and agile methodologies directly correlate with the job requirements...
{additional_context}"""

__USER_PROMPT__ = """(The following is the resume/cv of `{candidate_name}`)
        {resume}"""

__LANGCHAIN__ = ChatPromptTemplate.from_messages(
    [
        ("system", __SYSTEM_PROMPT__),
        ("user", __USER_PROMPT__),
        ("user", "{prompt}"),
    ]
) | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


class ResumeAgent:
    # The total tokens consumed by all agents combined
    total_tokens_consumed: int = 0

    # The total tokens consumed by all agents throughout the lifespan of the application
    total_lifetime_tokens_consumed: int = 0

    # Agents
    agents: list[Self] = []
    agents_dict: dict[str, Self] = {}

    def _reset_tokens():
        ResumeAgent.total_tokens_consumed = 0
        ResumeAgent._restart_timeout()

    def _restart_timeout():
        global __TIMEOUT__
        __TIMEOUT__.cancel()
        __TIMEOUT__ = threading.Timer(60.0, ResumeAgent._reset_tokens)
        __TIMEOUT__.start()

    def __init__(self, document: ExtractedDocument):
        self._name = nanoid.generate(NANOID_TEMPLATE, 4)
        self._document = document
        ResumeAgent.agents.append(self)
        ResumeAgent.agents_dict[document.candidate_name.strip().lower()] = self

    @cache
    def prompt(
        self,
        input: str = Field(description=f"A prompt about the resume or candidate"),
        use_short_answer=Field(
            False,
            description="Determines whether or not a short and straightforward response should be returned.",
        ),
    ) -> tuple[str, str]:
        additional_context = (
            """# Additional Requirements
        Only answer "yes" or "no" or "don't know" to answer the user's prompt."""
            if use_short_answer
            else ""
        )

        if ResumeAgent.total_tokens_consumed + 50_000 > OPENAI_TPM:
            print(
                f"Current tokens consumed is [bold bright_yellow]{ResumeAgent.total_tokens_consumed}[/]. [bright_red]Halting operations for 1 minute...[/]"
            )
            time.sleep(60.0)
            ResumeAgent._reset_tokens()
            print("Operation resumed.")

        try:
            response = __LANGCHAIN__.invoke(
                {
                    "candidate_name": self._document.candidate_name,
                    "resume": self._document.content,
                    "prompt": input,
                    "additional_context": additional_context,
                }
            ).content
        except:
            print("[bold bright_red]We've been rate-limited! Waiting 1 minute...[/]")
            time.sleep(60.0)
            print("[bright_yellow]Trying again...[/yellow]")
            response = __LANGCHAIN__.invoke(
                {
                    "candidate_name": self._document.candidate_name,
                    "resume": self._document.content,
                    "prompt": input,
                    "additional_context": additional_context,
                }
            ).content
        if use_short_answer:
            response = re.sub(r"[^A-Za-z\s']", "", response)

        ResumeAgent._restart_timeout()

        tokens = (
            num_of_tokens(input)
            + num_of_tokens(
                __USER_PROMPT__.format(
                    candidate_name=self._document.candidate_name,
                    resume=self._document.content,
                )
            )
            + num_of_tokens(
                __SYSTEM_PROMPT__.format(additional_context=additional_context)
            )
        )

        ResumeAgent.total_tokens_consumed += tokens
        ResumeAgent.total_lifetime_tokens_consumed += tokens

        return input, response

    def document(self):
        return self._document

    def name(self):
        return self._name

    def as_function_tool(self):
        return FunctionTool(
            self.prompt,
            metadata=ToolMetadata(
                name=self.name(),
                description=f"An agent who is responsible for handling the resume of {self._document.candidate_name}. Prompt them anything about this person's resume, and they will answer to the best of their ability. This tool returns a tuple, which is the prompt and the response.",
            ),
        )

    def get_candidate_names():
        return [*set([agent.document().candidate_name for agent in ResumeAgent.agents])]

    def get_resume_of_many(candidate_names: list[str]):
        """
        Gets the resumes from multiple candidates

        Arguments:
            `candidate_names`: The names of the candidates. To get a list of the candidates' names, refer to `get_candidate_names`

        Returns:
            A dictionary whose keys are the names of the candidate with the values being their respective resume(s). If a candidate does not exist, then they will have value of None. In rare cases, a candidate may have multiple resumes, which their value in the dictionary would be a string array containing their resumes/CVs.
        """

        findings: dict[str, str | list[str]] = {}

        def add_to_dict(candidate_name: str, agent: Self):
            content = agent.document().content
            if candidate_name not in findings:
                findings[candidate_name] = content
            elif type(findings[candidate_name]) != list:
                findings[candidate_name] = [findings[candidate_name], content]
            else:
                findings[candidate_name].append(content)

        for candidate_name in candidate_names:
            candidate_name = candidate_name.lower().strip()

            if candidate_name in ResumeAgent.agents_dict:
                add_to_dict(candidate_name, ResumeAgent.agents_dict[candidate_name])
            else:
                for agent in ResumeAgent.agents:
                    if (
                        candidate_name
                        in agent.document().candidate_name.strip().lower()
                    ):
                        add_to_dict(candidate_name, agent)

        return findings

    def get_resume_of_one(candidate_name: str):
        """
        Gets the resume of a candidate.

        Arguments:
            `candidate_name`: The name of the candidate to get the resume from

        Returns:
            The raw contents of the candidate's resume is returned. If not found, then None is returned.
        """

        candidate_name = candidate_name.lower().strip()

        if candidate_name in ResumeAgent.agents_dict:
            return ResumeAgent.agents_dict[candidate_name].document().content
        else:
            for agent in ResumeAgent.agents:
                if candidate_name in agent.document().candidate_name.strip().lower():
                    return agent.document().content

        return None

    def prompt_one(candidate_name: str, prompt: str):
        """
        Sends a prompt to a candidate to get information from

        Arguments:
            `candidate_name`: The name of the candidate. To get a list of candidates, refer to `get_candidate_names`
            `prompt`: The prompt about the candidate
        """

        candidate_name = candidate_name.lower().strip()

        if candidate_name in ResumeAgent.agents_dict:
            return ResumeAgent.agents_dict[candidate_name].prompt(input=prompt)
        else:
            for agent in ResumeAgent.agents:
                if candidate_name in agent.document().candidate_name.strip().lower():
                    return agent.prompt(input=prompt)

        return None

    def get_total_pages(results_per_page: int):
        """
        Gets the total number of pages for `prompt_many` based on `results_per_page`
        """
        return ceil(len(ResumeAgent.agents_dict) // results_per_page)

    def prompt_many(
        prompt_template: str,
        candidates: list[str] = [],
        results_per_page: int | None = None,
        page: int = 0,
    ):
        """
        Sends a prompt template to multiple candidates to get information from

        Arguments:
            `prompt_template`: A prompt template about a candidate or resume. For example, "Does {candidate_name} have any experience in Python?" is a proper prompt template. The `prompt_template` is sent to every LLM agent to get an answer from about the candidate. Do NOT use any pronouns, and use "{candidate_name}" instead.
            `candidates`: A list of candidate names to specify which the prompt should only be for. For example, ["John Smith", "Amy Wright"] would only get information about John Smith and Amy Wright among all other candidates. If the list is blank, then this will prompt all candidates by default, however, it will only return simple, straightforward answers. For more robust and detailed repsonses, this argument should not be blank.
            `results_per_page`: An unsigned integer that limits the # of prompt responses returned based on this parameter's value. If set to 100, up to 100 results are shown per page. By default this is set to None to show all results. Should you need to paginate results, set this argument to some unsigned integer such as 25.
            `page`: An unsigned integer that indicates the current page of the results window. By default this is set to 0, which is the first page. If `results_per_page` is set to None, this argument will render useless since all results are shown, requiring no pagination feature.

        Returns:
            A string containing all the answers to the prompts
        """
        use_short_answer = len(candidates) == 0
        agents = [*ResumeAgent.agents_dict.values()]

        responses: dict[str, list[str]] = {}

        def prompt_candidate(agent: ResumeAgent):
            return agent.prompt(
                prompt_template.format(candidate_name=agent.document().candidate_name),
                use_short_answer=use_short_answer,
            )

        if results_per_page is not None and (results_per_page * page) > len(agents):
            return None

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = executor.map(
                prompt_candidate,
                (
                    agents
                    if results_per_page is None
                    else agents[results_per_page * page : results_per_page * (page + 1)]
                ),
            )

            for result in results:
                prompt, response = result
                if response not in responses:
                    responses[prompt] = [response]
                else:
                    responses[prompt].append(response)

        if responses == {}:
            return "There are no candidates on this page to query from. Ensure that you're within the page boundaries. To get the max page, use the `get_total_pages` tool."

        # return responses
        string_response = ""

        for key, value in responses.items():
            string_response += key + " " + value[0] + "\n"

        string_response += (
            f"\nOnly up to {results_per_page} results are shown. There is a next page you can navigate to. The max page is {ResumeAgent.get_total_pages(results_per_page)}. To go to any page, use the `prompt_many` tool with a different `page` argument."
            if (results_per_page * (page + 1)) < len(agents)
            else f"\nOnly up to {results_per_page} results are shown. This is the last page, so there are no more pages to navigate to beyond this page."
        )
        return string_response


__TIMEOUT__: threading.Timer = threading.Timer(60.0, ResumeAgent._reset_tokens)
__TIMEOUT__.start()
