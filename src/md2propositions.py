from langchain_openai import ChatOpenAI
from prompts import PROPOSITION_GENERATOR_PROMPT
from langchain_core.prompts import ChatPromptTemplate
from parsers import md_json_parser
import os
import json


class Markdown2Propositions:
    def __init__(self, verbose=False, cache_dir=".cache") -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PROPOSITION_GENERATOR_PROMPT,
                ),
                ("human", "{input}"),
            ]
        )

        self._chain = self._prompt | self._llm | md_json_parser
        self._cache_dir = cache_dir
        self._verbose = verbose

    def _print(self, msg: str):
        if self._verbose:
            print(msg)

    def get_propositions(
        self, text2md_output: tuple[str, str]
    ) -> tuple[list[str], str]:
        md_content, md_hash = text2md_output
        json_path = f"{self._cache_dir}/{md_hash}.json"

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as json_file:
                propositions: list[str] = json.load(json_file)
                self._print(
                    f"An existing json propositions file has been found containing {len(propositions)} proposition{'s' if len(propositions) != 1 else ''}."
                )
                return propositions[1:], propositions[0].title()

        with open(json_path, "w", encoding="utf-8") as json_file:
            self._print(
                "Could not find existing json propositions file. Invoking proposition agent to generate one..."
            )
            propositions: list[str] = self._chain.invoke({"input": md_content})
            self._print(
                f"Generated {len(propositions)} proposition{'s' if len(propositions) != 1 else ''}"
            )
            json.dump(propositions, json_file, ensure_ascii=False, indent=4)
            return propositions[1:], propositions[0].title()
