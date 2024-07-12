from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import hashlib


class Text2Markdown:
    def __init__(self, cache_dir=".cache", verbose=False) -> None:
        self._llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Convert the resume/cv into a human-readable markdown format. Do not change any wording.",
                ),
                ("human", "{input}"),
            ]
        )

        self._chain = self._prompt | self._llm
        self._cache_dir = cache_dir
        self._verbose = verbose

    def _print(self, msg: str):
        if self._verbose:
            print(msg)

    def to_markdown(self, plain_text: str) -> tuple[str, str]:
        """
        Gets the markdown version of the plain text form of a document.

        Returns:
            A tuple containing the markdown string in the 0th index and
            the hash of the md document in the 1st index
        """
        md_hash = hashlib.sha256(str.encode(plain_text)).hexdigest()
        md_file_path = f"{self._cache_dir}/{md_hash}.md"
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir, exist_ok=True)

        if os.path.exists(md_file_path):
            self._print(
                "Using existing markdown file, so there is no need to convert it."
            )
            with open(md_file_path, "r", encoding="utf-8") as md_file:
                return (md_file.read(), md_hash)

        with open(md_file_path, "w", encoding="utf-8") as md_file:
            self._print(
                "Could not find markdown file. Creating it and storing it in the file system for cache..."
            )
            md_content = self._chain.invoke({"input": plain_text}).content
            self._print(
                f"Successfully generated a markdown from the provided document.\n\tMarkdown length: {len(md_content)}\n\tPlain-text length: {len(plain_text)}"
            )
            md_file.write(md_content)
            return (md_content, md_hash)
