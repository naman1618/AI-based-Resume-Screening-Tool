import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
import nanoid
from llama_index.core import Document as LlamaDocument

from prompts import CHUNK_DETERMINER_PROMPT, CHUNK_SUMMARY_PROMPT, CHUNK_TITLE_PROMPT


class Chunk(BaseModel):
    chunk_id: str
    chunk_idx: int
    propositions: set[str]
    summary: str
    title: str
    keywords: str


class AgenticSplitter:
    def __init__(self, verbose=False, cache_dir=".cache") -> None:
        self._llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self._keyword_generator_agent = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given a sequence of sentences, generate keywords that are about their topic/idea. If a keyword has an acronym, please include it, too. If a keyword is an acronym, generate an expanded version of it, please. These keywords must be separated by a comma.",
                    ),
                    ("user", "{propositions}"),
                ]
            )
            | self._llm
        )
        self._chunk_summary_generator_agent = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        CHUNK_SUMMARY_PROMPT,
                    ),
                    (
                        "user",
                        "Determine the summary of the chunk of the following propositions:\n{propositions}",
                    ),
                ]
            )
            | self._llm
        )

        self._chunk_title_generator_agent = (
            ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        CHUNK_TITLE_PROMPT,
                    ),
                    (
                        "user",
                        "Determine the title of the chunk that this summary belongs to:\n{summary}",
                    ),
                ]
            )
            | self._llm
        )
        self._chunk_finder_agent = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", CHUNK_DETERMINER_PROMPT),
                    (
                        "user",
                        "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--",
                    ),
                    (
                        "user",
                        "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}",
                    ),
                ]
            )
            | self._llm
        )
        self._chunks: dict[str, Chunk] = {}
        self._propositions: set[str] = set()
        self._verbose = verbose
        self._cache_dir = cache_dir

    def _print(self, msg: str):
        if self._verbose:
            print(msg)

    def _create_new_chunk(self, proposition: str, name: str):
        chunk_id = nanoid.generate(
            alphabet="012345679abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
            size=8,
        )
        summary = self._get_new_chunk_summary(proposition=proposition, name=name)
        title = self._get_new_chunk_title(summary)
        keywords = self._get_chunk_keywords([proposition])
        self._chunks[chunk_id] = Chunk(
            chunk_id=chunk_id,
            chunk_idx=len(self._chunks),
            propositions=set([proposition]),
            summary=summary,
            title=title,
            keywords=keywords,
        )
        self._print(f"Created new chunk ({chunk_id}): {title}")

    def _get_new_chunk_summary(self, proposition: str, name: str) -> str:
        return self._chunk_summary_generator_agent.invoke(
            {"propositions": proposition, "name": name}
        ).content

    def _get_new_chunk_title(self, summary: str) -> str:
        return self._chunk_title_generator_agent.invoke({"summary": summary}).content

    def _get_updated_chunk_summary(self, propositions: list[str], name: str) -> str:
        return self._chunk_summary_generator_agent.invoke(
            {"propositions": " ".join(propositions), "name": name}
        ).content

    def _get_updated_chunk_title(self, summary: str) -> str:
        return self._chunk_title_generator_agent.invoke({"summary": summary}).content

    def _get_chunk_keywords(self, propositions: list[str]) -> str:
        return self._keyword_generator_agent.invoke(
            {"propositions": " ".join(propositions)}
        ).content

    def _find_relevant_chunk(self, proposition: str) -> str | None:
        chunk_id = self._chunk_finder_agent.invoke(
            {
                "current_chunk_outline": self._get_chunks_outline(),
                "proposition": proposition,
            }
        ).content
        if chunk_id not in self._chunks:
            return None
        return chunk_id

    def _get_chunks_outline(self):
        chunk_outline = ""

        for chunk_id, chunk in self._chunks.items():
            chunk_str = f"""\t- Chunk ID: {chunk_id}\n\t- Chunk Name: **{chunk.title}**\n\t- Chunk Summary: *{chunk.summary}*\n\n"""
            chunk_outline += chunk_str

        return chunk_outline

    def _add_proposition_to_chunk(self, proposition: str, chunk_id: str, name: str):
        self._chunks[chunk_id].propositions.add(proposition)
        propositions = [*self._chunks[chunk_id].propositions]
        summary = self._get_updated_chunk_summary(propositions=propositions, name=name)
        keywords = self._get_chunk_keywords(propositions=propositions)
        title = self._get_updated_chunk_title(summary)
        self._chunks[chunk_id].title = title
        self._chunks[chunk_id].summary = summary
        self._chunks[chunk_id].keywords = keywords

    def reorganize_chunks(self):
        """
        Reorganizes duplicate chunks and groups those together. Sometimes an LLM is not-so perfect
        """
        entries = sorted(self._chunks.items(), key=lambda entry: entry[1].title)
        for entry in entries:
            chunk_id, chunk = entry
            print(chunk.title)

    def add_propositions(self, propositions: list[str], candidate_name: str):
        for proposition in propositions:
            self.add_proposition(proposition=proposition, name=candidate_name)

    def add_proposition(self, proposition: str, name: str):
        if proposition in self._propositions:
            self._print(
                f'The proposition "{proposition}" already exists and will not be added.'
            )
            return
        self._propositions.add(proposition)
        self._print(f"Adding: '{proposition}'")
        if len(self._chunks) == 0:
            self._print("Detected that this is the first chunk that will be added.")
            self._create_new_chunk(proposition, name)
            return

        chunk_id = self._find_relevant_chunk(proposition=proposition)

        if chunk_id == None:
            self._print("No chunk found")
            self._create_new_chunk(proposition, name)
        else:
            self._print(
                f"Chunk found ({chunk_id}), adding to {self._chunks[chunk_id].title}"
            )
            self._add_proposition_to_chunk(
                proposition=proposition, chunk_id=chunk_id, name=name
            )

    def get_chunks(self) -> list[list[str]]:
        return [[*self._chunks[x].propositions] for x in self._chunks]

    def persist(self, md_hash: str):
        """
        Saves chunks locally to disk so that it can be used again
        """
        file_path = f"{self._cache_dir}/{md_hash}-chunks.json"
        with open(file_path, "w", encoding="utf-8") as f:
            copy = {}
            for key in self._chunks:
                copy[key] = self._chunks[key].model_dump(mode="json")
            json.dump(copy, f, ensure_ascii=False, indent=4)
            self._print(f"Persisted chunks at {file_path}")

    def load_chunks(self, md_hash: str):
        """
        Loads chunks that are persisted
        """
        file_path = f"{self._cache_dir}/{md_hash}-chunks.json"

        if not os.path.exists(file_path):
            raise FileNotFoundError()

        with open(file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            for chunk_id in loaded:
                loaded[chunk_id] = Chunk.model_validate(loaded[chunk_id])

            self._print(
                f"Loaded {len(loaded)} chunk{'s' if len(loaded) != 1 else ''} from {file_path}"
            )

            self._chunks = loaded

    def to_documents(self, candidate_name: str) -> list[LlamaDocument]:
        """
        Converts the chunks into a list of llama documents
        """
        return [
            LlamaDocument(
                text=" ".join(doc.propositions),
                metadata={
                    "candidate": candidate_name,
                    "title": doc.title,
                    "summary": doc.summary,
                    "id": doc.chunk_id,
                    "index": doc.chunk_idx,
                    "keywords": doc.keywords,
                },
            )
            for _, doc in self._chunks.items()
        ]
