import json
from gliner import GLiNER
import re
from prompts import STRAIGHT_FORWARD_PROMPT
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_ner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
_labels = ["name", "phone number", "university/college", "college_degree"]
_llm = ChatOpenAI(temperature=0.5)


def _parse_direct_answer(gpt_response: dict[str, str]) -> None | str:
    return gpt_response.content if gpt_response.content == "None" else None


_chain = (
    ChatPromptTemplate.from_messages(
        [
            ("system", STRAIGHT_FORWARD_PROMPT),
            ("human", "{proposition}\n\nWhat is the name of the college degree only?"),
        ]
    )
    | _llm
    | _parse_direct_answer
)


def information_integrity_agent_output_parser(gpt_response: dict[str, str]) -> str:
    return gpt_response.content


def md_json_parser(gpt_response: dict[str, str]) -> list[str]:
    return json.loads(gpt_response.content[7:-3].strip())


def extract_person_metadata(propositions: list[str]) -> dict[str, str]:
    data: dict[str, list[set[str]]] = {
        "email": [set(), set()],
    }
    for proposition in propositions:
        # email regex
        found_email = re.search(
            "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])",
            proposition.lower(),
        )
        if found_email is not None:
            data["email"][1].add(found_email.group(0))
        entities = _ner.predict_entities(proposition, _labels, threshold=0.5)
        for entity in entities:
            key, value = entity["label"], entity["text"]
            if key not in data:
                data[key] = [set(), set()]

            match key:
                case "university/college":
                    value = value.lower()
                    if "college" in value or "university" in value:
                        data[key][1].add(value.replace("the", "").strip().title())
                case "college_degree":
                    value = value.lower()
                    if value not in data[key][1]:
                        data[key][0].add(proposition)
                        answer = _chain.invoke({"proposition": proposition})
                        if answer != None:
                            data[key][1].add(answer)
                case _:
                    data[key][1].add(value)
    for key in data:
        data[key] = ", ".join([x.rstrip(".") for x in data[key][1]])
    return data
