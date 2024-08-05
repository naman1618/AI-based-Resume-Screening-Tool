import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o-mini")


def num_of_tokens(text: str) -> int:
    return len(encoder.encode(text))
