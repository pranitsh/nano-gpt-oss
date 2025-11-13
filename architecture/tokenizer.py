import tiktoken


def get_tokenizer():
    o200k_base = tiktoken.get_encoding("o200k_base")
    tokenizer = tiktoken.Encoding(
        name="o200k_harmony",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={
            **o200k_base._special_tokens,
            **{f"<|reserved_{i}|>": i for i in range(200000, 201088)},
            "<|startoftext|>": 199998,
            "<|endoftext|>": 199999,
            "<|return|>": 200002,
            "<|constrain|>": 200003,
            "<|channel|>": 200005,
            "<|start|>": 200006,
            "<|end|>": 200007,
            "<|message|>": 200008,
            "<|call|>": 200012,
        },
    )
    return tokenizer
