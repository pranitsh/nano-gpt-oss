import torch, gc
from torch.utils.data import Dataset, DataLoader
from architecture.tokenizer import get_tokenizer
from datasets import load_dataset
from tqdm import tqdm

batch_size = 1
context_len = 4000

train_file = "train_tokens.npy"
val_file = "val_tokens.npy"

if os.path.exists(train_file) and os.path.exists(val_file):
    print("Loading tokens from disk...")
    train_tokens_np = np.load(train_file)
    val_tokens_np = np.load(val_file)
    train_tokens = train_tokens_np.tolist()
    val_tokens = val_tokens_np.tolist()
    del train_tokens_np, val_tokens_np
    print("Token files loaded.")
else:
    print("Token files not found. Tokenizing...")
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split={
            'train': 'train[:40000]',
            'validation': 'validation[:20000]'
        }
    )
    tokenizer = get_tokenizer()
    print("Tokenizing train data (story by story)...")
    train_tokens_list = []-
    for example in tqdm(dataset["train"]):
        story_tokens = tokenizer.encode(example["text"])
        train_tokens_list.extend(story_tokens)
    print("Tokenizing val data (story by story)...")
    val_tokens_list = []
    for example in tqdm(dataset["validation"]):
        story_tokens = tokenizer.encode(example["text"])
        val_tokens_list.extend(story_tokens)
    print("Converting to numpy arrays...")
    train_tokens_np = np.array(train_tokens_list, dtype=np.uint16)
    val_tokens_np = np.array(val_tokens_list, dtype=np.uint16)
    print("Saving token arrays to disk...")
    np.save(train_tokens_np, train_tokens)
    np.save(val_tokens_np, val_tokens)
    print("Token files saved.")
    del train_tokens_np, val_tokens_np


print(f"Total train tokens: {len(train_tokens):,}")
print(f"Total val tokens: {len(val_tokens):,}")


class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):
        self.input_ids = []
        self.target_ids = []
        for i in tqdm(range(0, len(tokens) - max_length, stride)):
            input_chunk = tokens[i : i + max_length]
            target_chunk = tokens[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
)

del dataset
gc.collect()
