from torch.utils.data import Dataset
import json


class TextDataset(Dataset):
    def __init__(self, data_path, encoder):
        with open(data_path, "r") as f:
            self.data = f.readlines()
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def encode(self, text):
        return self.encoder(text)


class JSONDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx])

    def encode(self, text):
        return self.tokenizer(
            text, max_length=self.max_length, truncation=True, padding=True
        )


if __name__ == "__main__":
    from encoder import Encoder
    from config import BASE_ENCODER_MODEL, EXAMPLE_DATA_PATH_TEXT

    encoder = Encoder(model_name_or_path=BASE_ENCODER_MODEL)
    dataset = TextDataset(
        data_path=EXAMPLE_DATA_PATH_TEXT,
        encoder=encoder,
    )
    for i in range(len(dataset)):
        encoded = dataset[i]
        print(encoded)
