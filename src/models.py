import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from utils import average_f1
from tqdm import tqdm

class CharTokenizer:
    """Simple character-based tokenizer where each character is a token."""
    
    def __init__(self):
        # We can define special tokens if needed
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3
        }
        self.pad_token_id = 0
        self.vocab_size = 65536  # Unicode range for character encoding
        
    def tokenize(self, text):
        """Tokenize text into individual characters."""
        if not isinstance(text, str):
            return []
        return list(text)
    
    def encode(self, text):
        """Convert text to token IDs (character codes)."""
        tokens = self.tokenize(text)
        return [ord(char) if ord(char) < self.vocab_size else self.special_tokens['[UNK]'] for char in tokens]
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return ''.join([chr(token_id) if token_id > 3 else '$' for token_id in token_ids])  # Skip special tokens
    
    def convert_tokens_to_ids(self, tokens):
        """Convert tokens (characters) to IDs."""
        return [ord(char) if ord(char) < self.vocab_size else self.special_tokens['[UNK]'] for char in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs back to tokens (characters)."""
        return [chr(id_val) if id_val > 3 else '[PAD]' for id_val in ids]

class SpaceDataset(Dataset):
	def __init__(self, csv_path, max_length=1024):
		self.data = pd.read_csv(csv_path)
		self.tokenizer = CharTokenizer()
		self.max_length = max_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		tokens = eval(self.data.iloc[idx]["tokens"])
		labels = eval(self.data.iloc[idx]["labels"])
		text = self.data.iloc[idx]["text"] if "text" in self.data.columns else ""

		# Convert tokens to ids
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_ids = input_ids[:self.max_length]
		labels = labels[:self.max_length]

		# Pad if needed
		pad_len = self.max_length - len(input_ids)
		if pad_len > 0:
			input_ids += [self.tokenizer.pad_token_id] * pad_len
			labels += [0] * pad_len
		return torch.tensor(input_ids), torch.tensor(labels), text

class SpaceModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)
		self.fc = nn.Linear(hidden_dim * 2, 2)  # 2 classes: space/no space

	def forward(self, x):
		x = self.embedding(x)
		x, _ = self.lstm(x)
		x = self.fc(x)
		return x

class Trainer:
	def __init__(self, model, train_loader, val_loader, device="cpu", lr=1e-3):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(model.parameters(), lr=lr)

	def train_epoch(self):
		self.model.train()
		total_loss = 0
		all_true = []
		all_pred = []
		all_texts = []
		all_tokens = []
		for batch in tqdm(self.train_loader, desc='Train', leave=False):
			input_ids, labels, texts = batch
			input_ids = input_ids.to(self.device)
			labels = labels.to(self.device)
			outputs = self.model(input_ids)
			loss = self.criterion(outputs.view(-1, 2), labels.view(-1))
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			total_loss += loss.item()
			preds = torch.argmax(outputs, dim=-1).cpu().numpy()
			labels_np = labels.cpu().numpy()
			# Get tokens from input_ids
			for i in range(preds.shape[0]):
				all_true.append(labels_np[i].tolist())
				all_pred.append(preds[i].tolist())
				all_texts.append(texts[i])
				# Convert input_ids to tokens
				tokens = self.train_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
				all_tokens.append(tokens)
		train_f1 = self.calculate_true_f1(all_texts, all_pred, all_tokens)
		return total_loss / len(self.train_loader), train_f1

	def validate(self):
		self.model.eval()
		total_loss = 0
		all_true = []
		all_pred = []
		all_texts = []
		all_tokens = []
		with torch.no_grad():
			for batch in tqdm(self.val_loader, desc='Val', leave=False):
				input_ids, labels, texts = batch
				input_ids = input_ids.to(self.device)
				labels = labels.to(self.device)
				outputs = self.model(input_ids)
				loss = self.criterion(outputs.view(-1, 2), labels.view(-1))
				total_loss += loss.item()
				preds = torch.argmax(outputs, dim=-1).cpu().numpy()
				labels_np = labels.cpu().numpy()
				for i in range(preds.shape[0]):
					all_true.append(labels_np[i].tolist())
					all_pred.append(preds[i].tolist())
					all_texts.append(texts[i])
					tokens = self.val_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
					all_tokens.append(tokens)
		f1 = self.calculate_true_f1(all_texts, all_pred, all_tokens)
		return total_loss / len(self.val_loader), f1

	def calculate_true_f1(self, true_texts, pred_labels, all_tokens):
		# true_texts: list of original texts
		# pred_labels: list of predicted label sequences
		# all_tokens: list of token lists
		pred_texts = []
		for labels, tokens in zip(pred_labels, all_tokens):
			s = ''
			for token, label in zip(tokens, labels):
				# Remove padding tokens
				if token == '[PAD]':
					continue
				s += token  # No need to replace '##' for character tokens
				if label == 1:
					s += ' '
			pred_texts.append(s.strip())
		return average_f1(true_texts, pred_texts)

	def fit(self, epochs=5):
		for epoch in range(epochs):
			train_loss, train_f1 = self.train_epoch()
			val_loss, val_f1 = self.validate()
			print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")