import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

import utils
from models import SpaceModel, CharTokenizer
import os
import argparse

class SpaceDatasetTest(Dataset):
	def __init__(self, csv_path, max_length=512):
		self.data = utils.preprocess_dataset(csv_path)
		self.tokenizer = CharTokenizer()
		self.max_length = max_length

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		tokens = self.tokenizer.tokenize(self.data.iloc[idx]["text_no_spaces"])
		# Convert tokens to ids

		input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
		input_ids = input_ids[:self.max_length]

		# Pad if needed
		pad_len = self.max_length - len(input_ids)
		if pad_len > 0:
			input_ids += [self.tokenizer.pad_token_id] * pad_len
		return torch.tensor(input_ids)

def main():
	parser = argparse.ArgumentParser(description="Train space restoration model")
	parser.add_argument('--test_path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset_1937770_3.txt'))
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--model_path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'space_model.pt'))
	parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data'))

	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = CharTokenizer()
	vocab_size = tokenizer.vocab_size

	test_dataset = SpaceDatasetTest(args.test_path)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	model = SpaceModel(vocab_size)
	model.load_state_dict(torch.load(args.model_path, weights_only=True))
	model.eval()
	
	all_tokens = []
	all_pred = []
	for batch in tqdm(test_loader, desc='Test', leave=False):
		input_ids = batch
		input_ids = input_ids.to(device)
		outputs = model(input_ids)
		preds = torch.argmax(outputs, dim=-1).cpu().numpy()
		# Get tokens from input_ids
		for i in range(preds.shape[0]):
			all_pred.append(preds[i].tolist())
			# Convert input_ids to tokens
			tokens = test_loader.dataset.tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy())
			all_tokens.append(tokens)
	pred_texts = []
	pred_pos = []
	for labels, tokens in zip(all_pred, all_tokens):
		s = ''
		ln = 0
		pred_pos_cur = []
		for token, label in zip(tokens, labels):
			# Remove padding tokens
			if token == '[PAD]':
				continue
			s += token  # No need to replace '##' for character tokens
			ln += len(token)  # Each character token has length 1
			if label == 1:
				pred_pos_cur.append(ln)
				s += ' '
		pred_pos.append(pred_pos_cur)
		pred_texts.append(s.strip())
	df_test = utils.preprocess_dataset(args.test_path)
	df_test['text'] = pred_texts
	df_test['predicted_positions'] = pred_pos
	df_test = df_test.set_index('id')
	df_test.to_csv(os.path.join(args.output_dir, 'output.csv'))
	df_test = df_test[['predicted_positions']]
	df_test.to_csv(os.path.join(args.output_dir, 'stepik.csv'))

if __name__ == "__main__":
	main()
