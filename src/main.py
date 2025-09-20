
import torch
from models import SpaceDataset, SpaceModel, Trainer, CharTokenizer
from torch.utils.data import DataLoader
import os
import argparse

def main():
	parser = argparse.ArgumentParser(description="Train space restoration model")
	parser.add_argument('--train_path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv'))
	parser.add_argument('--val_path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'val.csv'))
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--output_path', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'space_model.pt'))

	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	tokenizer = CharTokenizer()
	vocab_size = tokenizer.vocab_size

	train_dataset = SpaceDataset(args.train_path)
	val_dataset = SpaceDataset(args.val_path)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

	model = SpaceModel(vocab_size)
	trainer = Trainer(model, train_loader, val_loader, device=device)
	trainer.fit(epochs=args.epochs)

	# Save trained model
	model_save_path = args.output_path
	torch.save(model.state_dict(), model_save_path)
	print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
	main()
