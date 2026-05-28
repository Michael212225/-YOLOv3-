import csv
import os

import matplotlib.pyplot as plt


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, 'epoch_metrics.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Missing log file: {csv_path}')

    epochs = []
    train_losses = []
    valid_losses = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            valid_losses.append(float(row['valid_loss']))

    if not epochs:
        raise RuntimeError('No rows found in epoch_metrics.csv')

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, marker='o', linewidth=1.5, label='Train Loss')
    plt.plot(epochs, valid_losses, marker='s', linewidth=1.5, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('YOLOv3 Training Curve (80 Epochs)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(current_dir, 'training_curve.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
