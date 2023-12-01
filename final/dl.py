import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# Parameters


# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';', header=None, names=['text', 'emotion'])
    return df['text'].values, df['emotion'].values

train_texts, train_labels = load_data('data/train.txt')
test_texts, test_labels = load_data('data/test.txt')

# Build vocabulary
word_counter = Counter()
for text in train_texts:
    word_counter.update(text.split())
vocab = ['<PAD>', '<UNK>'] + [word for word, freq in word_counter.items() if freq > 1]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
vocab_size = len(vocab)

# Tokenize and encode labels
def tokenize(text):
    return [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text.split()]

def encode_labels(labels):
    label_encoder = LabelEncoder()
    return label_encoder.fit_transform(labels), label_encoder


# Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [torch.tensor(tokenize(text)) for text in texts]
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Padding function
def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = pad_sequence(texts, batch_first=True, padding_value=word_to_idx['<PAD>'])
    return texts, torch.tensor(labels)

# LSTM Model
class LSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(LSTMEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word_to_idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)
    
    # Evaluation function
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def main(BATCH_SIZE = 32, EMBEDDING_DIM = 100, HIDDEN_DIM = 128, EPOCHS = 5):
    loss_list = []

    encoded_train_labels, label_encoder = encode_labels(train_labels)
    encoded_test_labels, _ = encode_labels(test_labels)

    # Data Loaders
    train_dataset = EmotionDataset(train_texts, encoded_train_labels, word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    test_dataset = EmotionDataset(test_texts, encoded_test_labels, word_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model initialization
    model = LSTMEmotionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, len(label_encoder.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            labels = labels.long()  # Convert labels to LongTensor
            loss = criterion(outputs, labels)
            loss.backward()
            loss_list.append(loss.item())
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')



    # Test the model
    test_accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {test_accuracy*100:.2f}%')

    # Save the model to disk
    torch.save(model.state_dict(), 'lstm_emotion_classifier_model.pth')

    # Load the model from disk
    loaded_model = LSTMEmotionClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, len(label_encoder.classes_))
    loaded_model.load_state_dict(torch.load('lstm_emotion_classifier_model.pth'))

    # Validate the loaded model
    val_texts, val_labels = load_data('data/val.txt')
    encoded_val_labels, _ = encode_labels(val_labels)
    val_dataset = EmotionDataset(val_texts, encoded_val_labels, word_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    validation_accuracy = evaluate_model(loaded_model, val_loader)
    print(f'Validation Accuracy: {validation_accuracy*100:.2f}%')
    return test_accuracy, loss_list

hyperparameters = [
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 5
    },
    {
        'BATCH_SIZE': 64,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 5
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 200,
        'HIDDEN_DIM': 128,
        'EPOCHS': 5
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 256,
        'EPOCHS': 5
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 10
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 20
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 30
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 40
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 50
    },
    {
        'BATCH_SIZE': 32,
        'EMBEDDING_DIM': 100,
        'HIDDEN_DIM': 128,
        'EPOCHS': 100
    }
]

results = []
loss_per_epoch_matrix = []

for hyperparameter in hyperparameters:
    test_accuracy, loss_list = main(**hyperparameter)
    loss_per_epoch_matrix.append(loss_list)
    results.append(test_accuracy)

plt.figure(figsize=(10, 5))
for i in range(len(loss_per_epoch_matrix)):
    plt.plot(range(len(loss_per_epoch_matrix[i])), loss_per_epoch_matrix[i], label=f'Hyperparameter {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_lstm.png')

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(results)+1), results)
plt.xlabel('Hyperparameter')
plt.ylabel('Accuracy')
plt.savefig('accuracy_bar_lstm.png')

for i in range(len(results)):
    print(f'Hyperparameter {i+1}: {results[i]*100:.2f}%')


