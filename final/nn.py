import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import matplotlib.pyplot as plt

# Tokenize and pad/truncate
def tokenize(text, max_length):
    tokens = re.findall(r'\w+', text.lower())
    if len(tokens) > max_length:
        return tokens[:max_length]
    return tokens + ['<PAD>'] * (max_length - len(tokens))

def load_data(file_path, max_length):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            text, label = line.strip().split(';')
            texts.append(tokenize(text, max_length))
            labels.append(label)
    return texts, labels

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx):
        self.texts = [[word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, EMBED_DIM, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return self.fc(x)
    
# Function to evaluate model
def evaluate_model(model, data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in data_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
    
# Hyperparameters
hyperparameters = [
    {'MAX_SEQ_LENGTH': 25, 'EPOCHS': 50, 'EMBED_DIM': 50},
    {'MAX_SEQ_LENGTH': 25, 'EPOCHS': 50, 'EMBED_DIM': 100},
    {'MAX_SEQ_LENGTH': 25, 'EPOCHS': 50, 'EMBED_DIM': 200},
    {'MAX_SEQ_LENGTH': 25, 'EPOCHS': 100, 'EMBED_DIM': 100},
    {'MAX_SEQ_LENGTH': 25, 'EPOCHS': 100, 'EMBED_DIM': 200},
    {'MAX_SEQ_LENGTH': 50, 'EPOCHS': 50, 'EMBED_DIM': 100},
    {'MAX_SEQ_LENGTH': 50, 'EPOCHS': 50, 'EMBED_DIM': 200},
    {'MAX_SEQ_LENGTH': 50, 'EPOCHS': 100, 'EMBED_DIM': 100},
    {'MAX_SEQ_LENGTH': 50, 'EPOCHS': 100, 'EMBED_DIM': 200},
    {'MAX_SEQ_LENGTH': 100, 'EPOCHS': 50, 'EMBED_DIM': 100},
    {'MAX_SEQ_LENGTH': 100, 'EPOCHS': 50, 'EMBED_DIM': 200},
]

results = []
loss_per_epoch_matrix = []

def main(MAX_SEQ_LENGTH, EPOCHS, EMBED_DIM):
    loss_list = []
    # Load and process data
    train_texts, train_labels = load_data('data/train.txt', MAX_SEQ_LENGTH)
    test_texts, test_labels = load_data('data/test.txt', MAX_SEQ_LENGTH)

    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    # Build vocabulary
    all_words = [word for text in train_texts for word in text]
    word_counts = Counter(all_words)
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.items() if count > 1]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Model parameters
    vocab_size = len(vocab)

    num_classes = len(label_encoder.classes_)

    # Model, Loss, and Optimizer
    model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Data loaders
    train_dataset = EmotionDataset(train_texts, train_labels, word_to_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = EmotionDataset(test_texts, test_labels, word_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(EPOCHS):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        loss_list.append(loss.item())

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy*100:.2f}%')

    # Save the model to disk
    torch.save(model.state_dict(), 'emotion_classifier_model.pth')

    # Load the model from disk
    loaded_model = EmotionClassifier(vocab_size, EMBED_DIM, num_classes)
    loaded_model.load_state_dict(torch.load('emotion_classifier_model.pth'))



    # Load and process validation data
    val_texts, val_labels = load_data('data/val.txt', MAX_SEQ_LENGTH)
    val_labels = label_encoder.transform(val_labels)
    val_dataset = EmotionDataset(val_texts, val_labels, word_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Evaluate the loaded model on validation data
    accuracy = evaluate_model(loaded_model, val_loader)
    print(f'Validation Accuracy: {accuracy*100:.2f}%')

    results.append(accuracy)
    loss_per_epoch_matrix.append(loss_list)

for hyperparameter in hyperparameters:
    main(**hyperparameter)


plt.figure(figsize=(10, 5))
for i in range(len(loss_per_epoch_matrix)):
    plt.plot(range(1, hyperparameters[i]['EPOCHS']+1), loss_per_epoch_matrix[i], label=f'Hyperparameter {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot_nn.png')

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(results)+1), results)
plt.xlabel('Hyperparameter')
plt.ylabel('Accuracy')
plt.savefig('accuracy_bar_nn.png')


for i in range(len(results)):
    print(f'Hyperparameter {i+1}: {results[i]*100:.2f}%')
