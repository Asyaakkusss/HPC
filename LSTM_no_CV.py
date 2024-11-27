import os
import re
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.io
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def collate_fn(batch):
    signals, labels = zip(*batch)
    
    # Process the signals in smaller chunks
    input_size = 128  # Choose a smaller chunk size
    processed_signals = []
    
    for signal in signals:
        # Limit each signal to have the last dimension as `input_size`
        processed_signal = signal[:, :input_size]
        processed_signals.append(processed_signal)
    
    # Stack the signals into a batch tensor
    processed_signals = torch.stack(processed_signals, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return processed_signals, labels



class SignalDataset(Dataset):
    def __init__(self, data, labels, max_length=None):
        self.data = data
        self.labels = labels
        self.max_length = max_length if max_length else max(len(signal[0]) for signal in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        # Truncate or pad the signal to match the input size
        input_size = 128  # Ensure this matches the model's input size
        padded_signal = np.zeros((signal.shape[0], input_size))
        padded_signal[:, :min(signal.shape[1], input_size)] = signal[:, :input_size]

        return torch.tensor(padded_signal, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Preprocessing logic for .mat files
def preprocess_file(index, data_dir):
    """
    Preprocess a single .mat file and return the processed signal.
    Modify this function to implement the actual preprocessing logic.
    """
    try:
        file_name = os.listdir(data_dir)[index]
        full_path = os.path.join(data_dir, file_name)
        mat_data = scipy.io.loadmat(full_path)

        # Check if 'val' exists in the loaded .mat file
        if 'val' in mat_data:
            signals = mat_data['val']
            if signals.size > 0:
                return signals, len(signals)  # Placeholder for t_downsample
            else:
                print(f"Skipping file: Signal is empty in {file_name}")
                return None, None
        else:
            print(f"Skipping file: 'val' key not found in {file_name}")
            return None, None
    except Exception as e:
        print(f"Error processing file at index {index}: {e}")
        return None, None

# Function to extract preterm flag and age details
def get_preterm_flag(header_fullFileName):
    with open(header_fullFileName, 'r') as file:
        buffer = file.read()

    # Extract gestational age
    substr = 'Rectime'
    loc = buffer.find(substr)
    gestational_age_string = buffer[loc + len(substr):loc + len(substr) + 5].strip()

    if len(gestational_age_string) < 3 or not gestational_age_string[:2].isdigit():
        raise ValueError(f"Invalid gestational age format in {header_fullFileName}: {gestational_age_string}")

    gestational_age = int(gestational_age_string[:2]) * 7
    if len(gestational_age_string) > 3 and gestational_age_string[3].isdigit():
        gestational_age += round(int(gestational_age_string[3]) / 10 * 7)

    # Extract delivery age
    substr = 'Gestation'
    loc = buffer.find(substr)
    delivery_age_string = buffer[loc + len(substr):loc + len(substr) + 5].strip()

    if not delivery_age_string:
        substr = 'age at delivery(w/d):'
        loc = buffer.find(substr)
        delivery_age_string = buffer[loc + len(substr):loc + len(substr) + 5].strip()

    delivery_age_cleaned = re.sub(r'[^0-9]', '', delivery_age_string)

    if len(delivery_age_cleaned) >= 2:
        delivery_age = int(delivery_age_cleaned[:2]) * 7
        if len(delivery_age_cleaned) > 2:
            delivery_age += round(int(delivery_age_cleaned[2]) / 10 * 7)
    else:
        raise ValueError(f"Invalid delivery age format in {header_fullFileName}: {delivery_age_string}")

    is_preterm = 1 if delivery_age < 259 else 0
    return is_preterm, gestational_age, delivery_age

# Define the BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, bidirectional=True):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        h_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, 
                          x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers * 2 if self.bidirectional else self.num_layers, 
                          x.size(0), self.hidden_size).to(x.device)

        # LSTM output
        out, _ = self.bilstm(x, (h_0, c_0))  # out: [batch_size, seq_len, hidden_size * num_directions]

        # Fully connected layer
        out = self.fc(out[:, -1, :])  # Take the last time step
        return self.softmax(out)




data_dir = r"C:\Users\User\Desktop\CSDS438\term_DB"
header_dir = r"C:\Users\User\Desktop\CSDS438\term_DB"  # Directory for .hea files
train_data = []
train_labels = []

# Process the dataset
for k, file_name in enumerate(os.listdir(data_dir)):
    if file_name.endswith('.mat'):
        try:
            signals, _ = preprocess_file(k, data_dir)
            if signals is not None and len(signals) > 0:
                # Get corresponding header file
                header_file = os.path.join(header_dir, file_name.replace('.mat', '.hea'))
                if os.path.exists(header_file):
                    is_preterm, _, _ = get_preterm_flag(header_file)
                    train_data.append(signals)
                    train_labels.append(is_preterm)
                else:
                    print(f"Header file not found for {file_name}")
            else:
                print(f"Skipping file: Signal is empty or invalid in {file_name}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Check if valid training data exists
if len(train_data) == 0 or len(train_labels) == 0:
    raise ValueError("No valid training data available. Ensure the preprocessing logic is correct.")

# Split dataset
train_data, val_data, train_labels, val_labels = train_test_split( train_data, train_labels, test_size=0.2, random_state=42)

# Create DataLoader
batch_size = 16  # Set batch size as needed
train_dataset = SignalDataset(train_data, train_labels)
val_dataset=SignalDataset(val_data,val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
for signals, labels in train_loader:
    print(f"Batch signals shape: {signals.shape}")  # Should be [batch_size, seq_len, input_size]
    break

# define model ,loss,optimiser
# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
input_size = 128  # Adjust based on your data
hidden_size = 100
output_size = 2  # For preterm and term classification
num_layers = 4
learning_rate = 0.001
batch_size = 32
num_epochs = 100

# Instantiate model, loss function, and optimizer
model = BiLSTMModel(input_size, hidden_size, output_size).to(device)
#class_counts = np.bincount(train_labels.numpy())
class_weights = torch.tensor(1.0 / np.bincount(train_labels), dtype=torch.float32).to(device)
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Scheduler for learning rate decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
train_losses = []
val_accuracies = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for signals, labels in train_loader:
        signals, labels = signals.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(signals)
        
        print(f"Model output shape: {outputs.shape}")

        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()
        # After each epoch, calculate the average loss for that epoch
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store the average loss
    
    scheduler.step()  # Update learning rate

    # Epoch loss
    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


print("Training Complete")

# Save the model
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch}, 'model_checkpoint.pth')

print("Model saved!")

# Evaluation (placeholder for validation dataset)
model.eval()
y_true = []
y_pred = []
train_data, val_data, train_labels, val_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42
)
"""" 
train_data, val_data, train_labels, val_labels = train_test_split(
    [x.numpy() for x in train_data],
    train_labels.numpy(),
    test_size=0.2,
    random_state=42
)

# Convert back to tensors
train_data = [torch.tensor(x, dtype=torch.float32) for x in train_data]
train_labels = torch.tensor(train_labels, dtype=torch.long)
val_data = [torch.tensor(x, dtype=torch.float32) for x in val_data]
val_labels = torch.tensor(val_labels, dtype=torch.long)


"""
val_dataset = SignalDataset(val_data, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

correct = 0
total = 0
y_true, y_pred = [], []  # Reset before evaluation
y_true, y_pred = [], []  # Reset before evaluation

with torch.no_grad():
    for signals, labels in val_loader:
        signals, labels = signals.to(device), labels.to(device)

        outputs = model(signals)
        _, predicted = torch.max(outputs, 1)

        # Ensure batch size matches
        if len(labels) != len(predicted):
            print(f"Batch size mismatch: {len(labels)} != {len(predicted)}")
            continue  # Skip this batch

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# After loop, verify length consistency
print(f"Length of y_true: {len(y_true)}")
print(f"Length of y_pred: {len(y_pred)}")

if len(y_true) == len(y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"]))
else:
    print("Error: Mismatch in lengths of y_true and y_pred!")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

#