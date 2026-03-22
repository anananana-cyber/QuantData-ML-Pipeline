import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
import gc

# --- PHASE 1: RAM-EFFICIENT LOADING ---
print("Step 1: Loading Data...")
needed_cols = ['CLOSE_PRICE', 'TTL_TRD_QNTY', 'SYMBOL', 'Date']
try:
    chunks = [chunk for chunk in pd.read_csv("nse_5year_history.csv", usecols=needed_cols, chunksize=1000000)]
    df = pd.concat(chunks, axis=0)
    del chunks
    gc.collect()
except Exception as e:
    print(f"Error loading CSV: {e}")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(['SYMBOL', 'Date'])

# --- PHASE 2: FEATURE ENGINEERING (IN-PLACE) ---
print("Step 2: Building High-Precision Signals...")
df['RET_5D'] = df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(5).astype('float32')
df['RET_10D'] = df.groupby('SYMBOL')['CLOSE_PRICE'].pct_change(10).astype('float32')

def get_rsi(s, n=14):
    delta = s.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0; down[down > 0] = 0
    return 100.0 - (100.0 / (1.0 + (up.rolling(n).mean() / (down.abs().rolling(n).mean() + 1e-9))))

df['RSI'] = df.groupby('SYMBOL')['CLOSE_PRICE'].transform(get_rsi).astype('float32')
df['LOG_RET'] = np.log(df['CLOSE_PRICE'] / (df.groupby('SYMBOL')['CLOSE_PRICE'].shift(1) + 1e-9)).astype('float32')
df['VOL_Z'] = ((df['TTL_TRD_QNTY'] - df.groupby('SYMBOL')['TTL_TRD_QNTY'].transform(lambda x: x.rolling(20).mean())) / (df.groupby('SYMBOL')['TTL_TRD_QNTY'].transform(lambda x: x.rolling(20).std()) + 1e-9)).astype('float32')
df['MKT_REL'] = (df['CLOSE_PRICE'] / (df.groupby('Date')['CLOSE_PRICE'].transform('mean') + 1e-9)).astype('float32')
df['TARGET'] = (df.groupby('SYMBOL')['CLOSE_PRICE'].shift(-60) > (df['CLOSE_PRICE'] * 1.05)).astype('float32')

df.dropna(subset=['RSI', 'LOG_RET', 'VOL_Z', 'RET_5D', 'RET_10D', 'MKT_REL', 'TARGET'], inplace=True)

FEATURES = ['RSI', 'LOG_RET', 'VOL_Z', 'RET_5D', 'RET_10D', 'MKT_REL']
X = df[FEATURES].values
y = df['TARGET'].values.reshape(-1, 1)
temporal_weights = np.linspace(1.0, 2.5, len(df)).reshape(-1, 1).astype('float32')

del df
gc.collect()

# --- PHASE 3: RESIDUAL ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.BatchNorm1d(size)
    def forward(self, x):
        return x + F.leaky_relu(self.bn(self.fc(x)), 0.1)

class PrecisionResNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, 128)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.output = nn.Linear(128, 1)
    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x), 0.1)
        x = self.res1(x)
        x = self.res2(x)
        return self.output(x)

# --- PHASE 4: 75-EPOCH MONITORING ENGINE ---
print("Step 3: Starting Training Monitor (75 Epochs)...")
scaler = RobustScaler()
X_tensor = torch.from_numpy(scaler.fit_transform(X))
y_tensor = torch.from_numpy(y)
w_tensor = torch.from_numpy(temporal_weights)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor, w_tensor), batch_size=32768, shuffle=True)
model = PrecisionResNet(len(FEATURES))
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
criterion = nn.BCEWithLogitsLoss(reduction='none') 

best_acc = 0.0

print(f"{'Epoch':<8} | {'Loss':<10} | {'Train Acc':<10}")
print("-" * 35)

for epoch in range(75): # Increased to 75
    model.train()
    total_loss, correct, total_samples = 0, 0, 0
    
    for b_x, b_y, b_w in train_loader:
        optimizer.zero_grad()
        logits = model(b_x)
        raw_loss = criterion(logits, b_y)
        weighted_loss = (raw_loss * b_w).mean()
        weighted_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == b_y).sum().item()
            total_samples += b_y.size(0)
            total_loss += weighted_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = (correct / total_samples) * 100
    
    # Track Best Accuracy
    if avg_acc > best_acc:
        best_acc = avg_acc
        torch.save(model.state_dict(), 'best_alpha_model.pth')
    
    status = "*" if avg_acc == best_acc else ""
    print(f"{epoch+1:<8} | {avg_loss:<10.4f} | {avg_acc:<9.2f}% {status}")

# --- PHASE 5: EVALUATION ---
print("\nStep 4: Final Precision Audit (Threshold: 0.70)...")
model.load_state_dict(torch.load('best_alpha_model.pth')) # Load the best version
model.eval()
y_probs = []
eval_loader = DataLoader(TensorDataset(X_tensor), batch_size=65536)

with torch.no_grad():
    for batch in eval_loader:
        y_probs.append(torch.sigmoid(model(batch[0])))

y_prob_concat = torch.cat(y_probs).numpy()
y_pred = (y_prob_concat > 0.70).astype(float) # 0.70 Threshold for Precision

print("\n--- PERFORMANCE REPORT (CONFIDENCE > 0.70) ---")
print(classification_report(y, y_pred))
