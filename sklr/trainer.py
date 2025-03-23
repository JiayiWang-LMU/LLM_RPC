import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import ast
from sklearn.model_selection import KFold
import os  
import random 
import numpy as np 
from torch.cuda.amp import GradScaler, autocast  

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

class RankingDataset(Dataset):
    def __init__(self, csv_files=None, ranking_columns=None, data=None):
      
        if data is not None:
            self.data = data
        else:
            dataframes = []
            for file, ranking_col in zip(csv_files, ranking_columns):
                df = pd.read_csv(file)
                df = df[['question_id', 'content', ranking_col]].rename(columns={ranking_col: 'ranking'})
                df['ranking_type'] = ranking_col  
                dataframes.append(df)
            self.data = pd.concat(dataframes, ignore_index=True)
            
            # Drop rows with missing ranking information.
            self.data.dropna(subset=['ranking'], inplace=True)
            
            # Check if the dataset is empty after filtering.
            if self.data.empty:
                raise ValueError("The dataset is empty after filtering. Please check the input CSV files.")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        question_id = row['question_id']
        content = row['content']
        ranking_type = row['ranking_type']
        
        # Safely convert ranking string into a Python list.
        def safe_literal_eval(val):
            if pd.isna(val) or str(val).strip() == "":
                return []
            try:
                return ast.literal_eval(val)
            except Exception as e:
                print(f"Error parsing ranking value: {val}. Error: {e}")
                return []
        
        ranking = safe_literal_eval(row['ranking'])
        
        # Generate ground-truth pairwise comparisons.
        ground_truth_pairs = []
        num_llms = len(ranking)
        if num_llms > 0:
            for i in range(num_llms):
                for j in range(i+1, num_llms):
                    if ranking[i] < ranking[j]:
                        ground_truth_pairs.append((i, j, 1))
                    elif ranking[i] > ranking[j]:
                        ground_truth_pairs.append((j, i, 1))
                    else:
                        ground_truth_pairs.append((i, j, 0))
        
        sample = {
            'question_id': question_id,
            'content': content,
            'ranking': ranking,
            'ranking_type': ranking_type,
            'ground_truth_pairs': ground_truth_pairs
        }
        return sample

# Transformer-based question encoder.
class QuestionEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
    
    def forward(self, texts, device):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)  # Move inputs to the correct device
        outputs = self.encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

# Add dropout to the RankingModel for regularization
class RankingModel(nn.Module):
    def __init__(self, embed_dim, num_llms=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout with a probability of 0.3
            nn.Linear(128, num_llms)
        )
    
    def forward(self, q_embedding):
        scores = self.fc(q_embedding)
        return scores

# Pairwise loss function.
def pairwise_loss(scores, ground_truth_pairs):
    """
    scores: tensor of shape (batch_size, num_llms)
    ground_truth_pairs: list (length=batch_size) of lists of tuples (i, j, relation)
       relation = 1 indicates a strict preference (score[i] should be higher than score[j])
       relation = 0 indicates a tie (scores should be as similar as possible)
    """
    loss = 0.0
    count = 0
    batch_size, num_llms = scores.shape
    for b in range(batch_size):
        sample_pairs = ground_truth_pairs[b]
        for (i, j, relation) in sample_pairs:
            if relation == 1:
                margin = 1.0
                diff = scores[b, i] - scores[b, j]
                loss += torch.relu(margin - diff)
                count += 1
            else:
                diff = torch.abs(scores[b, i] - scores[b, j])
                loss += diff
                count += 1
    return loss / count if count > 0 else loss

# Custom collate function that returns the batch as a list.
def collate_fn(batch):
    return batch

# Load datasets from multiple CSV files.
csv_files = [
    'sklr/dataset/rankings_borda.csv',
    'sklr/dataset/rankings_copeland.csv',
    'sklr/dataset/rankings_mc4.csv',
    'sklr/dataset/rankings_ml.csv'
]
ranking_columns = ['ranking1', 'ranking2', 'ranking3', 'ranking4']
dataset = RankingDataset(csv_files, ranking_columns)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# Check if CUDA is available and set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# File paths for saving/loading models
ENCODER_MODEL_PATH = "sklr/model/encoder_model.pth"
RANKING_MODEL_PATH = "sklr/model/ranking_model.pth"

# Check if models exist and load them
models_loaded = False
if os.path.exists(ENCODER_MODEL_PATH) and os.path.exists(RANKING_MODEL_PATH):
    print("Loading saved models...")
    encoder_model = QuestionEncoder().to(device)
    encoder_model.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
    
    embed_dim = encoder_model.encoder.config.hidden_size  # Define embed_dim based on the encoder model
    ranking_model = RankingModel(embed_dim, num_llms=6).to(device)
    ranking_model.load_state_dict(torch.load(RANKING_MODEL_PATH, map_location=device))
    models_loaded = True
else:
    print("No saved models found. Initializing new models...")
    encoder_model = QuestionEncoder().to(device)
    embed_dim = encoder_model.encoder.config.hidden_size
    ranking_model = RankingModel(embed_dim, num_llms=6).to(device)

# Use weight decay for L2 regularization in the optimizer
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) + list(ranking_model.parameters()),
    lr=1e-4,
    weight_decay=1e-5  # Add weight decay for regularization
)

# Add a learning rate scheduler for stabilizing training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 100  # Adjust as needed.

# Parameters for cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Convert the dataset to a DataFrame for splitting
data_df = dataset.data

# Cross-validation loop
if not models_loaded:  # Skip training if models are already loaded
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_df)):
        print(f"Starting fold {fold + 1}/{k_folds}...")
        
        # Split the data into training and validation sets
        train_data = data_df.iloc[train_idx].reset_index(drop=True)
        val_data = data_df.iloc[val_idx].reset_index(drop=True)
        
        # Create DataLoaders for training and validation
        train_dataset = RankingDataset(data=train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
        
        val_dataset = RankingDataset(data=val_data)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
        
        # Initialize GradScaler for mixed precision
        scaler = GradScaler()

        # Gradient accumulation steps
        accumulation_steps = 4

        # Training loop
        for epoch in range(num_epochs):
            encoder_model.train()
            ranking_model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()  # Reset gradients at the start of each epoch
            for step, batch in enumerate(train_dataloader):
                texts = [sample['content'] for sample in batch]
                gt_pairs_dict = [sample['ground_truth_pairs'] for sample in batch]

                # Mixed precision forward pass
                with autocast():
                    q_embeddings = encoder_model(texts, device)
                    scores = ranking_model(q_embeddings)
                    total_loss = pairwise_loss(scores, gt_pairs_dict) / accumulation_steps

                # Backward pass with scaling
                scaler.scale(total_loss).backward()

                # Update weights after accumulation steps
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += total_loss.item() * accumulation_steps  # Scale back loss for logging
            scheduler.step()  # Update learning rate
            print(f"Epoch {epoch + 1}, Training Loss: {epoch_loss / len(train_dataloader)}")
        
        # Validation loop
        encoder_model.eval()
        ranking_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                texts = [sample['content'] for sample in batch]
                gt_pairs_dict = [sample['ground_truth_pairs'] for sample in batch]
                
                # Encode questions
                q_embeddings = encoder_model(texts, device)
                scores = ranking_model(q_embeddings)
                
                # Compute loss
                total_loss = pairwise_loss(scores, gt_pairs_dict)
                val_loss += total_loss.item()
        print(f"Fold {fold + 1}, Validation Loss: {val_loss / len(val_dataloader)}")

    # Save models after training
    torch.save(encoder_model.state_dict(), ENCODER_MODEL_PATH)
    torch.save(ranking_model.state_dict(), RANKING_MODEL_PATH)
else:
    print("Models are already trained. Skipping training process.")

