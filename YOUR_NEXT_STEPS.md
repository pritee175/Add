# YOUR EXACT NEXT STEPS TO RESEARCH PAPER
## Step-by-Step Guide to 85%+ Accuracy & Publication

**Current Status:** 566 images, 76.2% test accuracy (ResNet50)  
**Goal:** 1000+ images, 85%+ accuracy, paper-ready in 12-16 weeks

---

## STEP 1: Dataset Expansion (CRITICAL - Week 1-3)
### Why: More data = better accuracy (expect +5-7% improvement)

### Action 1.1: Download 400+ More Videos (Week 1-2)
**DO THIS NOW:**

```bash
# Run existing scripts to download more videos
python download_preventive_tv_ads.py
python download_promotional_tv_ads.py
```

**Manually search YouTube and add to scripts:**
- Search terms:
  - "Indian anti smoking ad hindi"
  - "Indian alcohol advertisement banned"
  - "Vimal pan masala ad"
  - "Royal Stag music CD ad" (surrogate)
  - "Pan Bahar elaichi ad" (surrogate)

**Target:** 50+ new videos per class (100 total)

### Action 1.2: Extract & Filter Frames (Week 2)
```bash
# Extract frames from new videos
python frame_extractor.py --video_folder Videos/Preventive_NEW --output ExtractedFrames/Preventive_NEW

python frame_extractor.py --video_folder Videos/Promotional_NEW --output ExtractedFrames/Promotional_NEW

# Preprocess
python preprocess_dataset.py --input ExtractedFrames/Preventive_NEW --output Dataset_Processed/Preventive_NEW

python preprocess_dataset.py --input ExtractedFrames/Promotional_NEW --output Dataset_Processed/Promotional_NEW
```

**Manual filtering:** Keep only clear, non-duplicate frames (you did this before)

**Target:** 
- 350+ new Preventive frames (total: 650+)
- 400+ new Promotional frames (total: 650+)

### Action 1.3: Merge & Rebuild Split (Week 3)
```bash
# Merge new images (after manual filtering)
cp Dataset_Processed/Preventive_NEW/*.jpg Dataset_Processed/Preventive/
cp Dataset_Processed/Promotional_NEW/*.jpg Dataset_Processed/Promotional/

# Update metadata
python backfill_metadata.py

# Rebuild split
python build_split.py --ratios 70 15 15 --seed 42
```

**Expected Dataset:** 1300+ images total (650+ per class)

---

## STEP 2: Data Augmentation (Easy Win - Week 3)
### Why: Artificially increase dataset size, expect +2-3% accuracy

### Action 2.1: Add Augmentation to Training
**Edit `train.py` to add more aggressive augmentation:**

```python
# In build_transforms() function, update train_transform:
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Increased from 10
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Added saturation
    transforms.RandomGrayscale(p=0.1),  # New: Sometimes grayscale
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # New: Slight translation
    transforms.ToTensor(),
    transforms.Normalize(mean=base.mean, std=base.std),
])
```

**No code needed - just edit train.py**


---

## STEP 3: Train Improved Baseline (Week 3-4)
### Why: Better baseline = stronger paper foundation

### Action 3.1: Retrain with Expanded Dataset & Augmentation
```bash
# Train ResNet50 (your best model)
python train.py --model resnet50 --epochs 20 --batch-size 16 --lr 1e-4 --stage2-epochs 15 --stage2-lr 5e-6 --unfreeze-blocks 3

# Train EfficientNet-B0 (backup)
python train.py --model efficientnet_b0 --epochs 20 --batch-size 16 --lr 1e-4 --stage2-epochs 15 --stage2-lr 5e-6 --unfreeze-blocks 3
```

**Expected Results:**
- ResNet50: 80-82% test accuracy (from 76.2%)
- EfficientNet-B0: 76-78% test accuracy (from 69.0%)

**Why Improvement:**
- More data (1300 vs 566 images) = +3-4%
- Better augmentation = +1-2%
- Longer training = +1%

---

## STEP 4: YOLO Object Detection (Week 4-6)
### Why: Detect products/warnings as features, expect +2-3% accuracy

### Action 4.1: Annotate 300 Frames for YOLO (Week 4-5)
**Install LabelImg:**
```bash
pip install labelImg
labelImg
```

**Annotation Process:**
1. Open 300 random frames from Dataset_Processed/
2. Draw bounding boxes around:
   - `cigarette` (any cigarette in frame)
   - `cigarette_pack` (cigarette box/packet)
   - `pan_masala_pouch` (gutka/pan masala sachet)
   - `alcohol_bottle` (any alcohol bottle)
   - `warning_symbol` (skull, cancer graphic, crossed cigarette)
   - `warning_text` (health warning text in Hindi/English)
   - `brand_logo` (Vimal, Rajnigandha, Royal Stag logos)
3. Save annotations in YOLO format (txt files)
4. Split: 210 train, 45 val, 45 test

**Create folder structure:**
```
YOLO_Data/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

**Create `YOLO_Data/data.yaml`:**
```yaml
path: ./YOLO_Data
train: train/images
val: val/images
test: test/images

nc: 7  # number of classes
names: ['cigarette', 'cigarette_pack', 'pan_masala_pouch', 'alcohol_bottle', 'warning_symbol', 'warning_text', 'brand_logo']
```

**Time:** 2 hours per 100 frames = 6 hours total

### Action 4.2: Train YOLOv8 (Week 5)
**Create `train_yolo.py`:**
```python
from ultralytics import YOLO

# Load pretrained YOLOv8 medium
model = YOLO('yolov8m.pt')

# Train
results = model.train(
    data='YOLO_Data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU (use 'cpu' if no GPU)
    project='yolo_runs',
    name='addiction_detector'
)

# Validate
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")

# Save
model.save('yolo_addiction_detector.pt')
```

**Run:**
```bash
pip install ultralytics
python train_yolo.py
```

**Expected:** mAP@0.5 > 0.70 (good enough for feature extraction)

### Action 4.3: Extract YOLO Features (Week 6)
**Create `extract_yolo_features.py`:**
```python
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

model = YOLO('yolo_addiction_detector.pt')

features = []
for split in ['train', 'val', 'test']:
    for category in ['Preventive', 'Promotional']:
        img_dir = Path(f'Dataset_Split/{split}/{category}')
        
        for img_path in img_dir.glob('*.jpg'):
            results = model(str(img_path), verbose=False)
            
            # Extract detections
            boxes = results[0].boxes
            classes = boxes.cls.cpu().numpy() if len(boxes) else []
            
            # Count each object type
            class_counts = {f'n_{name}': 0 for name in model.names.values()}
            for cls_id in classes:
                class_name = model.names[int(cls_id)]
                class_counts[f'n_{class_name}'] += 1
            
            # Binary presence
            class_presence = {f'has_{name}': int(f'n_{name}' in class_counts and class_counts[f'n_{name}'] > 0) 
                            for name in model.names.values()}
            
            features.append({
                'filename': img_path.name,
                'split': split,
                'category': category,
                **class_counts,
                **class_presence,
                'total_objects': len(boxes),
                'avg_confidence': boxes.conf.mean().item() if len(boxes) else 0.0
            })

df = pd.DataFrame(features)
df.to_csv('yolo_features.csv', index=False)
print(f"Extracted features for {len(df)} images")
print(df.head())
```

**Run:**
```bash
python extract_yolo_features.py
```

**Output:** `yolo_features.csv` with object counts per frame


---

## STEP 5: Add YOLO Features to Classification (Week 6)
### Why: Objects tell you a lot (warnings = preventive, brand = promotional)

### Action 5.1: Train Classifier with YOLO Features
**Create `train_with_yolo.py`:**
```python
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class ImageYOLODataset(Dataset):
    def __init__(self, image_paths, labels, yolo_features_df, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Merge with YOLO features
        self.yolo_df = yolo_features_df.set_index('filename')
        self.yolo_feature_cols = [col for col in yolo_features_df.columns 
                                   if col.startswith('n_') or col.startswith('has_') 
                                   or col in ['total_objects', 'avg_confidence']]
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Get YOLO features
        filename = Path(self.image_paths[idx]).name
        yolo_feat = self.yolo_df.loc[filename, self.yolo_feature_cols].values.astype(np.float32)
        
        return img, torch.tensor(yolo_feat), self.labels[idx]
    
    def __len__(self):
        return len(self.image_paths)

class ResNetWithYOLO(nn.Module):
    def __init__(self, num_yolo_features=16):
        super().__init__()
        # Visual branch
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final layer
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2048 + num_yolo_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, img, yolo_feat):
        visual_feat = self.resnet(img)
        combined = torch.cat([visual_feat, yolo_feat], dim=1)
        return self.fusion(combined)

# Training loop (similar to train.py but with YOLO features)
yolo_df = pd.read_csv('yolo_features.csv')
model = ResNetWithYOLO(num_yolo_features=len([c for c in yolo_df.columns if c.startswith('n_') or c.startswith('has_')]))

# Train for 15 epochs...
# (Full training code similar to train.py)
```

**Shortcut: Modify existing train.py instead**

**Expected:** 82-84% test accuracy (ResNet50 baseline + YOLO = +2-3%)

---

## STEP 6: Temporal Modeling with LSTM (Week 7-9)
### Why: Video sequences provide context, expect +2-3% accuracy

### Action 6.1: Create Sequence Dataset (Week 7)
**Create `video_sequence_dataset.py`:**
```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import re

class VideoSequenceDataset(Dataset):
    def __init__(self, split_dir, sequence_length=5, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = []
        
        # Parse filenames to group by video
        for category in ['Preventive', 'Promotional']:
            cat_dir = Path(split_dir) / category
            
            # Group frames by video_id
            video_groups = {}
            for img_path in sorted(cat_dir.glob('*.jpg')):
                # Extract video_id from filename
                # PREVENTIVE_7M7oRkAaujg_F001.jpg -> 7M7oRkAaujg
                match = re.match(r'[A-Z]+_([^_]+)_F\d+\.jpg', img_path.name)
                if match:
                    video_id = match.group(1)
                    video_groups.setdefault(video_id, []).append(str(img_path))
            
            # Create sequences
            label = 0 if category == 'Preventive' else 1
            for video_id, frames in video_groups.items():
                if len(frames) >= sequence_length:
                    # Sample uniformly
                    indices = np.linspace(0, len(frames)-1, sequence_length, dtype=int)
                    sequence = [frames[i] for i in indices]
                    self.sequences.append((sequence, label))
    
    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        
        frames = []
        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        
        return torch.stack(frames), label
    
    def __len__(self):
        return len(self.sequences)

# Usage
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = VideoSequenceDataset('Dataset_Split/train', sequence_length=5, transform=transform)
print(f"Created {len(train_dataset)} video sequences")
```


### Action 6.2: Train LSTM Temporal Model (Week 8-9)
**Create `train_lstm_temporal.py`:**
```python
import torch
import torch.nn as nn
from torchvision import models
from video_sequence_dataset import VideoSequenceDataset
from torch.utils.data import DataLoader

class LSTMTemporalModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=2):
        super().__init__()
        # Frame feature extractor
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze feature extractor initially
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, 3, 224, 224]
        batch_size, seq_len = x.shape[:2]
        
        # Extract features for each frame
        x = x.view(batch_size * seq_len, 3, 224, 224)
        with torch.no_grad():  # Frozen features initially
            features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        final_hidden = lstm_out[:, -1, :]
        
        # Classify
        logits = self.fc(final_hidden)
        return logits

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMTemporalModel().to(device)

train_dataset = VideoSequenceDataset('Dataset_Split/train', sequence_length=5)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_dataset = VideoSequenceDataset('Dataset_Split/val', sequence_length=5)
val_loader = DataLoader(val_dataset, batch_size=8)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)  # Only train LSTM+FC initially

# Stage 1: Train LSTM only (5 epochs)
for epoch in range(5):
    model.train()
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    print(f"Epoch {epoch+1}/5: Val Acc = {100.*correct/total:.2f}%")

# Stage 2: Unfreeze and fine-tune (10 epochs)
for param in model.feature_extractor.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Continue training for 10 more epochs...

torch.save(model.state_dict(), 'lstm_temporal_model.pt')
```

**Run:**
```bash
python train_lstm_temporal.py
```

**Expected:** 84-86% test accuracy (ResNet50 + LSTM temporal context = +4-6% over original baseline)

---

## STEP 7: Multi-Modal Fusion (Week 9-11)
### Why: Text (OCR) adds brand/warning info, expect +1-2% accuracy

### Action 7.1: Extract Text with OCR (Week 9)
**Install EasyOCR:**
```bash
pip install easyocr
```

**Create `extract_ocr_features.py`:**
```python
import easyocr
import pandas as pd
from pathlib import Path
from tqdm import tqdm

reader = easyocr.Reader(['en', 'hi'], gpu=True)  # English + Hindi

ocr_data = []
for split in ['train', 'val', 'test']:
    for category in ['Preventive', 'Promotional']:
        img_dir = Path(f'Dataset_Split/{split}/{category}')
        
        for img_path in tqdm(list(img_dir.glob('*.jpg')), desc=f'{split}/{category}'):
            # Extract text
            result = reader.readtext(str(img_path), detail=0)  # detail=0 returns only text
            text = ' '.join(result).lower()
            
            # Feature engineering
            ocr_data.append({
                'filename': img_path.name,
                'split': split,
                'category': category,
                'text': text,
                'has_warning_word': int(any(w in text for w in ['smoking', 'धूम्रपान', 'cancer', 'कैंसर', 'injurious', 'हानिकारक'])),
                'has_brand': int(any(b in text for b in ['vimal', 'rajnigandha', 'pan bahar', 'royal stag', 'mcdowell'])),
                'text_length': len(text.split())
            })

df = pd.DataFrame(ocr_data)
df.to_csv('ocr_features.csv', index=False)
print(f"Extracted OCR for {len(df)} images")
print(df[df['text_length'] > 0].head())
```

**Run:**
```bash
python extract_ocr_features.py
```

**Time:** ~2-3 hours for 1300 images

