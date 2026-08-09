# Team Blueprint: Context-Aware Advertisement Classification
## Research Project Roadmap to Publication

---

## Current Status (Completed)

✅ **Dataset:** 566 images (303 Preventive, 263 Promotional)  
✅ **Train/Val/Test Split:** 70/15/15 (video-level splitting)  
✅ **Baseline Models:** ResNet50 (76.2% test), EfficientNet-B0 (69.0% test)  
✅ **Infrastructure:** Download, preprocessing, training, evaluation pipelines  
✅ **Error Analysis:** Misclassification tracking and hard case identification

---

## Project Goal

**Research Contribution:** Context-aware multi-modal framework for classifying Indian addiction-related TV advertisements (Preventive vs Promotional) that handles surrogate advertising patterns

**Target Venues:** CVPR, ICCV, ACM Multimedia, or domain-specific conferences

---

## PHASE 1: Dataset Expansion & Quality Enhancement (Weeks 1-4)

### Priority: CRITICAL - Foundation for all other work

### Task 1.1: Expand to 600+ images per class
**Owner:** [Team Member Name]  
**Timeline:** 2 weeks

**Action Items:**
- Download 100+ more YouTube videos per class (target: 200 total videos)
- Focus on diversity:
  - Different brands: Vimal, Rajnigandha, Pan Bahar, Royal Stag, McDowell's, Bagpiper
  - Different time periods: 2015-2024
  - Different languages: Hindi, Tamil, Telugu, Bengali
  - Alcohol ads (currently underrepresented)
  - Surrogate ads (cardamom, soda, music CDs masking real products)
- Filter: Only 15-60 second TV commercials (no documentaries, street plays)
- Extract frames at 1fps, preprocess to 224x224
- Manual quality check before merging

**Deliverables:**
- 300+ new Preventive frames
- 350+ new Promotional frames
- Updated `metadata.csv` with video sources

### Task 1.2: Create Surrogate Ad Test Set
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Identify 50+ surrogate ad frames (Pan Bahar elaichi, Royal Stag soda, etc.)
- Manual annotation with ground truth labels
- Create separate test set: `Dataset_Surrogate/`
- Track which product is being masked (e.g., "Royal Stag Soda" → Whisky)

**Deliverables:**
- `Dataset_Surrogate/` folder with 50+ labeled frames
- `surrogate_metadata.csv` with annotations

### Task 1.3: Video Source Balancing
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Analyze current video distribution (some videos may dominate)
- Limit max 20 frames per video
- Ensure no single brand dominates (>15% of class)
- Re-run `build_split.py` after balancing

**Deliverables:**
- Updated `Dataset_Processed/` with balanced distribution
- Video distribution analysis report

---

## PHASE 2: YOLO Object Detection Integration (Weeks 3-5)

### Goal: Detect addiction products, brands, and warning symbols as features

### Task 2.1: Prepare YOLO Training Data
**Owner:** [Team Member Name]  
**Timeline:** 1.5 weeks

**Action Items:**
- Annotate 300+ frames using LabelImg or Roboflow
- Object classes to annotate:
  - `cigarette`, `cigarette_pack`, `pan_masala_pouch`, `gutka_pouch`
  - `alcohol_bottle`, `glass_alcohol`, `brand_logo`
  - `warning_symbol`, `warning_text`, `health_graphic`
  - `celebrity_face`, `hand_gesture` (optional)
- Split annotations: 70% train, 15% val, 15% test
- Export to YOLO format (txt files with bounding boxes)

**Deliverables:**
- `YOLO_Annotations/` folder with train/val/test splits
- `classes.txt` with object class names
- Annotation statistics report


### Task 2.2: Train YOLOv8 Object Detector
**Owner:** [Team Member Name]  
**Timeline:** 1.5 weeks

**Action Items:**
- Use YOLOv8-medium or YOLOv8-large (better accuracy for small objects)
- Fine-tune on annotated data for 100+ epochs
- Track mAP@0.5 and mAP@0.5:0.95 metrics
- Test on validation set, optimize threshold

**Code Template:**
```python
from ultralytics import YOLO

# Load pretrained YOLOv8
model = YOLO('yolov8m.pt')

# Train on custom dataset
model.train(
    data='yolo_config.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)

# Validate
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")

# Save
model.save('yolo_addiction_detector.pt')
```

**Deliverables:**
- Trained YOLO model: `yolo_addiction_detector.pt`
- Validation metrics report
- Sample detection visualizations

### Task 2.3: Extract YOLO Features for Classification
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Run YOLO on all dataset frames
- Extract feature vector per frame:
  - Object presence (binary): [has_cigarette, has_warning, has_logo, ...]
  - Object counts: [n_cigarettes, n_warnings, ...]
  - Bounding box statistics: [avg_box_size, total_coverage, ...]
- Save features: `yolo_features.csv` (one row per frame)

**Code Template:**
```python
model = YOLO('yolo_addiction_detector.pt')

features = []
for img_path in image_list:
    results = model(img_path)
    
    # Extract feature vector
    feat = {
        'filename': img_path,
        'has_cigarette': int('cigarette' in results[0].names),
        'n_warnings': sum(1 for cls in results[0].boxes.cls if cls == warning_id),
        'avg_confidence': results[0].boxes.conf.mean().item(),
        # ... more features
    }
    features.append(feat)

pd.DataFrame(features).to_csv('yolo_features.csv', index=False)
```

**Deliverables:**
- `yolo_features.csv` with extracted features
- Feature distribution analysis

---

## PHASE 3: Context-Aware Temporal Modeling (Weeks 5-8)

### Goal: Model video sequences instead of single frames


### Task 3.1: Create Video-Level Sequences
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Group frames by video_id (already in filenames)
- Create sequence dataset: each sample = 5-10 consecutive frames from same video
- Temporal sampling strategies:
  - **Uniform:** Sample 5 frames evenly spaced across video
  - **Beginning-Middle-End:** 2 frames from start, 1 middle, 2 end
  - **Sliding Window:** Overlapping windows of 5 frames
- Save as PyTorch Dataset class

**Code Template:**
```python
class VideoSequenceDataset(Dataset):
    def __init__(self, frames_dict, sequence_length=5):
        self.sequences = []
        
        # Group frames by video_id
        for video_id, frames in frames_dict.items():
            if len(frames) >= sequence_length:
                # Create sequences
                for i in range(0, len(frames) - sequence_length + 1, 2):
                    self.sequences.append(frames[i:i+sequence_length])
    
    def __getitem__(self, idx):
        frame_paths = self.sequences[idx]
        frames = [load_and_preprocess(p) for p in frame_paths]
        return torch.stack(frames), label
```

**Deliverables:**
- `video_sequence_dataset.py` with Dataset class
- Sequence statistics (avg length, total sequences)

### Task 3.2: Implement LSTM Temporal Model
**Owner:** [Team Member Name]  
**Timeline:** 2 weeks

**Action Items:**
- Architecture: ResNet50 backbone → LSTM → Classification head
- Extract frame-level features using ResNet50 (2048-dim)
- Feed sequence of features into LSTM (2-3 layers, hidden_size=512)
- Train end-to-end or freeze ResNet initially

**Code Template:**
```python
class TemporalLSTM(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512, num_layers=2):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classifier
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Linear(hidden_size, 2)  # Binary classification
    
    def forward(self, x):
        # x: [batch, seq_len, 3, 224, 224]
        batch_size, seq_len = x.shape[:2]
        
        # Extract features for each frame
        x = x.view(batch_size * seq_len, 3, 224, 224)
        features = self.resnet(x)  # [batch*seq_len, 2048]
        features = features.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features)
        final_hidden = lstm_out[:, -1, :]  # Last timestep
        
        # Classify
        logits = self.fc(final_hidden)
        return logits
```

**Training Strategy:**
- Stage 1: Freeze ResNet, train LSTM (5 epochs)
- Stage 2: Unfreeze last ResNet block, fine-tune (10 epochs)

**Deliverables:**
- `train_temporal_lstm.py` script
- Trained model: `lstm_temporal_model.pt`
- Results comparison: LSTM vs baseline ResNet50


### Task 3.3: Implement 3D CNN Alternative (Optional)
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Use 3D ResNet or I3D for spatiotemporal features
- Input: [batch, 3, seq_len, 224, 224] (video clips)
- Compare with LSTM approach

**Libraries:**
- `torchvision.models.video.r3d_18` (3D ResNet)
- `pytorchvideo` for I3D

**Deliverables:**
- `train_3d_cnn.py` script
- Comparison: 3D CNN vs LSTM

---

## PHASE 4: Multi-Modal Fusion (Weeks 7-10)

### Goal: Combine visual features with text (OCR) and object detection

### Task 4.1: Extract Text from Frames (OCR)
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Use EasyOCR or PaddleOCR for Hindi + English text extraction
- Extract text from all frames
- Clean and normalize text (remove special chars, lowercasing)
- Identify key patterns:
  - Brand names: "Vimal", "Rajnigandha", "Royal Stag"
  - Warning text: "धूम्रपान", "Smoking", "Cancer"
  - Government disclaimers

**Code Template:**
```python
import easyocr

reader = easyocr.Reader(['en', 'hi'])  # English + Hindi

ocr_data = []
for img_path in image_list:
    result = reader.readtext(img_path)
    
    # Concatenate all detected text
    text = ' '.join([detection[1] for detection in result])
    
    ocr_data.append({
        'filename': img_path,
        'text': text,
        'has_warning_word': int('smoking' in text.lower() or 'धूम्रपान' in text),
        'has_brand': int(any(brand in text.lower() for brand in brand_list))
    })

pd.DataFrame(ocr_data).to_csv('ocr_features.csv', index=False)
```

**Deliverables:**
- `ocr_features.csv` with extracted text
- Text statistics (avg words per frame, common terms)


### Task 4.2: Text Embedding with BERT
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Use multilingual BERT or IndicBERT for text encoding
- Generate 768-dim embeddings per frame
- Handle frames with no text (zero vectors or learned embedding)

**Code Template:**
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')

def get_text_embedding(text):
    if not text.strip():
        return torch.zeros(768)  # No text
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding

# Extract for all frames
text_embeddings = []
for row in ocr_data:
    emb = get_text_embedding(row['text'])
    text_embeddings.append(emb.numpy())

np.save('text_embeddings.npy', np.array(text_embeddings))
```

**Deliverables:**
- `text_embeddings.npy` (768-dim per frame)
- Text embedding visualization (t-SNE)

### Task 4.3: Multi-Modal Fusion Model
**Owner:** [Team Member Name]  
**Timeline:** 2 weeks

**Action Items:**
- Combine 3 modalities:
  - **Visual:** ResNet50/LSTM features (2048-dim or 512-dim)
  - **Text:** BERT embeddings (768-dim)
  - **Objects:** YOLO features (10-20 dim)
- Fusion strategies:
  - **Early Fusion:** Concatenate all features → MLP
  - **Late Fusion:** Separate heads → weighted average
  - **Cross-Attention:** Transformer-based fusion

**Recommended: Early Fusion (simplest)**

```python
class MultiModalClassifier(nn.Module):
    def __init__(self, visual_dim=2048, text_dim=768, object_dim=15):
        super().__init__()
        
        # Visual encoder (ResNet50 backbone)
        self.visual_encoder = resnet50(pretrained=True)
        self.visual_encoder.fc = nn.Identity()
        
        # Fusion layers
        fusion_dim = visual_dim + text_dim + object_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, image, text_emb, object_feat):
        # Extract visual features
        visual_feat = self.visual_encoder(image)
        
        # Concatenate all modalities
        combined = torch.cat([visual_feat, text_emb, object_feat], dim=1)
        
        # Classify
        logits = self.fusion_mlp(combined)
        return logits
```

**Deliverables:**
- `train_multimodal.py` script
- Trained model: `multimodal_fusion_model.pt`
- Ablation study: Visual-only vs +Text vs +Objects vs All


---

## PHASE 5: Advanced Context-Aware Features (Weeks 9-12)

### Task 5.1: Attention Mechanisms (Explainability)
**Owner:** [Team Member Name]  
**Timeline:** 2 weeks

**Action Items:**
- Implement Grad-CAM for visualization
- Show which regions the model focuses on (products, warnings, celebrities)
- Generate attention maps for 50+ test samples (5 per class)

**Code Template:**
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model = load_trained_model()
target_layers = [model.visual_encoder.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

for img_path in test_images:
    img = load_image(img_path)
    grayscale_cam = cam(input_tensor=img, targets=None)
    
    # Overlay on original image
    visualization = show_cam_on_image(img, grayscale_cam[0], use_rgb=True)
    save_visualization(visualization, f'attention_{img_path}')
```

**Deliverables:**
- Attention visualizations for paper figures
- Analysis: Does model focus on correct regions?

### Task 5.2: Indian Brand & Celebrity Detection
**Owner:** [Team Member Name]  
**Timeline:** 2 weeks

**Action Items:**
- Train face recognition on Bollywood celebrities (Shah Rukh Khan, Ajay Devgn, etc.)
- Use InsightFace or DeepFace
- Detect Indian brand logos (YOLO or template matching)
- Add as binary features: [has_celebrity, has_known_brand]

**Deliverables:**
- Celebrity detection model
- Brand detection accuracy report
- Feature integration into multi-modal model

### Task 5.3: Surrogate Advertising Classifier
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Build binary classifier: Is this a surrogate ad?
- Use surrogate test set from Phase 1
- Features:
  - Visual: Product looks like X but labeled as Y
  - Text: Brand name + unrelated product category
  - Context: Same brand in both promotional and "neutral" contexts

**Deliverables:**
- Surrogate ad classifier
- Precision/Recall on surrogate test set
- Error analysis: Which surrogate patterns are hardest?

---

## PHASE 6: Evaluation & Analysis (Weeks 11-13)

### Task 6.1: Comprehensive Evaluation
**Owner:** [Team Member Name]  
**Timeline:** 1.5 weeks

**Metrics to Report:**
- Accuracy, Precision, Recall, F1 (per class)
- Confusion matrix
- ROC-AUC
- Per-brand accuracy (aggregate by brand)
- Per-language accuracy (if language detected)
- Temporal consistency (same video frames agree?)

**Statistical Tests:**
- McNemar's test: Compare baseline vs context-aware models
- 5-fold cross-validation with confidence intervals
- Paired t-test for significance

**Deliverables:**
- Complete evaluation report
- Statistical significance results
- Performance breakdown by video characteristics


### Task 6.2: Ablation Studies (CRITICAL for Paper)
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Must Answer:**
1. Does temporal modeling help? (Frame-only vs LSTM vs 3D CNN)
2. Does multi-modal fusion help? (Visual-only vs +Text vs +Objects vs All)
3. Which modality contributes most? (Remove one at a time)
4. Does YOLO object detection add value? (With vs without)
5. Does Indian-specific context help? (With vs without celebrity/brand features)

**Deliverables:**
- Ablation study table (model variants vs metrics)
- Statistical significance for each comparison

### Task 6.3: Error Analysis & Hard Cases
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Identify top 20 misclassified examples
- Categorize failure modes:
  - Ambiguous frames (no clear product)
  - Surrogate ads (model confused by dual messaging)
  - Low-quality frames (blur, dark, occlusion)
  - Celebrity-only frames (no product visible)
  - Brand confusion (similar packaging)
- Manual review and annotation

**Deliverables:**
- Error analysis report with examples
- Recommendations for improvement

---

## PHASE 7: Paper Writing & Submission (Weeks 13-16)

### Task 7.1: Dataset Paper/Documentation
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Create dataset card (Hugging Face format)
- Document collection methodology
- Provide usage examples
- Address ethical considerations:
  - Public YouTube videos only
  - No personal information
  - Intended for regulatory research
- License: CC BY-NC 4.0 (research only)

**Deliverables:**
- Dataset README.md
- Sample usage notebook
- Ethical compliance statement

### Task 7.2: Paper Drafting
**Owner:** [All Team Members]  
**Timeline:** 3 weeks

**Paper Structure:**

**1. Abstract (200 words)**
- Problem: Surrogate advertising in Indian media
- Gap: No context-aware multi-modal approach
- Solution: YOLO + LSTM + Multi-modal fusion
- Results: X% accuracy, Y% on surrogate ads

**2. Introduction (1.5 pages)**
- Indian advertising regulations (COTPA 2003)
- Surrogate advertising challenge
- Technical gap: Single-frame vs context-aware
- Contributions:
  - First Indian addiction ad dataset (600+ images, 100+ videos)
  - Multi-modal context-aware framework
  - Surrogate ad detection capability
  - Public dataset release


**3. Related Work (2 pages)**
- Advertising classification (general)
- Tobacco/health content detection
- Video understanding (temporal models)
- Indian language/cultural context in CV
- Multi-modal learning (vision + text)

**4. Dataset (2 pages)**
- Collection methodology
  - YouTube search queries
  - Filtering criteria (15-60s TV ads only)
  - Manual quality check
- Statistics:
  - 600+ images from 100+ videos
  - Preventive vs Promotional distribution
  - Brand diversity, language diversity
  - Temporal characteristics (frames per video)
- Annotation protocol
  - Primary class label
  - Optional: Product type, brand, language
- Comparison with existing datasets (if any)

**5. Methodology (3-4 pages)**

**5.1 Problem Formulation**
- Binary classification task
- Challenges: Surrogate ads, temporal context, multi-lingual

**5.2 Baseline: Single-Frame CNN**
- ResNet50, EfficientNet-B0
- Transfer learning from ImageNet
- Two-stage training (head-only → fine-tune)

**5.3 YOLO Object Detection**
- Object classes (cigarette, warning, brand, etc.)
- Training procedure
- Feature extraction for classification

**5.4 Temporal Modeling**
- LSTM architecture
- Sequence construction (5-frame windows)
- Training strategy

**5.5 Multi-Modal Fusion**
- Visual: CNN/LSTM features
- Text: OCR + BERT embeddings
- Objects: YOLO features
- Fusion: Early concatenation + MLP

**5.6 Context-Aware Enhancements**
- Attention mechanisms
- Indian brand/celebrity detection
- Surrogate ad patterns

**6. Experiments (3 pages)**

**6.1 Experimental Setup**
- Train/Val/Test split (70/15/15)
- Video-level splitting (prevent leakage)
- Hyperparameters
- Training details (epochs, lr, optimizer)
- Hardware (GPU specs)

**6.2 Evaluation Metrics**
- Accuracy, Precision, Recall, F1
- Confusion matrix
- ROC-AUC
- Surrogate ad detection performance

**6.3 Baseline Results**
- ResNet50: 76.2% test accuracy
- EfficientNet-B0: 69.0% test accuracy
- Per-class breakdown

**6.4 Context-Aware Results**
- LSTM temporal: [Expected ~80-82%]
- Multi-modal fusion: [Expected ~83-85%]
- Full system: [Expected ~85-87%]

**6.5 Ablation Studies**
- Temporal vs single-frame
- Multi-modal vs visual-only
- YOLO contribution
- Per-modality analysis

**6.6 Surrogate Ad Performance**
- Accuracy on surrogate test set
- Comparison with baseline

**6.7 Qualitative Analysis**
- Attention visualizations
- Success cases
- Failure cases (error analysis)


**7. Discussion (1.5 pages)**
- What works well (temporal context, multi-modal)
- What doesn't (surrogate ads still challenging)
- Comparison with commercial APIs (if tested)
- Regulatory implications
  - How can this help ASCI/COTPA enforcement?
  - Scalability to real-time monitoring
- Limitations:
  - Dataset size (600 vs ideal 5000+)
  - Hindi/English only (missing regional languages)
  - TV ads only (no social media, print)
  - Binary classification (no fine-grained product types)

**8. Conclusion (0.5 pages)**
- Summary of contributions
- Future work:
  - Expand to 5000+ images
  - Add regional languages
  - Real-time video classification
  - Mobile deployment
  - Fine-grained classification (cigarette vs pan masala)

**9. References**
- 30-40 relevant papers
- Cite datasets, methods, regulatory papers

---

### Task 7.3: Figures & Tables
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Required Figures:**
1. Dataset sample grid (4x4, showing variety)
2. Class distribution (bar chart)
3. Model architecture diagram (multi-modal fusion)
4. Training curves (loss, accuracy over epochs)
5. Confusion matrix (baseline vs context-aware)
6. Attention map examples (Grad-CAM visualizations)
7. ROC curve (baseline vs proposed)
8. Ablation study bar chart
9. Per-brand performance (if space allows)
10. Surrogate ad examples with predictions

**Required Tables:**
1. Dataset statistics
2. Baseline results (ResNet50, EfficientNet)
3. Context-aware results (LSTM, Multi-modal)
4. Ablation study table
5. Surrogate ad performance
6. Comparison with prior work (if applicable)
7. Per-class metrics

---

### Task 7.4: Code & Dataset Release
**Owner:** [Team Member Name]  
**Timeline:** 1 week

**Action Items:**
- Clean code repository (remove debug, add comments)
- Create requirements.txt / environment.yml
- Write README with:
  - Installation instructions
  - Dataset download link
  - Training command examples
  - Inference examples
- Prepare dataset for release:
  - Anonymize any personal info
  - Check copyright compliance
  - Create download script
- Upload to:
  - **Code:** GitHub (with MIT/Apache license)
  - **Dataset:** Zenodo or IEEE DataPort (with DOI)
  - **Models:** Hugging Face Hub

**Deliverables:**
- Public GitHub repository
- Dataset with DOI
- Trained model checkpoints

---

## PHASE 8: Submission & Revision (Weeks 16-20)

### Task 8.1: Internal Review
**Timeline:** 1 week

- Proofread entire paper (3+ rounds)
- Check all figures render correctly
- Verify all citations
- Run plagiarism check (Turnitin or similar)
- Internal presentation to lab/advisors

### Task 8.2: Conference Selection & Submission
**Timeline:** 1 week

**Target Conferences (in priority order):**

**Tier 1 (Computer Vision):**
- CVPR (Computer Vision & Pattern Recognition) - June deadline
- ICCV (International Conference on Computer Vision) - March deadline
- ECCV (European Conference on Computer Vision) - March deadline
- ACM Multimedia - April deadline

**Tier 2 (Pattern Recognition / Multimedia):**
- ICPR (International Conference on Pattern Recognition)
- WACV (Winter Conference on Applications of Computer Vision)
- MMM (MultiMedia Modeling)

**Domain-Specific:**
- JMIR Public Health (journal, health informatics)
- Tobacco Control (journal, public health)

**Regional (for validation before top-tier):**
- NCVPRIPG (National Conference on Computer Vision, India)
- ICVGIP (Indian Conference on Computer Vision & Graphics)

**Submission Checklist:**
- Paper PDF (formatted per conference template)
- Supplementary material (if allowed)
- Code/data availability statement
- Author conflict of interest form


### Task 8.3: Respond to Reviews
**Timeline:** 2-3 weeks (after receiving reviews)

**Common Review Comments & How to Address:**

1. **"Dataset too small"**
   - Response: Acknowledge, show we're expanding to 1000+
   - Show our rigorous video-level splitting prevents overfitting

2. **"Missing comparison with commercial APIs"**
   - Response: Test on Google Cloud Vision, AWS Rekognition
   - Show our domain-specific model outperforms general APIs

3. **"Limited to Hindi/English"**
   - Response: Acknowledge as limitation, future work includes Tamil, Telugu

4. **"Surrogate ad detection unclear"**
   - Response: Add detailed surrogate ad definition, more examples
   - Separate evaluation section for surrogates

5. **"Need user study / expert validation"**
   - Response: Get domain expert (advertising professional) to validate samples

**Rebuttal Strategy:**
- Address ALL reviewer comments point-by-point
- Provide evidence (new experiments, citations)
- Be respectful and constructive
- If disagreement, explain politely with evidence

---

## Resource Requirements

### Computational Resources

**GPU Requirements:**
- Minimum: 1x NVIDIA RTX 3090 (24GB VRAM)
- Recommended: 2x NVIDIA A100 (40GB VRAM each)
- Cloud alternative: Google Colab Pro+ or AWS p3.2xlarge

**Storage:**
- Dataset: ~100GB (videos + frames + annotations)
- Models & checkpoints: ~20GB
- Intermediate features: ~10GB
- Total: ~150GB (recommend 200GB+ buffer)

### Software & Tools

**Required:**
- Python 3.8+
- PyTorch 1.12+
- torchvision, timm
- ultralytics (YOLOv8)
- transformers (Hugging Face)
- opencv-python
- easyocr or paddleocr
- labelImg (for YOLO annotations)
- scikit-learn, pandas, matplotlib

**Optional:**
- Weights & Biases (experiment tracking)
- Roboflow (annotation management)
- TensorBoard (training visualization)

### Human Resources & Time Allocation

**Team Size:** 3-4 members recommended

**Role Distribution:**
- **Data Engineer** (1 person, 30% time):
  - Dataset expansion
  - Annotation
  - Data quality checks
  
- **ML Engineer** (2 people, 60% time):
  - Model development (YOLO, LSTM, Multi-modal)
  - Training & experimentation
  - Ablation studies
  
- **Research Lead** (1 person, 50% time):
  - Overall direction
  - Paper writing
  - Experiment design
  - Review response

**Total Estimated Hours:** 800-1000 person-hours (5-6 months with 4-person team)

---

## Success Metrics & Milestones

### Milestone 1 (Week 4): Dataset Ready
✅ 600+ images per class  
✅ Balanced video distribution  
✅ Surrogate ad test set created  
✅ Updated train/val/test split

### Milestone 2 (Week 8): Context Models Trained
✅ YOLO object detector trained (mAP > 0.7)  
✅ LSTM temporal model trained (accuracy > baseline + 3%)  
✅ OCR features extracted

### Milestone 3 (Week 12): Multi-Modal System Complete
✅ Multi-modal fusion model trained  
✅ Ablation studies completed  
✅ Full evaluation done  
✅ Target: 85%+ test accuracy

### Milestone 4 (Week 16): Paper Draft Complete
✅ All sections written  
✅ All figures/tables created  
✅ Code & dataset ready for release  
✅ Internal review completed

### Milestone 5 (Week 20): Paper Submitted
✅ Conference selected  
✅ Paper submitted  
✅ Supplementary materials uploaded  
✅ Dataset publicly available


---

## Risk Management

### Risk 1: Dataset Expansion Takes Longer Than Expected
**Impact:** Delays entire timeline  
**Mitigation:**
- Start video collection immediately (parallel with other work)
- Use semi-automated filtering (CLIP-based)
- Accept 600+ per class as minimum viable (aim for 800+)

### Risk 2: YOLO Annotation Bottleneck
**Impact:** Delays Phase 2  
**Mitigation:**
- Use active learning (annotate hard samples first)
- Leverage pre-trained YOLO on COCO (fine-tune on fewer samples)
- Consider semi-supervised annotation tools (Label Studio)

### Risk 3: Multi-Modal Model Doesn't Improve Over Baseline
**Impact:** Weakens paper contribution  
**Mitigation:**
- Ensure proper feature normalization
- Try different fusion strategies (late fusion, attention-based)
- If modest improvement, emphasize explainability & surrogate detection

### Risk 4: Low Surrogate Ad Detection Performance
**Impact:** Weakens novelty claim  
**Mitigation:**
- Acknowledge as "challenging problem" (even humans struggle)
- Show qualitative analysis (attention maps reveal confusion)
- Position as "first benchmark for surrogate detection"

### Risk 5: Paper Rejection
**Impact:** Timeline extends by 3-6 months  
**Mitigation:**
- Prepare for 2-3 submission rounds
- Start with regional conference for feedback
- Have backup venues ready (Tier 2 conferences, journals)

---

## Communication & Collaboration

### Weekly Team Meetings
**Agenda:**
- Progress updates (each member, 5 min)
- Blockers & help needed
- Next week priorities
- Quick wins to celebrate

**Tools:**
- Slack / Discord for daily communication
- Google Drive for shared documents
- GitHub for code collaboration
- Weights & Biases for experiment tracking
- Notion / Trello for task management

### Monthly Advisor Meetings
**Agenda:**
- Demo of latest model results
- Discussion of next steps
- Paper writing feedback
- Conference selection strategy

---

## Quick Start Guide for New Team Members

### Week 1: Setup & Familiarization
1. Clone repository: `git clone [repo_url]`
2. Setup environment: `pip install -r requirements.txt`
3. Download dataset (current 566 images)
4. Run baseline training: `python train.py --model resnet50`
5. Review existing error analysis
6. Read PROJECT_ANALYSIS_AND_BLUEPRINT.md

### Week 2: First Contribution
- Pick a task from current phase
- Coordinate with team lead
- Create feature branch: `git checkout -b feature/your-task`
- Submit PR when ready
- Document your work in lab notebook

---

## Appendix: Useful Resources

### Papers to Read
1. **Temporal Modeling:**
   - "Two-Stream CNNs for Action Recognition" (Simonyan & Zisserman, NIPS 2014)
   - "Quo Vadis, Action Recognition?" (Carreira & Zisserman, CVPR 2017)

2. **Multi-Modal Learning:**
   - "VideoBERT" (Sun et al., ICCV 2019)
   - "CLIP: Learning Transferable Visual Models" (Radford et al., ICML 2021)

3. **Object Detection:**
   - "YOLOv8 Documentation" (Ultralytics)
   - "Faster R-CNN" (Ren et al., NIPS 2015)

4. **Advertising / Health:**
   - Search "tobacco advertising detection" on Google Scholar
   - Check JMIR Public Health for similar work

### Code Repositories
- YOLOv8: https://github.com/ultralytics/ultralytics
- PyTorch Video Models: https://github.com/pytorch/vision/tree/main/torchvision/models/video
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Grad-CAM: https://github.com/jacobgil/pytorch-grad-cam

### Datasets (Reference)
- COCO (object detection): https://cocodataset.org/
- Kinetics (video classification): https://www.deepmind.com/open-source/kinetics
- UCF101 (action recognition): https://www.crcv.ucf.edu/data/UCF101.php

---

## Contact & Ownership

**Project Lead:** [Name]  
**Email:** [email]  
**GitHub:** [repo_url]  
**Last Updated:** [Date]

---

**END OF BLUEPRINT**

_This is a living document. Update as project progresses. Good luck team! 🚀_
