\# Research Internship — Current Experimental Status



This README records only the work that has actually been implemented, run, measured, or directly verified in the current repository. Planned or untested interpretations are explicitly marked as not established.



\---



\## 1. Work Completed



\### Dataset



The current clean dataset contains \*\*564 frames from 98 videos\*\*.



The dataset is split into:



\- Training: 395 frames / 52 videos

\- Validation: 85 frames / 23 videos

\- Test: 84 frames / 23 videos



The split was checked at the video level to prevent frames from the same video from appearing across different splits.



\### Image Classification Pipeline



The implemented image-classification pipeline supports:



\- EfficientNet-B0

\- ResNet50



Both models use ImageNet-pretrained weights and are adapted to the two-class classification task.



The training pipeline in `train.py` implements:



1\. \*\*Stage 1\*\* — classifier-head training with the backbone frozen.

2\. \*\*Stage 2\*\* — optional fine-tuning of the final backbone blocks/layers using a lower learning rate.



Training uses augmentation, while validation and test use evaluation-only preprocessing.



The pretrained model's normalization statistics are used for evaluation.



The evaluation pipeline reports:



\- Accuracy

\- Per-class precision

\- Per-class recall

\- Per-class F1

\- Confusion matrix

\- Classification report



The best checkpoint is selected using validation accuracy.



\### Video-Level Experiments



Two video-level aggregation approaches have been implemented:



\- Mean pooling

\- Attention pooling



The implementations are:



\- `video\_mean\_pool.py`

\- `video\_attention\_pool.py`



Experiment outputs are stored under:



\- `experiments/mean\_pool\_resnet50/`

\- `experiments/attention\_pool\_resnet50/`



The attention experiment additionally records attention weights.



\### Error Analysis



The repository contains error-analysis outputs including:



\- prediction tables

\- error tables

\- hard cases

\- annotated hard cases

\- misclassified examples



These were generated to inspect model errors and difficult examples.



\---



\## 2. Evidence Status



| Claim | Status | Evidence |

|---|---|---|

| Clean dataset: 564 frames, 98 videos, with a 395/85/84 frame split and 52/23/23 video split | \*\*Verified\*\* | `Dataset\_Split/split\_manifest.csv`, computed directly |

| Zero same-video / same-label leakage | \*\*Verified\*\* | Grouped by `video\_id` alone and confirmed after the dataset split fix |

| ResNet50 clean-dataset baseline: \*\*79.76% test accuracy\*\*, confusion matrix `\[\[31,14],\[3,36]]` | \*\*Verified\*\* | `checkpoints/resnet50\_best.pt` and `checkpoints/resnet50\_results.json` |

| EfficientNet-B0: \*\*69.05% test accuracy\*\* | \*\*Verified, but only on the OLD contaminated 566-frame dataset\*\* | `efficientnet\_b0\_results.json`; EfficientNet-B0 has \*\*not yet been re-run on the clean 564-frame dataset\*\* |

| Mean pooling: \*\*82.61% (19/23 videos)\*\* | \*\*Verified\*\* | `experiments/mean\_pool\_resnet50/results.json` |

| Frame-majority baseline: \*\*86.96% (20/23 videos)\*\* | \*\*Verified\*\* | Directly reported/computed in the video-level evaluation |

| Attention pooling: \*\*86.96% (20/23 videos)\*\* | \*\*Verified\*\* | `experiments/attention\_pool\_resnet50/results.json` |

| Preventive recall with attention pooling: \*\*33% → 50%\*\* | \*\*Verified\*\* | `experiments/attention\_pool\_resnet50/results.json` |

| Attention pooling numerically matches the frame-majority baseline and exceeds mean pooling | \*\*Verified (numbers only)\*\* | Direct comparison of the recorded experiment results |

| Attention pooling represents a meaningfully more robust or generalizable improvement | \*\*Not established\*\* | Only 23 test videos; each video represents \~4.35 percentage points. Validation accuracy also saturated at 100% for \~70 epochs, making checkpoint selection effectively arbitrary among tied checkpoints |

| Attention weights are semantically meaningful and reliably identify important warning frames | \*\*Not established / mixed evidence\*\* | Manual inspection of 7 frames found examples where Preventive videos were still misclassified despite containing a legible, explicit on-screen warning (e.g. PREV006) |

| Results generalize beyond this dataset | \*\*Not established\*\* | No external or independent held-out dataset has been evaluated |



\---



\## Important Dataset Note



The \*\*79.76% ResNet50 result is the clean-dataset baseline\*\*.



The previously recorded \*\*69.05% EfficientNet-B0 result was obtained on the older contaminated 566-frame dataset\*\* and therefore should not be directly compared with the clean ResNet50 result.



EfficientNet-B0 has \*\*not yet been re-run on the clean 564-frame dataset\*\*.



\---



\## 3. What We Have Learned



\### Verified Findings



The project has established a clean dataset split with no observed same-video leakage and has produced a ResNet50 baseline on that clean split.



At the video level:



\- Mean pooling achieved \*\*82.61% (19/23)\*\*.

\- Attention pooling achieved \*\*86.96% (20/23)\*\*.

\- Frame-majority aggregation achieved \*\*86.96% (20/23)\*\*.



Therefore, attention pooling numerically improved over mean pooling on the current 23-video test set and matched the frame-majority baseline.



\### Findings That Require Further Confirmation



The current evidence does \*\*not\*\* establish that attention pooling is a robust or generalizable improvement.



It also does not establish that the learned attention weights reliably identify semantically important warning frames.



\---



\## 4. What Is Not Established Yet



The following remain unknown or unconfirmed:



\- Whether attention pooling is genuinely better than mean pooling beyond this 23-video test set.

\- Whether attention pooling provides a statistically meaningful improvement.

\- Whether attention weights correspond reliably to important warning frames.

\- Whether the results generalize to an external or independent dataset.

\- Whether EfficientNet-B0 performs differently when trained and evaluated on the clean 564-frame dataset.

\- Whether the observed error-analysis patterns generalize beyond the inspected examples.



\---



\## 5. Remaining Work



1\. Re-run EfficientNet-B0 on the clean 564-frame dataset.

2\. Compare EfficientNet-B0 and ResNet50 under the same clean dataset conditions.

3\. Compare mean pooling, attention pooling, and frame-majority aggregation under the same evaluation conditions.

4\. Further analyze attention weights against correctly classified and misclassified videos.

5\. Verify the final reported metrics directly from the saved result files.

6\. Document the final experimental configuration.

7\. Consider moving large model checkpoints to Git LFS before adding additional large checkpoints.



\---



\## Current Conclusion



The completed work establishes a clean \*\*564-frame / 98-video dataset split\*\*, a \*\*ResNet50 clean-dataset baseline of 79.76% test accuracy\*\*, and video-level experiments using mean and attention pooling.



On the current 23-video test set:



\- Mean pooling achieved \*\*82.61% (19/23)\*\*.

\- Attention pooling achieved \*\*86.96% (20/23)\*\*.

\- Frame-majority aggregation achieved \*\*86.96% (20/23)\*\*.



These results demonstrate a numerical improvement of attention pooling over mean pooling on this test set, but they do \*\*not yet establish\*\* that attention pooling provides a robust, statistically significant, or generalizable improvement.



The most important next controlled experiment is to re-run EfficientNet-B0 on the clean 564-frame dataset and then compare the models and video-level aggregation methods under the same conditions.

