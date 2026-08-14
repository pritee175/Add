# Today's Research Work Report

## Date

2026-08-14

## Objective

The objective of today's work was to perform error analysis on the
Preventive vs Promotional video-frame classification task.

## What was analyzed

Two trained image-classification models were considered:

- EfficientNet-B0
- ResNet50

The analysis focused on frames where both models made the same incorrect
prediction.

## Shared-error analysis

A total of 13 shared-error frames were manually reviewed.

The shared errors came from five videos:

- mHBb67I9uT0
- PREV004
- PREV006
- YU0COqQ9W-A
- Z4A8G6FzgSw

## Main observation

The manually reviewed frames often did not contain enough explicit visual
information to make the underlying class obvious.

Several frames contained people, objects, vehicles, scenes, or contextual
imagery, while the actual semantic information needed to distinguish
Preventive from Promotional content may occur elsewhere in the video.

Therefore, these errors should not be interpreted as evidence that the
frames themselves clearly depict an addictive product or advertisement.

## Important methodological decision

No unsupported product-level labels were assigned during manual review.

Instead, the errors were described using conservative visual categories:

- insufficient_visual_context
- weak_class_signal

This avoids inventing semantic information that is not clearly visible in
the individual frame.

## Interpretation

The analysis suggests that some classification errors may arise because
the task is being performed at the individual-frame level while the
meaning of an advertisement can depend on information distributed across
multiple frames of a video.

This provides a useful direction for future analysis: consider whether
temporal/video-level information could improve classification compared
with relying only on isolated frames.

## Files produced/updated

- error_analysis/manual_hard_case_review.csv
- error_analysis/final_error_summary.py
- error_analysis/final_error_summary.txt
- error_analysis/shared_error_review.jpg

## Status

Today's error-analysis task is complete.

The next stage can focus on documenting model performance and considering
video-level/temporal information as a possible future improvement.
