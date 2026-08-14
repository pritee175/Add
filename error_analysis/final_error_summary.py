import csv
from pathlib import Path
from collections import Counter

p = Path("error_analysis/manual_hard_case_review.csv")

with p.open(encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("\n===== SHARED ERROR ANALYSIS =====")
print(f"Shared-error frames: {len(rows)}")

both_wrong = Counter()
causes = Counter()

for r in rows:
    direction = f'{r["true_class"]} -> {r["resnet50_prediction"]}'
    both_wrong[direction] += 1
    causes[r["manual_evidence"]] += 1

print("\nError direction:")
for k, v in both_wrong.items():
    print(f"  {k}: {v}")

print("\nManual evidence:")
for k, v in causes.items():
    print(f"  {k}: {v}")

print("\nInterpretation:")
print(
    "The shared errors contain frames where the visual evidence for the "
    "underlying class is weak or incomplete. Several frames show people, "
    "objects, vehicles, scenes, or contextual imagery without an explicit "
    "class-defining message. Therefore, frame-level classification can be "
    "difficult when the relevant semantic information is distributed across "
    "the video rather than visible in an individual frame."
)
