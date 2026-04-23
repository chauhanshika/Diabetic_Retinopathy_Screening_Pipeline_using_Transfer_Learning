# Clinical Decision Guidelines & Triage Mechanics

How does this model actually fit into a hospital or tele-medicine screening system?

## 1. The Triage Workflow
This model acts strictly as a **First-Pass Screening Filter**, not an autonomous diagnostic arbiter. 
1. **Intake**: A remote clinic snaps a fundus image and uploads it.
2. **Inference**: The model scans the image.
3. **Filtering**: 
   - If the model is >85% confident the image is Normal, the image goes into a low-priority queue.
   - If the model predicts Abnormal, the image is marked **CRITICAL** and pushed directly to the front of a human ophthalmologist's queue.

## 2. Managing False Negatives Risk
The **64 False Negatives** are our greatest clinical liability. An improvement protocol requires **Threshold Sensitivity Tuning**. The sigmoid threshold should realistically be shifted from standard argmax (`>0.50`) to conservative diagnostic boundaries (`>0.30`). This sacrifices False Positives deliberately to catch fringe pathological cases. 

## 3. Assistive Alignment vs Autonomous Diagnostics
Specialists suffer from massive visual fatigue examining thousands of fundus scans per week. Human accuracy drops steeply late in shifts. This ML pipeline is not designed to replace the specialist, but rather to shield them from having to review thousands of perfectly healthy scans, allowing them to redirect their expertise exclusively towards the complex, borderline, and critically ill patients that the ML model flags into their fast-track queue.
