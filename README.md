# Thyroid Ultrasound Nodule Segmentation  

## Quantitative Results on Private Thyroid Ultrasound Dataset (N = 5,248)

| Model              | Dice ↑       | IoU ↑       | Precision ↑  | Recall ↑    |
|--------------------|--------------|-------------|--------------|-------------|
| U-Net++            | 0.6102       | 0.5525      | 0.6544       | 0.7860      |
| AAUNet             | 0.7233       | 0.6243      | 0.7967       | 0.7252      |
| **Full Proposed**  | **0.7441**   | **0.6714**  | **0.8643**   | **0.7502**  |

**Our Full Proposed model achieves +2.08% Dice and +4.71% IoU improvement over the strongest baseline (AAUNet).**

### Highlights
- Fully automatic thyroid nodule segmentation on real clinical ultrasound images  
- Novel multi-scale attention + adaptive asymmetry convolution module  
- Supports **machine unlearning** (SISA + Influence-based) – delete mislabeled or patient-withdrawn samples in minutes instead of days  
- nnU-Net style automatic preprocessing pipeline  
- 5-fold cross-validation + external test set evaluation  

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/yourname/thyroid-ultrasound-segmentation.git
cd thyroid-ultrasound-segmentation

# 2. Create environment
conda create -n thyroid python=3.9 -y
conda activate thyroid
pip install -r requirements.txt

# 3. Training (example: 5-fold, Full Proposed model)
python train.py --config configs/full_proposed.yaml --fold 0

# 4. Inference
python inference.py --checkpoint checkpoints/fold0_best.pth --output results/

