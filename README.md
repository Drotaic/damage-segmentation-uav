# Damage Segmentation UAV

Pixel-level building damage classification from UAV imagery using semantic segmentation with U-Net/DeepLabV3+ models.

## 🧱 Use Case
After natural disasters like earthquakes, drones capture aerial images. This tool classifies building conditions pixel-by-pixel to support rapid response and resource allocation.

## 🧠 Model
- Semantic segmentation using U-Net or DeepLabV3+
- Input: Aerial drone images
- Output: Segmentation masks:
  - Green = Intact
  - Yellow = Minor damage
  - Red = Collapsed

## 📁 Structure
- `data/`: Training images and labeled masks
- `models/`: Trained segmentation models
- `notebooks/`: Interactive training + visualization
- `scripts/`: Inference and mask overlay tools
- `assets/`: Sample images and results

## 🛠️ How to Use
1. Place training data in `data/`
2. Train the model via `segmentation_notebook.ipynb`
3. Run `segment.py` to infer masks and overlay

## 🔮 Future Work
- Use real datasets like xBD
- Integration with drone AI for live damage map feeds
