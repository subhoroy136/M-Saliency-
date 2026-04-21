# M-Saliency: Biologically-Constrained Saliency in Histopathology

A mathematically rigorous and biologically feasible framework for interpretable histopathological cancer detection, integrating pathological primitives directly into deep learning architectures.

**Author:** Subhodip Roy, Independent Researcher, West Bengal, India  
**Date:** April 21, 2026  
**License:** MIT

## Overview

M-Saliency addresses a critical problem in computational pathology: contemporary deep learning models achieve high quantitative performance on benchmark datasets but frequently exploit non-diagnostic artifacts (staining intensities, background textures) rather than morphologically meaningful features. When deployed across institutional boundaries with different staining protocols and scanning hardware, these models often fail catastrophically.

M-Saliency solves this through a Unified Morphological Scoring Function that constrains feature extraction to established pathological primitives, ensuring diagnoses are "Right for the Right Reasons."

## Key Contributions

- **Mathematically Rigorous Framework:** Four mathematical gates grounded in cellular pathology theory
- **Multi-Dataset Validation:** Consistent performance across BreakHis, LC25000, PCam, and CRC datasets
- **Pathologist-Validated Interpretability:** Saliency maps confirmed to highlight genuine diagnostic features
- **Institutional Robustness:** Staining-invariant design ensures generalization across clinical settings
- **Computational Efficiency:** 25.5M parameters vs. foundation models with 300M+

## Architecture

### Unified Morphological Scoring Function

```
M(x,y) = μc(x,y) · [λ₁/(λ₂ + ε)] · ||∇L(x,y)||² · [1 + γ·4μc(x,y)(1-μc(x,y))]
```

**Gate I: Presence Gate** – Tissue identification via sigmoid activation
**Gate II: Symmetry Ratio** – Boundary asymmetry quantification via Structure Tensor eigenvalues
**Gate III: Boundary Energy** – Morphological contrast detection via luminance gradients
**Gate IV: Threshold Amplifier** – Uncertainty-directed focus via Gini Impurity

## Performance

| Dataset | Cancer Type | Samples | AUC | Accuracy |
|---------|------------|---------|-----|----------|
| BreakHis | Breast | 7,909 | 0.9941 | 0.9834 |
| LC25000 | Lung | 25,000 | 0.9953 | 0.9887 |
| PCam | Prostate | 327,680 | 0.9889 | 0.9756 |
| CRC-VAL-HE-7K | Colorectal | 7,000 | 0.9916 | 0.9812 |

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0.1
- CUDA 11.0+ (for GPU support)
- scikit-learn, Pillow, NumPy, Matplotlib

### Quick Setup

```bash
# Clone repository
git clone https://github.com/subhodiproy/M-Saliency.git
cd M-Saliency

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from m_saliency_framework import MSaliencyFramework, train_model, evaluate_model
import torch

# Initialize model
model = MSaliencyFramework(backbone='resnet50', device='cuda')

# Train
history = train_model(model, train_loader, val_loader, epochs=5)

# Evaluate
metrics = evaluate_model(model, test_loader)
print(f"AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
```

### Kaggle GPU Training

```bash
# Launch training on Kaggle with dual GPU support
python train_m_saliency.py \
    --dataset breakhis \
    --epochs 5 \
    --batch_size 32 \
    --gpu_ids 0 1 \
    --validate_on multiple
```

### Generating Saliency Maps

```python
model.eval()
with torch.no_grad():
    logits, saliency_dict = model.forward(input_image, return_saliency=True)
    
    # Access individual components
    presence = saliency_dict['presence']
    symmetry = saliency_dict['symmetry']
    boundary = saliency_dict['boundary']
    combined = saliency_dict['combined']
```

## Project Structure

```
M-Saliency/
├── m_saliency_framework.py      # Core framework implementation
├── train_m_saliency.py          # Training script with multi-GPU support
├── evaluate_framework.py        # Evaluation utilities and metrics
├── visualize_saliency.py        # Saliency map visualization tools
├── requirements.txt             # Package dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
├── Biologically-Constrained_Saliency_Research_Paper.pdf
└── examples/
    ├── basic_training.py        # Basic training example
    ├── multi_dataset_validation.py
    └── saliency_visualization.py
```

## Key Features

### Mathematical Soundness

Each gate implements established pathological principles:
- **Presence Gate:** Non-tissue regions rejected before morphological analysis
- **Symmetry Ratio:** Asymmetric patterns characteristic of malignancy identified via Structure Tensor eigenvalues
- **Boundary Energy:** Sharp transitions of malignancy vs. gradual transitions of normal tissue
- **Threshold Amplifier:** Hard cases where network is uncertain receive expert attention

### Biological Feasibility

All components validated through systematic ablation studies demonstrating necessity and contribution to biological grounding.

### Clinical Interpretability

Pathologists can audit model decisions by examining each mathematical gate independently. Saliency maps highlight regions matching pathological theory, not arbitrary "attended" pixels.

### Institutional Robustness

L* channel invariance eliminates staining intensity variation. Morphological constraints ensure generalization across institutional variations in staining protocols and scanning hardware.

## Experimental Results

### Single-Dataset Performance
- Baseline ResNet-50: 0.9979 AUC
- M-Saliency: 0.9916 AUC
- **Transparency Tax:** 0.63% AUC reduction for biological constraints

### Ablation Study Results
| Variant | AUC | Biological Grounding |
|---------|-----|-------------------|
| Full M-Saliency | 0.9916 | High |
| −Presence Gate | 0.8930 | Minimal |
| −Symmetry Ratio | 0.9963 | Low |
| −Boundary Energy | 0.9964 | Low |
| −Threshold Amplifier | 0.9842 | Medium |

### Performance Paradox
Removing biological constraints increases AUC to 0.9963-0.9964 but collapses biological grounding to "Low," demonstrating model exploits non-diagnostic artifacts.

## Parameter Tuning

### Gamma (γ = 0.35)
Controls uncertainty boost magnitude in the Threshold Amplifier. Values below 0.25 provide insufficient focus on ambiguous regions. Values above 0.50 create artifactual peaks in confident regions.

### Epsilon (ε = 1 × 10⁻⁵)
Prevents division-by-zero in Symmetry Ratio at perfect edges where λ₂ approaches zero. Smaller values risk numerical instability. Larger values artificially bias toward isotropic tissue.

## Computational Requirements

- **Memory:** 4GB GPU VRAM (single GPU), scales to multi-GPU environments
- **Training Time:** ~7 hours per epoch on dual GPU setup (Kaggle P100)
- **Inference:** 2-3% overhead vs. baseline ResNet-50
- **Model Size:** 25.5M parameters (ResNet-50)

## Citation

If you use M-Saliency in your research, please cite:

```bibtex
@article{Roy2026MSaliency,
  title={Biologically-Constrained Saliency in Histopathology},
  author={Roy, Subhodip},
  year={2026},
  institution={Independent Researcher, West Bengal, India}
}
```

## Limitations and Future Work

### Current Limitations
- Evaluation on patch-based samples rather than whole-slide images
- Pathologist validation by two experts (n=2)
- Public datasets rather than prospective clinical samples
- Limited evaluation on genuinely ambiguous diagnostic cases

### Future Directions
- Whole-slide image processing pipeline
- Multi-institutional pathologist validation panel
- Prospective clinical trial validation
- Extension to additional cancer types
- Real-time clinical deployment integration

## Contributing

We welcome contributions from the research community. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Submit Pull Request with detailed description

## Acknowledgments

- NCT-CRC-HE-100K dataset creators
- BreakHis, LC25000, and PCam dataset providers
- Pathologists who validated saliency maps
- Open-source research community

## Contact

**Author:** Subhodip Roy  
**Email:** [contact information]  
**GitHub:** [@subhodiproy](https://github.com/subhodiproy)  
**Location:** West Bengal, India

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

This framework is designed for research and educational purposes. It should not be used for clinical diagnosis without regulatory approval and clinical validation. Always consult qualified pathologists for diagnostic decisions.

---

**Status:** Active Development  
**Last Updated:** April 21, 2026  
**Version:** 2.1.0
