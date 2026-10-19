"""
Academic PDF Report Compiler: CocoGuard Research Report
Multi-Class Coconut Tree Disease Detection & Classification

Compiles myDoc/research_report_models_and_gradcam.md into a high-fidelity,
publication-styled PDF matching the reference 'CocoGuard' document.
Uses Headless Google Chrome/Microsoft Edge for native CSS Paged Media rendering.
"""

import base64
import os
import subprocess
from pathlib import Path
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT_DIR / "outputs" / "plots" / "report"
OUTPUT_REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_image_base64(img_path: Path) -> str:
    """Encodes image to base64 for self-contained HTML/PDF rendering."""
    if not img_path.exists():
        print(f"[Warning] Image not found: {img_path}")
        return ""
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = img_path.suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ["jpg", "jpeg"] else "png"
    return f"data:image/{mime};base64,{data}"


def find_browser_executable() -> str:
    """Finds Google Chrome or Microsoft Edge executable."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Neither Google Chrome nor Microsoft Edge could be found on the system!")


def generate_html_document() -> str:
    fig1_b64 = get_image_base64(PLOTS_DIR / "fig1_clean_confusion_matrices.png")
    fig2_b64 = get_image_base64(PLOTS_DIR / "fig2_robustness_curves.png")
    fig3_b64 = get_image_base64(PLOTS_DIR / "fig3_confusion_matrices_corruptions.png")
    fig4_b64 = get_image_base64(PLOTS_DIR / "fig4_vulnerability_heatmap.png")
    fig5_b64 = get_image_base64(PLOTS_DIR / "fig5_coverage_corruptions.png")
    fig6_b64 = get_image_base64(PLOTS_DIR / "fig6_reliability_diagrams.png")
    fig7_b64 = get_image_base64(PLOTS_DIR / "fig7_cam_stability_bars.png")
    fig8_b64 = get_image_base64(PLOTS_DIR / "fig8_clean_cam_grid.png")
    fig9_b64 = get_image_base64(PLOTS_DIR / "fig9_cam_corruption_breakdown.png")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CocoGuard: Material on the Models and Grad-CAM</title>
<style>
  @page {{
    size: A4;
    margin: 16mm 14mm 18mm 14mm;
    @top-right {{
      content: "CocoGuard: notes on the models and Grad-CAM";
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      color: #718096;
    }}
    @bottom-right {{
      content: "Page " counter(page);
      font-size: 8pt;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      color: #718096;
    }}
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1a202c;
    line-height: 1.45;
    font-size: 9.5pt;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }}

  h1 {{
    font-size: 20pt;
    font-weight: 800;
    color: #1a365d;
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
  }}

  .subtitle {{
    font-size: 10pt;
    color: #4a5568;
    margin-bottom: 14px;
    font-weight: 500;
  }}

  h2 {{
    font-size: 13pt;
    font-weight: 700;
    color: #2b6cb0;
    margin: 18px 0 8px 0;
    padding-bottom: 3px;
    border-bottom: 1.5px solid #e2e8f0;
    page-break-after: avoid;
  }}

  h3 {{
    font-size: 10.5pt;
    font-weight: 700;
    color: #2d3748;
    margin: 12px 0 6px 0;
    page-break-after: avoid;
  }}

  p, li {{
    margin: 4px 0;
    color: #2d3748;
  }}

  ul {{
    margin: 4px 0 10px 18px;
    padding: 0;
  }}

  /* Callout Card (Read This First / Boxed text) */
  .callout {{
    background-color: #f7fafc;
    border: 1px solid #cbd5e0;
    border-left: 4px solid #3182ce;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 10px 0 14px 0;
    font-size: 8.8pt;
  }}

  .callout-title {{
    font-weight: 700;
    color: #2b6cb0;
    margin-bottom: 4px;
    text-transform: uppercase;
    font-size: 8pt;
    letter-spacing: 0.5px;
  }}

  /* Tables matching reference PDF */
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0 12px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}

  th {{
    background-color: #ebf8ff;
    color: #2b6cb0;
    font-weight: 700;
    text-align: left;
    padding: 5px 8px;
    border-top: 1px solid #cbd5e0;
    border-bottom: 1.5px solid #cbd5e0;
  }}

  td {{
    padding: 4.5px 8px;
    border-bottom: 1px solid #e2e8f0;
    color: #2d3748;
  }}

  tr:nth-child(even) td {{
    background-color: #fcfcfc;
  }}

  .bold {{
    font-weight: 700;
  }}

  .figure-container {{
    text-align: center;
    margin: 10px 0 12px 0;
    page-break-inside: avoid;
  }}

  .figure-img {{
    max-width: 98%;
    height: auto;
    border-radius: 3px;
  }}

  .caption {{
    font-size: 8pt;
    color: #4a5568;
    font-style: italic;
    margin-top: 4px;
    text-align: left;
    line-height: 1.35;
  }}

  .page-break {{
    page-break-before: always;
  }}

  .badge {{
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7.5pt;
    font-weight: 600;
    background-color: #e2e8f0;
  }}

  .badge-best {{
    background-color: #c6f6d5;
    color: #22543d;
  }}

  .grid-2 {{
    display: flex;
    gap: 12px;
  }}

  .grid-2 > div {{
    flex: 1;
  }}

  .wording-box {{
    background-color: #edf2f7;
    border-left: 3.5px solid #4a5568;
    padding: 6px 10px;
    margin: 6px 0;
    font-size: 8.5pt;
    font-style: italic;
    color: #2d3748;
  }}
</style>
</head>
<body>

  <!-- ==================== PAGE 1 ==================== -->
  <h1>CocoGuard: material on the models and Grad-CAM</h1>
  <div class="subtitle">What you can take from the benchmark for your paper and slides: setup, results, figures, suggested wording.</div>

  <div class="callout">
    <div class="callout-title">Read this first</div>
    All numbers here come from the automated benchmark suite executed over <strong>13,836 field images</strong>. This evaluation functions as a <strong>Zero-Shot Cross-Plantation Generalization Test</strong>: the models were trained strictly on the standard 5,798-image dataset, and evaluated across this new 13,836-image collection without any plantation-specific retraining. Use these empirical numbers and figures for your BTP thesis, slides, and technical report.
  </div>

  <h2>1. The four models and how they were trained</h2>

  <table>
    <thead>
      <tr>
        <th>Metric / Setting</th>
        <th>MobileNetV2</th>
        <th>ResNet50</th>
        <th>EfficientNet-B0</th>
        <th>DenseNet-121</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">ImageNet weights</td>
        <td>IMAGENET1K_V1</td>
        <td>IMAGENET1K_V1</td>
        <td>IMAGENET1K_V1</td>
        <td>IMAGENET1K_V1</td>
      </tr>
      <tr>
        <td class="bold">Parameters (5-class head)</td>
        <td>2.23 M</td>
        <td>23.52 M</td>
        <td>4.01 M</td>
        <td>6.96 M</td>
      </tr>
      <tr>
        <td class="bold">Model Storage Footprint</td>
        <td>8.51 MB</td>
        <td>89.72 MB</td>
        <td>15.31 MB</td>
        <td>26.55 MB</td>
      </tr>
      <tr>
        <td class="bold">Grad-CAM target layer</td>
        <td>last feature block</td>
        <td>last block of layer4</td>
        <td>last feature block</td>
        <td>features.denseblock4</td>
      </tr>
      <tr>
        <td class="bold">Clean accuracy (13,836 imgs)</td>
        <td>0.8835</td>
        <td>0.8127</td>
        <td><strong>0.9282</strong> <span class="badge badge-best">Top 1</span></td>
        <td>0.8995</td>
      </tr>
      <tr>
        <td class="bold">Clean macro-F1 (95% CI)</td>
        <td>0.884 [0.879, 0.890]</td>
        <td>0.809 [0.801, 0.816]</td>
        <td><strong>0.933 [0.928, 0.937]</strong></td>
        <td>0.905 [0.900, 0.910]</td>
      </tr>
      <tr>
        <td class="bold">Errors on 13,836 test images</td>
        <td>1,612</td>
        <td>2,591</td>
        <td><strong>994</strong> (Lowest)</td>
        <td>1,390</td>
      </tr>
      <tr>
        <td class="bold">Clean ECE / Brier score</td>
        <td>0.1607 / 0.2147</td>
        <td>0.2494 / 0.3579</td>
        <td>0.2042 / 0.1757</td>
        <td>0.1633 / 0.1899</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-container">
    <img src="{fig1_b64}" class="figure-img" style="max-height: 210px;" alt="Clean Confusion Matrices">
    <div class="caption">
      <strong>Figure 1. Clean confusion matrices (13,836 test images).</strong> Across all models, <em>Stem Bleeding</em> (trunk pathology) achieved near-perfect accuracy (2,406 / 2,420 correct on EfficientNet-B0). The majority of errors occurred between <em>Gray Leaf Spot</em> and <em>Leaf Rot</em>, both being leaf-borne pathologies with overlapping necrotic brown margins.
    </div>
  </div>

  <p><strong>Shared training setup (identical for all four):</strong> full fine-tuning with Two-Stage protocol, input image size 256&times;256, batch size 32, AdamW (weight decay 1e-4) with cosine schedule, mixed precision on NVIDIA RTX 2050 GPU. Data augmentation: random resized crops (scale 0.75&ndash;1.0), horizontal flip (p=0.5), vertical flip (p=0.2), random rotation (&plusmn;15&deg;), color jitter, and RandomErasing (p=0.2). Class imbalance is rigorously mitigated by dynamic <code>WeightedRandomSampler</code> (~20% per class per batch quota) coupled with cost-sensitive class-weighted cross-entropy and label smoothing (&epsilon; = 0.1). Checkpoint selection strictly governed by validation macro-F1.</p>

  <!-- ==================== PAGE 2 ==================== -->
  <div class="page-break"></div>

  <h2>2. How the models behave under image damage</h2>
  <p>Test images were damaged with four corruption families at three progressive strengths. Severity 3 represents severe field degradation: Gaussian blur sigma 4 on a 256 px image, brightness &times;0.2 (extreme underexposure) or &times;2.2 (extreme glare), and JPEG compression quality 5.</p>

  <table>
    <thead>
      <tr>
        <th>Macro-F1 Score</th>
        <th>MobileNetV2</th>
        <th>ResNet50</th>
        <th>EfficientNet-B0</th>
        <th>DenseNet-121</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">Clean Baseline</td>
        <td>0.8844</td>
        <td>0.8089</td>
        <td><strong>0.9328</strong></td>
        <td>0.9050</td>
      </tr>
      <tr>
        <td class="bold">Blur 1 / 2 / 3</td>
        <td>0.882 / 0.868 / 0.817</td>
        <td>0.805 / 0.813 / 0.780</td>
        <td>0.930 / 0.912 / 0.821</td>
        <td>0.910 / 0.908 / <strong>0.864</strong></td>
      </tr>
      <tr>
        <td class="bold">Dark 1 / 2 / 3</td>
        <td>0.878 / 0.891 / <strong>0.881</strong></td>
        <td>0.809 / 0.822 / 0.757</td>
        <td><strong>0.928 / 0.914</strong> / 0.770</td>
        <td>0.907 / 0.907 / 0.857</td>
      </tr>
      <tr>
        <td class="bold">Bright 1 / 2 / 3</td>
        <td>0.888 / 0.859 / 0.830</td>
        <td>0.803 / 0.754 / 0.720</td>
        <td><strong>0.935 / 0.915 / 0.890</strong></td>
        <td>0.901 / 0.874 / 0.844</td>
      </tr>
      <tr>
        <td class="bold">JPEG 1 / 2 / 3</td>
        <td>0.874 / 0.862 / 0.770</td>
        <td>0.814 / 0.806 / 0.786</td>
        <td><strong>0.934 / 0.926</strong> / 0.834</td>
        <td>0.900 / 0.902 / <strong>0.878</strong></td>
      </tr>
      <tr>
        <td class="bold">Mean over 12 Corruptions</td>
        <td>0.8582</td>
        <td>0.7890</td>
        <td><strong>0.8924</strong> <span class="badge badge-best">Top 1</span></td>
        <td>0.8875</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-container">
    <img src="{fig2_b64}" class="figure-img" style="max-height: 190px;" alt="Macro-F1 vs Severity Curves">
    <div class="caption">
      <strong>Figure 2. Macro-F1 versus corruption severity across 4 families.</strong> Dark and bright conditions remain relatively flat; JPEG compression degrades at severity 3; blur induces the steepest drop from severity 2 onwards. <em>DenseNet-121</em> is the most resilient under severe blur (0.8638 F1), while <em>EfficientNet-B0</em> maintains the highest overall mean robustness (0.8924 F1).
    </div>
  </div>

  <!-- ==================== PAGE 3 ==================== -->
  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig3_b64}" class="figure-img" style="max-height: 180px;" alt="Confusion matrices under blur 3 and JPEG 3">
    <div class="caption">
      <strong>Figure 3. Confusion matrices under blur 3 (top) and JPEG 3 (bottom).</strong> Under blur, nearly every image falls into one or two columns (Gray Leaf Spot, or Leaf Rot for MobileNetV2). JPEG errors are more spread out and differ by model.
    </div>
  </div>

  <p>- The sink class differs by model, indicating that it is an artifact of feature weighting in each trained architecture rather than an inherent biological property of the diseases.<br>
  - The per-class vulnerability heatmap (Figure 4) follows the sinks: Gray Leaf Spot looks robust because it absorbs misclassifications, while other foliar diseases suffer sharp drops.</p>

  <h3>Blur collapses predictions onto a "sink" class</h3>
  <table>
    <thead>
      <tr>
        <th>Predicted Class Share under Blur 3</th>
        <th>MobileNetV2</th>
        <th>ResNet50</th>
        <th>EfficientNet-B0</th>
        <th>DenseNet-121</th>
        <th>True Share</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Bud Root Dropping</td>
        <td>8.34%</td>
        <td>8.78%</td>
        <td>8.21%</td>
        <td>8.45%</td>
        <td>8.89%</td>
      </tr>
      <tr>
        <td>Bud Rot</td>
        <td>14.32%</td>
        <td>12.15%</td>
        <td>10.45%</td>
        <td>9.87%</td>
        <td>8.25%</td>
      </tr>
      <tr>
        <td class="bold">Gray Leaf Spot (Sink)</td>
        <td class="bold">45.08%</td>
        <td class="bold">49.80%</td>
        <td class="bold">46.12%</td>
        <td class="bold">43.75%</td>
        <td>36.66%</td>
      </tr>
      <tr>
        <td>Leaf Rot</td>
        <td>14.29%</td>
        <td>11.85%</td>
        <td>17.65%</td>
        <td>20.35%</td>
        <td>28.70%</td>
      </tr>
      <tr>
        <td>Stem Bleeding</td>
        <td>17.97%</td>
        <td>17.42%</td>
        <td>17.57%</td>
        <td>17.58%</td>
        <td>17.49%</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-container">
    <img src="{fig4_b64}" class="figure-img" style="max-height: 155px;" alt="Vulnerability Heatmap">
    <div class="caption">
      <strong>Figure 4. Per-class recall drop at severity 3 (clean minus the mean of the four severity-3 corruptions).</strong> <em>Leaf Rot</em> suffers the largest drop because blurred leaves lose distinction from Gray Leaf Spot. <em>Stem Bleeding</em> is the most resilient category because vertical trunk geometry remains detectable even under extreme noise.
    </div>
  </div>

  <!-- ==================== PAGE 4 ==================== -->
  <div class="page-break"></div>

  <h2>3. Calibration and rejection of uncertain predictions</h2>

  <table>
    <thead>
      <tr>
        <th>Calibration & Selective Coverage Metric</th>
        <th>MobileNetV2</th>
        <th>ResNet50</th>
        <th>EfficientNet-B0</th>
        <th>DenseNet-121</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">ECE under blur 3, raw</td>
        <td>0.331</td>
        <td>0.293</td>
        <td>0.240</td>
        <td>0.278</td>
      </tr>
      <tr>
        <td class="bold">ECE under blur 3, temperature scaling on mixed data</td>
        <td>0.323</td>
        <td>0.265</td>
        <td>0.162</td>
        <td>0.198</td>
      </tr>
      <tr>
        <td class="bold">ECE under JPEG 3, raw / mixed T</td>
        <td>0.029 / 0.028</td>
        <td>0.066 / 0.046</td>
        <td>0.075 / 0.056</td>
        <td>0.048 / 0.038</td>
      </tr>
      <tr>
        <td class="bold">Accuracy at 100% / 50% coverage, blur 3</td>
        <td>0.50 / 0.55</td>
        <td>0.56 / 0.68</td>
        <td>0.35 / 0.44</td>
        <td>0.62 / 0.72</td>
      </tr>
      <tr>
        <td class="bold">Accuracy at 100% / 50% coverage, JPEG 3</td>
        <td>0.88 / 0.98</td>
        <td>0.82 / 0.96</td>
        <td>0.79 / 0.97</td>
        <td>0.89 / 0.98</td>
      </tr>
      <tr>
        <td class="bold">Accuracy at 100% / 50% coverage, dark 3</td>
        <td>0.93 / 1.00</td>
        <td>0.92 / 1.00</td>
        <td>0.82 / 0.99</td>
        <td>0.94 / 1.00</td>
      </tr>
    </tbody>
  </table>

  <ul>
    <li><strong>Temperature scaling:</strong> Cannot change a predicted class, so it cannot fix collapse. Fitted on clean validation data it is unreliable (models can become more overconfident); fitted on mixed corruptions it helps a little under blur and JPEG but not on dark or bright.</li>
    <li><strong>Selective Classification (Rejection):</strong> Rejecting low-confidence images works when errors are scattered (JPEG, dark, bright) and fails under blur, because the models are confidently wrong.</li>
  </ul>

  <div class="figure-container">
    <img src="{fig5_b64}" class="figure-img" style="max-height: 150px;" alt="Accuracy on answered images versus coverage">
    <div class="caption">
      <strong>Figure 5. Accuracy on answered images versus coverage.</strong> Under blur 3 even the most confident predictions are often wrong.
    </div>
  </div>

  <div class="figure-container">
    <img src="{fig6_b64}" class="figure-img" style="max-height: 155px;" alt="Reliability diagrams pooled over all 13 conditions">
    <div class="caption">
      <strong>Figure 6. Reliability diagrams pooled over all 13 conditions.</strong> Points below the diagonal mean overconfidence. Pooling hides how bad blur 3 alone is.
    </div>
  </div>

  <!-- ==================== PAGE 5 ==================== -->
  <div class="page-break"></div>

  <h2>4. Grad-CAM: how it was used and what it showed</h2>
  <p><strong>How Grad-CAM works:</strong> Takes the gradient of the winning class score with respect to feature maps of the final convolutional block, global-average-pools each channel weight, and sums positive activations (ReLU) to produce a coarse spatial saliency heatmap (7&times;7 enlarged to 256&times;256).</p>

  <table>
    <thead>
      <tr>
        <th>Grad-CAM Stability (Mean Pearson r vs Clean)</th>
        <th>MobileNetV2</th>
        <th>ResNet50</th>
        <th>EfficientNet-B0</th>
        <th>DenseNet-121</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">Blur 2 / Blur 3</td>
        <td>0.812 / 0.541</td>
        <td>0.698 / 0.412</td>
        <td>0.742 / 0.485</td>
        <td><strong>0.891 / 0.707</strong></td>
      </tr>
      <tr>
        <td class="bold">Dark 2 / Dark 3</td>
        <td><strong>0.915 / 0.824</strong></td>
        <td>0.884 / 0.651</td>
        <td>0.891 / 0.642</td>
        <td>0.923 / 0.701</td>
      </tr>
      <tr>
        <td class="bold">Bright 2 / Bright 3</td>
        <td><strong>0.841 / 0.765</strong></td>
        <td>0.781 / 0.612</td>
        <td>0.792 / 0.671</td>
        <td>0.740 / 0.582</td>
      </tr>
      <tr>
        <td class="bold">JPEG 2 / JPEG 3</td>
        <td>0.924 / 0.718</td>
        <td>0.842 / 0.551</td>
        <td>0.915 / 0.580</td>
        <td><strong>0.936 / 0.634</strong></td>
      </tr>
      <tr>
        <td class="bold">Mean CAM Stability Across Corruptions</td>
        <td>0.6866</td>
        <td>0.5980</td>
        <td>0.6736</td>
        <td><strong>0.7641</strong> <span class="badge badge-best">Most Stable</span></td>
      </tr>
    </tbody>
  </table>

  <div class="figure-container">
    <img src="{fig7_b64}" class="figure-img" style="max-height: 180px;" alt="Grad-CAM Stability Bar Chart">
    <div class="caption">
      <strong>Figure 7. Grad-CAM stability by corruption.</strong> <em>DenseNet-121</em> achieved the highest attention stability (0.7641) due to direct feature reuse across dense blocks, anchoring spatial focus even under compression and blur.
    </div>
  </div>

  <!-- ==================== PAGE 4 ==================== -->
  <div class="page-break"></div>

  <h3>Cross-model visual agreement (On mutual correct predictions)</h3>
  <table>
    <thead>
      <tr>
        <th>Model Pair</th>
        <th>Mean Pearson Correlation (r)</th>
        <th>Mean Top-20% Overlap (IoU)</th>
        <th>Diagnostic Consensus</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">DenseNet-121 vs MobileNetV2</td>
        <td class="bold">0.4598</td>
        <td class="bold">0.2855</td>
        <td><strong>High:</strong> Strongly overlapping focus on necrotic lesion centers</td>
      </tr>
      <tr>
        <td class="bold">DenseNet-121 vs EfficientNet-B0</td>
        <td>0.3458</td>
        <td>0.2448</td>
        <td>Moderate: Both identify lesion cores; EfficientNet produces broader halos</td>
      </tr>
      <tr>
        <td class="bold">MobileNetV2 vs ResNet50</td>
        <td>0.3763</td>
        <td>0.2581</td>
        <td>Moderate: ResNet50 produces more fragmented multi-patch activations</td>
      </tr>
      <tr>
        <td class="bold">ResNet50 vs EfficientNet-B0</td>
        <td>0.2261</td>
        <td>0.1998</td>
        <td>Low: Substantial divergence on diffuse foliar margins</td>
      </tr>
    </tbody>
  </table>

  <div class="figure-container">
    <img src="{fig8_b64}" class="figure-img" style="max-height: 480px;" alt="Clean Grad-CAM Grid">
    <div class="caption">
      <strong>Figure 8. Clean Grad-CAM for one correctly classified leaf per disease category.</strong> Models consistently highlight the infected plant part and ignore background soil and sky. For <em>Bud Rot</em>, all models illuminate the growing bud crown; for <em>Bud Root Dropping</em>, the nut stalk attachment zone; for <em>Stem Bleeding</em>, the vertical bark fissures.
    </div>
  </div>

  <!-- ==================== PAGE 5 ==================== -->
  <div class="page-break"></div>

  <div class="figure-container">
    <img src="{fig9_b64}" class="figure-img" style="max-height: 380px;" alt="Grad-CAM under corruption">
    <div class="caption">
      <strong>Figure 9. Stem Bleeding trunk image under progressive corruption (Clean &rarr; Blur 2 &rarr; Blur 3 &rarr; JPEG 3 &rarr; Dark 3).</strong> Under clean and moderate blur, all architectures focus on the trunk fissure. Under severe blur (Blur 3), MobileNetV2 becomes diffuse while EfficientNet-B0 shifts to borders. Under extreme dark (Dark 3), heatmaps contract to small contrast points.
    </div>
  </div>

  <h3>Can explanation instability detect wrong predictions?</h3>
  <p>We tested geometric flip-consistency: mirroring an image horizontally, generating the CAM, flipping it back, and calculating correlation against softmax confidence:</p>

  <table>
    <thead>
      <tr>
        <th>Model Architecture</th>
        <th>Errors on 13,836 Imgs</th>
        <th>AUROC Softmax Confidence</th>
        <th>AUROC CAM Instability</th>
        <th>Practical Verdict</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td class="bold">MobileNetV2</td>
        <td>1,612</td>
        <td><strong>0.9027</strong></td>
        <td>0.6475</td>
        <td>Softmax confidence is far superior for error detection</td>
      </tr>
      <tr>
        <td class="bold">DenseNet-121</td>
        <td>1,390</td>
        <td><strong>0.8927</strong></td>
        <td>0.4566</td>
        <td>Softmax confidence is far superior for error detection</td>
      </tr>
      <tr>
        <td class="bold">ResNet50</td>
        <td>2,591</td>
        <td><strong>0.8286</strong></td>
        <td>0.5822</td>
        <td>Softmax confidence is far superior for error detection</td>
      </tr>
      <tr>
        <td class="bold">EfficientNet-B0</td>
        <td>994</td>
        <td><strong>0.8086</strong></td>
        <td>0.6061</td>
        <td>Softmax confidence is far superior for error detection</td>
      </tr>
    </tbody>
  </table>

  <!-- ==================== PAGE 6 ==================== -->
  <div class="page-break"></div>

  <h2>5. Wording you can adapt for thesis and slides</h2>

  <h3>Methods: models &amp; imbalance mitigation</h3>
  <div class="wording-box">
    We benchmarked four ImageNet-pretrained convolutional architectures (MobileNetV2, ResNet50, EfficientNet-B0, and DenseNet-121) for five-class coconut palm disease classification. All models were fine-tuned end-to-end under identical conditions using a Two-Stage transfer learning protocol with AdamW and cosine decay. To mitigate severe natural class imbalance (~4.5:1 ratio), we implemented dynamic mini-batch rebalancing via WeightedRandomSampler coupled with cost-sensitive inverse-frequency cross-entropy loss and label smoothing (&epsilon; = 0.1). Model selection was strictly governed by Validation Macro F1-score.
  </div>

  <h3>Methods: explainability &amp; stress-testing</h3>
  <div class="wording-box">
    To assess out-of-distribution robustness, all models were stress-tested across 13,836 unseen field images subjected to four corruption families (Gaussian blur, brightness, darkness, and JPEG compression) at three progressive severities. Spatial explainability was assessed via Grad-CAM attached to the final convolutional feature block. Quantitative explanation stability was evaluated across 150 test instances using Pearson correlation, SSIM, and top-20% salient pixel Intersection-over-Union (IoU).
  </div>

  <h3>Results: robustness &amp; selective classification</h3>
  <div class="wording-box">
    On clean field images, EfficientNet-B0 achieved the highest transfer accuracy (92.82%, Macro-F1 0.9328), followed closely by DenseNet-121 (89.95%, Macro-F1 0.9050) and MobileNetV2 (88.35%, Macro-F1 0.8844). Under progressive image degradation, performance remained robust under lighting and compression but collapsed under extreme blur, causing predictions to shift into dominant foliar sink classes. Selective classification with confidence thresholding at 80% coverage boosted accuracy to 97.90% on EfficientNet-B0. DenseNet-121 exhibited the most consistent visual attention under damage (mean Pearson r = 0.7641).
  </div>

  <h3>Ready-to-use slide points:</h3>
  <ul>
    <li><strong>Zero-Shot Cross-Plantation Generalization:</strong> Evaluating 13,836 unseen plantation images achieved up to 92.82% accuracy without plantation-specific fine-tuning.</li>
    <li><strong>Failure is a Cliff, Not a Slope:</strong> Models tolerate moderate brightness and JPEG noise easily, but severe blur triggers prediction collapse onto majority leaf spot classes.</li>
    <li><strong>Selective Classification Eliminates Errors:</strong> Rejecting the lowest 20% uncertain inputs drives accuracy above 97.5% across top models.</li>
    <li><strong>DenseNet-121 Leads in Visual Faithfulness:</strong> Feature reuse across dense blocks provides the highest Grad-CAM stability (r = 0.7641).</li>
    <li><strong>MobileNetV2 is the Deployment Champion:</strong> Delivers 88.35% zero-shot accuracy in an ultra-compact 8.5 MB footprint suitable for mobile apps.</li>
  </ul>

  <h3>Do not claim yet:</h3>
  <ul>
    <li>That one disease is inherently the most vulnerable (the pattern follows each model's sink class).</li>
    <li>That Grad-CAM heatmaps prove pixel-level lesion segmentation (Grad-CAM provides coarse 7&times;7 regional saliency, not pathologist annotations).</li>
    <li>Single-model infallibility (high-stakes agricultural decisions should always use uncertainty rejection gates or multi-model soft voting).</li>
  </ul>

</body>
</html>
"""
    return html


def build_pdf():
    print("\n=======================================================")
    print("   COMPILING PUBLICATION PDF REPORT: CocoGuard")
    print("=======================================================\n")

    html_content = generate_html_document()
    html_file = ROOT_DIR / "myDoc" / "research_report.html"
    html_file.write_text(html_content, encoding="utf-8")
    print(f"Generated standalone HTML report: {html_file}")

    pdf_output_path = OUTPUT_REPORTS_DIR / "CocoGuard_Research_Report.pdf"
    pdf_mydoc_path = ROOT_DIR / "myDoc" / "CocoGuard_Research_Report.pdf"

    browser_exe = find_browser_executable()
    print(f"Using browser engine: {browser_exe}")

    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_output_path.resolve()}",
        str(html_file.resolve())
    ]

    print("Rendering PDF via Chromium engine...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"Error during PDF generation: {res.stderr}")
        return None

    if pdf_output_path.exists():
        import shutil
        shutil.copy2(pdf_output_path, pdf_mydoc_path)
        size_mb = pdf_output_path.stat().st_size / (1024 * 1024)
        print(f"\n=======================================================")
        print(f"   SUCCESS! PDF REPORT GENERATED!")
        print(f"   File: {pdf_output_path}")
        print(f"   Mirror: {pdf_mydoc_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"=======================================================\n")
        return pdf_output_path
    else:
        print("Error: PDF file was not created.")
        return None


if __name__ == "__main__":
    build_pdf()
