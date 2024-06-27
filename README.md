
# NEW: FLAME Diffuser: Wildfire Image Synthesis using Mask Guided Diffusion

## Introduction

Wildfires have devastating impacts on natural environments and human settlements. Existing fire detection systems rely on large, annotated datasets that often lack geographic diversity, leading to decreased generalizability. To address this, we introduce the **FLAME Diffuser**, a diffusion-based framework that synthesizes high-quality wildfire images with precise flame location control. This training-free framework eliminates the need for model fine-tuning and provides a dataset of 10,000 high-quality images, enhancing the development of robust wildfire detection models.


## FLAME-SD Dataset

<img src="../images/FLAME_SD/sample.jpg" width="500px">

- **Content:** 10,000 RGB images, each paired with binary masks and metadata.
- **Quality Control:** CLIP-based filtering ensures high confidence and relevance of wildfire content.

- **Dataset and Code:** Available at [GitHub - FLAME_SD](https://arazi2.github.io/aisends.github.io/project/flame)

---

---

## Key Features

- **Training-Free Diffusion Framework:** Generates wildfire images without the need for model training or fine-tuning.
- **Precise Flame Control:** Utilizes noise-based masks for accurate flame placement.
- **Diverse Backgrounds:** Creates images with varied and realistic backgrounds, enhancing model generalizability.
- **High-Quality Dataset:** Introduces FLAME-SD with 10,000 synthesized images for robust model training.

## Methodology

<img src="../images/FLAME_SD/frame.png" width="500px">

1. **Mask Generation:** 
   - Masks are generated to define areas for fire elements using fundamental shapes like rectangles and circles.
   - Noise is added to the masks to create a smoother integration process.

2. **Diffusion Process:**
   - Combines masks with raw images, processed through a Variational Autoencoder (VAE) to generate latent variables.
   - The denoising U-Net refines these variables to produce realistic images guided by text prompts.

3. **Data Filtering:**
   - Utilizes CLIP for filtering synthesized images, ensuring high-quality and relevant wildfire content.

## Experimental Results

<img src="../images/FLAME_SD/sample_2.jpg" width="500px">

- **Mask-Guided Control:** Demonstrated effective integration of fire elements into images, maintaining the original style.
- **Background and Style Control:** Adjusting mask noise and text prompts allowed precise control over image content and style.

For more details, visit the [GitHub - FLAME_SD](https://arazi2.github.io/aisends.github.io/project/flame).

---
**Authors:** Hao Wang, Sayed Pedram Haeri Boroujeni, Xiwen Chen, Ashish Bastola, Huayu Li, Wenhui Zhu, and Abolfazl Razi
**Affiliations:** Clemson University, The University of Arizona, Arizona State University
**Corresponding Author:** Abolfazl Razi (Email: [contact information])

**Funding:** This project is supported by the National Science Foundation under Grant Number CNS-2204445. Special thanks to USDA Forest Service and Kaibab National Forest administration.
