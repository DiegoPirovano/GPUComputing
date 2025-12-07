# CUDA Medical Image Segmentation Analysis

This project implements a high-performance, GPU-accelerated pipeline to analyze and compare medical image segmentations stored in HDF5 format. It calculates statistical metrics to evaluate the variability and agreement between different annotators (specifically comparing **Expert** vs. **Non-Expert** groups).

The application is designed to benchmark parallel CUDA performance against a sequential implementation, ensuring correctness via a verification step.

## Features

* **HDF5 Data Ingestion:** loads flattened segmentation masks and metadata (`expertise`, `Origin` attributes) from `annotations.h5`.
* **Dual Pipeline:**
    1.  **Parallel (GPU):** Optimized CUDA kernels for batch processing.
    2.  **Sequential (CPU):** Reference implementation for validation.
* **Automated Verification:** Compares GPU and CPU results with a strict tolerance ($10^{-4}$) to ensure accuracy.
* **Performance Profiling:** Reports execution time and speedup factors.

## Metrics Calculated

For every image group (Expert/Non-Expert), the following metrics are computed:

### 1. Average Dice Coefficient
Measures the spatial overlap accuracy between pairs of segmentations.

$$Dice = \frac{2 \times |A \cap B|}{|A| + |B|}$$


### 2. Mean Agreement Entropy
Quantifies the pixel-wise uncertainty or disagreement across a group of masks.

### 3. Normalized Area Standard Deviation
Measures the variability in the volume of the segmented structures.


