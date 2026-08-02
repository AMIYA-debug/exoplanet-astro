# Automated Exoplanet Detection from NASA Kepler Light Curves Using the InceptionTime Deep Learning Architecture


# Chapter 1
# Project Specification and System Architecture

---

# 1.1 Project Objective

The objective of this project is to develop an end-to-end deep learning system capable of detecting whether an observed star hosts an exoplanet using its photometric light curve. The complete workflow shall begin with the acquisition of raw Kepler observations from NASA's Mikulski Archive for Space Telescopes (MAST) and conclude with a deployable inference service capable of predicting the presence of an exoplanet from previously unseen stellar observations.

Unlike conventional image classification problems, the input to the system consists of one-dimensional temporal brightness measurements (light curves). Therefore, the primary challenge is not feature extraction from images but learning temporal patterns corresponding to planetary transit events.

The completed system shall perform the following operations automatically:

- Download raw Kepler observations
- Extract usable light curves
- Clean and preprocess observations
- Generate machine-learning ready sequences
- Train a deep learning classifier
- Evaluate classification performance
- Deploy the trained model through an inference API

---

# 1.2 Project Scope

This project is limited to **binary exoplanet classification**.

Every observed star shall belong to one of two classes.

| Label | Description |
|--------|-------------|
| 0 | No Confirmed Exoplanet |
| 1 | Confirmed Exoplanet |

The project shall not estimate orbital parameters, planetary radius, stellar properties, or orbital periods. Those tasks require astrophysical modelling and are considered outside the scope of this implementation.

The primary objective is accurate binary classification.

---

# 1.3 Dataset Selection

After evaluating publicly available astronomical datasets, the **NASA Kepler Mission Light Curve Dataset** has been selected as the sole data source.

The Kepler mission provides the highest-quality publicly available photometric observations specifically collected for exoplanet discovery.

Alternative datasets such as TESS, ZTF and PLAsTiCC shall not be used because they either target different scientific objectives or provide shorter observation durations.

The dataset shall be downloaded directly from the **Mikulski Archive for Space Telescopes (MAST)** to ensure access to the original calibrated observations.

Dataset source

https://mast.stsci.edu

---

# 1.4 Raw Dataset Format

Kepler observations are distributed as **FITS (Flexible Image Transport System)** files.

Each FITS file contains multiple Header Data Units (HDUs) that store metadata and observational measurements.

For this project, only the **Light Curve HDU** shall be processed.

The following columns shall be extracted from every observation.

| Column | Purpose | Used |
|----------|----------|------|
| TIME | Observation timestamp | ✓ |
| PDCSAP_FLUX | Corrected stellar brightness | ✓ |
| PDCSAP_FLUX_ERR | Flux uncertainty | ✓ |
| QUALITY | Observation quality flags | ✓ |
| SAP_FLUX | Raw brightness | ✗ |
| SAP_FLUX_ERR | Raw uncertainty | ✗ |

The **PDCSAP_FLUX** column shall be used throughout the project because instrumental systematics have already been removed by the Kepler processing pipeline. Using SAP_FLUX would require additional calibration and detrending, which is unnecessary for this implementation.

---

# 1.5 Software Stack

The implementation shall be developed entirely in Python.

The following libraries have been selected.

| Library | Purpose |
|-----------|-----------|
| Astropy | Reading FITS files |
| Lightkurve | Light curve manipulation |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Visualization |
| Scikit-Learn | Dataset splitting and evaluation |
| PyTorch | Deep Learning |
| InceptionTime | Time-series classifier |
| FastAPI | Deployment |
| Streamlit | User Interface |

Each library has been selected because it is widely adopted within both the astronomical and machine learning communities.

---

# 1.6 Why Astropy and Lightkurve?

Although FITS files can technically be parsed using Astropy alone, implementing light curve extraction, normalization, stitching multiple observing quarters, and quality filtering manually would significantly increase development effort.

Instead, the following responsibilities shall be assigned.

### Astropy

- Reading FITS files
- Inspecting HDU structures
- Accessing metadata
- Low-level table extraction

### Lightkurve

- Downloading Kepler observations
- Reading light curves
- Stitching multiple quarters
- Removing NaN values
- Quality filtering
- Normalization
- Detrending
- Visualization

Lightkurve internally uses Astropy while providing astronomy-specific utilities, making it the preferred interface for this project.

---

# 1.7 Preprocessing Pipeline

Every downloaded observation shall pass through the following preprocessing pipeline before being supplied to the neural network.

```
Raw FITS Files
        │
        ▼
Read FITS using Lightkurve
        │
        ▼
Merge Observation Quarters
        │
        ▼
Remove Invalid Quality Flags
        │
        ▼
Remove Missing Flux Values
        │
        ▼
Normalize Flux
        │
        ▼
Detrend Long-Term Stellar Variability
        │
        ▼
Generate Fixed-Length Sequences
        │
        ▼
Create PyTorch Dataset
```

No preprocessing stage shall be omitted.

---

# 1.8 Model Selection

The **InceptionTime** architecture has been selected as the primary classifier.

Several architectures were evaluated before making the final decision.

| Model | Decision |
|---------|----------|
| CNN | Baseline only |
| CNN + LSTM | Rejected |
| ResNet1D | Alternative |
| Transformer | Future Work |
| InceptionTime | Selected |

InceptionTime has been selected because it has been specifically designed for one-dimensional time-series classification and consistently achieves state-of-the-art performance across benchmark datasets.

Unlike image-based convolutional neural networks, InceptionTime processes the raw light curve directly without requiring conversion into recurrence plots, spectrograms or Gramian Angular Fields.

The network employs multiple convolution kernels operating in parallel, enabling transit events of different temporal durations to be detected simultaneously.

---

# 1.9 Final Input Tensor

After preprocessing, every stellar observation shall be converted into a fixed-length tensor.

Expected tensor shape

```
Sequence Length × Features

2048 × 1
```

where

Feature

```
Normalized PDCSAP Flux
```

If auxiliary features are later incorporated,

the tensor becomes

```
2048 × N
```

where N represents the number of selected features.

---

# 1.10 Deployment Architecture

The trained model shall be exported as a PyTorch checkpoint.

Deployment shall be performed using FastAPI.

The deployment pipeline shall follow the architecture below.

```
User

↓

Upload Light Curve

↓

FastAPI

↓

Preprocessing Pipeline

↓

InceptionTime

↓

Prediction

↓

Planet / No Planet
```

A lightweight Streamlit interface shall be developed for demonstration purposes.

---

# 1.11 Project Directory

```
project/

│

├── data/

│   ├── raw/

│   ├── processed/

│

├── notebooks/

│

├── src/

│   ├── preprocessing/

│   ├── models/

│   ├── training/

│   ├── evaluation/

│   └── deployment/

│

├── checkpoints/

│

├── app/

│

└── requirements.txt
```

---

# 1.12 Expected Deliverables

At the completion of the project the following artefacts shall be available.

- Raw Kepler dataset
- Processed light curve dataset
- PyTorch Dataset class
- Trained InceptionTime model
- Saved model weights
- Evaluation report
- Confusion matrix
- ROC curve
- FastAPI inference service
- Streamlit demonstration interface

---
# Chapter 2
# Dataset Acquisition and Raw Data Preparation

---

# 2.1 Objective

The objective of this stage is to convert the raw Kepler observations into a structured dataset that can be processed by the preprocessing pipeline.

At the completion of this chapter, every observed star shall be represented as a cleaned light curve containing only valid photometric observations.

The output generated during this stage will serve as the direct input to the preprocessing pipeline described in the following chapter.

---

# 2.2 Dataset Acquisition

The project shall use the **Kepler Light Curve Collection** available from the Mikulski Archive for Space Telescopes (MAST).

Instead of manually downloading thousands of FITS files from the MAST website, the **Lightkurve** package shall be used to programmatically access the archive. Lightkurve provides a high-level interface for querying, downloading, and managing Kepler observations while internally relying on Astropy for FITS file handling.

The required libraries are:

```python
lightkurve
astropy
numpy
pandas
```

---

# 2.3 Dataset Organization

After downloading the observations, the project directory should follow the structure below.

```
project/

├── data/
│   ├── raw/
│   │   ├── fits/
│   │   └── metadata/
│   │
│   └── processed/
│
├── notebooks/
│
├── src/
│
└── checkpoints/
```

The **raw** directory shall contain only the downloaded Kepler FITS files. No modifications shall be performed on these files to preserve the original observations.

---

# 2.4 Understanding the Raw FITS Files

Each FITS file corresponds to one observing quarter for a particular star.

A single star observed over four years may therefore have multiple FITS files, each representing a different observation period.

Example:

```
Star KIC 11442793

Quarter 1
Quarter 2
Quarter 3
...
Quarter 17
```

Each quarter contains approximately ninety days of observations.

During preprocessing, these quarters shall be combined into one continuous light curve.

---

# 2.5 Reading FITS Files

Although FITS files can be opened directly using **Astropy**, the project shall primarily use **Lightkurve** because it provides astronomy-specific abstractions while preserving access to the underlying FITS data.

Internally, the following sequence is performed:

```
FITS File

↓

Astropy

↓

LightCurve Object

↓

Pandas DataFrame
```

The final preprocessing pipeline shall operate exclusively on Pandas DataFrames.

---

# 2.6 Extracting Required Columns

The LightCurve object contains numerous observational fields. Only the columns required for machine learning shall be retained.

| Column | Description | Used |
|----------|------------------------|------|
| TIME | Observation timestamp | ✓ |
| PDCSAP_FLUX | Corrected stellar brightness | ✓ |
| PDCSAP_FLUX_ERR | Measurement uncertainty | ✓ |
| QUALITY | Observation quality flag | ✓ |
| CADENCENO | Observation identifier | Optional |
| SAP_FLUX | Raw brightness | ✗ |
| SAP_FLUX_ERR | Raw uncertainty | ✗ |

The **PDCSAP_FLUX** column shall be selected because it has already undergone systematic error correction through the Kepler processing pipeline. This removes many instrumental artifacts while preserving astrophysical transit signals.

---

# 2.7 Why PDCSAP_FLUX Instead of SAP_FLUX?

Two brightness measurements are provided for every observation.

### SAP_FLUX

- Raw photometric measurements.
- Contains telescope systematics.
- Requires additional correction.
- More suitable for astrophysical calibration studies.

### PDCSAP_FLUX

- Corrected brightness measurements.
- Instrumental trends already removed.
- Standard choice for machine learning applications.
- Used throughout this implementation.

For these reasons, **PDCSAP_FLUX** shall be used as the primary feature.

---

# 2.8 Initial DataFrame

After extraction, every star shall be represented as a Pandas DataFrame similar to the example below.

| TIME | PDCSAP_FLUX | PDCSAP_FLUX_ERR | QUALITY |
|-------|-------------|-----------------|----------|
|0.00|112432.4|24.1|0|
|0.02|112418.8|23.9|0|
|0.04|112401.6|24.5|0|
|0.06|NaN|NaN|128|
|0.08|112375.1|25.4|0|

At this stage, missing values and poor-quality observations are still present.

No preprocessing has yet been performed.

---

# 2.9 Quality Flag Inspection

The **QUALITY** column records known issues associated with each observation, such as spacecraft maneuvers, cosmic ray events, reaction wheel zero crossings, or data corruption.

Observations with non-zero quality flags are generally less reliable.

During preprocessing, these entries shall either be removed or masked depending on the proportion of affected observations.

Example:

| QUALITY | Interpretation |
|----------|----------------|
|0|Valid observation|
|128|Potential anomaly|
|2048|Possible pointing issue|

Initially, the distribution of quality flags should be inspected to determine the proportion of affected measurements.

---

# 2.10 Data Verification

Before preprocessing begins, the following checks shall be performed.

### Missing Values

```
Expected

NaN values exist.
```

### Duplicate Timestamps

```
Expected

No duplicate timestamps.
```

### Time Ordering

```
Expected

Ascending order.
```

### Flux Values

```
Expected

Positive floating-point values.
```

### Quality Flags

```
Expected

Majority equal to zero.
```

---

# 2.11 Output of This Stage

At the completion of this chapter, every star should be represented by a DataFrame similar to the structure below.

```
TIME

↓

PDCSAP_FLUX

↓

PDCSAP_FLUX_ERR

↓

QUALITY
```

No normalization, interpolation, detrending, or sequence generation shall have been performed yet.

Only extraction and verification are completed during this stage.

---

# Chapter 3
# Light Curve Preprocessing Pipeline

---

# 3.1 Objective

The objective of this stage is to transform the raw Kepler observations into clean, standardized light curves suitable for deep learning.

Although the Kepler pipeline provides calibrated observations, the downloaded light curves still contain missing measurements, poor-quality observations, isolated spikes, long-term stellar variability, and inconsistent sequence lengths. These artifacts can obscure planetary transit signatures and negatively affect model performance.

A standardized preprocessing pipeline shall therefore be applied to every observed star. The output generated during this stage shall serve as the direct input to the dataset construction process described in Chapter 4.

---

# 3.2 Input

Input to this stage consists of multiple FITS files corresponding to different observation quarters for a single star.

Example

```
Star KIC 11442793

├── Quarter 1
├── Quarter 2
├── Quarter 3
...
└── Quarter 17
```

Each file contains a calibrated light curve represented as a `LightCurve` object.

---

# 3.3 Preprocessing Pipeline

Every observation shall pass through the following pipeline.

```

Raw FITS Files

        │

        ▼

Read using Lightkurve

        │

        ▼

Stitch Quarters

        │

        ▼

Remove Invalid Quality Flags

        │

        ▼

Remove Missing Values

        │

        ▼

Remove Outliers

        │

        ▼

Flatten (Detrend)

        │

        ▼

Normalize Flux

        │

        ▼

Generate Continuous Flux Array

```

Each stage depends on the successful completion of the previous stage.

---

# 3.4 Quarter Stitching

## Purpose

Kepler observations are divided into multiple observing quarters. Since the neural network must analyse a complete stellar light curve, all quarters belonging to the same star shall be merged.

## Implementation

The `LightCurveCollection.stitch()` method provided by Lightkurve shall be used.

```
Quarter 1

↓

Quarter 2

↓

Quarter 3

↓

...

↓

Quarter 17

↓

Continuous Light Curve
```

## Expected Output

Before stitching

```
17 independent LightCurve objects
```

After stitching

```
1 LightCurve object
```

## Verification

- Observation time must increase continuously.
- Duplicate timestamps should not exist.
- Flux values should remain unchanged.

---

# 3.5 Quality Filtering

## Purpose

Every Kepler observation includes a quality flag indicating whether the spacecraft experienced an event that may have corrupted the measurement.

Examples include

- spacecraft pointing errors,
- cosmic ray hits,
- reaction wheel desaturation,
- detector anomalies.

These measurements should not be supplied to the neural network.

## Implementation

Only observations with

```
QUALITY == 0
```

shall be retained.

## Example

Before

| Time | Flux | Quality |
|------|------|---------|
|0.00|112431|0|
|0.02|112425|0|
|0.04|112410|2048|
|0.06|112402|0|

After

| Time | Flux |
|------|------|
|0.00|112431|
|0.02|112425|
|0.06|112402|

## Verification

The majority of observations should remain.

If more than 20% of observations are removed, the corresponding star should be inspected manually.

---

# 3.6 Missing Value Removal

## Purpose

Some observations contain undefined flux values.

These values cannot be processed by PyTorch and would cause the training process to fail.

## Implementation

Missing observations shall be removed using

```
remove_nans()
```

provided by Lightkurve.

## Example

Before

```
112431

NaN

112410

112401
```

After

```
112431

112410

112401
```

## Verification

```
Total NaN values

Before

148

After

0
```

---

# 3.7 Outlier Removal

## Purpose

Occasionally the telescope records isolated brightness spikes caused by cosmic rays or detector artifacts.

These spikes are not astrophysical events and should therefore be removed.

## Implementation

The built-in

```
remove_outliers()

```

function shall be used.

Recommended parameter

```
sigma = 5
```

## Example

Before

```
1.001

0.999

5.421

1.000

0.998
```

After

```
1.001

0.999

1.000

0.998
```

## Verification

The resulting light curve should remain smooth while preserving genuine transit dips.

---

# 3.8 Flattening (Detrending)

## Purpose

Stars naturally exhibit long-term brightness variations due to stellar rotation, magnetic activity, and residual instrumental effects.

These slow variations may hide shallow planetary transits.

Flattening removes only the slowly varying baseline while preserving short-duration transit events.

## Implementation

The built-in

```
flatten()

```

method shall be used.

Recommended parameters

```
window_length = 401

polyorder = 2
```

## Example

Before

```
      ╱╲

     ╱  ╲

____╱    ╲____
```

After

```
────────╲____╱────────
```

The long-term trend disappears while the transit remains.

## Verification

- Baseline approximately horizontal.
- Transit dips still visible.

---

# 3.9 Flux Normalization

## Purpose

Different stars possess different intrinsic brightness levels.

Training directly on absolute flux values would cause the network to learn stellar brightness rather than transit morphology.

Each light curve shall therefore be normalized independently.

## Implementation

The built-in

```
normalize()

```

function shall be applied.

## Example

Before

```
112431

112420

112405
```

After

```
1.0003

1.0001

0.9998
```

## Verification

The normalized light curve should fluctuate around

```
Flux ≈ 1.0
```

---

# 3.10 Conversion to NumPy

The LightCurve object shall now be converted into a one-dimensional NumPy array.

Output

```
[

1.0002,

1.0001,

0.9999,

0.9986,

...

]
```

Expected Shape

```
(number_of_observations,)
```

No labels are attached at this stage.

---

# 3.11 Output

At the completion of preprocessing, each star shall contain

```
Normalized Flux Array

↓

One-dimensional NumPy Array

↓

Ready for Dataset Construction
```

No train-validation-test split has been performed yet.

No tensor conversion has been performed yet.

No labels have been attached yet.

---

# 3.12 Verification Checklist

Before proceeding to Chapter 4, the following conditions shall be satisfied.

✓ All quarters stitched

✓ QUALITY filtering completed

✓ NaN values removed

✓ Outliers removed

✓ Long-term trends removed

✓ Flux normalized

✓ NumPy arrays generated

The dataset is now ready for dataset construction and label assignment.


---
# Chapter 4
# Dataset Construction and Preparation for Deep Learning

---

# 4.1 Objective

At the completion of the preprocessing stage, each observed star has been converted into a cleaned and normalized light curve. However, these light curves are not yet suitable for supervised learning because they have not been assigned labels, partitioned into independent datasets, or converted into tensors.

The objective of this stage is to construct the final machine learning dataset by assigning labels, creating independent training, validation, and testing subsets, generating fixed-length input sequences, and converting the processed observations into PyTorch datasets.

The output produced during this stage will serve as the direct input to the InceptionTime model.

---

# 4.2 Label Assignment

The Kepler archive provides catalog information indicating whether a star contains one or more confirmed exoplanets.

Each observed star shall therefore receive one binary label.

| Label | Description |
|--------|-------------|
| 0 | No Confirmed Exoplanet |
| 1 | Confirmed Exoplanet |

Example

| Star ID | Planet Count | Assigned Label |
|----------|--------------|----------------|
| KIC 11442793 | 1 | 1 |
| KIC 8923761 | 0 | 0 |
| KIC 10358789 | 2 | 1 |

The actual number of planets shall not be predicted during this project. The classification task is limited to determining whether at least one confirmed exoplanet is present.

---

# 4.3 Dataset Structure

After labeling, every sample shall consist of

```
Star

↓

Light Curve

↓

Binary Label
```

Example

```
KIC 11442793

↓

Normalized Flux Sequence

↓

Label = 1
```

---

# 4.4 Dataset Splitting

The dataset shall be divided into three independent subsets.

| Dataset | Percentage |
|----------|------------|
| Training | 70 % |
| Validation | 15 % |
| Testing | 15 % |

The split shall be performed **before window generation**.

Only complete stars shall be assigned to each subset.

This ensures that no observations originating from the same star appear in multiple datasets.

Example

```
Training

Star A

Star B

Star C

Validation

Star D

Star E

Testing

Star F

Star G
```

This approach completely eliminates data leakage.

---

# 4.5 Window Generation

After the split has been completed, fixed-length windows shall be generated independently within each subset.

Example

```
Training

Star A

↓

Window 1

Window 2

Window 3

Window 4

Validation

Star D

↓

Window 1

Window 2

Testing

Star F

↓

Window 1

Window 2
```

Since each subset contains different stars, no overlap exists between training and testing samples.

---

# 4.6 Why Fixed-Length Windows?

The Kepler mission observed different stars for different durations.

Consequently,

```
Star A

58 000 observations

Star B

47 000 observations

Star C

63 000 observations
```

Deep learning models require every sample within a batch to possess identical dimensions.

Window generation therefore standardizes every input sequence.

Expected input

```
2048 observations
```

Expected tensor

```
2048 × 1
```

---

# 4.7 Class Imbalance

The Kepler dataset contains significantly more stars without confirmed exoplanets than stars with confirmed exoplanets.

Example

```
Planet

2 100

No Planet

11 500
```

Training directly on this distribution would bias the model toward predicting "No Planet."

To mitigate this issue, the following strategy shall be adopted.

- Compute class frequencies.
- Apply class-weighted Cross Entropy Loss during training.
- Preserve the original distribution within the validation and testing datasets.

No synthetic oversampling techniques shall be applied.

---

# 4.8 PyTorch Dataset

Each subset shall be converted into a custom PyTorch Dataset.

Every item returned by the dataset shall contain

```
Input Tensor

↓

Shape

(2048,1)

↓

Target Label

↓

0 or 1
```

The DataLoader shall automatically construct mini-batches during training.

Example

```
Batch Size

32

↓

Tensor Shape

(32,

2048,

1)

↓

Labels

(32)
```

---

# 4.9 Expected Dataset Directory

```
dataset/

│

├── train/

│   ├── X_train.npy

│   ├── y_train.npy

│

├── validation/

│   ├── X_validation.npy

│   ├── y_validation.npy

│

├── test/

│   ├── X_test.npy

│   ├── y_test.npy

│

└── metadata.csv
```

---

# 4.10 Verification

Before proceeding to model development, the following conditions shall be verified.

### Dataset Split

```
Training

70 %

Validation

15 %

Testing

15 %
```

### Data Leakage

```
Expected

None
```

### Input Shape

```
(number_of_windows,

2048,

1)
```

### Labels

```
Binary

0

1
```

### Class Distribution

```
Training

Similar to original distribution

Validation

Similar

Testing

Similar
```

---

# 4.11 Output of This Stage

At the completion of this stage, the following artefacts shall exist.

```
train_loader

validation_loader

test_loader
```

These objects will be supplied directly to the InceptionTime training pipeline.

---
# Chapter 5
# Model Architecture and Training Pipeline

---

# 5.1 Objective

At the completion of Chapter 4, the Kepler light curves have been converted into standardized PyTorch datasets suitable for supervised learning. The objective of this stage is to design, configure, and train the deep learning model responsible for identifying planetary transit signatures within the processed light curves.

The implementation shall use the **InceptionTime** architecture as the primary classifier. The model has been selected because it is specifically designed for one-dimensional time-series classification and consistently achieves state-of-the-art performance across benchmark datasets without requiring manual feature engineering.

The output of this stage shall be a trained model capable of predicting whether an unseen stellar light curve contains an exoplanet.

---

# 5.2 Model Input

Each sample entering the neural network consists of a normalized one-dimensional flux sequence generated during preprocessing.

Input Tensor

```
Batch Size × Channels × Sequence Length
```

Example

```
32 × 1 × 2048
```

where

| Dimension | Description |
|-----------|-------------|
|32|Batch Size|
|1|Flux Channel|
|2048|Number of Observations|

No handcrafted features shall be supplied to the model.

The network shall learn all discriminative representations directly from the normalized light curves.

---

# 5.3 Why InceptionTime?

Unlike conventional CNNs that analyse a signal using a single convolution kernel size, InceptionTime processes the same light curve using multiple kernel sizes simultaneously.

A planetary transit may occupy only a few observations for short-period planets, while long-period planets produce wider transit signatures. A single convolution kernel cannot efficiently detect both.

The InceptionTime architecture therefore applies several convolutions in parallel.

```
Normalized Light Curve

                │
────────────────┼──────────────────
                │
      ┌─────────┼─────────┐
      │         │         │
      ▼         ▼         ▼

Kernel 9   Kernel 19   Kernel 39

      │         │         │
      └─────────┼─────────┘
                ▼

Feature Concatenation

                ▼

Residual Block

                ▼

Next Inception Module
```

This multi-scale feature extraction enables both shallow and deep transit events to be learned simultaneously.

---

# 5.4 Network Architecture

The complete network consists of the following stages.

```
Input Tensor

↓

Inception Block

↓

Residual Connection

↓

Inception Block

↓

Residual Connection

↓

Global Average Pooling

↓

Fully Connected Layer

↓

Softmax

↓

Planet / No Planet
```

Three Inception blocks shall be implemented, each followed by residual connections to improve gradient flow during training.

---

# 5.5 Inception Module

Each Inception module consists of four parallel branches.

```
Input

│

├──────────── Conv1D(kernel=9)

│

├──────────── Conv1D(kernel=19)

│

├──────────── Conv1D(kernel=39)

│

└──────────── MaxPool → Conv1D

↓

Concatenate

↓

Batch Normalization

↓

ReLU
```

Each branch extracts temporal features at a different scale.

The concatenated output forms the input to the next residual block.

---

# 5.6 Training Configuration

The following training configuration shall be adopted.

| Parameter | Value |
|-----------|------|
|Optimizer|AdamW|
|Learning Rate|0.001|
|Batch Size|32|
|Epochs|100|
|Weight Decay|1e-4|
|Learning Rate Scheduler|Cosine Annealing|
|Early Stopping|10 Epochs|

The optimizer shall update the model parameters using only the training dataset.

The validation dataset shall be used exclusively for monitoring generalization performance.

---

# 5.7 Loss Function

The dataset contains more negative samples than positive samples.

To reduce prediction bias, weighted Cross Entropy Loss shall be employed.

```
Class 0 Weight

↓

Lower

Class 1 Weight

↓

Higher
```

The class weights shall be computed automatically from the training dataset.

---

# 5.8 Forward Pass

During every iteration, the following operations shall occur.

```
Light Curve

↓

Forward Pass

↓

Predicted Probability

↓

Cross Entropy Loss

↓

Backpropagation

↓

Parameter Update

↓

Next Batch
```

This process shall continue until all batches have been processed.

One complete traversal of the training dataset constitutes one epoch.

---

# 5.9 Model Checkpointing

After every epoch, the validation loss shall be evaluated.

If the validation loss decreases,

```
Current Model

↓

Save Checkpoint

↓

best_model.pth
```

Otherwise,

```
Continue Training
```

Only the model achieving the lowest validation loss shall be retained.

---

# 5.10 Expected Training Curves

A successful training run should produce curves similar to the following.

```
Loss

1.2 │\
    │ \
0.8 │  \
    │   \
0.4 │    \____

      Epoch
```

```
Accuracy

100│            _______

90 │         __/

80 │      __/

70 │____/

      Epoch
```

Validation accuracy should stabilize while validation loss continues decreasing.

---

# 5.11 Output of This Stage

The following files shall be produced.

```
checkpoints/

best_model.pth

training_history.csv

loss_curve.png

accuracy_curve.png
```

These artifacts will be used during model evaluation.

---
# Chapter 6
# Model Evaluation and Performance Analysis

---

# 6.1 Objective

At the completion of the training stage, the model has learned a mapping between normalized stellar light curves and binary class labels representing the presence or absence of an exoplanet. However, the training accuracy alone is insufficient to determine whether the model has generalized successfully.

The objective of this stage is to evaluate the trained InceptionTime model using the independent testing dataset, quantify its predictive performance, identify possible overfitting, and analyse the types of classification errors made by the model.

The output of this stage shall determine whether the trained model is suitable for deployment.

---

# 6.2 Evaluation Workflow

The trained model shall be evaluated using only the testing dataset.

The evaluation workflow shall follow the sequence below.

```

Load Best Model

↓

Load Test Dataset

↓

Generate Predictions

↓

Compute Metrics

↓

Generate Evaluation Plots

↓

Analyse Misclassified Samples

↓

Save Evaluation Report

```

No parameter updates shall occur during this stage.

The model shall remain in inference mode throughout the evaluation process.

---

# 6.3 Testing Dataset

The testing dataset consists exclusively of stars that were never observed during training or validation.

This separation ensures that the reported performance reflects the model's ability to generalize to previously unseen observations.

Expected Dataset

| Dataset | Purpose |
|----------|---------|
| Training | Model Learning |
| Validation | Hyperparameter Selection |
| Testing | Final Evaluation |

No overlap shall exist between these subsets.

---

# 6.4 Prediction Generation

Each light curve shall be passed through the trained InceptionTime model.

Example

```

Input

↓

Tensor

(1,1,2048)

↓

InceptionTime

↓

Probability

[0.021,0.979]

↓

Prediction

Planet Detected

```

The predicted class corresponds to the probability with the highest confidence.

---

# 6.5 Confusion Matrix

A confusion matrix shall be generated to summarize the classification performance.

Example

```

                 Predicted

                No      Planet

Actual

No Planet      2780      112

Planet          143      865

```

Definitions

True Positive

Planet correctly detected.

True Negative

No-planet star correctly identified.

False Positive

Planet predicted where none exists.

False Negative

Existing planet missed.

For this project, minimizing **False Negatives** is particularly important because missing a real exoplanet may prevent subsequent scientific investigation.

---

# 6.6 Evaluation Metrics

The following performance metrics shall be computed.

| Metric | Purpose |
|----------|----------------|
| Accuracy | Overall correctness |
| Precision | Reliability of positive predictions |
| Recall | Ability to detect actual planets |
| F1-Score | Balance between precision and recall |
| ROC-AUC | Overall classification capability |

Example

```

Accuracy

96.3%

Precision

95.4%

Recall

94.8%

F1 Score

95.1%

ROC-AUC

0.987

```

Among these metrics, **Recall** and **F1-Score** shall receive greater emphasis because they directly measure the model's ability to identify planetary systems.

---

# 6.7 Training History

The training history generated during Chapter 5 shall be analysed.

The following curves shall be plotted.

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Expected Behaviour

```

Loss

1.0 │\
    │ \
0.5 │  \____

      Epoch

```

```

Accuracy

100│         _______

95 │      __/

90 │   __/

85 │__/

      Epoch

```

Training and validation curves should converge without significant divergence.

---

# 6.8 Overfitting Analysis

The relationship between training and validation performance shall be analysed.

Indicators of overfitting include

- continuously decreasing training loss,
- increasing validation loss,
- large gap between training and validation accuracy.

Example

```

Training Accuracy

99%

Validation Accuracy

86%

```

Such behaviour indicates that the model has memorized the training observations rather than learning generalized transit patterns.

If overfitting is observed, the following corrective actions should be considered.

- Increase dropout.
- Increase weight decay.
- Reduce network complexity.
- Apply stronger augmentation.
- Reduce training epochs.

---

# 6.9 Misclassification Analysis

Incorrect predictions shall be examined individually.

For every misclassified sample,

the following information should be recorded.

| Property | Example |
|-----------|---------|
| Star ID | KIC 8923761 |
| Actual Label | Planet |
| Predicted Label | No Planet |
| Confidence | 0.62 |

The corresponding light curves shall also be visualized.

This analysis helps determine whether the error originates from

- shallow transit signals,
- excessive stellar variability,
- residual preprocessing artifacts,
- poor signal-to-noise ratio.

---

# 6.10 Model Acceptance Criteria

The trained model shall be accepted only if the following conditions are satisfied.

| Requirement | Target |
|--------------|--------|
| Accuracy | >95% |
| Precision | >94% |
| Recall | >94% |
| F1 Score | >94% |
| ROC-AUC | >0.98 |

These values represent the expected performance goals for the implementation.

Failure to achieve these thresholds shall require further refinement of preprocessing, hyperparameter tuning, or architectural modifications.

---

# 6.11 Output of This Stage

The evaluation stage shall generate the following files.

```

evaluation/

│

├── confusion_matrix.png

├── roc_curve.png

├── precision_recall_curve.png

├── loss_curve.png

├── accuracy_curve.png

└── evaluation_report.csv

```

These artifacts shall be included in the final project report and used to justify the deployment of the trained model.

---

# Chapter 7
# Model Publication and Deployment

---

# 7.1 Objective

After successful training and evaluation, the best-performing InceptionTime model shall be published so that external applications can perform exoplanet predictions without requiring model retraining.

The web application shall communicate with the deployed model through a REST API.

---

# 7.2 Model Export

The model achieving the highest validation performance shall be exported.

Expected output

```

checkpoints/

best_model.pth

```

Along with the model, the following files shall also be saved.

```

config.yaml

class_mapping.json

preprocessing.pkl

requirements.txt

README.md

```

The preprocessing configuration must remain identical to that used during training to ensure consistent predictions.

---

# 7.3 Hugging Face Repository Structure

The following repository structure shall be created.

```

exoplanet-inceptiontime/

│

├── README.md

├── app.py

├── model.py

├── inference.py

├── preprocessing.py

├── requirements.txt

├── config.yaml

├── best_model.pth

└── class_mapping.json

```

---

# 7.4 Inference Pipeline

Every prediction request shall follow the same preprocessing pipeline used during training.

```

User Uploads FITS

↓

Read using Lightkurve

↓

Quarter Stitching

↓

Quality Filtering

↓

Remove NaN Values

↓

Outlier Removal

↓

Flatten

↓

Normalize

↓

Tensor Generation

↓

Load InceptionTime

↓

Prediction

↓

Planet / No Planet

```

No preprocessing step shall differ from the training pipeline.

---

# 7.5 Hugging Face Deployment

The trained model shall be deployed using **Hugging Face Spaces** together with a lightweight FastAPI application.

The deployment pipeline shall be

```

Browser

↓

Frontend Team

↓

REST Request

↓

Hugging Face Endpoint

↓

Inference Pipeline

↓

Prediction

↓

JSON Response

```

The frontend shall never communicate directly with the PyTorch model.

All inference shall occur on the Hugging Face server.

---

# 7.6 API Request

The frontend shall send a POST request.

Input

```

FITS File

```

or

```

CSV Light Curve

```

depending on the final frontend implementation.

---

# 7.7 API Response

The inference server shall return

```json
{
    "star_id": "KIC11442793",
    "prediction": "Planet Detected",
    "confidence": 0.984,
    "probabilities": {
        "planet": 0.984,
        "no_planet": 0.016
    }
}
```

If preprocessing fails,

```json
{
    "status":"error",
    "message":"Invalid FITS file."
}
```

shall be returned.

---

# 7.8 Model Versioning

Every trained model shall receive a version number.

Example

```

v1.0

Initial Release

↓

v1.1

Improved Preprocessing

↓

v1.2

Hyperparameter Optimization

```

Only the latest stable model shall be used for production inference.

---

# 7.9 Final Deployment Architecture

```

                Frontend

                    │

                    ▼

             REST API Request

                    │

                    ▼

         Hugging Face Space

                    │

                    ▼

        FastAPI Inference Server

                    │

                    ▼

      Lightkurve Preprocessing

                    │

                    ▼

          InceptionTime Model

                    │

                    ▼

          Prediction JSON

                    │

                    ▼

              Frontend UI

```

---

# 7.10 Deliverables

At the completion of deployment, the following shall be available.

- Trained InceptionTime model
- Hugging Face repository
- Hugging Face inference endpoint
- REST API
- Model documentation
- API documentation
- Example prediction requests
- Example prediction responses



