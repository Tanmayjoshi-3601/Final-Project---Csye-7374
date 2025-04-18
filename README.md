# Medical Image to Text Report Generation

This repository contains the implementation of various deep learning approaches for generating textual reports from chest X-ray images. The project was developed as the final assignment for the Applied Deep Learning for Healthcare (CSYE-7374) course.

## Project Overview

The goal of this project is to build deep learning models that can automatically generate descriptive text reports from chest X-ray images. This technology has practical applications in assisting radiologists and improving healthcare workflow efficiency.

We implement and compare four different architectures:

1. **Vision Transformer + Transformer Decoder**: Using a ViT model for feature extraction and a transformer decoder for text generation.
2. **ResNet + Transformer with Memory**: Using a ResNet backbone with a transformer architecture that has memory components.
3. **Autoencoder + Transformer Decoder**: A custom approach with a two-stage training process that first trains an autoencoder for feature extraction.
4. **GAN-Inspired Autoencoder + Transformer**: An approach inspired by stable diffusion techniques using GAN principles.

## Dataset

We use the Indiana University Chest X-ray Collection (IU X-Ray dataset), which contains chest X-ray images paired with their corresponding radiological reports.

## Requirements

### Prerequisites
- Conda (Miniconda or Anaconda)
- Git
- Internet connection to download the dataset
- CUDA-compatible GPU (recommended)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Tanmayjoshi-3601/Final-Project---Csye-7374.git
cd Final-Project---Csye-7374
```

2. Create and activate a Conda environment:
```bash
conda create -n xray-report python=3.10
conda activate xray-report
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Dataset Download and Extraction

1. Download the dataset using Kaggle API:
```bash
curl -L -o ~/Downloads/chest-xrays-indiana-university.zip \
  https://www.kaggle.com/api/v1/datasets/download/raddar/chest-xrays-indiana-university
```

2. Extract the downloaded zip file:
```bash
unzip ~/Downloads/chest-xrays-indiana-university.zip -d ./data
```

> **Note**: If you have trouble with the Kaggle API, you can download the dataset directly from the [Kaggle website](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) and extract it to the data folder.

## Project Structure

```
Final-Project---Csye-7374/
├── data/                          # Dataset folder (created after extraction)
├── data-extract.ipynb             # Data preprocessing and EDA notebook
├── experiments/                   # Experiment implementations
│   ├── Vision-transformer/        # Experiment 1: ViT + Transformer Decoder
│   ├── Resnet-transformer/        # Experiment 2: ResNet + Transformer
│   ├── autoencoder-transformer/   # Experiment 3: Autoencoder + Transformer
│   └── stable-diffusion/          # Experiment 4: GAN-inspired approach
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

## Running the Experiments

### Data Preparation (Optional)

The preprocessed data files are already included in each experiment folder, but if you want to run the data preprocessing yourself:

```bash
jupyter notebook data-extract.ipynb
```

This will:
- Perform exploratory data analysis on the X-ray dataset
- Preprocess the images and captions
- Create the final dataset file used by the experiments

### Running Individual Experiments

Each experiment is contained in its own folder with a Jupyter notebook:

#### Experiment 1: Vision Transformer + Transformer Decoder
```bash
cd experiments/Vision-transformer
jupyter notebook Vision-transformer.ipynb
```

#### Experiment 2: ResNet + Transformer with Memory
```bash
cd experiments/Resnet-transformer
jupyter notebook Resnet-transformer.ipynb
```

#### Experiment 3: Autoencoder + Transformer Decoder
```bash
cd experiments/autoencoder-transformer
jupyter notebook autoencoder-transformer.ipynb
```

#### Experiment 4: GAN-Inspired Approach
```bash
cd experiments/stable-diffusion
jupyter notebook stable-diffusion-approach.ipynb
```

## Results

Each experiment produces:
- Training and validation loss curves
- BLEU score metrics
- Example generated captions with their corresponding X-ray images
- Detailed performance metrics saved in the results directory

Results from our experiments show varying performance across different architectures:

- The Vision Transformer + Transformer Decoder approach achieves competitive BLEU-4 scores around 0.44
- The ResNet + Transformer architecture demonstrates strong performance with stable training dynamics
- The Autoencoder + Transformer approach provides interesting latent space representations
- The GAN-inspired approach, while conceptually advanced, was more challenging to train effectively

## Computational Requirements

- Training these models is computationally intensive and is recommended to be done on a machine with a CUDA-compatible GPU
- Approximate training times (on NVIDIA A100 GPU):
  - Vision Transformer: ~3-4 hours
  - ResNet + Transformer: ~2-3 hours
  - Autoencoder + Transformer: ~4-5 hours
  - GAN-Inspired Approach: ~8-10 hours

## Acknowledgements

- Indiana University for providing the chest X-ray dataset
- The course instructors and teaching assistants for their guidance

## License

This project is for educational purposes. The dataset has its own license from Indiana University.

## Authors

- Tanmay Joshi

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. In Advances in Neural Information Processing Systems.
2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
4. Rombach, R., et al. (2022). High-Resolution Image Synthesis With Latent Diffusion Models.
