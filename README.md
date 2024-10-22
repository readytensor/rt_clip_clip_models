# CLIP from Scratch: A PyTorch Implementation

This repository contains an implementation of the **Contrastive Language-Image Pretraining (CLIP)** model from scratch using PyTorch. The goal of CLIP is to bridge the gap between images and natural language descriptions by learning a shared embedding space where images and corresponding text are aligned. This project follows the original architecture proposed by OpenAI, with a few modifications to improve clarity and ease of understanding.

## Project Structure

- `clip.py`: The core implementation of the CLIP model, including the image and text encoders, attention mechanism, and contrastive learning.
- `train.ipynb`: A Jupyter notebook that includes the pipeline of data loading, model training and saving the weights.
- `predict.ipynb`: A notebook that demonstrates how to load the pre-trained model and use it with an example of image-caption pairing.

## Model Overview

CLIP consists of two main components:
1. **Image Encoder**: This can be based on either ResNet or Vision Transformers (ViT). In this implementation, the Vision Transformer architecture is used, where images are split into patches, and self-attention is applied to learn meaningful features.
2. **Text Encoder**: A transformer-based encoder such as GPT or BERT is used to encode the input text into a shared embedding space. The encoded representations of images and text are then aligned using a contrastive loss.

## Key Features
- **Patch-based Image Encoding**: The image encoder splits images into patches and encodes each patch individually, treating it as a token similar to words in a sentence.
- **Contrastive Loss**: A contrastive learning approach is used to align image and text embeddings in a common space, maximizing the similarity between matching pairs and minimizing the similarity between non-matching pairs.
- **Multi-Head Self-Attention**: Both the image and text encoders use self-attention to capture dependencies and relationships across different patches and tokens.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/clip-from-scratch.git
   cd rt_clip_clip_models
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks:
   Open `train.ipynb` and `predict.ipynb` to explore the full implementation and see how the model can be fine-tuned or trained on a dataset.


## Dataset

This implementation uses the `Attila1011/img_caption_EN_AppleFlair_Blip` dataset, which consists of image-text pairs. The dataset is split into training and test sets.


## Acknowledgments

This implementation is inspired by the original CLIP paper by OpenAI and follows a similar architecture for educational purposes.
