# nano-gpt-copy

This project implements a Bigram Language Model using PyTorch. The model can be trained on text data and used to generate text based on learned patterns.

## Features
- Train a bigram-based language model on a given text dataset.
- Generate text using a trained model.
- Command-line interface for easy usage.

## Installation
Ensure you have Python and the required dependencies installed:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run:
```bash
python src/main.py --mode train --model_path models/bigram_model.pth --max_iters 5000
```
- `--mode train` : Specifies training mode.
- `--model_path` : Path to save or load the model.
- `--max_iters` : Number of training iterations.

### Generating Text
To generate text using a trained model, run:
```bash
python src/main.py --mode generate --model_path models/bigram_model.pth
```
- `--mode generate` : Specifies text generation mode.
- `--model_path` : Path to the trained model.

## Project Structure
```
.
├── .venv/                 # Virtual environment (optional)
├── models/                # Directory to store trained models
├── src/                   # Source code directory
│   ├── main.py            # Main script for training and generating text
│   ├── bigram_model.py    # Bigram language model implementation
│   ├── train.py           # Training function
├── input.txt              # Input text file for training 
├── LICENSE                # Project license
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
```

## Input Data
The input text file is based on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
 dataset:

## Acknowledgment
This project is inspired by and based on the work of [Andrej Karpathy](https://karpathy.ai/). A huge appreciation for his contributions to deep learning and open-source education! 🙌

