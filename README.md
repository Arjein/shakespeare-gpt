# nano-gpt-copy

This project implements a Bigram-based Language Model using PyTorch. The model is trained on the **TinyShakespeare** dataset and can be used for both training and text generation.

## Overview

The model uses a transformer-based architecture with the following components:
- **Embedding layers** for tokens and positional encodings
- **Multi-Head Attention** for context capture
- **Feedforward layers** for further processing
- **LayerNorm** for stabilization
- **Cross-entropy loss** for training optimization

The trained model can generate text based on a given input, predicting the next token in a sequence using the learned patterns from the TinyShakespeare dataset.

## Requirements

- Python 3.12
- PyTorch (install with `pip install torch`)
- Command-line tools to run scripts

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

- `--mode train`: Specifies training mode.
- `--model_path`: Path to save or load the model.
- `--max_iters`: Number of training iterations.

### Generating Text
To generate text using a trained model, run:

```bash
python src/main.py --mode generate --model_path models/bigram_model.pth --max_new_tokens 1000
```

- `--mode generate`: Specifies text generation mode.
- `--model_path`: Path to the trained model.
- `--max_new_tokens`: Number of characters to be generated.

## Sample Output
Here is an example of generated text using the trained Bigram model after running the `--mode generate` command:

**sample_output.txt**
```

Have you with the leity depase of you born.

First Servingman:
I go content that villain I say thee.

PETRUCHIO:
Trust! a sweet lady?

THUMIO:
Shall I not stand bear my arm: but I had respected
that shall bear in the mercy speed three
that I smother at think.

LEONTES:
How love?

LEONTES:

BENVOLA:
Lurd?

FLORIO:
No, sir:
My mistress aboutding before
That prock-locks he first,
As is in the sun yalk off; as the fount
Advanced, as so belonged; her sistays for the lork.

EMILLIO:
Why, I do swear to pity.

LEONTES:
Is it sup?

Pass:
Good Juliet.

DUKE VINCENTIO:
Sir, madam; good-morrow is hither?

PAULINA:
The case is the worse to tell; I would there
More than leather.

LEONTES:
When I have my loved to visit him,
I dare to marry our children yiells? For relish?

PAULINA:
It should safehed for Bupiils Surrey, say, it than it.

LADY CAPULE:
No, sir, I should no, sirrce weep.

PETER:
How, my brother lady lace to make sad my love and
That seel methinks to sing ovy mind, see'th honourable,
So k

```

## Model Architecture

The Bigram Language Model consists of:
- **Token Embedding Layer**: Maps characters to vectors.
- **Positional Embedding Layer**: Adds positional information to the embeddings.
- **Multiple Blocks**: Each block contains multi-head attention and feed-forward layers.
- **Linear Layer**: Outputs the predicted probabilities for each character in the vocabulary.

### Training Loss
The model uses **Cross-Entropy Loss** for training, which is calculated after flattening the logits and targets.

### Generation
During text generation, the model predicts one token at a time using softmax sampling. The `generate` function generates text by repeatedly predicting the next token and appending it to the input sequence.

## Hyperparameters

- `BLOCK_SIZE = 256`
- `n_embd = 384`
- `dropout = 0.2`

These hyperparameters control the model's architecture and training dynamics. Feel free to experiment with different values to explore model performance.

## Acknowledgment

This project is inspired by and based on the work of [Andrej Karpathy](https://karpathy.ai/). A huge appreciation for his contributions to deep learning and open-source education! 🙌

## License

This project is open-source and available under the MIT License.
"""