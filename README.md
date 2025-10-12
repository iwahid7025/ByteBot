# ByteBot

Deep learning chatbot using Python and TensorFlow, capable of advanced natural language processing.

## Overview

ByteBot is an intelligent conversational AI that uses neural networks to understand user intents and provide relevant responses. It leverages TensorFlow and tflearn for deep learning, and NLTK for natural language processing.

## Features

- **Intent Recognition**: Identifies user intentions using a trained neural network
- **Natural Language Processing**: Uses NLTK for tokenization and word stemming
- **Bag of Words Model**: Converts user input into numerical representations
- **Persistent Training**: Saves preprocessed data and trained models for faster startup
- **Easy Customization**: Modify intents and responses via JSON configuration

## Requirements

- Python 3.x
- nltk
- numpy
- tflearn
- tensorflow
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ByteBot.git
cd ByteBot
```

2. Install dependencies:
```bash
pip install nltk numpy tflearn tensorflow scipy
```

3. Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt')"
```

## Usage

Run the chatbot:
```bash
cd ByteBot
python main.py
```

Type your messages and press Enter. Type `quit` to exit.

## Configuration

Edit `intents.json` to customize the chatbot's responses. The file should contain:
- `tag`: Intent category name
- `patterns`: Example user inputs
- `responses`: Possible bot responses

## How It Works

1. **Data Preprocessing**: Tokenizes and stems words from intent patterns
2. **Training**: Creates bag-of-words representations and trains a neural network
3. **Prediction**: Converts user input to bag-of-words and predicts the intent
4. **Response**: Randomly selects a response from the matched intent category

## Project Structure

```
ByteBot/
├── ByteBot/
│   ├── main.py          # Main chatbot application
│   └── intents.json     # Intent definitions and responses
└── README.md            # This file
```

## License

This project is open source and available for educational purposes.
