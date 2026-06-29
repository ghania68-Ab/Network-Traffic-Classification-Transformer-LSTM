```markdown
# Network Traffic Classification Using Transformer-Enhanced LSTM Models

## Project Description

This project focuses on network traffic classification using deep learning models. The main objective is to compare a Simple LSTM baseline model with a Transformer-Enhanced LSTM model and evaluate whether the addition of an attention mechanism improves classification performance.

CIC-Darknet2020 is used as the main dataset, while UNSW-NB15 is used as a comparison dataset. Both models are trained and evaluated using the same preprocessing pipeline, dataset split, class weighting strategy, training callbacks, and evaluation metrics to ensure a fair comparison.

The project is designed as a university research project and includes preprocessing, model training, evaluation, visualization, and final performance comparison.

## Dataset Links

- CIC-Darknet2020: [https://www.kaggle.com/datasets/dhoogla/cicdarknet2020]
- UNSW-NB15: [https://www.kaggle.com/datasets/dhoogla/unswnb15]


Expected dataset files:

```text
dataset/cicdarknet2020.parquet
dataset/UNSW_NB15_training-set.parquet
dataset/UNSW_NB15_testing-set.parquet
```

## Installation Steps

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The project uses Python with TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib, and Jupyter Notebook.

## How To Run The Code

### Run From Terminal

Train and evaluate the models on CIC-Darknet2020:

```bash
python src/train.py --dataset cic --epochs 20 --batch-size 256
```

Train and evaluate the models on UNSW-NB15:

```bash
python src/train.py --dataset unsw --epochs 20 --batch-size 256
```

Train and evaluate the models on both datasets:

```bash
python src/train.py --all-datasets --epochs 20 --batch-size 256
```

### Run From Notebook

Open the experiment notebook:

```bash
jupyter notebook notebooks/experiment.ipynb
```

Inside the notebook, select the dataset mode:

```python
RUN_MODE = "cic"   # options: "cic", "unsw", "all"
EPOCHS = 20
BATCH_SIZE = 256
```

Then run all notebook cells from top to bottom. The notebook performs preprocessing, training, evaluation, metric display, and graph visualization.

## Model Details

### Simple LSTM Baseline

The Simple LSTM model is used as the baseline model. It is designed to provide a fair standard comparison against the proposed model.

The architecture includes:

- Input layer
- LSTM layer
- Dropout layer
- Dense layer
- Softmax output layer

The LSTM layer learns sequential patterns from the reshaped network traffic features. Dropout is used to reduce overfitting, and the final softmax layer performs multi-class classification.

### Transformer-Enhanced LSTM

The Transformer-Enhanced LSTM is the proposed model. It combines the sequence-learning ability of LSTM with the feature-interaction learning ability of Transformer-style attention.

The architecture includes:

- Input layer
- LSTM sequence encoder
- MultiHeadAttention layer
- Residual connections
- Layer normalization
- Dropout layers
- Dense classification layers
- Softmax output layer

The LSTM layer first converts the preprocessed traffic features into a sequence representation. The MultiHeadAttention layer then learns relationships between different feature positions. Residual connections and layer normalization help stabilize training, while dropout helps reduce overfitting. The final dense and softmax layers classify the network traffic into the target classes.

Both models are trained using Adam optimizer, validation monitoring, EarlyStopping, and ReduceLROnPlateau for fair evaluation.

## Results Summary

The models are evaluated using accuracy, precision, recall, F1-score, classification report, and confusion matrix. Since network traffic datasets can be imbalanced, both weighted F1-score and macro F1-score are useful for understanding model performance.

The generated results are saved in the `results/` folder, and the training graphs are saved in the `figures/` folder.

Main output files include:

```text
results/metrics.txt
results/confusion_matrix.png
figures/accuracy_curve.png
figures/loss_curve.png
figures/comparison_model_graph.png
```

Dataset-specific metric and figure files are also generated when CIC-Darknet2020 or UNSW-NB15 is trained separately. The final comparison shows how the Simple LSTM baseline performs against the Transformer-Enhanced LSTM model under the same experimental conditions.

## Team Members

- Ghania Jawed (62745)
- Samia Shahzad (64248)
- Laraib Ali (65132)
```