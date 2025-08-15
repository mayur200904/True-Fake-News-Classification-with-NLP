### True & Fake News Classification with NLP

---

### Introduction
This project implements a Natural Language Processing (NLP) model to classify news articles as either **true** or **fake**. It leverages a deep learning approach, combining advanced word embeddings with a sophisticated recurrent neural network to achieve high accuracy in distinguishing between authentic and fabricated content.

---

### Features
* **Data Preprocessing:** Cleans and prepares textual data for machine learning.
* **Word Embeddings:** Utilizes pre-trained **GloVe** (Global Vectors for Word Representation) to convert words into dense vector representations, capturing semantic relationships.
* **Deep Learning Model:** Employs a **Long Short-Term Memory (LSTM)** network, a type of recurrent neural network (RNN) well-suited for processing sequential data like text.
* **Performance Evaluation:** Generates a comprehensive classification report and a visual confusion matrix to evaluate model performance.

---

### Technologies Used
* **Python 3.x**
* **TensorFlow** / **Keras**
* **NLTK** (Natural Language Toolkit)
* **GloVe** (pre-trained word vectors)
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **scikit-learn**

---

### Installation
This project is designed to be run in a **Jupyter Notebook** or **Google Colaboratory** environment. Follow these steps to set up the project:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```
2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy tensorflow scikit-learn nltk matplotlib seaborn
    ```
3.  **Download GloVe Embeddings:**
    The notebook requires pre-trained GloVe embeddings. You will need to download the `glove.6B.zip` file (available online) and place the `glove.6B.100d.txt` file in your project directory.

---

### Usage
To run the project, open the `True__Fake_News_NLP_GloVe__LSTM-2.ipynb.ipynb` file in a Jupyter environment and execute the cells sequentially. The notebook will guide you through:
1.  Loading and exploring the dataset.
2.  Preprocessing the text data.
3.  Loading the GloVe embeddings.
4.  Building and training the LSTM model.
5.  Evaluating the model's performance.

---

### Results
Upon execution, the notebook will output a **classification report** detailing the model's precision, recall, and F1-score. It will also display a **confusion matrix**  which provides a visual summary of the model's predictions, showing the number of correctly and incorrectly classified articles for both the 'Fake' and 'Original' classes.
