from PySide6.QtWidgets import QApplication, QVBoxLayout, QLabel, QPushButton, QTextEdit, QWidget
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Load the model
model = tf.keras.models.load_model('LLM.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_length = 15  # Change this to the actual value used during training

def generate_text(seed_text, next_words=100):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    output_text = seed_text
    
    for _ in range(next_words):
        token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
        predicted_probabilities = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probabilities)
        output_word = tokenizer.index_word.get(predicted_index, "<UNK>")
        output_text += " " + output_word
        token_list = token_list[0].tolist()
        token_list.append(predicted_index)
        token_list = token_list[1:]
    
    return output_text

# Create PySide6 GUI
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Text Generator")
        
        # Create layout
        layout = QVBoxLayout()
        
        # Text input
        self.input_field = QTextEdit()
        layout.addWidget(self.input_field)
        
        # Button
        self.button = QPushButton("Generate Text")
        self.button.clicked.connect(self.on_generate)
        layout.addWidget(self.button)
        
        # Text output
        self.output_label = QLabel("Generated text will appear here")
        layout.addWidget(self.output_label)
        
        self.setLayout(layout)
    
    def on_generate(self):
        seed_text = self.input_field.toPlainText()
        generated_text = generate_text(seed_text, next_words=15)
        self.output_label.setText(generated_text)

# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
