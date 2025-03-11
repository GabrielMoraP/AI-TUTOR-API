from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Importar CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el modelo y el tokenizer
model = tf.keras.models.load_model('model.keras')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Definir la longitud máxima de las secuencias (debe ser la misma que se usó durante el entrenamiento)
max_len = 20

# Crear la aplicación FastAPI
app = FastAPI()

# Configurar CORS para permitir cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir solicitudes desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

# Modelo de datos para la entrada
class TextInput(BaseModel):
    text: str

# Ruta para la predicción
@app.post('/predict')
def predict(input_data: TextInput):
    # Obtener el texto de entrada
    input_text = input_data.text

    # Tokenizar y aplicar padding a la entrada de texto
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post')

    # Hacer la predicción
    prediction = model.predict(input_padded)
    prediction = (prediction > 0.5).astype(int)

    # Convertir la predicción a un texto legible
    if prediction[0][0] == 1:
        response_text = "La entrada corresponde a convertir de octal a decimal."
    else:
        response_text = "La entrada corresponde a convertir de decimal a octal."

    # Devolver la respuesta en formato JSON
    return {
        "input_text": input_text,
        "prediction": int(prediction[0][0]),  # Corregí el typo aquí (prediction en lugar de prediction)
        "response": response_text
    }