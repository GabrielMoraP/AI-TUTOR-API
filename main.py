from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re  # Importar el módulo de expresiones regulares

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

# Función para convertir de octal a decimal
def octal_to_decimal(octal):
    decimal = 0
    steps = []
    for i, digit in enumerate(reversed(octal)):
        step_value = int(digit) * (8 ** i)
        step_explanation = (
            f"Paso {i + 1}: Tomar el dígito {digit} en la posición {i} (de derecha a izquierda), "
            f"multiplicarlo por 8^{i} (que es {8 ** i}) y sumarlo al resultado acumulado. "
            f"{digit} * 8^{i} = {step_value}"
        )
        steps.append(step_explanation)
        decimal += step_value
    return decimal, steps

# Función para convertir de decimal a octal
def decimal_to_octal(decimal):
    octal = ""
    steps = []
    step_count = 1
    while decimal > 0:
        remainder = decimal % 8
        step_explanation = (
            f"Paso {step_count}: Dividir {decimal} entre 8. "
            f"El cociente es {decimal // 8} y el residuo es {remainder}. "
            f"El residuo {remainder} es el siguiente dígito octal (de derecha a izquierda)."
        )
        steps.append(step_explanation)
        octal = str(remainder) + octal
        decimal = decimal // 8
        step_count += 1
    return octal, steps

# Función para extraer el número y el tipo de conversión
def parse_input(input_text):
    # Buscar un número en la cadena de texto
    number_match = re.search(r'\d+', input_text)
    if not number_match:
        raise ValueError("No se encontró un número en la entrada.")

    number = number_match.group()

    # Determinar el tipo de conversión
    if "decimal a octal" in input_text.lower():
        conversion_type = "decimal_to_octal"
    elif "octal a decimal" in input_text.lower():
        conversion_type = "octal_to_decimal"
    else:
        # Si no se especifica, usar la predicción del modelo
        input_sequence = tokenizer.texts_to_sequences([input_text])
        input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post')
        prediction = model.predict(input_padded)[0][0]  # Obtener el valor real de la predicción
        conversion_type = "octal_to_decimal" if prediction > 0.5 else "decimal_to_octal"

    return number, conversion_type, float(prediction) if "prediction" in locals() else None

# Ruta para la predicción
@app.post('/predict')
def predict(input_data: TextInput):
    # Obtener el texto de entrada
    input_text = input_data.text

    try:
        # Extraer el número, el tipo de conversión y la predicción (si aplica)
        number, conversion_type, prediction = parse_input(input_text)

        # Realizar la conversión correspondiente
        if conversion_type == "octal_to_decimal":
            decimal, steps = octal_to_decimal(number)
            response = {
                "tipo_conversion": "Conversión Octal a Decimal",
                "pasos": steps,
                "resultado": decimal,
                "prediccion": prediction if prediction is not None else None
            }
        elif conversion_type == "decimal_to_octal":
            octal, steps = decimal_to_octal(int(number))
            response = {
                "tipo_conversion": "Conversión Decimal a Octal",
                "pasos": steps,
                "resultado": octal,
                "prediccion": prediction if prediction is not None else None
            }
        else:
            response = {
                "error": "Tipo de conversión no válido."
            }
    except ValueError as e:
        response = {
            "error": str(e)
        }

    # Devolver la respuesta en formato JSON
    return {
        "input_text": input_text,
        "response": response
    }