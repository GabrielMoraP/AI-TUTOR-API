from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load the model and tokenizer
model = tf.keras.models.load_model('model.keras')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Maximum sequence length
max_len = 20

# Create the FastAPI application
app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data model for input
class TextInput(BaseModel):
    text: str

# Function to convert octal to decimal with steps
def octal_to_decimal_with_steps(octal):
    if not re.match(r'^[0-7]+$', octal):
        raise ValueError("El número octal debe contener solo dígitos del 0 al 7.")

    decimal = 0
    steps = []
    results = []
    for i, digit in enumerate(reversed(octal)):
        step_value = int(digit) * (8 ** i)
        step_explanation = f"Multiplica {digit} por 8^{i}:"
        step_operation = f"{digit} * 8^{i} = {step_value}"
        steps.append({"Explicación": step_explanation, "Operación": step_operation})
        decimal += step_value
        results.append(step_value)

    sum_explanation = "Suma todos los resultados:"
    sum_operation = " + ".join(map(str, results)) + f" = {decimal}"
    steps.append({"Explicación": sum_explanation, "Operación": sum_operation})
    return decimal, steps

# Function to convert decimal to octal with steps
def decimal_to_octal_with_steps(decimal):
    octal = ""
    steps = []
    remainders = []
    while decimal > 0:
        remainder = decimal % 8
        step_explanation = f"Divide {decimal} entre 8:"
        step_operation = f"{decimal} / 8 = {decimal // 8} <- Próximo Dividendo | Residuo = {remainder}"
        steps.append({"Explicación": step_explanation, "Operación": step_operation})
        octal = str(remainder) + octal
        decimal = decimal // 8
        remainders.append(str(remainder))

    join_explanation = "Junta los digitos del residuo en orden inverso:"
    join_operation = "".join(reversed(remainders))
    steps.append({"Explicación": join_explanation, "Operación": join_operation})
    return octal, steps

# Function to extract the number and make the prediction
def parse_input(input_text):
    number_match = re.search(r'\d+', input_text)
    if not number_match:
        raise ValueError("Proporciona un valor númerico, por favor.")

    number = number_match.group()
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_padded = pad_sequences(input_sequence, maxlen=max_len, padding='post')
    prediction = model.predict(input_padded)[0][0]
    return number, prediction

# Prediction route
@app.post('/predict')
def predict(input_data: TextInput):
    input_text = input_data.text

    try:
        number, prediction = parse_input(input_text)

        if prediction > 0.5:
            result, steps = octal_to_decimal_with_steps(number)
            conversion_type = "Octal a Decimal"
        else:
            result, steps = decimal_to_octal_with_steps(int(number))
            conversion_type = "Decimal a Octal"

        return {
            "input_number": number,
            "conversion_type": conversion_type,
            "result": result,
            "steps": steps,
            "prediction": float(prediction)
        }

    except:
        raise HTTPException(status_code=500, detail="Intentelo Nuevamente.")