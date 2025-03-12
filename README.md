# Octal/Decimal Prediction API

This is an API built with FastAPI that predicts whether a text input corresponds to converting from octal to decimal or vice versa.

## How it Works

The API uses a TensorFlow model to predict whether a given text input is a request to convert from octal to decimal or from decimal to octal. It extracts the numerical value from the input text and then performs the corresponding conversion, providing a detailed step-by-step explanation.

## How to Use

1.  Clone the repository.
2.  Install the dependencies with `pip install -r requirements.txt`.
3.  Run the API with `uvicorn main:app --reload`.
4.  Access the API at `http://127.0.0.1:8000`.

## Example Usage

Send a POST request to `/predict` with a JSON payload containing the text to be analyzed:

json
{
  "text": "Convierte 123 a decimal"
}

## Response Example

{
  "input_number": "123",
  "conversion_type": "Octal to Decimal",
  "result": 83,
  "steps": [
    {
      "Explicación": "Multiplica 3 por 8^0:",
      "Operación": "3 * 8^0 = 3"
    },
    {
      "Explicación": "Multiplica 2 por 8^1:",
      "Operación": "2 * 8^1 = 16"
    },
    {
      "Explicación": "Multiplica 1 por 8^2:",
      "Operación": "1 * 8^2 = 64"
    },
    {
      "Explicación": "Suma todos los resultados:",
      "Operación": "3 + 16 + 64 = 83"
    }
  ],
  "prediction": 0.95
}

