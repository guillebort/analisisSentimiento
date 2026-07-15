from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# 1. Definimos la estructura de los datos que debe enviar el cliente
class TextRequest(BaseModel):
    text: str

# 2. Inicializamos la aplicación
app = FastAPI(
    title="API de Análisis de Sentimiento", 
    description="Procesa texto usando un modelo BERT de Hugging Face"
)

# 3. Cargamos el modelo globalmente
#cargamos el modelo en la RAM al arrancar el servidor.

print("Cargando modelo de IA...")
nlp_classifier = pipeline("sentiment-analysis")
print("Modelo listo.")

# 4. Creamos el endpoint que recibe los datos
@app.post("/predict")
async def analyze_text(request: TextRequest):
    # Validar que no nos envíen texto en blanco
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío.")
    
    # Procesar el texto con el modelo
    # El modelo devuelve una lista, ej: [{'label': 'POSITIVE', 'score': 0.9998}]
    result = nlp_classifier(request.text)[0]
    
    # Construir y devolver la respuesta (FastAPI la convierte automáticamente a JSON)
    return {
        "input_text": request.text,
        "prediction": result["label"],
        "confidence_score": round(result["score"], 4)
    }