from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pyttsx3
import sqlite3
import random
import os
import json
import datetime
from duckduckgo_search import DDGS
import requests

# Inicialización de la aplicación Flask
app = Flask(__name__)

# Configuración del modelo de IA
MODELO_BASE = "modelo_kazu_v2"
tokenizer = AutoTokenizer.from_pretrained(MODELO_BASE)
modelo = AutoModelForCausalLM.from_pretrained(MODELO_BASE)
modelo_ia = pipeline("text-generation", model=modelo, tokenizer=tokenizer, device=0)

# Voz
voz = pyttsx3.init()
voz.setProperty("rate", 170)
voz.setProperty("volume", 1)

# Base de datos SQLite
conexion = sqlite3.connect("kazu_memoria.db", check_same_thread=False)
cursor = conexion.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS notas (id INTEGER PRIMARY KEY, texto TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS lista_compras (id INTEGER PRIMARY KEY, producto TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS aprendizaje (pregunta TEXT PRIMARY KEY, respuesta TEXT)")
conexion.commit()

# Respuestas predefinidas
respuestas_predefinidas = {
    "saludos": ["¡Hola, mi pana! ¿Cómo estás?", "¡Buenas! ¿Qué tal todo?"],
    "como_estas": ["Estoy pilas, gracias por preguntar.", "Todo chévere por aquí, ¿y tú?"],
    "quien_eres": ["Soy Kazu_ia, tu asistente inteligente.", "Soy Kazu_ia, tu asistente ecuatoriano."],
    "bien_y_tu": ["¡Me alegra! Yo también estoy bien, mi pana.", "Contento de hablar contigo."],
    "bromas": ["¿Por qué los programadores confunden Halloween con Navidad? OCT 31 = DEC 25."]
}

def hablar(texto):
    voz.say(texto)
    voz.runAndWait()

# API de escucha: Recibe el mensaje y lo procesa
@app.route('/api/escuchar', methods=['POST'])
def escuchar():
    try:
        # Recibimos los datos del JSON (mensaje)
        data = request.json
        mensaje = data['mensaje']
        
        if not mensaje:
            return jsonify({"error": "No se recibió mensaje."}), 400

        # Procesamos el mensaje con la IA
        respuesta = generar_respuesta_ia(mensaje)
        hablar(respuesta)

        # Devolvemos la respuesta al cliente
        return jsonify({"respuesta": respuesta}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Función para generar la respuesta de la IA
def generar_respuesta_ia(pregunta):
    pregunta_limpia = pregunta.strip()

    if "quién eres" in pregunta_limpia.lower() or "quien eres" in pregunta_limpia.lower():
        return random.choice(respuestas_predefinidas["quien_eres"])

    if "dime un poema de amor" in pregunta_limpia.lower():
        return "El amor es un fuego que arde sin verse..."

    prompt = f"""Eres Kazu_ia, un asistente amigable. Responde en español, de forma completa y natural, usando lenguaje cercano y coloquial.
Usuario: {pregunta_limpia}
Kazu_ia:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(modelo.device) for k, v in inputs.items()}

    salida = modelo.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        no_repeat_ngram_size=3
    )

    respuesta = tokenizer.decode(salida[0], skip_special_tokens=True).strip()
    return respuesta if respuesta else "No entendí bien, ¿puedes reformular?"

# Iniciar la API en el puerto 5000
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
