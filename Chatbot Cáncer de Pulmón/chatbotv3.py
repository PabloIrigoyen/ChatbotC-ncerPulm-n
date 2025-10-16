import os
import numpy as np
import pandas as pd
import spacy
import difflib
import logging
import joblib
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import random
from collections import defaultdict

# -----------------------------
# Configuración general
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')


class LungHealthChatbot:
    """Chatbot de salud pulmonar especializado."""

    def __init__(self, dataset_path='datasetchatbot_referencias.csv'):
        try:
            self.nlp = self._load_spacy_model()
            self.load_qa_data(dataset_path)
            self._setup_qa_structures()
            self.reset_conversation_state()
            logging.info("Chatbot inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando chatbot: {e}")
            raise

    def _load_spacy_model(self):
        """Carga el modelo de lenguaje español de spaCy."""
        try:
            nlp = spacy.load("es_core_news_sm")
            logging.info("Modelo spaCy cargado exitosamente")
            return nlp
        except OSError:
            try:
                nlp = spacy.load("es_core_news_sm")
                return nlp
            except OSError:
                import subprocess, sys
                logging.info("Descargando modelo de spaCy...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"], check=True)
                nlp = spacy.load("es_core_news_sm")
                return nlp

    def load_qa_data(self, filename):
        """Cargar dataset de preguntas y respuestas."""
        try:
            if not os.path.exists(filename):
                logging.error(f"Archivo {filename} no encontrado")
                raise FileNotFoundError(f"Dataset {filename} no encontrado")

            self.qa_data = pd.read_csv(filename)
            logging.info(f"Dataset cargado: {len(self.qa_data)} preguntas")

            # Limpiar y preparar datos
            for col in ['pregunta', 'respuesta', 'intencion', 'entidades']:
                self.qa_data[col] = self.qa_data[col].astype(str).str.strip()

            # Procesar entidades
            self.qa_data['entidades_lista'] = self.qa_data['entidades'].str.split('|')

        except Exception as e:
            logging.error(f"Error cargando dataset: {e}")
            raise

    def _setup_qa_structures(self):
        """Configurar estructuras de búsqueda."""
        try:
            # Diccionarios básicos
            self.qa_dict = dict(zip(self.qa_data['pregunta'], self.qa_data['respuesta']))
            self.intent_dict = dict(zip(self.qa_data['pregunta'], self.qa_data['intencion']))

            # Índice de entidades
            self.entity_to_questions = defaultdict(list)
            for idx, row in self.qa_data.iterrows():
                entities = row['entidades_lista']
                if isinstance(entities, list):
                    for entity in entities:
                        if entity and isinstance(entity, str):
                            self.entity_to_questions[entity.strip().lower()].append(idx)

            logging.info("Estructuras de búsqueda configuradas")

        except Exception as e:
            logging.error(f"Error configurando estructuras: {e}")
            raise

    def reset_conversation_state(self):
        """Reiniciar estado de la conversación."""
        self.conversation_history = []
        self.risk_assessment_active = False
        self.waiting_for_pdf_confirmation = False

    def _extract_entities(self, query):
        """Extraer entidades de la consulta."""
        try:
            query_lower = query.lower().strip()
            found_entities = set()

            # Obtener todas las entidades del dataset
            all_entities = set()
            for entities in self.qa_data['entidades_lista']:
                if isinstance(entities, list):
                    for entity in entities:
                        if entity and isinstance(entity, str):
                            all_entities.add(entity.strip().lower())

            # Buscar coincidencias
            for entity in all_entities:
                if entity in query_lower:
                    found_entities.add(entity)

            return list(found_entities)

        except Exception as e:
            logging.error(f"Error extrayendo entidades: {e}")
            return []

    def find_best_match(self, query):
        """Encontrar la mejor coincidencia para la consulta."""
        try:
            query_lower = query.lower().strip()

            # 1. Búsqueda exacta
            for i, question in enumerate(self.qa_data['pregunta']):
                if query_lower == question.lower():
                    return {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": 1.0,
                        "tipo": "exacta"
                    }

            # 2. Búsqueda por similitud de texto
            best_match = None
            best_score = 0

            for i, question in enumerate(self.qa_data['pregunta']):
                similarity = difflib.SequenceMatcher(None, query_lower, question.lower()).ratio()
                if similarity > best_score and similarity > 0.6:
                    best_score = similarity
                    best_match = {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": similarity,
                        "tipo": "similaridad"
                    }

            if best_match:
                return best_match

            # 3. Búsqueda por entidades
            entities = self._extract_entities(query)
            if entities:
                entity_matches = []
                for entity in entities:
                    if entity in self.entity_to_questions:
                        for idx in self.entity_to_questions[entity]:
                            entity_matches.append({
                                "pregunta": self.qa_data.iloc[idx]['pregunta'],
                                "respuesta": self.qa_data.iloc[idx]['respuesta'],
                                "score": 0.7,
                                "tipo": f"entidad_{entity}"
                            })

                if entity_matches:
                    # Eliminar duplicados
                    unique_matches = {}
                    for match in entity_matches:
                        key = match['pregunta']
                        if key not in unique_matches:
                            unique_matches[key] = match

                    matches_list = list(unique_matches.values())
                    if matches_list:
                        return matches_list[0]  # Devolver el primero

            return None

        except Exception as e:
            logging.error(f"Error en find_best_match: {e}")
            return None

    def process_message(self, message):
        """Procesar mensaje del usuario."""
        try:
            if not message or not isinstance(message, str):
                return "Por favor escribe un mensaje válido."

            message = message.strip()
            if not message:
                return "Por favor escribe un mensaje válido."

            # Guardar en historial
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_message': message,
                'type': 'user'
            })

            # Comandos especiales
            lower_message = message.lower()
            if any(cmd in lower_message for cmd in ['hola', 'hi', 'buenos días']):
                response = self.get_welcome_message()
            elif any(cmd in lower_message for cmd in ['ayuda', 'comandos']):
                response = self.get_help_message()
            elif any(cmd in lower_message for cmd in ['síntomas', 'sintomas']):
                response = self.get_symptoms_response()
            elif any(cmd in lower_message for cmd in ['diagnóstico', 'diagnostico']):
                response = self.get_diagnosis_response()
            elif any(cmd in lower_message for cmd in ['tratamiento']):
                response = self.get_treatment_response()
            else:
                # Búsqueda en el dataset
                match = self.find_best_match(message)
                if match:
                    response = match['respuesta']
                    logging.info(f"Match encontrado: {match['tipo']} (score: {match['score']:.2f})")
                else:
                    response = self._get_default_response()

            # Guardar respuesta
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'bot_response': response,
                'type': 'bot'
            })

            return response

        except Exception as e:
            logging.error(f"Error en process_message: {e}")
            return "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."

    def _get_default_response(self):
        """Respuesta por defecto."""
        default_responses = [
            "No encontré información específica sobre tu consulta en mi base de datos. ¿Puedes reformular tu pregunta?",
            "Sobre ese tema no tengo información detallada. Puedo ayudarte con información sobre síntomas, diagnóstico o tratamiento del cáncer de pulmón.",
            "Mi especialidad es el cáncer de pulmón. ¿Te interesa saber sobre algún aspecto específico como síntomas, diagnóstico o tratamientos?"
        ]
        return random.choice(default_responses)

    def get_welcome_message(self):
        """Mensaje de bienvenida."""
        return """
👋 ¡Hola! Soy tu asistente especializado en **cáncer de pulmón**.

Puedo ayudarte con información sobre:
• Síntomas y señales de alerta
• Métodos de diagnóstico  
• Opciones de tratamiento
• Factores de riesgo

💬 **Ejemplos de preguntas:**
• "¿Qué es el cáncer de pulmón?"
• "¿Cuáles son los síntomas?"
• "¿Cómo se diagnostica?"
• "¿Qué tratamientos existen?"

Escribe **'ayuda'** para ver todos los comandos disponibles.
"""

    def get_help_message(self):
        """Mensaje de ayuda."""
        return """
🤖 **TEMAS DISPONIBLES**

**🏥 Información médica:**
• Síntomas del cáncer de pulmón
• Métodos de diagnóstico  
• Opciones de tratamiento
• Tipos de cáncer de pulmón
• Factores de riesgo

**💡 Ejemplos de preguntas:**
• "¿Qué es el cáncer de pulmón microcítico?"
• "¿Cuáles son los síntomas tempranos?"
• "¿Cómo se trata el adenocarcinoma?"
• "¿Qué es la quimioterapia?"

Puedes hacer preguntas en tus propias palabras.
"""

    def get_symptoms_response(self):
        """Respuesta sobre síntomas."""
        try:
            sintomas_data = self.qa_data[self.qa_data['intencion'] == 'sintomas del cáncer del pulmón']
            if len(sintomas_data) > 0:
                return sintomas_data.iloc[0]['respuesta']
            else:
                return "Los síntomas del cáncer de pulmón pueden incluir tos persistente, dolor en el pecho, dificultad para respirar, tos con sangre, fatiga y pérdida de peso inexplicable."
        except:
            return "Los síntomas del cáncer de pulmón pueden incluir tos persistente, dolor en el pecho y dificultad para respirar."

    def get_diagnosis_response(self):
        """Respuesta sobre diagnóstico."""
        try:
            diagnostico_data = self.qa_data[self.qa_data['intencion'] == 'diagnostico del cáncer del pulmón']
            if len(diagnostico_data) > 0:
                return diagnostico_data.iloc[0]['respuesta']
            else:
                return "El cáncer de pulmón se diagnostica mediante radiografías de tórax, tomografías computarizadas, biopsias y otros métodos de imagen."
        except:
            return "El diagnóstico del cáncer de pulmón incluye radiografías, tomografías y biopsias."

    def get_treatment_response(self):
        """Respuesta sobre tratamiento."""
        try:
            tratamiento_data = self.qa_data[self.qa_data['intencion'] == 'tratamiento del cáncer de pulmón']
            if len(tratamiento_data) > 0:
                return tratamiento_data.iloc[0]['respuesta']
            else:
                return "Los tratamientos para el cáncer de pulmón incluyen cirugía, quimioterapia, radioterapia, inmunoterapia y terapias dirigidas."
        except:
            return "Los tratamientos incluyen cirugía, quimioterapia y radioterapia."


# Inicializar chatbot
try:
    chatbot = LungHealthChatbot('datasetchatbot_referencias.csv')
    logging.info("✅ Chatbot inicializado exitosamente")
except Exception as e:
    logging.error(f"❌ Error inicializando chatbot: {e}")
    chatbot = None


# Rutas Flask
@app.route('/')
def home():
    welcome_msg = chatbot.get_welcome_message() if chatbot else "El chatbot no está disponible en este momento."
    return render_template('chat.html',
                           welcome_message=welcome_msg,
                           current_time=datetime.now().strftime("%H:%M"))


@app.route('/send_message', methods=['POST'])
def send_message():
    if not chatbot:
        return jsonify({'error': 'Chatbot no disponible'}), 500

    if not request.json or 'message' not in request.json:
        return jsonify({'error': 'Mensaje no proporcionado'}), 400

    try:
        user_message = request.json['message']
        bot_response = chatbot.process_message(user_message)

        return jsonify({
            'user_message': user_message,
            'bot_response': bot_response,
            'timestamp': datetime.now().strftime("%H:%M")
        })

    except Exception as e:
        logging.error(f"Error en send_message: {e}")
        return jsonify({
            'user_message': user_message,
            'bot_response': "Lo siento, hubo un error procesando tu mensaje.",
            'timestamp': datetime.now().strftime("%H:%M")
        })


@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    if chatbot:
        chatbot.reset_conversation_state()
        return jsonify({'status': 'success', 'message': 'Conversación reiniciada'})
    return jsonify({'error': 'Chatbot no disponible'}), 500


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if chatbot else 'error',
        'chatbot_loaded': chatbot is not None,
        'dataset_size': len(chatbot.qa_data) if chatbot else 0
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

    logging.info(f"🚀 Iniciando servidor en puerto {port} (debug: {debug_mode})")

    # Para producción en Render, usar gunicorn en lugar de app.run()
    if os.getenv('RENDER'):  # Render establece esta variable
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=debug_mode, host='0.0.0.0', port=port)