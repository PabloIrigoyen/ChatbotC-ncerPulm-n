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
    """Chatbot de salud pulmonar especializado con evaluación de riesgo."""

    def __init__(self, dataset_path='datasetchatbot_referencias.csv', model_path='lung_health_model.joblib'):
        try:
            self.nlp = self._load_spacy_model()
            self.load_qa_data(dataset_path)
            self._setup_qa_structures()
            self.load_prediction_model(model_path)
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

    def load_prediction_model(self, model_path):
        """Cargar modelo de machine learning entrenado."""
        try:
            if not os.path.exists(model_path):
                logging.warning(f"Modelo {model_path} no encontrado. La evaluación de riesgo no estará disponible.")
                self.model_data = None
                return

            self.model_data = joblib.load(model_path)
            logging.info(f"✅ Modelo de ML cargado: {self.model_data['mean_accuracy']:.2%} accuracy")

        except Exception as e:
            logging.error(f"Error cargando modelo ML: {e}")
            self.model_data = None

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
        self.risk_questions = self._initialize_risk_questions()
        self.current_risk_question = 0
        self.risk_responses = {}

    def _initialize_risk_questions(self):
        """Inicializar cuestionario de evaluación de riesgo."""
        return [
            {
                "id": "age",
                "pregunta": "¿Cuál es tu edad?",
                "tipo": "numero",
                "opciones": None
            },
            {
                "id": "gender",
                "pregunta": "¿Cuál es tu género?",
                "tipo": "opcion",
                "opciones": ["Masculino", "Femenino"]
            },
            {
                "id": "smoking",
                "pregunta": "¿Fumas actualmente o has fumado en el pasado?",
                "tipo": "opcion",
                "opciones": ["Nunca he fumado", "Fumé en el pasado", "Fumo actualmente"]
            },
            {
                "id": "yellow_fingers",
                "pregunta": "¿Has notado coloración amarillenta en tus dedos?",
                "tipo": "opcion",
                "opciones": ["No", "Sí, leve", "Sí, notable"]
            },
            {
                "id": "anxiety",
                "pregunta": "¿Experimentas ansiedad con frecuencia?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Frecuentemente"]
            },
            {
                "id": "peer_pressure",
                "pregunta": "¿Estás expuesto a presión social para fumar?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Sí, frecuentemente"]
            },
            {
                "id": "chronic_disease",
                "pregunta": "¿Tienes alguna enfermedad crónica respiratoria?",
                "tipo": "opcion",
                "opciones": ["No", "Asma", "EPOC", "Otra enfermedad respiratoria"]
            },
            {
                "id": "fatigue",
                "pregunta": "¿Experimentas fatiga o cansancio constante?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Sí, frecuentemente"]
            },
            {
                "id": "allergy",
                "pregunta": "¿Tienes alergias respiratorias?",
                "tipo": "opcion",
                "opciones": ["No", "Alergias leves", "Alergias severas"]
            },
            {
                "id": "wheezing",
                "pregunta": "¿Has experimentado silbidos al respirar (sibilancias)?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Frecuentemente"]
            },
            {
                "id": "alcohol",
                "pregunta": "¿Consumes alcohol regularmente?",
                "tipo": "opcion",
                "opciones": ["No consumo", "Ocasionalmente", "Regularmente"]
            },
            {
                "id": "coughing",
                "pregunta": "¿Tienes tos persistente?",
                "tipo": "opcion",
                "opciones": ["No", "Tos ocasional", "Tos persistente"]
            },
            {
                "id": "shortness_breath",
                "pregunta": "¿Experimentas dificultad para respirar?",
                "tipo": "opcion",
                "opciones": ["No", "Al hacer ejercicio", "En reposo"]
            },
            {
                "id": "swallowing_difficulty",
                "pregunta": "¿Tienes dificultad para tragar?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Frecuentemente"]
            },
            {
                "id": "chest_pain",
                "pregunta": "¿Experimentas dolor en el pecho?",
                "tipo": "opcion",
                "opciones": ["No", "Ocasionalmente", "Frecuentemente"]
            }
        ]

    def start_risk_assessment(self):
        """Iniciar evaluación de riesgo."""
        self.risk_assessment_active = True
        self.current_risk_question = 0
        self.risk_responses = {}

        first_question = self.risk_questions[0]
        return {
            "type": "risk_assessment_start",
            "question": first_question["pregunta"],
            "question_id": first_question["id"],
            "question_type": first_question["tipo"],
            "options": first_question["opciones"],
            "progress": f"1/{len(self.risk_questions)}"
        }

    def process_risk_response(self, response):
        """Procesar respuesta del cuestionario de riesgo."""
        if not self.risk_assessment_active:
            return {"error": "No hay evaluación de riesgo activa"}

        current_q = self.risk_questions[self.current_risk_question]
        self.risk_responses[current_q["id"]] = response

        # Avanzar a la siguiente pregunta
        self.current_risk_question += 1

        if self.current_risk_question < len(self.risk_questions):
            next_q = self.risk_questions[self.current_risk_question]
            return {
                "type": "risk_assessment_question",
                "question": next_q["pregunta"],
                "question_id": next_q["id"],
                "question_type": next_q["tipo"],
                "options": next_q["opciones"],
                "progress": f"{self.current_risk_question + 1}/{len(self.risk_questions)}"
            }
        else:
            # Completar evaluación
            return self._complete_risk_assessment()

    def _complete_risk_assessment(self):
        """Completar evaluación de riesgo y generar resultado."""
        self.risk_assessment_active = False

        # Convertir respuestas a formato del modelo
        if self.model_data:
            risk_prediction = self._predict_risk()
        else:
            risk_prediction = self._calculate_basic_risk()

        # Guardar en historial
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'type': 'risk_assessment_complete',
            'risk_result': risk_prediction
        })

        return risk_prediction

    def _map_response_to_model_input(self):
        """Mapear respuestas al formato esperado por el modelo."""
        if not self.model_data:
            return None

        mapping = {
            'age': lambda x: int(x),
            'gender': lambda x: 1 if x == "Masculino" else 0,
            'smoking': lambda x: 2 if x == "Nunca he fumado" else (1 if x == "Fumé en el pasado" else 2),
            'yellow_fingers': lambda x: 2 if x == "No" else (1 if x == "Sí, leve" else 2),
            'anxiety': lambda x: 2 if x == "No" else (1 if x == "Ocasionalmente" else 2),
            'peer_pressure': lambda x: 1 if x == "No" else (2 if x == "Ocasionalmente" else 2),
            'chronic_disease': lambda x: 1 if x == "No" else 2,
            'fatigue': lambda x: 2 if x == "No" else (1 if x == "Ocasionalmente" else 2),
            'allergy': lambda x: 2 if x == "No" else (1 if x == "Alergias leves" else 2),
            'wheezing': lambda x: 2 if x == "No" else (1 if x == "Ocasionalmente" else 2),
            'alcohol': lambda x: 2 if x == "No consumo" else (1 if x == "Ocasionalmente" else 2),
            'coughing': lambda x: 2 if x == "No" else (1 if x == "Tos ocasional" else 2),
            'shortness_breath': lambda x: 2 if x == "No" else (1 if x == "Al hacer ejercicio" else 2),
            'swallowing_difficulty': lambda x: 2 if x == "No" else (1 if x == "Ocasionalmente" else 2),
            'chest_pain': lambda x: 2 if x == "No" else (1 if x == "Ocasionalmente" else 2)
        }

        try:
            model_input = []
            for feature in self.model_data['feature_names']:
                if feature.lower() in self.risk_responses:
                    value = self.risk_responses[feature.lower()]
                    mapped_value = mapping[feature.lower()](value)
                    model_input.append(mapped_value)
                else:
                    model_input.append(2)  # Valor por defecto (No/No aplica)

            return np.array([model_input])

        except Exception as e:
            logging.error(f"Error mapeando respuestas: {e}")
            return None

    def _predict_risk(self):
        """Realizar predicción usando el modelo ML."""
        try:
            model_input = self._map_response_to_model_input()
            if model_input is None:
                return self._calculate_basic_risk()

            # Escalar datos
            scaled_input = self.model_data['scaler'].transform(model_input)

            # Predecir
            prediction = self.model_data['model'].predict(scaled_input)[0]
            probability = self.model_data['model'].predict_proba(scaled_input)[0]

            # Decodificar resultado
            if 'label_encoder' in self.model_data:
                risk_level = self.model_data['label_encoder'].inverse_transform([prediction])[0]
            else:
                risk_level = "ALTO" if prediction == 1 else "BAJO"

            confidence = max(probability)

            return {
                "type": "risk_assessment_result",
                "risk_level": risk_level,
                "confidence": f"{confidence:.1%}",
                "probability_high": f"{probability[1]:.1%}",
                "recommendations": self._get_recommendations(risk_level, confidence),
                "based_on_ml": True
            }

        except Exception as e:
            logging.error(f"Error en predicción ML: {e}")
            return self._calculate_basic_risk()

    def _calculate_basic_risk(self):
        """Calcular riesgo básico cuando no hay modelo ML."""
        risk_score = 0
        total_questions = len(self.risk_responses)

        # Puntuar respuestas de riesgo
        high_risk_responses = ['Fumo actualmente', 'Sí, notable', 'Frecuentemente', 'En reposo', 'Frecuentemente']

        for response in self.risk_responses.values():
            if response in high_risk_responses:
                risk_score += 2
            elif any(word in str(response).lower() for word in ['ocasionalmente', 'leve', 'asma', 'epoc']):
                risk_score += 1

        risk_percentage = min(100, (risk_score / (total_questions * 2)) * 100)

        if risk_percentage > 70:
            risk_level = "ALTO"
        elif risk_percentage > 40:
            risk_level = "MODERADO"
        else:
            risk_level = "BAJO"

        return {
            "type": "risk_assessment_result",
            "risk_level": risk_level,
            "risk_score": f"{risk_percentage:.0f}%",
            "confidence": "Estimación básica",
            "recommendations": self._get_recommendations(risk_level, risk_percentage / 100),
            "based_on_ml": False
        }

    def _get_recommendations(self, risk_level, confidence):
        """Generar recomendaciones basadas en el nivel de riesgo."""
        base_recommendations = [
            "Consulta con un especialista en neumología",
            "Realiza controles médicos periódicos",
            "Mantén un estilo de vida saludable"
        ]

        if risk_level == "ALTO":
            specific_recs = [
                "🚨 **Consulta médica URGENTE** con neumólogo",
                "💊 Considera realizar una tomografía computarizada",
                "🚭 Suspende el tabaquismo inmediatamente",
                "🏥 Programa evaluación pulmonar completa"
            ]
        elif risk_level == "MODERADO":
            specific_recs = [
                "📅 Consulta médica programada",
                "🔍 Considera radiografía de tórax",
                "🌱 Reduce factores de riesgo",
                "📊 Monitorea síntomas regularmente"
            ]
        else:
            specific_recs = [
                "👍 Mantén hábitos saludables",
                "🚭 Evita exposición al humo",
                "💪 Realiza ejercicio regular",
                "🥗 Sigue una dieta balanceada"
            ]

        return base_recommendations + specific_recs

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

            # Evaluación de riesgo
            if any(cmd in lower_message for cmd in ['evaluar riesgo', 'test riesgo', 'cuestionario', 'evaluación']):
                risk_start = self.start_risk_assessment()
                response = risk_start
                response['bot_response'] = f"🔍 {risk_start['question']}"
                if risk_start['options']:
                    response['bot_response'] += f"\nOpciones: {', '.join(risk_start['options'])}"

            elif self.risk_assessment_active:
                # Procesar respuesta del cuestionario
                risk_response = self.process_risk_response(message)
                if 'question' in risk_response:
                    response = risk_response
                    response['bot_response'] = f"📝 {risk_response['question']}"
                    if risk_response['options']:
                        response['bot_response'] += f"\nOpciones: {', '.join(risk_response['options'])}"
                else:
                    # Resultado final
                    result = risk_response
                    risk_emoji = "🔴" if result['risk_level'] == "ALTO" else "🟡" if result[
                                                                                       'risk_level'] == "MODERADO" else "🟢"
                    response = {
                        'bot_response': f"""{risk_emoji} **EVALUACIÓN COMPLETADA**

**Nivel de riesgo:** {result['risk_level']}
**Confianza:** {result['confidence']}

**📋 RECOMENDACIONES:**
""" + "\n".join([f"• {rec}" for rec in result['recommendations']]) + """

💡 _Esta evaluación es informativa. Consulta siempre con un profesional de la salud._"""
                    }

            elif any(cmd in lower_message for cmd in ['hola', 'hi', 'buenos días']):
                response = {'bot_response': self.get_welcome_message()}
            elif any(cmd in lower_message for cmd in ['ayuda', 'comandos']):
                response = {'bot_response': self.get_help_message()}
            elif any(cmd in lower_message for cmd in ['síntomas', 'sintomas']):
                response = {'bot_response': self.get_symptoms_response()}
            elif any(cmd in lower_message for cmd in ['diagnóstico', 'diagnostico']):
                response = {'bot_response': self.get_diagnosis_response()}
            elif any(cmd in lower_message for cmd in ['tratamiento']):
                response = {'bot_response': self.get_treatment_response()}
            else:
                # Búsqueda en el dataset
                match = self.find_best_match(message)
                if match:
                    response = {'bot_response': match['respuesta']}
                    logging.info(f"Match encontrado: {match['tipo']} (score: {match['score']:.2f})")
                else:
                    response = {'bot_response': self._get_default_response()}

            # Guardar respuesta
            if 'bot_response' in response:
                self.conversation_history.append({
                    'timestamp': datetime.now(),
                    'bot_response': response['bot_response'],
                    'type': 'bot'
                })

            return response

        except Exception as e:
            logging.error(f"Error en process_message: {e}")
            return {'bot_response': "Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo."}

    def _get_default_response(self):
        """Respuesta por defecto."""
        default_responses = [
            "No encontré información específica sobre tu consulta. ¿Te gustaría realizar una evaluación de riesgo? Escribe 'evaluar riesgo' para comenzar.",
            "Sobre ese tema no tengo información detallada. Puedo ayudarte con una evaluación de riesgo de cáncer pulmonar o información sobre síntomas/diagnóstico.",
            "Mi especialidad es el cáncer de pulmón. ¿Te interesa saber sobre síntomas, diagnóstico, o prefieres una evaluación de riesgo personalizada?"
        ]
        return random.choice(default_responses)

    def get_welcome_message(self):
        """Mensaje de bienvenida."""
        model_status = "✅ **Con evaluación de riesgo avanzada**" if self.model_data else "⚠️ **Evaluación básica disponible**"

        return f"""
👋 ¡Hola! Soy tu asistente especializado en **cáncer de pulmón**.

{model_status}

**Puedo ayudarte con:**
• 🏥 Información médica sobre cáncer pulmonar
• 🔍 **Evaluación de riesgo personalizada** 
• 💡 Respuestas a tus preguntas específicas

**💬 Comandos disponibles:**
• "evaluar riesgo" - Cuestionario de evaluación
• "síntomas" - Síntomas del cáncer de pulmón  
• "diagnóstico" - Métodos de diagnóstico
• "tratamiento" - Opciones de tratamiento
• "ayuda" - Ver todos los comandos

¡Estoy aquí para ayudarte! 😊
"""

    def get_help_message(self):
        """Mensaje de ayuda."""
        return """
🤖 **TEMAS Y COMANDOS DISPONIBLES**

**🔍 EVALUACIÓN DE RIESGO:**
• "evaluar riesgo" - Cuestionario completo de 15 preguntas
• "test riesgo" - Evaluación personalizada

**🏥 INFORMACIÓN MÉDICA:**
• "síntomas" - Síntomas y señales de alerta
• "diagnóstico" - Métodos de diagnóstico  
• "tratamiento" - Opciones de tratamiento
• Tipos de cáncer de pulmón
• Factores de riesgo

**💡 EJEMPLOS DE PREGUNTAS:**
• "¿Qué es el cáncer de pulmón microcítico?"
• "¿Cuáles son los síntomas tempranos?"
• "¿Cómo se trata el adenocarcinoma?"

¡Puedes hacer preguntas en tus propias palabras!
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
    chatbot = LungHealthChatbot('datasetchatbot_referencias.csv', 'lung_health_model.joblib')
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
        response = chatbot.process_message(user_message)

        return jsonify({
            'user_message': user_message,
            'bot_response': response.get('bot_response', ''),
            'risk_assessment_active': chatbot.risk_assessment_active,
            'risk_question_data': {k: v for k, v in response.items() if k != 'bot_response'},
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
        'model_loaded': chatbot.model_data is not None if chatbot else False,
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