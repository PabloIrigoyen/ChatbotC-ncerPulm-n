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
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Configuración general
# -----------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')


class LungHealthChatbot:
    """Chatbot de salud pulmonar especializado con evaluación de riesgo mejorada."""

    def __init__(self, dataset_path='datasetchatbot_referencias.csv', model_path='lung_cancer_decision_tree.joblib'):
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
            logging.info(f"✅ Modelo de ML cargado: {model_path}")

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
                            clean_entity = entity.strip().lower()
                            self.entity_to_questions[clean_entity].append({
                                'index': idx,
                                'pregunta': row['pregunta'],
                                'respuesta': row['respuesta'],
                                'intencion': row['intencion'],
                                'all_entities': entities
                            })

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
        """Inicializar cuestionario de evaluación de riesgo ampliado."""
        return [
            {
                "id": "age",
                "pregunta": "👤 ¿Cuál es su edad?",
                "tipo": "numero",
                "opciones": None
            },
            {
                "id": "gender",
                "pregunta": "🚻 ¿Cuál es su género?",
                "tipo": "opcion",
                "opciones": ["Masculino", "Femenino"]
            },
            {
                "id": "air_pollution",
                "pregunta": "🏭 ¿Cómo calificaría su exposición a la contaminación del aire?\n1-Nula/Mínima 2-Baja 3-Moderada 4-Alta 5-Muy Alta 6-Extremadamente Alta 7-Severa 8-Crítica",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "alcohol_use",
                "pregunta": "🍷 ¿Cuál es su nivel de consumo de alcohol?\n1-Nunca 2-Ocasional 3-Moderado 4-Regular 5-Frecuente 6-Excesivo 7-Muy Excesivo 8-Crítico",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "dust_allergy",
                "pregunta": "🌫️ ¿Tiene alergia al polvo? ¿Cómo la calificaría?\n1-Nula/Mínima 2-Baja 3-Moderada 4-Alta 5-Muy Alta 6-Extremadamente Alta 7-Severa 8-Crítica",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "occupational_hazards",
                "pregunta": "👷 ¿Está expuesto a riesgos ocupacionales? (productos químicos, polvo industrial, etc.)\n1-Nula/Mínima 2-Baja 3-Moderada 4-Alta 5-Muy Alta 6-Extremadamente Alta 7-Severa 8-Crítica",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "genetic_risk",
                "pregunta": "🧬 ¿Tiene antecedentes familiares de cáncer de pulmón? ¿Cómo calificaría su riesgo genético?\n1-Nulo 2-Muy bajo 3-Bajo 4-Moderado 5-Alto 6-Muy alto 7-Extremadamente alto 8-Crítico",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "chronic_lung_disease",
                "pregunta": "🫁 ¿Tiene enfermedad pulmonar crónica? (asma, EPOC, etc.)\n1-Ausente 2-Muy leve 3-Leve 4-Moderada 5-Moderadamente severa 6-Severa 7-Muy severa 8-Crítica",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "balanced_diet",
                "pregunta": "🥗 ¿Cómo calificaría su dieta?\n1-Muy pobre 2-Pobre 3-Regular 4-Adecuada 5-Buena 6-Muy buena 7-Excelente 8-Óptima",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "obesity",
                "pregunta": "⚖️ ¿Cuál es su nivel de obesidad/IMC?\n1-Bajo peso 2-Normal 3-Sobrepeso leve 4-Sobrepeso 5-Obesidad grado I 6-Obesidad grado II 7-Obesidad grado III 8-Obesidad mórbida",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "smoking",
                "pregunta": "🚬 ¿Cuál es su historial de tabaquismo?\n1-Nunca fumó 2-Ex-fumador (>5 años) 3-Ex-fumador (<5 años) 4-Fumador ocasional 5-Fumador moderado 6-Fumador frecuente 7-Fumador intenso 8-Fumador muy intenso",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "passive_smoker",
                "pregunta": "💨 ¿Está expuesto a humo de segunda mano (fumadores pasivos)?\n1-Nula 2-Mínima 3-Ocasional 4-Regular 5-Frecuente 6-Intensa 7-Muy Intensa 8-Constante",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "chest_pain",
                "pregunta": "💓 ¿Experimenta dolor en el pecho?\n1-Ausente 2-Muy leve/ocasional 3-Leve 4-Moderado 5-Moderadamente severo 6-Severo 7-Muy severo 8-Debilitante 9-Extremo",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "coughing_blood",
                "pregunta": "🩸 ¿Ha tenido tos con sangre?\n1-Ausente 2-Muy rara 3-Ocasional 4-Intermitente 5-Frecuente 6-Muy frecuente 7-Diaria 8-Múltiples veces al día 9-Constante",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "fatigue",
                "pregunta": "😴 ¿Cómo calificaría su nivel de fatiga?\n1-Ausente 2-Mínima 3-Leve 4-Moderada 5-Notable 6-Severa 7-Muy severa 8-Debilitante 9-Completa",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "weight_loss",
                "pregunta": "⚖️ ¿Ha experimentado pérdida de peso?\n1-Ninguna 2-Mínima (<2 kg) 3-Leve (2-4 kg) 4-Moderada (5-7 kg) 5-Significativa (8-10 kg) 6-Severa (11-15 kg) 7-Muy severa (16-20 kg) 8-Extrema (21-25 kg) 9-Crítica (>25 kg)",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "shortness_breath",
                "pregunta": "🌬️ ¿Tiene dificultad para respirar?\n1-Ausente 2-Solo con ejercicio intenso 3-Con ejercicio moderado 4-Con actividades diarias 5-Con actividades livianas 6-En reposo ocasional 7-En reposo frecuente 8-En reposo constante 9-Incapacitante",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "wheezing",
                "pregunta": "🎵 ¿Experimenta sibilancias (silbidos al respirar)?\n1-Ausentes 2-Muy raras 3-Ocasionales 4-Intermitentes 5-Frecuentes 6-Muy frecuentes 7-Diarias 8-Constantes leves 9-Constantes severas",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "swallowing_difficulty",
                "pregunta": "🥛 ¿Tiene dificultad para tragar?\n1-Ausente 2-Muy leve 3-Leve 4-Moderada 5-Moderadamente severa 6-Severa 7-Muy severa 8-Solo líquidos 9-Incapacitante",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "clubbing_finger_nails",
                "pregunta": "🖐️ ¿Tiene acropaquia (dedos en palillo de tambor)?\n1-Ausente 2-Muy leve 3-Leve 4-Moderada 5-Notable 6-Severa 7-Muy severa 8-Extrema",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8"]
            },
            {
                "id": "frequent_cold",
                "pregunta": "🤧 ¿Con qué frecuencia tiene resfriados?\n1-Muy raros (<1/año) 2-Ocasionales (1-2/año) 3-Regulares (3-4/año) 4-Frecuentes (5-6/año) 5-Muy frecuentes (7-8/año) 6-Constantes (9-10/año) 7-Muy constantes (>10/año)",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7"]
            },
            {
                "id": "dry_cough",
                "pregunta": "🤭 ¿Tiene tos seca?\n1-Ausente 2-Muy ocasional 3-Ocasional 4-Intermitente 5-Frecuente 6-Muy frecuente 7-Constante leve 8-Constante moderada 9-Constante severa",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            {
                "id": "snoring",
                "pregunta": "😴 ¿Ronca?\n1-Ausentes 2-Muy leves/ocasionales 3-Leves 4-Moderados 5-Frecuentes 6-Intensos 7-Muy intensos",
                "tipo": "escala",
                "opciones": ["1", "2", "3", "4", "5", "6", "7"]
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

        # Validar respuesta según el tipo
        if current_q["tipo"] == "numero":
            try:
                response = int(response)
                if response < 1 or response > 120:
                    return {"error": "Por favor ingrese una edad válida entre 1 y 120 años"}
            except ValueError:
                return {"error": "Por favor ingrese un número válido para la edad"}

        elif current_q["tipo"] in ["escala", "opcion"]:
            if response not in current_q["opciones"]:
                return {"error": f"Por favor seleccione una opción válida: {', '.join(current_q['opciones'])}"}

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
        try:
            self.risk_assessment_active = False
            logging.info("🏁 Completando evaluación de riesgo...")
            logging.info(f"📋 Respuestas recolectadas: {self.risk_responses}")

            # Convertir respuestas a formato del modelo
            if self.model_data:
                logging.info("🔍 Usando modelo ML para predicción...")
                risk_prediction = self._predict_risk()
            else:
                logging.info("🔍 Modelo ML no disponible, usando cálculo básico...")
                risk_prediction = self._calculate_basic_risk()

            # Guardar en historial
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'type': 'risk_assessment_complete',
                'risk_result': risk_prediction
            })

            logging.info(f"✅ Evaluación completada exitosamente")
            return risk_prediction

        except Exception as e:
            logging.error(f"❌ Error en _complete_risk_assessment: {e}")
            return {
                "type": "risk_assessment_result",
                "risk_level": "MODERADO",
                "confidence": "Estimación básica",
                "recommendations": self._get_recommendations("MODERADO"),
                "based_on_ml": False
            }

    def _map_response_to_model_input(self):
        """Mapear respuestas al formato esperado por el modelo."""
        if not self.model_data:
            return None

        # Mapeo para el modelo de cáncer de pulmón
        mapping = {
            'age': lambda x: int(x),
            'gender': lambda x: 1 if x == "Masculino" else 2,
            'air_pollution': lambda x: int(x),
            'alcohol_use': lambda x: int(x),
            'dust_allergy': lambda x: int(x),
            'occupational_hazards': lambda x: int(x),
            'genetic_risk': lambda x: int(x),
            'chronic_lung_disease': lambda x: int(x),
            'balanced_diet': lambda x: int(x),
            'obesity': lambda x: int(x),
            'smoking': lambda x: int(x),
            'passive_smoker': lambda x: int(x),
            'chest_pain': lambda x: int(x),
            'coughing_blood': lambda x: int(x),
            'fatigue': lambda x: int(x),
            'weight_loss': lambda x: int(x),
            'shortness_breath': lambda x: int(x),
            'wheezing': lambda x: int(x),
            'swallowing_difficulty': lambda x: int(x),
            'clubbing_finger_nails': lambda x: int(x),
            'frequent_cold': lambda x: int(x),
            'dry_cough': lambda x: int(x),
            'snoring': lambda x: int(x)
        }

        try:
            model_input = []

            # Orden esperado por el modelo (basado en el dataset original)
            expected_order = [
                'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
                'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
                'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain',
                'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath',
                'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails',
                'Frequent Cold', 'Dry Cough', 'Snoring'
            ]

            # Mapear nombres de campos
            field_mapping = {
                'age': 'Age',
                'gender': 'Gender',
                'air_pollution': 'Air Pollution',
                'alcohol_use': 'Alcohol use',
                'dust_allergy': 'Dust Allergy',
                'occupational_hazards': 'OccuPational Hazards',
                'genetic_risk': 'Genetic Risk',
                'chronic_lung_disease': 'chronic Lung Disease',
                'balanced_diet': 'Balanced Diet',
                'obesity': 'Obesity',
                'smoking': 'Smoking',
                'passive_smoker': 'Passive Smoker',
                'chest_pain': 'Chest Pain',
                'coughing_blood': 'Coughing of Blood',
                'fatigue': 'Fatigue',
                'weight_loss': 'Weight Loss',
                'shortness_breath': 'Shortness of Breath',
                'wheezing': 'Wheezing',
                'swallowing_difficulty': 'Swallowing Difficulty',
                'clubbing_finger_nails': 'Clubbing of Finger Nails',
                'frequent_cold': 'Frequent Cold',
                'dry_cough': 'Dry Cough',
                'snoring': 'Snoring'
            }

            for field in expected_order:
                # Encontrar la clave correspondiente en las respuestas
                response_key = None
                for key, value in field_mapping.items():
                    if value == field:
                        response_key = key
                        break

                if response_key and response_key in self.risk_responses:
                    value = self.risk_responses[response_key]
                    mapped_value = mapping[response_key](value)
                    model_input.append(mapped_value)
                    logging.info(f"   {field}: '{value}' -> {mapped_value}")
                else:
                    logging.warning(f"Campo {field} no encontrado en respuestas, usando valor por defecto")
                    model_input.append(2)  # Valor por defecto

            # Convertir a numpy array para el modelo
            input_array = np.array([model_input])
            logging.info(f"🎯 Input final para el modelo: {input_array}")
            return input_array

        except Exception as e:
            logging.error(f"Error en _map_response_to_model_input: {e}")
            return None

    def _predict_risk(self):
        """Realizar predicción usando el modelo ML."""
        try:
            model_input = self._map_response_to_model_input()
            if model_input is None:
                logging.warning("No se pudo generar input para el modelo, usando cálculo básico")
                return self._calculate_basic_risk()

            # Predecir usando el modelo cargado
            prediction = self.model_data.predict(model_input)[0]

            # Si el modelo tiene predict_proba, usarlo para confianza
            try:
                probability = self.model_data.predict_proba(model_input)[0]
                confidence = max(probability)
                probability_high = probability[1] if len(probability) > 1 else 0.5
            except:
                confidence = 0.75
                probability_high = 0.5

            # Mapear predicción a nivel de riesgo
            risk_levels = {0: "BAJO", 1: "MEDIO", 2: "ALTO"}
            risk_level = risk_levels.get(prediction, "MODERADO")

            return {
                "type": "risk_assessment_result",
                "risk_level": risk_level,
                "confidence": f"{confidence:.1%}",
                "probability_high": f"{probability_high:.1%}",
                "recommendations": self._get_recommendations(risk_level),
                "based_on_ml": True,
                "risk_score": self._calculate_risk_score()
            }

        except Exception as e:
            logging.error(f"Error en _predict_risk: {e}")
            return self._calculate_basic_risk()

    def _calculate_risk_score(self):
        """Calcular puntuación de riesgo basada en las respuestas."""
        try:
            score = 0
            max_score = len(self.risk_responses) * 9  # Máximo teórico

            for key, value in self.risk_responses.items():
                if key != 'age' and key != 'gender':
                    try:
                        score += int(value)
                    except:
                        pass

            risk_percentage = min(100, (score / max_score) * 100)
            return f"{risk_percentage:.0f}%"
        except:
            return "50%"

    def _calculate_basic_risk(self):
        """Calcular riesgo básico cuando no hay modelo ML."""
        risk_score = 0
        total_questions = len(self.risk_responses)

        for response in self.risk_responses.values():
            if isinstance(response, str) and response.isdigit():
                value = int(response)
                if value >= 6:  # Valores altos en la escala
                    risk_score += 2
                elif value >= 4:  # Valores medios
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
            "recommendations": self._get_recommendations(risk_level),
            "based_on_ml": False
        }

    def _get_recommendations(self, risk_level):
        """Generar recomendaciones basadas en el nivel de riesgo."""
        if risk_level == "ALTO":
            return [
                "🚨 Consulta médica URGENTE con neumólogo",
                "💊 Considera realizar una tomografía computarizada de tórax",
                "🚭 Suspende el tabaquismo inmediatamente si fumas",
                "🏥 Programa evaluación pulmonar completa",
                "🔍 Monitorea síntomas regularmente",
                "🌡️ Evita exposición a contaminantes y humos"
            ]
        elif risk_level == "MODERADO":
            return [
                "📅 Consulta médica programada con tu médico de cabecera",
                "🔍 Considera radiografía de tórax en tu próximo chequeo",
                "🌱 Reduce factores de riesgo modificables",
                "🚭 Evita la exposición al humo de tabaco",
                "💪 Mantén un estilo de vida saludable",
                "📊 Realiza controles médicos anuales"
            ]
        else:
            return [
                "👍 Mantén tus hábitos saludables actuales",
                "🚭 Evita exposición al humo y contaminantes",
                "💪 Realiza ejercicio regularmente",
                "🥗 Sigue una dieta balanceada rica en frutas y verduras",
                "😴 Mantén una buena calidad de sueño",
                "📅 Realiza chequeos médicos preventivos anuales"
            ]

    def _extract_entities(self, query):
        """Extraer entidades de la consulta."""
        try:
            query_lower = query.lower().strip()
            found_entities = set()

            for entity in self.entity_to_questions.keys():
                if f" {entity} " in f" {query_lower} ":
                    found_entities.add(entity)
                elif entity in query_lower.split():
                    found_entities.add(entity)
                elif any(part in query_lower for part in entity.split('_')):
                    found_entities.add(entity)

            return list(found_entities)

        except Exception as e:
            logging.error(f"Error extrayendo entidades: {e}")
            return []

    def find_best_match(self, query):
        """Encontrar la mejor coincidencia para la consulta."""
        try:
            query_lower = query.lower().strip()
            logging.info(f"🔍 Buscando match para: '{query}'")

            # 1. Búsqueda exacta
            for i, question in enumerate(self.qa_data['pregunta']):
                if query_lower == question.lower():
                    logging.info(f"✅ Match exacto encontrado: {question}")
                    return {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": 1.0,
                        "tipo": "exacta"
                    }

            # 2. Búsqueda por entidades
            entities = self._extract_entities(query)
            logging.info(f"📊 Entidades encontradas en query: {entities}")

            if entities:
                entity_matches = []
                all_relevant_questions = set()

                for entity in entities:
                    if entity in self.entity_to_questions:
                        for q_info in self.entity_to_questions[entity]:
                            all_relevant_questions.add(q_info['index'])

                for idx in all_relevant_questions:
                    row = self.qa_data.iloc[idx]
                    question_entities = row['entidades_lista']

                    if not isinstance(question_entities, list):
                        continue

                    matching_entities = sum(
                        1 for entity in entities if any(entity in str(e).lower() for e in question_entities))
                    total_query_entities = len(entities)

                    entity_score = matching_entities / total_query_entities if total_query_entities > 0 else 0

                    if entity_score > 0.3:
                        entity_matches.append({
                            "pregunta": row['pregunta'],
                            "respuesta": row['respuesta'],
                            "score": entity_score,
                            "tipo": f"entidades_{matching_entities}"
                        })

                if entity_matches:
                    entity_matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = entity_matches[0]
                    logging.info(
                        f"✅ Mejor match por entidades: {best_match['pregunta']} (score: {best_match['score']:.2f})")
                    return best_match

            # 3. Búsqueda por similitud
            best_similarity_match = None
            best_similarity_score = 0

            for i, question in enumerate(self.qa_data['pregunta']):
                similarity = difflib.SequenceMatcher(None, query_lower, question.lower()).ratio()

                if similarity > best_similarity_score and similarity > 0.5:
                    best_similarity_score = similarity
                    best_similarity_match = {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": similarity,
                        "tipo": "similaridad"
                    }

            if best_similarity_match:
                logging.info(
                    f"✅ Match por similitud: {best_similarity_match['pregunta']} (score: {best_similarity_match['score']:.2f})")
                return best_similarity_match

            logging.info("❌ No se encontró match adecuado")
            return None

        except Exception as e:
            logging.error(f"Error en find_best_match: {e}")
            return None

    def process_message(self, message):
        """Procesar mensaje del usuario."""
        try:
            if not message or not isinstance(message, str):
                return {"bot_response": "Por favor escribe un mensaje válido."}

            message = message.strip()
            if not message:
                return {"bot_response": "Por favor escribe un mensaje válido."}

            # Guardar en historial
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_message': message,
                'type': 'user'
            })

            lower_message = message.lower()

            # Comandos especiales
            if any(cmd in lower_message for cmd in ['evaluar riesgo', 'test riesgo', 'cuestionario', 'evaluación']):
                risk_start = self.start_risk_assessment()
                response = risk_start
                response['bot_response'] = f"🔍 {risk_start['question']}"
                if risk_start['options']:
                    response['bot_response'] += f"\n\n💡 Opciones: {', '.join(risk_start['options'])}"

            elif self.risk_assessment_active:
                risk_response = self.process_risk_response(message)

                if 'error' in risk_response:
                    response = {'bot_response': f"❌ {risk_response['error']}"}
                elif 'question' in risk_response:
                    response = risk_response
                    response['bot_response'] = f"📝 {risk_response['question']}"
                    if risk_response['options']:
                        response['bot_response'] += f"\n\n💡 Opciones: {', '.join(risk_response['options'])}"
                else:
                    # Resultado final
                    result = risk_response
                    risk_emoji = "🔴" if result['risk_level'] == "ALTO" else "🟡" if result[
                                                                                       'risk_level'] == "MODERADO" else "🟢"

                    response = {
                        'bot_response': f"""{risk_emoji} EVALUACIÓN DE RIESGO COMPLETADA

Nivel de riesgo: {result['risk_level']}
Puntuación de riesgo: {result.get('risk_score', 'N/A')}
Confianza del modelo: {result['confidence']}

📋 RECOMENDACIONES:
""" + "\n".join([f"• {rec}" for rec in result['recommendations']]) + """

💡 _Esta evaluación es informativa y no reemplaza la consulta médica profesional. Consulta siempre con un especialista._"""
                    }

            elif any(cmd in lower_message for cmd in ['hola', 'hi', 'buenos días', 'buenas']):
                response = {'bot_response': self.get_welcome_message()}
            elif any(cmd in lower_message for cmd in ['ayuda', 'comandos']):
                response = {'bot_response': self.get_help_message()}
            else:
                match = self.find_best_match(message)
                if match:
                    response = {'bot_response': match['respuesta']}
                    logging.info(f"🎯 Match final: {match['tipo']} - Pregunta: '{match['pregunta']}'")
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
            "No encontré información específica sobre tu consulta. ¿Te gustaría realizar una evaluación de riesgo de cáncer de pulmón? Escribe 'evaluar riesgo' para comenzar el cuestionario de 23 preguntas.",
            "Sobre ese tema no tengo información detallada en mi base de conocimientos. Puedo ayudarte con una evaluación completa de riesgo de cáncer pulmonar o información sobre síntomas, diagnóstico y tratamiento.",
            "Mi especialidad es el cáncer de pulmón. ¿Te interesa realizar una evaluación de riesgo personalizada? Escribe 'evaluar riesgo' para comenzar."
        ]
        return random.choice(default_responses)

    def get_welcome_message(self):
        """Mensaje de bienvenida."""
        model_status = "✅ Con evaluación de riesgo avanzada (ML)" if self.model_data else "⚠️ Evaluación básica disponible"

        return f"""
👋 ¡Hola! Soy tu asistente especializado en cáncer de pulmón.

{model_status}

Puedo ayudarte con:
• 🏥 Información médica sobre cáncer pulmonar
• 🔍 Evaluación de riesgo personalizada (23 preguntas)
• 💡 Respuestas a tus preguntas específicas
• 📊 Análisis de síntomas y factores de riesgo

💬 Comandos disponibles:
• "evaluar riesgo" - Cuestionario completo de evaluación
• "ayuda" - Ver todos los comandos
• Cualquier pregunta sobre cáncer de pulmón

¡Estoy aquí para ayudarte! 😊
"""

    def get_help_message(self):
        """Mensaje de ayuda."""
        return """
🤖 COMANDOS DISPONIBLES

🔍 EVALUACIÓN DE RIESGO:
• "evaluar riesgo" - Cuestionario completo de 23 preguntas
• "test riesgo" - Evaluación personalizada con modelo de ML

💡 EJEMPLOS DE PREGUNTAS:
• "¿Qué es el cáncer de pulmón microcítico?"
• "¿Cuáles son los síntomas tempranos?"
• "¿Cómo se trata el adenocarcinoma?"
• "Factores de riesgo del cáncer de pulmón"
• "Diagnóstico y estadificación"
• "Tratamientos disponibles"

🏥 INFORMACIÓN ESPECÍFICA:
• Tipos de cáncer de pulmón
• Síntomas y detección temprana
• Opciones de tratamiento
• Prevención y factores de riesgo
• Cuidados paliativos

¡Puedes hacer preguntas en tus propias palabras! Soy especialista en cáncer de pulmón.
"""


# Inicializar chatbot
try:
    chatbot = LungHealthChatbot('datasetchatbot_referencias.csv', 'lung_cancer_decision_tree.joblib')
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
        'dataset_size': len(chatbot.qa_data) if chatbot else 0,
        'risk_questions': len(chatbot.risk_questions) if chatbot else 0
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

    logging.info(f"🚀 Iniciando servidor en puerto {port} (debug: {debug_mode})")

    if os.getenv('RENDER'):
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=debug_mode, host='0.0.0.0', port=port)

