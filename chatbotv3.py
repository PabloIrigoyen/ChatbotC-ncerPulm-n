import os
import numpy as np
import pandas as pd
import spacy
import difflib
import logging
import joblib
import torch
import torch.nn as nn
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


# Clase GloVeEmbeddings para cargar los embeddings guardados
class GloVeEmbeddings:

    def __init__(self, embeddings_dict=None):
        self.word_vectors = embeddings_dict if embeddings_dict else {}
        self.embedding_dim = 300

    def get_word_embedding(self, word):
        """Obtener embedding para una palabra"""
        word_lower = word.lower()
        if word_lower in self.word_vectors:
            return self.word_vectors[word_lower]
        else:
            return np.zeros(self.embedding_dim)

    def get_sentence_embedding(self, tokens, method='mean'):
        """Obtener embedding para una oración"""
        if not tokens:
            return np.zeros(self.embedding_dim)

        embeddings = [self.get_word_embedding(token) for token in tokens]

        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            return np.mean(embeddings, axis=0)


# Arquitectura del modelo neuronal
class HighAccuracyLungCancerClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dims=[512, 512, 256, 128], num_classes=4,
                 dropout_rate=0.4, use_batch_norm=True):
        super(HighAccuracyLungCancerClassifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        layers = []
        current_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.output_layer(features)
        return logits


class AdvancedMedicalPreprocessor:
    def __init__(self):
        try:
            self.nlp = spacy.load("es_core_news_sm")
        except OSError:
            print("Descargando modelo spaCy español...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
            self.nlp = spacy.load("es_core_news_sm")

        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('spanish'))
        additional_stopwords = {'vez', 'tal', 'etc', 'qué', 'cómo', 'dónde', 'cuándo', 'porqué', 'cual'}
        self.stop_words.update(additional_stopwords)

        self.medical_terms = {
            'cáncer', 'pulmón', 'síntoma', 'diagnóstico', 'tratamiento', 'prevención',
            'tabaquismo', 'radiografía', 'tomografía', 'biopsia', 'quimioterapia',
            'radioterapia', 'inmunoterapia', 'metástasis', 'tumor', 'célula',
            'epitelial', 'adenocarcinoma', 'carcinoma', 'neoplasia', 'broncoscopia',
            'toracoscopia', 'oncólogo', 'neumólogo', 'patología', 'estadio',
            'pronóstico', 'supervivencia', 'mortalidad', 'incidencia', 'prevalencia',
            'pulmonar', 'respiratorio', 'alveolos', 'bronquios', 'pleura',
            'linfático', 'histología', 'citología', 'biomarcador', 'mutación',
            'microcítico', 'no_microcítico', 'carcinoide', 'neuroendocrino',
            'radón', 'asbesto', 'amianto', 'contaminación', 'fumador_pasivo',
            'toracocentesis', 'mediastinoscopia', 'esputo', 'ganglios_linfáticos',
            'lobectomía', 'neumonectomía', 'criocirugía', 'electrocauterización',
            'pancoast', 'paraneoplásicos', 'antidiurética', 'cushing', 'disfagia',
            'citopatológico', 'percutánea', 'inmunohistoquímica', 'oligometastásico',
            'indiferenciado', 'neuroendocrino', 'pulmonares', 'respiratorios',
            'oncológico', 'clínico', 'médico', 'paciente', 'enfermedad', 'salud',
            'hospital', 'consulta'
        }

    def normalize_text(self, text):
        if not isinstance(text, str) or pd.isna(text):
            return ""

        text = text.lower()
        import re
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        text = re.sub(r'[^\w\sáéíóúñüÁÉÍÓÚÑÜ]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        lemmas = []

        for token in doc:
            if token.text.lower() in self.medical_terms:
                lemmas.append(token.text.lower())
            elif not token.is_stop and not token.is_punct and not token.is_space:
                lemma = token.lemma_.lower().strip()
                if len(lemma) > 1 and lemma not in self.stop_words:
                    lemmas.append(lemma)

        return lemmas

    def preprocess_text(self, text):
        try:
            normalized_text = self.normalize_text(text)
            lemmas = self.lemmatize_text(normalized_text)

            filtered_lemmas = []
            for lemma in lemmas:
                if len(lemma) >= 2 or lemma in self.medical_terms:
                    filtered_lemmas.append(lemma)

            return {
                'original_text': text,
                'normalized_text': normalized_text,
                'tokens': filtered_lemmas,
                'processed_text': ' '.join(filtered_lemmas) if filtered_lemmas else ''
            }
        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return {
                'original_text': text,
                'normalized_text': '',
                'tokens': [],
                'processed_text': ''
            }


class LungHealthChatbot:
    def __init__(self, dataset_path='datasetchatbot_referencias.csv', model_dir='saved_models'):
        # Inicializar atributos primero
        self.neural_model = None
        self.neural_resources = None
        self.model_data = None
        self.qa_data = None
        self.nlp = None

        try:
            self.nlp = self._load_spacy_model()
            self.load_qa_data(dataset_path)
            self._setup_qa_structures()
            self.load_neural_model(model_dir)
            self.load_prediction_model('lung_health_model.joblib')
            self.reset_conversation_state()
            logging.info("Chatbot inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando chatbot: {e}")
            # Asegurarse de que los atributos estén definidos incluso si hay error
            if self.neural_model is None:
                self.neural_model = None
            if self.model_data is None:
                self.model_data = None

    def _load_spacy_model(self):
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

    def load_neural_model(self, model_dir):
        """Cargar modelo neuronal desde los archivos .pkl y .pth"""
        try:
            if not os.path.exists(model_dir):
                logging.warning(f"Directorio de modelo {model_dir} no encontrado.")
                self.neural_model = None
                self.neural_resources = None
                return

            # Cargar embeddings GloVe
            glove_path = os.path.join(model_dir, 'glove_embeddings.pkl')
            if os.path.exists(glove_path):
                glove_embeddings_dict = joblib.load(glove_path)
                embedding_model = GloVeEmbeddings(glove_embeddings_dict)
                logging.info("✅ Embeddings GloVe cargados")
            else:
                logging.warning("Archivo glove_embeddings.pkl no encontrado")
                embedding_model = GloVeEmbeddings()

            # Cargar preprocessor y label_encoder desde otros archivos .pkl
            preprocessor = AdvancedMedicalPreprocessor()
            label_encoder = None

            # Buscar archivos .pkl adicionales
            pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f != 'glove_embeddings.pkl']
            for pkl_file in pkl_files:
                try:
                    loaded_data = joblib.load(os.path.join(model_dir, pkl_file))
                    if hasattr(loaded_data, 'classes_') or (
                            isinstance(loaded_data, dict) and 'classes_' in loaded_data):
                        label_encoder = loaded_data
                        logging.info(f"✅ Label encoder cargado desde {pkl_file}")
                except:
                    continue
            # Crear recursos del modelo
            self.neural_resources = {
                'preprocessor': preprocessor,
                'embedding_model': embedding_model,
                'label_encoder': label_encoder,
                'model_config': {
                    'input_dim': 300,
                    'hidden_dims': [512, 512, 256, 128],
                    'num_classes': 4
                }
            }
            # Cargar modelo neuronal
            model_path = os.path.join(model_dir, 'best_glove_lung_cancer_model.pth')
            if os.path.exists(model_path):
                model_config = self.neural_resources['model_config']
                self.neural_model = HighAccuracyLungCancerClassifier(
                    input_dim=model_config['input_dim'],
                    hidden_dims=model_config['hidden_dims'],
                    num_classes=model_config['num_classes']
                )

                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.neural_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.neural_model.load_state_dict(checkpoint)

                self.neural_model.eval()
                logging.info("✅ Modelo neuronal cargado exitosamente")
            else:
                logging.warning("Archivo best_glove_lung_cancer_model.pth no encontrado")
                self.neural_model = None

        except Exception as e:
            logging.error(f"Error cargando modelo neuronal: {e}")
            self.neural_model = None
            self.neural_resources = None

    def classify_intent_neural(self, text):
        if self.neural_model is None or self.neural_resources is None:
            return None, 0.0

        try:
            preprocessor = self.neural_resources['preprocessor']
            embedding_model = self.neural_resources['embedding_model']
            label_encoder = self.neural_resources['label_encoder']

            processed = preprocessor.preprocess_text(text)
            embedding = embedding_model.get_sentence_embedding(processed['tokens'], method='mean')

            if len(embedding) != 300:
                embedding = np.zeros(300)

            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = self.neural_model(embedding_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predicted_idx = predicted.cpu().numpy()[0]
                confidence_value = confidence.item()

            # Si no hay label_encoder, usar índices básicos
            if label_encoder is not None:
                if hasattr(label_encoder, 'inverse_transform'):
                    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
                else:
                    predicted_label = f"clase_{predicted_idx}"
            else:
                predicted_label = f"clase_{predicted_idx}"

            return predicted_label, confidence_value

        except Exception as e:
            logging.error(f"Error en clasificación neuronal: {e}")
            return None, 0.0

    def load_prediction_model(self, model_path):
        """Cargar modelo .joblib mejorado con manejo robusto"""
        try:
            if not os.path.exists(model_path):
                logging.warning(f"Modelo {model_path} no encontrado. La evaluación de riesgo no estará disponible.")
                self.model_data = None
                return

            # Cargar el modelo joblib
            loaded_data = joblib.load(model_path)

            # Manejar diferentes formatos de modelo joblib
            if isinstance(loaded_data, dict):
                # Si es un diccionario con todos los componentes
                self.model_data = loaded_data
                logging.info(f"✅ Modelo ML cargado desde dict: {len(loaded_data)} componentes")
            else:
                # Si es solo el modelo, crear estructura mínima
                self.model_data = {
                    'model': loaded_data,
                    'scaler': None,
                    'label_encoder': None,
                    'feature_names': None,
                    'mean_accuracy': 'N/A'
                }
                logging.info("✅ Modelo ML básico cargado")

            # Verificar componentes críticos
            if 'model' not in self.model_data:
                logging.error("❌ El modelo joblib no contiene el modelo entrenado")
                self.model_data = None
                return

            # Log de información del modelo
            model_info = f"Modelo: {type(self.model_data['model']).__name__}"
            if hasattr(self.model_data['model'], 'classes_'):
                model_info += f", Clases: {self.model_data['model'].classes_}"
            if 'mean_accuracy' in self.model_data:
                model_info += f", Precisión: {self.model_data.get('mean_accuracy', 'N/A')}"

            logging.info(f"📊 Información del modelo: {model_info}")

        except Exception as e:
            logging.error(f"❌ Error cargando modelo ML: {e}")
            self.model_data = None

    def load_qa_data(self, filename):
        try:
            if not os.path.exists(filename):
                logging.error(f"Archivo {filename} no encontrado")
                # Crear dataset básico si no existe
                self.qa_data = pd.DataFrame({
                    'pregunta': ['hola', 'evaluar riesgo', 'ayuda'],
                    'respuesta': [
                        '¡Hola! Soy tu asistente de salud pulmonar. Puedo ayudarte con información sobre cáncer de pulmón o realizar una evaluación de riesgo.',
                        'Iniciando evaluación de riesgo...',
                        'Puedo ayudarte con información sobre cáncer de pulmón o realizar una evaluación de riesgo personalizada.'
                    ],
                    'intencion': ['saludo', 'evaluacion_riesgo', 'ayuda'],
                    'entidades': ['saludo', 'evaluacion', 'ayuda']
                })
                logging.warning("Dataset básico creado")
                return

            self.qa_data = pd.read_csv(filename)
            logging.info(f"Dataset cargado: {len(self.qa_data)} preguntas")

            for col in ['pregunta', 'respuesta', 'intencion', 'entidades']:
                self.qa_data[col] = self.qa_data[col].astype(str).str.strip()

            self.qa_data['entidades_lista'] = self.qa_data['entidades'].str.split('|')

        except Exception as e:
            logging.error(f"Error cargando dataset: {e}")
            # Crear dataset básico como fallback
            self.qa_data = pd.DataFrame({
                'pregunta': ['hola'],
                'respuesta': ['¡Hola! Soy tu asistente de salud pulmonar.'],
                'intencion': ['saludo'],
                'entidades': ['saludo']
            })

    def _setup_qa_structures(self):
        try:
            self.qa_dict = dict(zip(self.qa_data['pregunta'], self.qa_data['respuesta']))
            self.intent_dict = dict(zip(self.qa_data['pregunta'], self.qa_data['intencion']))

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
            # Estructuras básicas como fallback
            self.qa_dict = {}
            self.intent_dict = {}
            self.entity_to_questions = defaultdict(list)

    def reset_conversation_state(self):
        self.conversation_history = []
        self.risk_assessment_active = False
        self.waiting_for_pdf_confirmation = False
        self.risk_questions = self._initialize_risk_questions()
        self.current_risk_question = 0
        self.risk_responses = {}

    def _initialize_risk_questions(self):
        """CUESTIONARIO MEJORADO con 7 niveles de opciones"""
        return [
            {
                "id": "age",
                "pregunta": "¿Cuál es tu edad? (en años)",
                "tipo": "numero",
                "opciones": None,
                "rango": (0, 100),
                "feature_name": "Age"
            },
            {
                "id": "gender",
                "pregunta": "¿Cuál es tu género?",
                "tipo": "opcion",
                "opciones": ["Masculino", "Femenino"],
                "feature_name": "Gender"
            },
            {
                "id": "air_pollution",
                "pregunta": "¿Cuál es tu nivel de exposición a la contaminación del aire?",
                "tipo": "escala",
                "opciones": [
                    "1 - Muy baja (zona rural sin industria)",
                    "2 - Baja (zona residencial tranquila)",
                    "3 - Moderada (ciudad pequeña)",
                    "4 - Alta (ciudad mediana)",
                    "5 - Muy alta (ciudad grande)",
                    "6 - Extremadamente alta (zona industrial)",
                    "7 - Máxima exposición (trabajo en minería/construcción)"
                ],
                "explicacion": "1 = Sin exposición, 7 = Exposición máxima constante",
                "feature_name": "Air Pollution"
            },
            {
                "id": "alcohol_use",
                "pregunta": "¿Cuál es tu nivel de consumo de alcohol?",
                "tipo": "escala",
                "opciones": [
                    "1 - Nunca consumo",
                    "2 - Muy ocasional (1-2 veces al año)",
                    "3 - Ocasional (1-2 veces al mes)",
                    "4 - Moderado (1-2 veces por semana)",
                    "5 - Regular (3-4 veces por semana)",
                    "6 - Frecuente (5-6 veces por semana)",
                    "7 - Diario (todos los días)"
                ],
                "explicacion": "1 = Nunca, 7 = Consumo diario",
                "feature_name": "Alcohol use"
            },
            {
                "id": "dust_allergy",
                "pregunta": "¿Tienes alergia al polvo?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin alergia",
                    "2 - Muy leve (rara vez molesta)",
                    "3 - Leve (ocasionalmente molesta)",
                    "4 - Moderada (frecuentemente molesta)",
                    "5 - Severa (afecta actividades diarias)",
                    "6 - Muy severa (requiere medicación constante)",
                    "7 - Alergia extrema (hospitalizaciones previas)"
                ],
                "explicacion": "1 = No hay alergia, 7 = Alergia muy severa",
                "feature_name": "Dust Allergy"
            },
            {
                "id": "occupational_hazards",
                "pregunta": "¿Estás expuesto a riesgos ocupacionales (químicos, polvo industrial, etc.)?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin exposición",
                    "2 - Mínima (oficina sin riesgos)",
                    "3 - Baja (trabajo ligero en servicios)",
                    "4 - Moderada (manufactura básica)",
                    "5 - Alta (construcción, industria)",
                    "6 - Muy alta (minería, químicos)",
                    "7 - Exposición extrema (amianto, asbestos, radiación)"
                ],
                "explicacion": "1 = Sin exposición laboral, 7 = Exposición constante a carcinógenos",
                "feature_name": "OccuPational Hazards"
            },
            {
                "id": "genetic_risk",
                "pregunta": "¿Tienes antecedentes familiares de cáncer de pulmón?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin antecedentes",
                    "2 - Muy bajo riesgo (familiar muy lejano)",
                    "3 - Bajo riesgo (primo/tío)",
                    "4 - Moderado (abuelos)",
                    "5 - Alto (padres/hermanos)",
                    "6 - Muy alto (múltiples familiares cercanos)",
                    "7 - Riesgo genético confirmado (síndromes hereditarios)"
                ],
                "explicacion": "1 = Sin antecedentes familiares, 7 = Historia familiar muy fuerte",
                "feature_name": "Genetic Risk"
            },
            {
                "id": "chronic_lung_disease",
                "pregunta": "¿Tienes enfermedad pulmonar crónica (EPOC, asma, etc.)?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin enfermedad",
                    "2 - Muy leve (asma ocasional)",
                    "3 - Leve (asma controlada)",
                    "4 - Moderada (EPOC leve, asma persistente)",
                    "5 - Severa (EPOC moderada, limitación actividades)",
                    "6 - Muy severa (EPOC severa, oxígeno ocasional)",
                    "7 - Enfermedad pulmonar avanzada (oxígeno permanente)"
                ],
                "explicacion": "1 = Sin enfermedad pulmonar, 7 = Enfermedad pulmonar crónica avanzada",
                "feature_name": "chronic Lung Disease"
            },
            {
                "id": "balanced_diet",
                "pregunta": "¿Cómo calificarías tu dieta balanceada?",
                "tipo": "escala_inversa",
                "opciones": [
                    "1 - Muy balanceada (frutas/verduras diarias, proteínas magras)",
                    "2 - Balanceada (dieta mediterránea)",
                    "3 - Moderadamente balanceada (ocasional comida procesada)",
                    "4 - Neutral (mezcla de saludable y no saludable)",
                    "5 - Poco balanceada (frecuente comida procesada)",
                    "6 - Desbalanceada (predominio comida rápida)",
                    "7 - Muy desbalanceada (dieta principalmente procesada)"
                ],
                "explicacion": "1 = Dieta muy saludable, 7 = Dieta muy poco saludable",
                "feature_name": "Balanced Diet"
            },
            {
                "id": "obesity",
                "pregunta": "¿Cuál es tu nivel de obesidad/sobrepeso?",
                "tipo": "escala",
                "opciones": [
                    "1 - Peso normal (IMC 18.5-24.9)",
                    "2 - Ligero sobrepeso (IMC 25-26.9)",
                    "3 - Sobrepeso (IMC 27-29.9)",
                    "4 - Obesidad grado I (IMC 30-34.9)",
                    "5 - Obesidad grado II (IMC 35-39.9)",
                    "6 - Obesidad grado III (IMC 40-49.9)",
                    "7 - Obesidad mórbida (IMC 50+)"
                ],
                "explicacion": "1 = Peso saludable, 7 = Obesidad severa",
                "feature_name": "Obesity"
            },
            {
                "id": "smoking",
                "pregunta": "¿Cuál es tu nivel de tabaquismo?",
                "tipo": "escala",
                "opciones": [
                    "1 - Nunca he fumado",
                    "2 - Ex-fumador leve (menos de 5 años)",
                    "3 - Ex-fumador moderado (5-10 años)",
                    "4 - Fumador ocasional (menos de 5 cigarrillos/día)",
                    "5 - Fumador regular (5-15 cigarrillos/día)",
                    "6 - Fumador frecuente (16-25 cigarrillos/día)",
                    "7 - Fumador empedernido (más de 25 cigarrillos/día)"
                ],
                "explicacion": "1 = No fumador, 7 = Fumador muy intenso",
                "feature_name": "Smoking"
            },
            {
                "id": "passive_smoker",
                "pregunta": "¿Estás expuesto al humo de segunda mano (fumador pasivo)?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin exposición",
                    "2 - Exposición mínima (ocasional en público)",
                    "3 - Exposición ocasional (1-2 veces/semana)",
                    "4 - Exposición regular (3-4 veces/semana)",
                    "5 - Exposición frecuente (diaria en trabajo/transporte)",
                    "6 - Exposición muy frecuente (varias horas diarias)",
                    "7 - Exposición constante (vive con fumador)"
                ],
                "explicacion": "1 = Sin exposición al humo, 7 = Exposición constante al humo",
                "feature_name": "Passive Smoker"
            },
            {
                "id": "chest_pain",
                "pregunta": "¿Experimentas dolor en el pecho?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin dolor",
                    "2 - Muy leve (raramente, sin molestia)",
                    "3 - Leve (ocasional, mínima molestia)",
                    "4 - Moderado (frecuente, molesto)",
                    "5 - Moderado-severo (diario, afecta actividades)",
                    "6 - Severo (constante, limita movimiento)",
                    "7 - Muy severo (dolor incapacitante)"
                ],
                "explicacion": "1 = Sin dolor torácico, 7 = Dolor muy intenso",
                "feature_name": "Chest Pain"
            },
            {
                "id": "coughing_blood",
                "pregunta": "¿Has tosido con sangre?",
                "tipo": "escala",
                "opciones": [
                    "1 - Nunca",
                    "2 - Una vez hace mucho tiempo",
                    "3 - Ocasionalmente en el pasado",
                    "4 - Recientemente (últimos meses)",
                    "5 - Frecuentemente (varias veces al mes)",
                    "6 - Muy frecuentemente (semanal)",
                    "7 - Constantemente (diario o casi diario)"
                ],
                "explicacion": "1 = Sin tos con sangre, 7 = Hemoptisis frecuente",
                "feature_name": "Coughing of Blood"
            },
            {
                "id": "fatigue",
                "pregunta": "¿Experimentas fatiga o cansancio?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin fatiga",
                    "2 - Muy leve (ocasional al final del día)",
                    "3 - Leve (frecuente al final del día)",
                    "4 - Moderada (interfiere con actividades)",
                    "5 - Moderada-severa (limita actividades diarias)",
                    "6 - Severa (dificulta trabajo/quehaceres)",
                    "7 - Muy severa (incapacitante, reposo necesario)"
                ],
                "explicacion": "1 = Sin fatiga, 7 = Fatiga que limita actividades",
                "feature_name": "Fatigue"
            },
            {
                "id": "weight_loss",
                "pregunta": "¿Has experimentado pérdida de peso inexplicable?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin pérdida",
                    "2 - Muy poca (1-2 kg sin explicación)",
                    "3 - Leve (3-4 kg sin explicación)",
                    "4 - Moderada (5-7 kg sin explicación)",
                    "5 - Significativa (8-10 kg sin explicación)",
                    "6 - Muy significativa (11-15 kg sin explicación)",
                    "7 - Pérdida extrema (más de 15 kg sin explicación)"
                ],
                "explicacion": "1 = Sin pérdida de peso, 7 = Pérdida de peso muy severa",
                "feature_name": "Weight Loss"
            },
            {
                "id": "shortness_breath",
                "pregunta": "¿Tienes dificultad para respirar?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin dificultad",
                    "2 - Solo con ejercicio intenso",
                    "3 - Con ejercicio moderado",
                    "4 - Con actividades diarias ligeras",
                    "5 - Al caminar plano a paso normal",
                    "6 - En reposo ocasionalmente",
                    "7 - En reposo constantemente"
                ],
                "explicacion": "1 = Sin dificultad respiratoria, 7 = Dificultad respiratoria extrema",
                "feature_name": "Shortness of Breath"
            },
            {
                "id": "wheezing",
                "pregunta": "¿Experimentas silbidos al respirar (sibilancias)?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin silbidos",
                    "2 - Muy leves (solo con resfriados)",
                    "3 - Leves (ocasionalmente)",
                    "4 - Moderados (frecuentemente)",
                    "5 - Moderados-severos (diarios)",
                    "6 - Severos (constantes, molestos)",
                    "7 - Muy severos (audibles, afectan sueño)"
                ],
                "explicacion": "1 = Sin sibilancias, 7 = Sibilancias constantes",
                "feature_name": "Wheezing"
            },
            {
                "id": "swallowing_difficulty",
                "pregunta": "¿Tienes dificultad para tragar?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin dificultad",
                    "2 - Muy leve (ocasional con alimentos secos)",
                    "3 - Leve (frecuente con alimentos secos)",
                    "4 - Moderada (con alimentos normales)",
                    "5 - Moderada-severa (con líquidos)",
                    "6 - Severa (dificultad constante)",
                    "7 - Muy severa (imposible tragar)"
                ],
                "explicacion": "1 = Sin dificultad para tragar, 7 = Dificultad severa",
                "feature_name": "Swallowing Difficulty"
            },
            {
                "id": "clubbing_finger_nails",
                "pregunta": "¿Has notado cambios en las uñas (engrosamiento, forma de palillo de tambor)?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin cambios",
                    "2 - Muy leves (ligero redondeamiento)",
                    "3 - Leves (redondeamiento notable)",
                    "4 - Moderados (engrosamiento visible)",
                    "5 - Moderados-severos (cambio forma claro)",
                    "6 - Severos (deformación evidente)",
                    "7 - Muy severos (acropaquia avanzada)"
                ],
                "explicacion": "1 = Uñas normales, 7 = Acropaquia severa",
                "feature_name": "Clubbing of Finger Nails"
            },
            {
                "id": "frequent_cold",
                "pregunta": "¿Tienes resfriados frecuentes?",
                "tipo": "escala",
                "opciones": [
                    "1 - Muy raros (menos de 1 al año)",
                    "2 - Raros (1-2 al año)",
                    "3 - Ocasionales (3-4 al año)",
                    "4 - Regulares (5-6 al año)",
                    "5 - Frecuentes (7-8 al año)",
                    "6 - Muy frecuentes (9-10 al año)",
                    "7 - Constantes (más de 10 al año)"
                ],
                "explicacion": "1 = Resfriados muy infrecuentes, 7 = Resfriados constantes",
                "feature_name": "Frequent Cold"
            },
            {
                "id": "dry_cough",
                "pregunta": "¿Tienes tos seca?",
                "tipo": "escala",
                "opciones": [
                    "1 - Sin tos seca",
                    "2 - Muy ocasional (1-2 veces al mes)",
                    "3 - Ocasional (1-2 veces por semana)",
                    "4 - Regular (3-4 veces por semana)",
                    "5 - Frecuente (diaria leve)",
                    "6 - Muy frecuente (varias veces al día)",
                    "7 - Tos seca constante (persistente todo el día)"
                ],
                "explicacion": "1 = Sin tos seca, 7 = Tos seca constante",
                "feature_name": "Dry Cough"
            },
            {
                "id": "snoring",
                "pregunta": "¿Roncas al dormir?",
                "tipo": "escala",
                "opciones": [
                    "1 - Nunca",
                    "2 - Muy raramente (ocasional con resfriado)",
                    "3 - Ocasionalmente (1-2 veces por semana)",
                    "4 - Regularmente (3-4 veces por semana)",
                    "5 - Frecuentemente (5-6 veces por semana)",
                    "6 - Muy frecuentemente (casi todas las noches)",
                    "7 - Siempre (todas las noches, fuerte)"
                ],
                "explicacion": "1 = Sin ronquidos, 7 = Ronquidos constantes",
                "feature_name": "Snoring"
            }
        ]

    def start_risk_assessment(self):
        """Iniciar evaluación de riesgo mejorada"""
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
            "explicacion": first_question.get("explicacion", ""),
            "progress": f"1/{len(self.risk_questions)}",
            "total_questions": len(self.risk_questions)
        }

    def process_risk_response(self, response):
        """Procesar respuesta del cuestionario mejorado"""
        if not self.risk_assessment_active:
            return {"error": "No hay evaluación de riesgo activa"}

        current_q = self.risk_questions[self.current_risk_question]

        # Validar respuesta
        validated_response = self._validate_response(response, current_q)
        if not validated_response:
            return {
                "type": "validation_error",
                "message": f"Respuesta inválida. Por favor ingresa una opción válida.",
                "options": current_q["opciones"]
            }

        self.risk_responses[current_q["id"]] = validated_response

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
                "explicacion": next_q.get("explicacion", ""),
                "progress": f"{self.current_risk_question + 1}/{len(self.risk_questions)}",
                "total_questions": len(self.risk_questions)
            }
        else:
            # Completar evaluación
            return self._complete_risk_assessment()

    def _validate_response(self, response, question):
        """Validar respuesta según el tipo de pregunta - MEJORADO"""
        try:
            response_clean = str(response).strip().lower()

            if question["tipo"] == "numero":
                value = int(response_clean)
                if "rango" in question:
                    min_val, max_val = question["rango"]
                    if min_val <= value <= max_val:
                        return value
                else:
                    return value

            elif question["tipo"] in ["opcion", "escala", "escala_inversa"]:
                # Buscar coincidencia exacta en opciones
                for i, option in enumerate(question["opciones"]):
                    option_lower = option.lower()
                    # Buscar por número
                    if response_clean == str(i + 1):
                        return i + 1
                    # Buscar por texto
                    if response_clean in option_lower:
                        return i + 1
                    # Buscar por palabras clave
                    option_words = option_lower.split()
                    for word in option_words:
                        if word.isdigit() and response_clean == word:
                            return int(word)
                        if len(word) > 3 and word in response_clean:
                            return i + 1

                # Si es género, manejar específicamente
                if question["id"] == "gender":
                    if any(word in response_clean for word in ['hombre', 'masculino', 'varón', 'hombre']):
                        return 1  # Masculino
                    elif any(word in response_clean for word in ['mujer', 'femenino', 'hembra', 'mujer']):
                        return 2  # Femenino
                    else:
                        # Por defecto para género no reconocido
                        return 1

            return None

        except Exception as e:
            logging.error(f"Error en validación: {e}")
            return None

    def _complete_risk_assessment(self):
        """Completar evaluación de riesgo usando el modelo joblib"""
        self.risk_assessment_active = False

        # Procesar respuestas para el modelo
        processed_responses = self._process_responses_for_joblib_model()

        # Usar el modelo joblib para predicción
        risk_prediction = self._predict_with_joblib_model(processed_responses)

        # Guardar en historial
        self.conversation_history.append({
            'timestamp': datetime.now(),
            'type': 'risk_assessment_complete',
            'risk_result': risk_prediction
        })

        return risk_prediction

    def _process_responses_for_joblib_model(self):
        """Procesar respuestas para el formato del modelo joblib"""
        processed = {}

        # Mapeo directo usando feature_name de cada pregunta
        for question in self.risk_questions:
            q_id = question["id"]
            feature_name = question.get("feature_name")

            if feature_name and q_id in self.risk_responses:
                response_value = self.risk_responses[q_id]

                # Conversión específica para género
                if q_id == 'gender':
                    processed[feature_name] = 1 if response_value == 1 else 2  # 1=Masculino, 2=Femenino
                else:
                    # Para escalas de 7 niveles, mantener el valor original
                    processed[feature_name] = response_value

        # Log para debugging
        logging.info(f"Respuestas procesadas para joblib: {processed}")
        return processed

    def _predict_with_joblib_model(self, responses):
        """Predicción usando el modelo joblib cargado - CORREGIDO"""
        try:
            # Verificar que el modelo esté cargado correctamente
            if not self.model_data or 'model' not in self.model_data:
                logging.warning("Modelo joblib no disponible, usando cálculo de respaldo")
                return self._calculate_fallback_risk(responses)

            model = self.model_data['model']
            scaler = self.model_data.get('scaler')
            label_encoder = self.model_data.get('label_encoder')
            feature_names = self.model_data.get('feature_names')

            logging.info(f"Usando modelo: {type(model).__name__}")
            logging.info(f"Características esperadas: {feature_names}")

            # DEFINIR el orden correcto de características basado en el dataset original
            expected_features = [
                'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
                'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
                'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
                'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
            ]

            # Construir el vector de características en el orden correcto
            model_input = []
            missing_features = []

            for feature in expected_features:
                if feature in responses:
                    value = responses[feature]
                    # Asegurar que el valor sea numérico
                    if value is None:
                        model_input.append(1)  # Valor por defecto bajo
                        missing_features.append(feature)
                    else:
                        model_input.append(float(value))
                else:
                    model_input.append(1)  # Valor por defecto bajo
                    missing_features.append(feature)

            if missing_features:
                logging.warning(f"Características faltantes con valores por defecto: {missing_features}")

            # Convertir a numpy array y crear DataFrame con nombres de características
            model_input_array = np.array([model_input])

            # CREAR DATAFRAME con nombres de características para evitar el warning
            if feature_names is not None:
                # Usar los nombres de características del modelo si están disponibles
                input_df = pd.DataFrame(model_input_array, columns=feature_names)
                logging.info(f"Usando nombres de características del modelo: {feature_names}")
            else:
                # Usar los nombres esperados por defecto
                input_df = pd.DataFrame(model_input_array, columns=expected_features)
                logging.info(f"Usando nombres de características por defecto: {expected_features}")

            logging.info(f"Input para modelo joblib: {model_input_array[0]}")

            # Escalar los datos si hay scaler disponible
            if scaler is not None:
                try:
                    scaled_input = scaler.transform(input_df)
                    logging.info("Datos escalados correctamente")
                except Exception as e:
                    logging.error(f"Error escalando datos: {e}")
                    # Fallback: usar datos sin escalar
                    scaled_input = model_input_array
                    logging.info("Usando datos sin escalar debido a error")
            else:
                scaled_input = model_input_array
                logging.info("Usando datos sin escalar (no hay scaler)")

            # Realizar predicción
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(scaled_input)[0]
                prediction = model.predict(scaled_input)[0]
                confidence = max(probabilities)

                logging.info(f"Predicción: {prediction}, Probabilidades: {probabilities}")
            else:
                prediction = model.predict(scaled_input)[0]
                probabilities = [0.33, 0.33, 0.34]  # Placeholder
                confidence = 0.5
                logging.info(f"Predicción sin probabilidades: {prediction}")

            # Determinar nivel de riesgo
            if label_encoder is not None:
                try:
                    risk_level = label_encoder.inverse_transform([prediction])[0]
                    logging.info(f"Nivel de riesgo decodificado: {risk_level}")
                except Exception as e:
                    logging.error(f"Error decodificando etiqueta: {e}")
                    risk_level = self._map_risk_level_from_label(risk_level)
            else:
                risk_level = self._map_risk_level(prediction)

            # Convertir "Low" a "BAJO" para consistencia en español
            risk_level_spanish = self._translate_risk_level(risk_level)

            return {
                "type": "risk_assessment_result",
                "risk_level": risk_level_spanish,
                "confidence": f"{confidence:.1%}",
                "probability_high": f"{probabilities[2]:.1%}" if len(probabilities) > 2 else "N/A",
                "probability_medium": f"{probabilities[1]:.1%}" if len(probabilities) > 1 else "N/A",
                "probability_low": f"{probabilities[0]:.1%}",
                "recommendations": self._get_recommendations_based_on_risk(risk_level_spanish, responses),
                "risk_factors": self._identify_risk_factors(responses),
                "based_on_ml": True,
                "model_type": type(model).__name__
            }

        except Exception as e:
            logging.error(f"Error crítico en predicción joblib: {e}")
            logging.error(f"Respuestas recibidas: {responses}")
            # Fallback a cálculo básico
            return self._calculate_fallback_risk(responses)

    def _translate_risk_level(self, risk_level):
        """Traducir nivel de riesgo a español"""
        translation_map = {
            'Low': 'BAJO',
            'Medium': 'MEDIO',
            'High': 'ALTO',
            'LOW': 'BAJO',
            'MEDIUM': 'MEDIO',
            'HIGH': 'ALTO'
        }
        return translation_map.get(risk_level, risk_level)

    def _map_risk_level_from_label(self, label):
        """Mapear etiqueta de riesgo basado en el texto"""
        label_str = str(label).lower()
        if 'low' in label_str:
            return 'BAJO'
        elif 'medium' in label_str or 'medio' in label_str:
            return 'MEDIO'
        elif 'high' in label_str or 'alto' in label_str:
            return 'ALTO'
        else:
            return 'MEDIO'

    def _map_risk_level(self, prediction):
        """Mapear predicción numérica a nivel de riesgo"""
        risk_mapping = {
            0: "BAJO",
            1: "MEDIO",
            2: "ALTO"
        }
        return risk_mapping.get(prediction, "MEDIO")

    def _calculate_fallback_risk(self, responses):
        """Cálculo de respaldo cuando el modelo joblib falla"""
        try:
            # Factores clave con pesos
            key_factors = {
                'Smoking': 3.0,
                'Coughing of Blood': 2.5,
                'Genetic Risk': 1.5,
                'chronic Lung Disease': 1.6,
                'Air Pollution': 1.2,
                'Occupational Hazards': 1.8
            }

            total_score = 0
            max_score = 0

            for factor, weight in key_factors.items():
                if factor in responses:
                    value = responses[factor]
                    total_score += (value / 7.0) * weight  # Normalizar a 0-1 (7 niveles)
                    max_score += weight

            risk_percentage = (total_score / max_score) * 100 if max_score > 0 else 0

            # Umbrales ajustados para mejor distribución
            if risk_percentage < 20:
                risk_level = "BAJO"
            elif risk_percentage < 50:
                risk_level = "MEDIO"
            else:
                risk_level = "ALTO"

            return {
                "type": "risk_assessment_result",
                "risk_level": risk_level,
                "risk_score": f"{risk_percentage:.1f}%",
                "confidence": "Estimación basada en factores clave",
                "recommendations": self._get_recommendations_based_on_risk(risk_level, responses),
                "risk_factors": self._identify_risk_factors(responses),
                "based_on_ml": False
            }

        except Exception as e:
            logging.error(f"Error en cálculo de respaldo: {e}")
            return {
                "type": "risk_assessment_result",
                "risk_level": "MEDIO",
                "risk_score": "50%",
                "confidence": "Estimación básica por fallo técnico",
                "recommendations": [
                    "Consulta con un especialista en neumología",
                    "Realiza controles médicos periódicos",
                    "Mantén un estilo de vida saludable"
                ],
                "risk_factors": [],
                "based_on_ml": False
            }

    def _identify_risk_factors(self, responses):
        """Identificar factores de riesgo específicos"""
        risk_factors = []

        high_risk_threshold = 5  # Valor >= 5 se considera alto riesgo (en escala 1-7)

        risk_mapping = {
            'Smoking': 'Tabaquismo',
            'Air Pollution': 'Contaminación del aire',
            'Genetic Risk': 'Riesgo genético',
            'Coughing of Blood': 'Tos con sangre',
            'Shortness of Breath': 'Dificultad respiratoria',
            'Occupational Hazards': 'Riesgos laborales',
            'chronic Lung Disease': 'Enfermedad pulmonar crónica',
            'Chest Pain': 'Dolor en el pecho'
        }

        for factor, spanish_name in risk_mapping.items():
            if factor in responses and responses[factor] >= high_risk_threshold:
                risk_factors.append({
                    'factor': spanish_name,
                    'level': responses[factor],
                    'severity': 'ALTO' if responses[factor] >= 6 else 'MODERADO'
                })

        return risk_factors

    def _get_recommendations_based_on_risk(self, risk_level, responses):
        """Recomendaciones basadas en el nivel de riesgo"""
        base_recommendations = [
            "Consulta con un especialista en neumología",
            "Realiza controles médicos periódicos",
            "Mantén un estilo de vida saludable",
            "Evita la exposición a contaminantes ambientales"
        ]

        specific_recs = []

        if risk_level == "ALTO":
            specific_recs = [
                "🚨 Consulta médica URGENTE con neumólogo",
                "💊 Considera realizar una tomografía computarizada de tórax",
                "🚭 Suspende el tabaquismo inmediatamente si fumas",
                "🏥 Programa evaluación pulmonar completa",
                "🔍 Monitorea síntomas regularmente",
                "📋 Realiza seguimiento médico cada 3-6 meses"
            ]
        elif risk_level == "MEDIO":
            specific_recs = [
                "📅 Consulta médica programada con neumólogo",
                "🔍 Considera radiografía de tórax anual",
                "🌱 Reduce factores de riesgo identificados",
                "💪 Implementa programa de ejercicio regular",
                "🥗 Mejora hábitos alimenticios",
                "📊 Monitorea síntomas mensualmente"
            ]
        else:  # BAJO
            specific_recs = [
                "👍 Mantén hábitos saludables actuales",
                "🚭 Evita exposición al humo de tabaco",
                "💪 Realiza ejercicio regularmente",
                "🥗 Sigue una dieta balanceada rica en antioxidantes",
                "🌳 Reduce exposición a contaminantes ambientales",
                "📝 Realiza chequeo médico anual preventivo"
            ]

        # Recomendaciones específicas basadas en factores de riesgo
        if 'Smoking' in responses and responses['Smoking'] >= 4:
            specific_recs.append("🎯 Programa de cesación tabáquica - Busca ayuda profesional para dejar de fumar")

        if 'Coughing of Blood' in responses and responses['Coughing of Blood'] >= 3:
            specific_recs.append("🩺 EVALUACIÓN URGENTE - La tos con sangre requiere atención inmediata")

        if 'Genetic Risk' in responses and responses['Genetic Risk'] >= 4:
            specific_recs.append("🧬 CONSEJO GENÉTICO - Considera asesoramiento genético especializado")

        if 'Occupational Hazards' in responses and responses['Occupational Hazards'] >= 5:
            specific_recs.append("🏭 PROTECCIÓN LABORAL - Usa equipo de protección respiratoria")

        return base_recommendations + specific_recs

    # ... (los métodos restantes de process_message se mantienen igual) ...

    def process_message(self, message):
        """Procesamiento mejorado sin comandos que interfieran"""
        try:
            if not message or not isinstance(message, str):
                return "Por favor escribe un mensaje válido."

            message = message.strip()
            if not message:
                return "Por favor escribe un mensaje válido."

            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_message': message,
                'type': 'user'
            })

            lower_message = message.lower()

            # SOLO comandos esenciales que no interfieren con búsqueda
            if any(cmd in lower_message for cmd in ['evaluar riesgo', 'test riesgo', 'cuestionario', 'evaluación']):
                risk_start = self.start_risk_assessment()
                response = risk_start
                response['bot_response'] = f"🔍 **{risk_start['question']}**"
                if risk_start['options']:
                    response['bot_response'] += f"\n\n📋 Opciones:\n" + "\n".join(
                        [f"• {opt}" for opt in risk_start['options']])
                if risk_start.get('explicacion'):
                    response['bot_response'] += f"\n\n💡 {risk_start['explicacion']}"
                response['bot_response'] += f"\n\n📊 Progreso: {risk_start['progress']}"

            elif self.risk_assessment_active:
                risk_response = self.process_risk_response(message)

                if risk_response.get('type') == 'validation_error':
                    response = risk_response
                    response['bot_response'] = f"❌ {risk_response['message']}"
                    if risk_response['options']:
                        response['bot_response'] += f"\n\n📋 Opciones válidas:\n" + "\n".join(
                            [f"• {opt}" for opt in risk_response['options']])

                elif 'question' in risk_response:
                    response = risk_response
                    response['bot_response'] = f"🔍 **{risk_response['question']}**"
                    if risk_response['options']:
                        response['bot_response'] += f"\n\n📋 Opciones:\n" + "\n".join(
                            [f"• {opt}" for opt in risk_response['options']])
                    if risk_response.get('explicacion'):
                        response['bot_response'] += f"\n\n💡 {risk_response['explicacion']}"
                    response['bot_response'] += f"\n\n📊 Progreso: {risk_response['progress']}"
                else:
                    # MEJORA: Verificar que risk_response tenga la estructura correcta
                    if isinstance(risk_response, dict) and 'risk_level' in risk_response:
                        result = risk_response
                        risk_emoji = "🔴" if result['risk_level'] == "ALTO" else "🟡" if result[
                                                                                           'risk_level'] == "MEDIO" else "🟢"

                        response_text = f"""{risk_emoji} EVALUACIÓN COMPLETADA - RESULTADOS

📈 Nivel de riesgo: {result['risk_level']}
🎯 Confianza: {result['confidence']}"""

                        response_text += "\n\n⚠️ Factores de riesgo identificados:"

                        if result.get('risk_factors'):
                            for factor in result['risk_factors']:
                                response_text += f"\n• {factor['factor']} (Nivel {factor['level']} - {factor['severity']})"
                        else:
                            response_text += "\n• No se identificaron factores de riesgo significativos"

                        response_text += f"\n\n📋 RECOMENDACIONES:"
                        for rec in result['recommendations']:
                            response_text += f"\n• {rec}"

                        response_text += "\n\n💡 _Esta evaluación es informativa. Consulta siempre con un profesional de la salud._"

                        if result.get('based_on_ml'):
                            response_text += f"\n\n🤖 _Evaluación generada por modelo de IA ({result.get('model_type', 'ML')})_"

                        response = {'bot_response': response_text}
                    else:
                        # MEJORA: Manejar caso donde risk_response no tiene la estructura esperada
                        logging.warning(f"Estructura inesperada en risk_response: {risk_response}")
                        response = {
                            'bot_response': "✅ Evaluación completada. Sin embargo, hubo un problema al generar el reporte detallado. Los datos han sido registrados para análisis."}

            elif any(cmd in lower_message for cmd in ['hola', 'hi', 'buenos días', 'buenas']):
                response = {'bot_response': self.get_welcome_message()}
            elif any(cmd in lower_message for cmd in ['gracias', 'grax', 'thnak you', 'muchas gracias']):
                response = {
                    'bot_response': 'De nada cualquier, estoy para responder cualquier otra consulta sobre cáncer de pulmón'}
            elif any(cmd in lower_message for cmd in ['adiós', 'bye', 'hasta luego', 'nos vemos']):
                response = {'bot_response': 'Hasta la proximá consulta'}
            elif any(cmd in lower_message for cmd in ['ayuda', 'comandos', 'qué puedes hacer']):
                response = {'bot_response': self.get_help_message()}
            else:
                # BÚSQUEDA INTELIGENTE SIN INTERFERENCIAS
                match = self.find_best_match(message)
                if match:
                    if match['tipo'].startswith('neuronal'):
                        confidence_info = f" (Confianza neuronal: {match.get('confidence', 'N/A')})"
                    else:
                        confidence_info = ""

                    response = {'bot_response': match['respuesta'] + confidence_info}
                    logging.info(f"Match encontrado: {match['tipo']} (score: {match['score']:.2f})")
                else:
                    response = {'bot_response': self._get_default_response()}

            if 'bot_response' in response:
                self.conversation_history.append({
                    'timestamp': datetime.now(),
                    'bot_response': response['bot_response'],
                    'type': 'bot'
                })

            return response

        except Exception as e:
            logging.error(f"Error en process_message: {e}")
            # MEJORA: Mensaje de error más específico
            error_msg = f"Lo siento, hubo un error procesando tu mensaje: {str(e)}. Por favor intenta de nuevo o escribe 'evaluar riesgo' para comenzar una nueva evaluación."
            return {'bot_response': error_msg}

    def _get_default_response(self):
        """Respuesta por defecto mejorada"""
        default_responses = [
            "No encontré información específica sobre tu consulta en mi base de datos. ¿Te gustaría realizar una evaluación de riesgo completa? Escribe 'evaluar riesgo' para comenzar.",
            "Sobre ese tema específico no tengo información detallada. Puedo ayudarte con una evaluación de riesgo personalizada o puedes intentar reformular tu pregunta.",
            "Mi especialidad es el cáncer de pulmón y la evaluación de riesgo. ¿Te interesa saber más sobre algún aspecto específico o prefieres una evaluación personalizada?"
        ]
        return random.choice(default_responses)

    def get_welcome_message(self):
        # Verificar de forma segura si los modelos están cargados
        model_loaded = hasattr(self, 'model_data') and self.model_data is not None
        neural_loaded = hasattr(self, 'neural_model') and self.neural_model is not None

        model_status = "✅ Con evaluación de riesgo avanzada (joblib)" if model_loaded else "⚠️ Evaluación básica disponible"
        neural_status = "🧠 Con inteligencia neuronal avanzada" if neural_loaded else ""

        return f"""
👋 ¡Hola! Soy tu asistente especializado en Cáncer de Pulmón.

{model_status}
{neural_status}

🏥 CUESTIONARIO MEJORADO DISPONIBLE:
• 23 preguntas completas con 7 niveles de detalle
• Evaluación con modelo de Machine Learning
• Resultados precisos con recomendaciones personalizadas

💬 Puedo ayudarte con:
• Información específica sobre cáncer de pulmón
• Evaluación de riesgo personalizada usando IA
• Respuestas a preguntas médicas específicas

¡Escribe tu pregunta o escribe 'evaluar riesgo' para comenzar! 😊
"""

    def get_help_message(self):
        return """
🤖 CÓMO PUEDO AYUDARTE

🔍 EVALUACIÓN DE RIESGO:
• "evaluar riesgo" - Cuestionario completo de 23 preguntas
• Evaluación personalizada basada en modelo de ML
• Resultados detallados con recomendaciones

💡 INFORMACIÓN ESPECÍFICA:
Puedes preguntarme sobre cualquier aspecto del cáncer de pulmón:

• Síntomas y detección:
  "síntomas tempranos", "señales de alerta", "detección precoz"

• Diagnóstico:
  "pruebas diagnósticas", "biopsia pulmonar", "estadificación"

• Tratamiento:
  "opciones de tratamiento", "quimioterapia", "cirugía pulmonar"

• Factores de riesgo:
  "tabaquismo y cáncer", "factores ambientales", "genética"

📝 EJEMPLOS:
• "¿Qué es el cáncer de pulmón de células pequeñas?"
• "¿Cómo afecta el tabaquismo al riesgo?"
• "¿Cuáles son los tratamientos más modernos?"

¡Pregúntame lo que necesites saber! 🎯
"""

    # ... (métodos de búsqueda y entidades se mantienen igual) ...

    def _extract_entities(self, query):
        """Extraer entidades mejorado con búsqueda más inteligente"""
        try:
            query_lower = query.lower().strip()
            found_entities = set()

            all_entities = set()
            for entities in self.qa_data['entidades_lista']:
                if isinstance(entities, list):
                    for entity in entities:
                        if entity and isinstance(entity, str):
                            all_entities.add(entity.strip().lower())

            # Búsqueda más flexible por palabras clave
            for entity in all_entities:
                # Buscar coincidencias parciales
                if entity in query_lower:
                    found_entities.add(entity)
                else:
                    # Buscar por palabras individuales
                    entity_words = entity.split()
                    if len(entity_words) > 1:
                        # Si la entidad tiene múltiples palabras, buscar coincidencias parciales
                        for word in entity_words:
                            if len(word) > 3 and word in query_lower:
                                found_entities.add(entity)
                                break

            return list(found_entities)

        except Exception as e:
            logging.error(f"Error extrayendo entidades: {e}")
            return []

    def _improve_similarity_search(self, query, question):
        """Búsqueda de similitud mejorada"""
        query_lower = query.lower()
        question_lower = question.lower()

        # Similitud básica
        basic_similarity = difflib.SequenceMatcher(None, query_lower, question_lower).ratio()

        # Bonus por coincidencias de palabras clave médicas
        medical_keywords = ['cáncer', 'pulmón', 'síntoma', 'diagnóstico', 'tratamiento',
                            'tabaco', 'fumar', 'tos', 'dolor', 'respirar']

        keyword_bonus = 0
        for keyword in medical_keywords:
            if keyword in query_lower and keyword in question_lower:
                keyword_bonus += 0.1

        return min(1.0, basic_similarity + keyword_bonus)

    def find_best_match(self, query):
        """Búsqueda mejorada sin interferencia de comandos genéricos"""
        try:
            query_lower = query.lower().strip()

            # 1. Primero intentar clasificación neuronal si está disponible
            if self.neural_model is not None:
                predicted_intent, confidence = self.classify_intent_neural(query)
                if predicted_intent and confidence > 0.7:
                    # Buscar preguntas con esta intención
                    intent_matches = self.qa_data[self.qa_data['intencion'] == predicted_intent]
                    if len(intent_matches) > 0:
                        best_match = intent_matches.iloc[0]
                        return {
                            "pregunta": best_match['pregunta'],
                            "respuesta": best_match['respuesta'],
                            "score": confidence,
                            "tipo": f"neuronal_{predicted_intent}",
                            "confidence": f"{confidence:.2f}"
                        }

            # 2. Búsqueda exacta
            for i, question in enumerate(self.qa_data['pregunta']):
                if query_lower == question.lower():
                    return {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": 1.0,
                        "tipo": "exacta"
                    }

            # 3. Búsqueda por similitud de texto mejorada
            best_match = None
            best_score = 0

            for i, question in enumerate(self.qa_data['pregunta']):
                similarity = self._improve_similarity_search(query, question)
                if similarity > best_score and similarity > 0.5:  # Umbral más bajo para más matches
                    best_score = similarity
                    best_match = {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": similarity,
                        "tipo": "similaridad"
                    }

            if best_match and best_score > 0.6:
                return best_match

            # 4. Búsqueda por entidades mejorada
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
                    # Eliminar duplicados y ordenar por relevancia
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


# Inicialización del chatbot
try:
    chatbot = LungHealthChatbot('datasetchatbot_referencias.csv', 'saved_models')
    logging.info("✅ Chatbot inicializado exitosamente")
except Exception as e:
    logging.error(f"❌ Error inicializando chatbot: {e}")
    # Crear instancia básica del chatbot
    chatbot = LungHealthChatbot.__new__(LungHealthChatbot)
    chatbot.neural_model = None
    chatbot.model_data = None
    chatbot.qa_data = pd.DataFrame({
        'pregunta': ['hola'],
        'respuesta': ['¡Hola! Soy tu asistente de salud pulmonar.'],
        'intencion': ['saludo'],
        'entidades': ['saludo']
    })
    chatbot.conversation_history = []
    chatbot.risk_assessment_active = False
    logging.info("✅ Chatbot básico inicializado como fallback")


# Rutas Flask
@app.route('/')
def home():
    if chatbot:
        welcome_msg = chatbot.get_welcome_message()
    else:
        welcome_msg = "El chatbot no está disponible en este momento."
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
    model_loaded = hasattr(chatbot, 'model_data') and chatbot.model_data is not None
    neural_loaded = hasattr(chatbot, 'neural_model') and chatbot.neural_model is not None
    dataset_size = len(chatbot.qa_data) if hasattr(chatbot, 'qa_data') and chatbot.qa_data is not None else 0

    return jsonify({
        'status': 'healthy' if chatbot else 'error',
        'chatbot_loaded': chatbot is not None,
        'model_loaded': model_loaded,
        'neural_model_loaded': neural_loaded,
        'dataset_size': dataset_size
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'

    logging.info(f"🚀 Iniciando servidor en puerto {port} (debug: {debug_mode})")

    if os.getenv('RENDER'):
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
