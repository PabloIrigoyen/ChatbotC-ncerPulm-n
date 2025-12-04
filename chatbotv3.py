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
import re
import sys
import io
import warnings

if sys.platform == "win32":
    # Forzar codificación UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Configurar el sistema para soportar emojis
    os.system('chcp 65001 > nul')

# Suprimir warnings específicos
warnings.filterwarnings('ignore', category=UserWarning)


class UnicodeSafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Si falla con emojis, usar versión sin emojis
            safe_msg = record.getMessage()
            # Reemplazar emojis comunes con texto
            safe_msg = safe_msg.replace('✅', '[OK]').replace('❌', '[ERROR]')
            safe_msg = safe_msg.replace('📋', '[LIST]').replace('📊', '[CHART]')
            safe_msg = safe_msg.replace('🚀', '[ROCKET]').replace('🧠', '[BRAIN]')
            safe_msg = safe_msg.replace('💡', '[IDEA]').replace('🔍', '[SEARCH]')
            safe_msg = safe_msg.replace('⚠️', '[WARNING]').replace('🎯', '[TARGET]')
            safe_msg = safe_msg.replace('🏥', '[HOSPITAL]').replace('💊', '[MEDICINE]')
            safe_msg = safe_msg.replace('🚭', '[NO SMOKE]').replace('🔴', '[RED]')
            safe_msg = safe_msg.replace('🟡', '[YELLOW]').replace('🟢', '[GREEN]')
            safe_msg = safe_msg.replace('🟠', '[ORANGE]').replace('🩺', '[STETHOSCOPE]')
            safe_msg = safe_msg.replace('🧬', '[DNA]').replace('🌡️', '[THERMOMETER]')
            safe_msg = safe_msg.replace('🌱', '[LEAF]').replace('💪', '[MUSCLE]')
            safe_msg = safe_msg.replace('📅', '[CALENDAR]').replace('🔍', '[MAGNIFY]')
            safe_msg = safe_msg.replace('📈', '[CHART UP]').replace('🤖', '[ROBOT]')

            stream.write(f"{record.asctime} [{record.levelname}] {safe_msg}\n")
            self.flush()
        except Exception:
            self.handleError(record)


# Configurar logging
log_format = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    handlers=[
                        UnicodeSafeStreamHandler(),
                        logging.FileHandler('lung_health_chatbot.log', encoding='utf-8')
                    ])

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')


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
            logging.info("📥 Descargando modelo spaCy español...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"],
                           check=True, capture_output=True)
            self.nlp = spacy.load("es_core_news_sm")

        # Cargar stopwords en español
        try:
            import nltk
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('spanish'))
        except:
            # Lista básica de stopwords en español
            self.stop_words = set(
                ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un', 'para', 'con', 'no',
                 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque', 'esta',
                 'entre', 'cuando', 'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde',
                 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
                 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él',
                 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar',
                 'estas', 'algunas', 'algo', 'nosotros', 'mi', 'mis', 'tú', 'te', 'ti', 'tu', 'tus', 'ellas',
                 'nosotras', 'vosotros', 'vosotras', 'os', 'mío', 'mía', 'míos', 'mías', 'tuyo', 'tuya', 'tuyos',
                 'tuyas', 'suyo', 'suya', 'suyos', 'suyas', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro',
                 'vuestra', 'vuestros', 'vuestras', 'esos', 'esas', 'estoy', 'estás', 'está', 'estamos', 'estáis',
                 'están', 'esté', 'estés', 'estemos', 'estéis', 'estén', 'estaré', 'estarás', 'estará', 'estaremos',
                 'estaréis', 'estarán', 'estaría', 'estarías', 'estaríamos', 'estaríais', 'estarían', 'estaba',
                 'estabas', 'estábamos', 'estabais', 'estaban', 'estuve', 'estuviste', 'estuvo', 'estuvimos',
                 'estuvisteis', 'estuvieron', 'estuviera', 'estuvieras', 'estuviéramos', 'estuvierais', 'estuvieran',
                 'estuviese', 'estuvieses', 'estuviésemos', 'estuvieseis', 'estuviesen', 'estando', 'estado', 'estada',
                 'estados', 'estadas', 'estad', 'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'haya', 'hayas', 'hayamos',
                 'hayáis', 'hayan', 'habré', 'habrás', 'habrá', 'habremos', 'habréis', 'habrán', 'habría', 'habrías',
                 'habríamos', 'habríais', 'habrían', 'había', 'habías', 'habíamos', 'habíais', 'habían', 'hube',
                 'hubiste', 'hubo', 'hubimos', 'hubisteis', 'hubieron', 'hubiera', 'hubieras', 'hubiéramos',
                 'hubierais', 'hubieran', 'hubiese', 'hubieses', 'hubiésemos', 'hubieseis', 'hubiesen', 'habiendo',
                 'habido', 'habida', 'habidos', 'habidas', 'soy', 'eres', 'es', 'somos', 'sois', 'son', 'sea', 'seas',
                 'seamos', 'seáis', 'sean', 'seré', 'serás', 'será', 'seremos', 'seréis', 'serán', 'sería', 'serías',
                 'seríamos', 'seríais', 'serían', 'era', 'eras', 'éramos', 'erais', 'eran', 'fui', 'fuiste', 'fue',
                 'fuimos', 'fuisteis', 'fueron', 'fuera', 'fueras', 'fuéramos', 'fuerais', 'fueran', 'fuese', 'fueses',
                 'fuésemos', 'fueseis', 'fuesen', 'sintiendo', 'sentido', 'sentida', 'sentidos', 'sentidas', 'siente',
                 'sentid', 'tengo', 'tienes', 'tiene', 'tenemos', 'tenéis', 'tienen', 'tenga', 'tengas', 'tengamos',
                 'tengáis', 'tengan', 'tendré', 'tendrás', 'tendrá', 'tendremos', 'tendréis', 'tendrán', 'tendría',
                 'tendrías', 'tendríamos', 'tendríais', 'tendrían', 'tenía', 'tenías', 'teníamos', 'teníais', 'tenían',
                 'tuve', 'tuviste', 'tuvo', 'tuvimos', 'tuvisteis', 'tuvieron', 'tuviera', 'tuvieras', 'tuviéramos',
                 'tuvierais', 'tuvieran', 'tuviese', 'tuvieses', 'tuviésemos', 'tuvieseis', 'tuviesen', 'teniendo',
                 'tenido', 'tenida', 'tenidos', 'tenidas', 'tened'])

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
            logging.error(f"❌ Error en preprocesamiento: {e}")
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
            logging.info("🤖 Chatbot inicializado correctamente")
        except Exception as e:
            logging.error(f"❌ Error inicializando chatbot: {e}")
            # Asegurarse de que los atributos estén definidos incluso si hay error
            if self.neural_model is None:
                self.neural_model = None
            if self.model_data is None:
                self.model_data = None

    def _load_spacy_model(self):
        try:
            nlp = spacy.load("es_core_news_sm")
            logging.info("✅ Modelo spaCy cargado exitosamente")
            return nlp
        except OSError:
            try:
                nlp = spacy.load("es_core_news_sm")
                return nlp
            except OSError:
                import subprocess
                logging.info("📥 Descargando modelo de spaCy...")
                subprocess.run([sys.executable, "-m", "spacy", "download", "es_core_news_sm"],
                               check=True, capture_output=True)
                nlp = spacy.load("es_core_news_sm")
                return nlp

    def load_neural_model(self, model_dir):
        """Cargar modelo neuronal desde los archivos .pkl y .pth"""
        try:
            if not os.path.exists(model_dir):
                logging.warning(f"⚠️ Directorio de modelo {model_dir} no encontrado.")
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
                logging.warning("⚠️ Archivo glove_embeddings.pkl no encontrado")
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
                logging.warning("⚠️ Archivo best_glove_lung_cancer_model.pth no encontrado")
                self.neural_model = None

        except Exception as e:
            logging.error(f"❌ Error cargando modelo neuronal: {e}")
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
            logging.error(f"❌ Error en clasificación neuronal: {e}")
            return None, 0.0

    def load_prediction_model(self, model_path):
        """Cargar modelo .joblib mejorado con manejo robusto"""
        try:
            if not os.path.exists(model_path):
                logging.warning(f"⚠️ Modelo {model_path} no encontrado. La evaluación de riesgo no estará disponible.")
                self.model_data = None
                return

            # Cargar el modelo joblib
            loaded_data = joblib.load(model_path)

            # Manejar diferentes formatos de modelo joblib
            if isinstance(loaded_data, dict):
                # Si es un diccionario con todos los componentes
                self.model_data = loaded_data
                logging.info(f"✅ Modelo ML cargado desde dict: {len(loaded_data)} componentes")

                # Verificar estructura del modelo
                logging.info(f"📋 Componentes del modelo: {list(self.model_data.keys())}")

                if 'scaler' in self.model_data:
                    logging.info(f"✅ Scaler disponible: {type(self.model_data['scaler']).__name__}")
                    if hasattr(self.model_data['scaler'], 'feature_names_in_'):
                        logging.info(f"📊 Features del scaler: {list(self.model_data['scaler'].feature_names_in_)}")
                if 'label_encoder' in self.model_data:
                    logging.info(f"✅ Label encoder disponible: {type(self.model_data['label_encoder']).__name__}")
                    if hasattr(self.model_data['label_encoder'], 'classes_'):
                        logging.info(f"   Clases: {self.model_data['label_encoder'].classes_}")
                if 'feature_names' in self.model_data:
                    logging.info(f"📊 Feature names: {self.model_data['feature_names']}")
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
            model = self.model_data['model']
            model_info = f"Modelo: {type(model).__name__}"
            if hasattr(model, 'classes_'):
                model_info += f", Clases: {model.classes_}"
                logging.info(f"📊 El modelo tiene {len(model.classes_)} clases: {list(model.classes_)}")
            if 'mean_accuracy' in self.model_data:
                model_info += f", Precisión: {self.model_data.get('mean_accuracy', 'N/A')}"

            logging.info(f"📊 Información del modelo: {model_info}")

        except Exception as e:
            logging.error(f"❌ Error cargando modelo ML: {e}")
            self.model_data = None

    def load_qa_data(self, filename):
        try:
            if not os.path.exists(filename):
                logging.error(f"❌ Archivo {filename} no encontrado")
                # Crear dataset básico si no existe
                self.qa_data = pd.DataFrame({
                    'pregunta': ['hola', 'evaluar riesgo', 'ayuda'],
                    'respuesta': [
                        '¡Hola! Soy tu asistente de salud pulmonar. Puedo ayudarte con información sobre cáncer de pulmón o realizar una evaluación de riesgo.',
                        'Iniciando evaluación de riesgo...',
                        'Puede ayudarte con información sobre cáncer de pulmón o realizar una evaluación de riesgo personalizada.'
                    ],
                    'intencion': ['saludo', 'evaluacion_riesgo', 'ayuda'],
                    'entidades': ['saludo', 'evaluacion', 'ayuda']
                })
                logging.warning("⚠️ Dataset básico creado")
                return

            self.qa_data = pd.read_csv(filename)
            logging.info(f"📂 Dataset cargado: {len(self.qa_data)} preguntas")

            for col in ['pregunta', 'respuesta', 'intencion', 'entidades']:
                self.qa_data[col] = self.qa_data[col].astype(str).str.strip()

            self.qa_data['entidades_lista'] = self.qa_data['entidades'].str.split('|')

        except Exception as e:
            logging.error(f"❌ Error cargando dataset: {e}")
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

            logging.info("🔧 Estructuras de búsqueda configuradas")

        except Exception as e:
            logging.error(f"❌ Error configurando estructuras: {e}")
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
                "explicacion": "💡 1 = Sin exposición, 7 = Exposición máxima constante",
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
                "explicacion": "💡 1 = Nunca, 7 = Consumo diario",
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
                "explicacion": "💡 1 = No hay alergia, 7 = Alergia muy severa",
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
                "explicacion": "💡 1 = Sin exposición laboral, 7 = Exposición constante a carcinógenos",
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
                "explicacion": "💡 1 = Sin antecedentes familiares, 7 = Historia familiar muy fuerte",
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
                "explicacion": "💡 1 = Sin enfermedad pulmonar, 7 = Enfermedad pulmonar crónica avanzada",
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
                "explicacion": "💡 1 = Dieta muy saludable, 7 = Dieta muy poco saludable",
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
                "explicacion": "💡 1 = Peso saludable, 7 = Obesidad severa",
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
                "explicacion": "💡 1 = No fumador, 7 = Fumador muy intenso",
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
                "explicacion": "💡 1 = Sin exposición al humo, 7 = Exposición constante al humo",
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
                "explicacion": "💡 1 = Sin dolor torácico, 7 = Dolor muy intenso",
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
                "explicacion": "💡 1 = Sin tos con sangre, 7 = Hemoptisis frecuente",
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
                "explicacion": "💡 1 = Sin fatiga, 7 = Fatiga que limita actividades",
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
                "explicacion": "💡 1 = Sin pérdida de peso, 7 = Pérdida de peso muy severa",
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
                "explicacion": "💡 1 = Sin dificultad respiratoria, 7 = Dificultad respiratoria extrema",
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
                "explicacion": "💡 1 = Sin sibilancias, 7 = Sibilancias constantes",
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
                "explicacion": "💡 1 = Sin dificultad para tragar, 7 = Dificultad severa",
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
                "explicacion": "💡 1 = Uñas normales, 7 = Acropaquia severa",
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
                "explicacion": "💡 1 = Resfriados muy infrecuentes, 7 = Resfriados constantes",
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
                "explicacion": "💡 1 = Sin tos seca, 7 = Tos seca constante",
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
                "explicacion": "💡 1 = Sin ronquidos, 7 = Ronquidos constantes",
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

    def _validate_response(self, response, question):
        """Validar respuesta según el tipo de pregunta"""
        try:
            # Convertir a string y limpiar
            response_str = str(response).strip()

            # Debug: Mostrar qué se está recibiendo
            logging.info(f"🔍 Validando respuesta: '{response_str}' para pregunta '{question['id']}'")

            # 1. PARA NÚMEROS (edad)
            if question["tipo"] == "numero":
                try:
                    value = int(response_str)
                    if "rango" in question:
                        min_val, max_val = question["rango"]
                        if min_val <= value <= max_val:
                            return value
                    return value
                except ValueError:
                    return None

            # 2. PARA OPCIONES/ESCALAS
            elif question["tipo"] in ["opcion", "escala", "escala_inversa"]:
                # CASO 1: Si es un número directo (1-7)
                if response_str.isdigit():
                    num_value = int(response_str)
                    # Verificar si el número está en el rango válido (1-7 para escalas)
                    if 1 <= num_value <= 7:
                        return num_value
                    elif num_value == 1 and question["id"] == "gender":
                        return 1
                    elif num_value == 2 and question["id"] == "gender":
                        return 2

                # CASO 2: Si viene con "Opción X" (del frontend mejorado)
                if "opción" in response_str.lower() or "opcion" in response_str.lower():
                    match = re.search(r'(\d+)', response_str)
                    if match:
                        num_value = int(match.group(1))
                        if 1 <= num_value <= 7:
                            return num_value

                # CASO 3: Manejo especial para género
                if question["id"] == "gender":
                    response_lower = response_str.lower()
                    # Masculino
                    if any(term in response_lower for term in ['masculino', 'hombre', 'varón', 'male', 'm', '1']):
                        return 1
                    # Femenino
                    elif any(term in response_lower for term in ['femenino', 'mujer', 'hembra', 'female', 'f', '2']):
                        return 2

                # CASO 4: Si el usuario escribe el texto completo, buscar el número en las opciones
                if question["opciones"]:
                    for i, option in enumerate(question["opciones"], 1):
                        # Extraer número de la opción (ej: "1 - Muy baja")
                        option_num = option.split('-')[0].strip()
                        if option_num == response_str:
                            return i
                        # Si el usuario escribió el texto completo (sin número)
                        option_text = option.split('-', 1)[1].strip() if '-' in option else option
                        if response_str.lower() == option_text.lower():
                            return i
                        # Si hay palabras clave coincidentes
                        option_words = set(option_text.lower().split())
                        response_words = set(response_str.lower().split())
                        if len(option_words.intersection(response_words)) >= 2:
                            return i

                return None
            return None

        except Exception as e:
            logging.error(f"❌ Error en validación: {e}")
            return None

    def process_risk_response(self, response):
        """Procesar respuesta del cuestionario"""
        if not self.risk_assessment_active:
            return {"error": "No hay evaluación de riesgo activa"}

        current_q = self.risk_questions[self.current_risk_question]

        # Usar validación simplificada
        validated_response = self._validate_response(response, current_q)

        if validated_response is None:
            # Crear mensaje de error específico
            error_msg = f"❌ Respuesta inválida para: '{current_q['pregunta']}'"

            if current_q["opciones"]:
                error_msg += "\n\n📋 Opciones válidas:\n"
                for j, opt in enumerate(current_q["opciones"], 1):
                    error_msg += f"{j}. {opt}\n"

            if current_q.get("explicacion"):
                error_msg += f"\n💡 {current_q['explicacion']}"

            return {
                "type": "validation_error",
                "message": error_msg,
                "options": current_q["opciones"],
                "question_id": current_q["id"],
                "expected_type": current_q["tipo"]
            }

        # VALIDACIÓN ADICIONAL: Verificar que valores numéricos estén en rango
        if current_q["tipo"] == "escala" and not (1 <= validated_response <= 7):
            return {
                "type": "validation_error",
                "message": f"❌ Valor fuera de rango. Debe ser entre 1 y 7. Recibido: {validated_response}",
                "options": current_q["opciones"]
            }

        # Guardar respuesta validada
        self.risk_responses[current_q["id"]] = validated_response
        logging.info(f"✅ Respuesta validada: {current_q['id']} = {validated_response} (de: '{response}')")

        # Avanzar a la siguiente pregunta
        self.current_risk_question += 1

        if self.current_risk_question < len(self.risk_questions):
            next_q = self.risk_questions[self.current_risk_question]

            response_data = {
                "type": "risk_assessment_question",
                "question": next_q["pregunta"],
                "question_id": next_q["id"],
                "question_type": next_q["tipo"],
                "options": next_q["opciones"],
                "progress": f"{self.current_risk_question + 1}/{len(self.risk_questions)}",
                "total_questions": len(self.risk_questions)
            }

            # Agregar explicación si existe
            if next_q.get("explicacion"):
                response_data["explicacion"] = next_q["explicacion"]

            # Agregar rango para preguntas numéricas
            if next_q["tipo"] == "numero" and "rango" in next_q:
                response_data["range"] = next_q["rango"]

            return response_data
        else:
            # Completar evaluación
            return self._complete_risk_assessment()

    def _complete_risk_assessment(self):
        """Completar evaluación de riesgo usando el modelo joblib"""
        self.risk_assessment_active = False

        # Procesar respuestas para el modelo
        processed_responses = self._process_responses_for_joblib_model()

        # Usar el modelo joblib para predicción
        risk_prediction = self._predict_with_joblib(processed_responses)

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

        logging.info("📊 PROCESANDO RESPUESTAS PARA MODELO")

        for question in self.risk_questions:
            q_id = question["id"]
            feature_name = question.get("feature_name")

            if feature_name and q_id in self.risk_responses:
                response_value = self.risk_responses[q_id]

                logging.info(f"📝 Procesando: {q_id} -> {feature_name} = {response_value}")

                # Para todas las características, mantener el valor original (1-7)
                processed[feature_name] = response_value

        # Log detallado
        logging.info(f"📦 Respuestas procesadas ({len(processed)}):")
        for key, value in processed.items():
            logging.info(f"  {key}: {value}")

        return processed

    def _convert_to_model_scale(self, user_value):
        """Convertir de escala usuario (1-7) a escala modelo (1-9)"""
        try:
            user_value_int = int(user_value)
            # Asegurarnos que esté en rango 1-7
            user_value_int = max(1, min(7, user_value_int))

            # Fórmula lineal: y = 1 + (x-1)*(8/6) donde x es 1-7, y es 1-9
            model_value = 1 + round((user_value_int - 1) * (8 / 6))

            # Asegurar que esté en rango 1-9
            model_value = int(max(1, min(9, model_value)))

            logging.info(f"📐 Conversión escala: {user_value_int} (UI 1-7) → {model_value} (Modelo 1-9)")
            return model_value

        except Exception as e:
            logging.error(f"❌ Error en conversión de escala: {e}")
            return 5  # Valor medio por defecto

    def _get_model_feature_names(self):
        """Obtener los nombres de características que espera el modelo"""
        if not self.model_data:
            return None

        # Prioridad 1: Feature names del scaler
        if 'scaler' in self.model_data and self.model_data['scaler'] is not None:
            if hasattr(self.model_data['scaler'], 'feature_names_in_'):
                return list(self.model_data['scaler'].feature_names_in_)

        # Prioridad 2: Feature names explícitos
        if 'feature_names' in self.model_data and self.model_data['feature_names'] is not None:
            return self.model_data['feature_names']

        # Prioridad 3: Orden por defecto (basado en dataset original)
        return [
            'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
            'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
            'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
            'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
            'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
        ]

    def _predict_with_joblib(self, responses):
        """Predicción usando Random Forest - CON CORRECCIÓN DEL ERROR"""
        try:
            # Verificar que el modelo esté cargado correctamente
            if not self.model_data or 'model' not in self.model_data:
                logging.error("❌ Modelo joblib no disponible")
                return self._get_fallback_prediction("Modelo no disponible")

            model = self.model_data['model']
            scaler = self.model_data.get('scaler')
            label_encoder = self.model_data.get('label_encoder')

            logging.info(f"🌲 PREDICCIÓN CON RANDOM FOREST")
            logging.info(f"🤖 Modelo tipo: {type(model).__name__}")
            logging.info(f"📝 Respuestas UI (1-7): {responses}")

            # Obtener los nombres de características que espera el modelo
            feature_names = self._get_model_feature_names()
            if feature_names is None:
                logging.error("❌ No se pudieron obtener los nombres de características")
                return self._get_fallback_prediction("No se encontraron nombres de características")

            logging.info(f"📊 Features esperadas ({len(feature_names)}): {feature_names}")

            # Construir vector de características en el orden correcto
            model_input = []
            debug_info = []

            for i, feature in enumerate(feature_names):
                feature_clean = feature.strip()
                found = False

                # Buscar la característica en las respuestas (comparación flexible)
                for resp_key, resp_value in responses.items():
                    resp_key_clean = resp_key.strip()

                    # Comparar directamente o con variaciones
                    if (feature_clean.lower() == resp_key_clean.lower() or
                            feature_clean.replace(' ', '').lower() == resp_key_clean.replace(' ', '').lower() or
                            feature_clean.lower() in resp_key_clean.lower() or
                            resp_key_clean.lower() in feature_clean.lower()):

                        found = True
                        user_value = resp_value

                        try:
                            # Edad: mantener valor original
                            if feature_clean.lower() == 'age':
                                model_value = int(float(user_value))
                                model_value = max(1, min(100, model_value))
                                conversion_info = f"{user_value} (edad directa)"

                            # Género: 1=Masculino, 2=Femenino
                            elif feature_clean.lower() == 'gender':
                                gender_val = int(float(user_value))
                                model_value = 1 if gender_val == 1 else 2
                                conversion_info = f"{user_value} → {model_value} (1=M, 2=F)"

                            # Otras características: convertir de 1-7 a 1-9
                            else:
                                user_value_int = int(float(user_value))
                                model_value = self._convert_to_model_scale(user_value_int)
                                conversion_info = f"{user_value} (UI) → {model_value} (Modelo 1-9)"

                            model_input.append(model_value)
                            debug_info.append(f"{i + 1}. {feature}: {conversion_info}")
                            break

                        except Exception as e:
                            logging.error(f"❌ Error procesando {feature}: {e}")
                            # Valor por defecto según tipo
                            if feature_clean.lower() == 'age':
                                model_value = 50
                            elif feature_clean.lower() == 'gender':
                                model_value = 1
                            else:
                                model_value = 5  # Valor medio en escala 1-9
                            model_input.append(model_value)
                            debug_info.append(f"{i + 1}. {feature}: ERROR → {model_value} (defecto)")
                            break

                if not found:
                    # Característica no encontrada, usar valor por defecto
                    if feature_clean.lower() == 'age':
                        model_value = 50
                    elif feature_clean.lower() == 'gender':
                        model_value = 1
                    else:
                        model_value = 5  # Valor medio en escala 1-9

                    model_input.append(model_value)
                    debug_info.append(f"{i + 1}. {feature}: NO ENCONTRADO → {model_value} (defecto)")
                    logging.warning(f"⚠️  {feature}: No encontrado en respuestas")

            # Log detallado de conversiones
            logging.info("🔄 CONVERSIÓN DE VALORES")
            for info in debug_info:
                logging.info(f"  {info}")

            # Verificar que tenemos el número correcto de características
            expected_count = len(feature_names)
            actual_count = len(model_input)

            if actual_count != expected_count:
                logging.error(
                    f"❌ ERROR: Número incorrecto de características. Esperadas: {expected_count}, Obtenidas: {actual_count}")
                return self._get_fallback_prediction(
                    f"Error en número de características: {actual_count}/{expected_count}")

            # Convertir a numpy array
            model_input_array = np.array([model_input])

            # Estadísticas para debugging
            avg_value = sum(model_input) / len(model_input)
            max_value = max(model_input)
            min_value = min(model_input)
            high_values = sum(1 for x in model_input if x >= 7)  # Alto riesgo en escala 1-9

            logging.info(f"📈 ESTADÍSTICAS DEL VECTOR")
            logging.info(f"📦 Vector completo: {model_input}")
            logging.info(f"📐 Shape: {model_input_array.shape}")
            logging.info(f"📊 Promedio: {avg_value:.2f}")
            logging.info(f"📉 Mínimo: {min_value}")
            logging.info(f"📈 Máximo: {max_value}")
            logging.info(f"⚠️ Valores altos (>=7): {high_values}/{len(model_input)}")

            # Casos especiales para debugging
            if all(x == 1 for x in model_input[2:]):  # Ignorar Age y Gender
                logging.info("⚠️ CASO ESPECIAL: Todos los valores (excepto edad/género) = 1 (mínimo riesgo)")
            elif all(x == 9 for x in model_input[2:]):
                logging.info("⚠️ CASO ESPECIAL: Todos los valores (excepto edad/género) = 9 (máximo riesgo)")

            # Aplicar escalador si existe
            if scaler is not None:
                try:
                    # Verificar compatibilidad
                    if hasattr(scaler, 'n_features_in_'):
                        expected_scaler_features = scaler.n_features_in_
                        if expected_scaler_features != len(model_input):
                            logging.error(
                                f"❌ Scaler espera {expected_scaler_features} características, tenemos {len(model_input)}")
                            # Intentar ajustar (esto es un fallback)
                            if len(model_input) > expected_scaler_features:
                                model_input = model_input[:expected_scaler_features]
                            else:
                                model_input = model_input + [1] * (expected_scaler_features - len(model_input))
                            model_input_array = np.array([model_input])

                    model_input_array = scaler.transform(model_input_array)
                    logging.info("✅ Datos escalados correctamente")

                except Exception as e:
                    logging.error(f"❌ Error escalando datos: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    logging.warning("⚠️ Continuando con datos sin escalar")

            # Realizar predicción
            try:
                # Verificar si el modelo tiene predict_proba
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(model_input_array)[0]
                    prediction_idx = model.predict(model_input_array)[0]
                    confidence = max(probabilities)

                    logging.info(f"🎯 RESULTADOS DE PREDICCIÓN")
                    logging.info(f"📊 Probabilidades: {probabilities}")
                    logging.info(f"🎯 Predicción (índice): {prediction_idx}")
                    logging.info(f"🎯 Confianza máxima: {confidence:.4f}")


                    predicted_class = None
                    if label_encoder is not None and hasattr(label_encoder, 'inverse_transform'):
                        try:
                            predicted_class = label_encoder.inverse_transform([prediction_idx])[0]
                            logging.info(f"🏷️ Clase decodificada: {predicted_class}")

                            # Log del orden de clases para debugging
                            if hasattr(label_encoder, 'classes_'):
                                logging.info(f"📋 Orden de clases en label_encoder: {list(label_encoder.classes_)}")
                        except Exception as e:
                            logging.error(f"❌ Error decodificando clase: {e}")
                            predicted_class = str(prediction_idx)
                    else:
                        predicted_class = str(prediction_idx)

                    # DEBUG: Mostrar mapeo de probabilidades
                    logging.info(f"📊 MAPEO DE PROBABILIDADES")
                    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                        for i, prob in enumerate(probabilities):
                            if i < len(label_encoder.classes_):
                                class_name = label_encoder.classes_[i]
                                logging.info(f"  {class_name}: {prob:.4f}")
                            else:
                                logging.info(f"  Clase_{i}: {prob:.4f}")

                    # CORRECCIÓN: Usar la nueva función que maneja arrays numpy correctamente
                    risk_level = self._determine_risk_level_corregido(predicted_class, probabilities, label_encoder)

                else:
                    # Modelo sin predict_proba
                    prediction_idx = model.predict(model_input_array)[0]
                    confidence = 0.5
                    probabilities = [0.33, 0.33, 0.34]
                    risk_level = self._determine_risk_level_simple(prediction_idx)
                    logging.info(f"🎯 Predicción simple: {prediction_idx} -> {risk_level}")

                # Log de resultado final
                logging.info(f"🏁 RESULTADO FINAL")
                logging.info(f"📊 Nivel de riesgo: {risk_level}")
                logging.info(f"🎯 Confianza: {confidence:.1%}")

                # Identificar factores de riesgo
                risk_factors = self._identify_significant_risk_factors(responses)

                # Generar recomendaciones
                recommendations = self._get_personalized_recommendations(risk_level, responses)

                return {
                    "type": "risk_assessment_result",
                    "risk_level": risk_level,
                    "prediction_value": int(prediction_idx) if hasattr(prediction_idx, '__int__') else 0,
                    "confidence": f"{confidence:.1%}",
                    "probability_low": f"{probabilities[0]:.1%}" if len(probabilities) > 0 else "N/A",
                    "probability_medium": f"{probabilities[1]:.1%}" if len(probabilities) > 1 else "N/A",
                    "probability_high": f"{probabilities[2]:.1%}" if len(probabilities) > 2 else "N/A",
                    "recommendations": recommendations,
                    "risk_factors": risk_factors,
                    "based_on_ml": True,
                    "model_type": type(model).__name__,
                    "debug_info": {
                        "input_values": model_input,
                        "avg_value": f"{avg_value:.2f}",
                        "min_value": min_value,
                        "max_value": max_value,
                        "high_count": high_values
                    }
                }

            except Exception as e:
                logging.error(f"❌ Error en predicción: {e}")
                import traceback
                logging.error(traceback.format_exc())
                return self._get_fallback_prediction(f"Error en predicción: {str(e)}")

        except Exception as e:
            logging.error(f"❌ Error en predicción: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return self._get_fallback_prediction(f"Error general: {str(e)}")

    def _determine_risk_level_corregido(self, predicted_class, probabilities, label_encoder=None):
        try:
            logging.info(f"🔍 Determinando riesgo CORREGIDO para: {predicted_class}, prob: {probabilities}")
            if isinstance(probabilities, np.ndarray):
                if probabilities.size >= 3:
                    max_idx = int(np.argmax(probabilities))
                    logging.info(f"📈 Máxima probabilidad en índice: {max_idx} = {probabilities[max_idx]:.4f}")
                    if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                        classes = list(label_encoder.classes_)
                        logging.info(f"📋 Orden de clases en label_encoder: {classes}")

                        if max_idx < len(classes):
                            class_name = str(classes[max_idx]).lower()
                            if 'high' in class_name or 'alto' in class_name:
                                return "ALTO"
                            elif 'medium' in class_name or 'medio' in class_name:
                                return "MEDIO"
                            elif 'low' in class_name or 'bajo' in class_name:
                                return "BAJO"

                    if max_idx == 0:
                        return "ALTO"
                    elif max_idx == 1:
                        return "MEDIO"
                    elif max_idx == 2:
                        return "BAJO"

            # Fallback usando predicted_class
            if isinstance(predicted_class, str):
                predicted_lower = predicted_class.lower()

                # Basado en tu debug: "High" -> ALTO
                if 'high' in predicted_lower or 'alto' in predicted_lower:
                    return "ALTO"
                elif 'medium' in predicted_lower or 'medio' in predicted_lower:
                    return "MEDIO"
                elif 'low' in predicted_lower or 'bajo' in predicted_lower:
                    return "BAJO"

            # Fallback numérico
            if isinstance(predicted_class, (int, float)):
                # Si 0=High como en tu debug
                if predicted_class == 0:
                    return "ALTO"
                elif predicted_class == 1:
                    return "MEDIO"
                elif predicted_class >= 2:
                    return "BAJO"

            # Default
            logging.warning(f"⚠️ No se pudo determinar riesgo para: {predicted_class}, usando MEDIO")
            return "MEDIO"

        except Exception as e:
            logging.error(f"❌ Error en determine_risk_level_corregido: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return "MEDIO"

    def _determine_risk_level_simple(self, prediction):
    
        if prediction == 0:
            return "BAJO"
        elif prediction == 1:
            return "MEDIO"
        elif prediction >= 2:
            return "ALTO"
        else:
            return "MEDIO"

    def _get_fallback_prediction(self, reason):
       
        logging.error(f"⚠️ Usando predicción de respaldo: {reason}")

        return {
            "type": "risk_assessment_result",
            "risk_level": "MEDIO",
            "prediction_value": 1,
            "confidence": "50%",
            "probability_low": "33%",
            "probability_medium": "34%",
            "probability_high": "33%",
            "recommendations": [
                "⚠️ Hubo un problema técnico con la evaluación automatizada.",
                "💊 Consulta con un profesional de la salud para evaluación precisa.",
                "🏥 Realiza chequeos médicos regulares para monitoreo preventivo."
            ],
            "risk_factors": [],
            "based_on_ml": False,
            "model_type": f"Fallback - {reason}"
        }

    def _identify_significant_risk_factors(self, responses):
      
        risk_factors = []

        # Mapeo de nombres de factores a español
        factor_names = {
            'Smoking': 'Tabaquismo',
            'Coughing of Blood': 'Tos con sangre',
            'Genetic Risk': 'Antecedentes familiares',
            'chronic Lung Disease': 'Enfermedad pulmonar crónica',
            'Air Pollution': 'Contaminación del aire',
            'Chest Pain': 'Dolor en el pecho',
            'Shortness of Breath': 'Dificultad respiratoria',
            'OccuPational Hazards': 'Riesgos laborales',
            'Weight Loss': 'Pérdida de peso inexplicable',
            'Dry Cough': 'Tos seca persistente',
            'Passive Smoker': 'Fumador pasivo',
            'Obesity': 'Obesidad',
            'Fatigue': 'Fatiga severa',
            'Age': 'Edad avanzada'
        }

        for factor_key, value in responses.items():
            if isinstance(value, (int, float)):
                severity = None
                if value == 7:
                    severity = 'MÁXIMO'
                elif value >= 6:
                    severity = 'ALTO'
                elif value >= 5:
                    severity = 'MODERADO-ALTO'

                if severity:
                    risk_factors.append({
                        'factor': factor_names.get(factor_key, factor_key),
                        'level': value,
                        'severity': severity
                    })

        # Ordenar por nivel descendente
        risk_factors.sort(key=lambda x: x['level'], reverse=True)

        return risk_factors

    def _get_personalized_recommendations(self, risk_level, responses):
        recommendations = []
        if risk_level == "ALTO":
            recommendations = [
                "🚨 CONSULTA MÉDICA URGENTE con neumólogo especialista",
                "💊 Tomografía computarizada de tórax INMEDIATA",
                "🚭 Suspensión total del tabaquismo - Programa intensivo",
                "🏥 Evaluación pulmonar COMPLETA de emergencia",
                "🔍 Monitoreo diario de síntomas y signos de alerta"
            ]
        elif risk_level == "MEDIO":
            recommendations = [
                "📅 Consulta médica PRIORITARIA en los próximos 15 días",
                "🔍 Radiografía de tórax y evaluación básica pulmonar",
                "🌱 Programa intensivo de reducción de factores de riesgo",
                "💪 Ejercicio supervisado y rehabilitación pulmonar",
                "📊 Monitoreo semanal de síntomas y evolución"
            ]
        else:  
            recommendations = [
                "✅ Excelente perfil de salud pulmonar",
                "💪 Mantenimiento de hábitos saludables actuales",
                "🚭 Evitación COMPLETA del tabaco y humo ambiental",
                "📅 Chequeo médico preventivo anual",
                "🏃 Ejercicio regular para mantener capacidad pulmonar"
            ]

        for factor_key, value in responses.items():
            if isinstance(value, (int, float)) and value >= 6:
                if factor_key == 'Smoking':
                    recommendations.append("🎯 PROGRAMA URGENTE DE DEJAR DE FUMAR - Terapia combinada")
                elif factor_key == 'Coughing of Blood':
                    recommendations.append(
                        "🩺 ATENCIÓN DE EMERGENCIA INMEDIATA - Hemoptisis requiere evaluación URGENTE")
                elif factor_key == 'Genetic Risk':
                    recommendations.append("🧬 CONSEJO GENÉTICO ONCOLÓGICO - Evaluación familiar completa")
                elif factor_key == 'chronic Lung Disease':
                    recommendations.append("🌡️ CONTROL ESPECIALIZADO MULTIDISCIPLINARIO - Neumólogo + Rehabilitador")

        return recommendations

    def process_message(self, message):
        try:
            if not message or not isinstance(message, str):
                return {"bot_response": "❌ Por favor escribe un mensaje válido."}

            message = message.strip()
            if not message:
                return {"bot_response": "❌ Por favor escribe un mensaje válido."}

            self.conversation_history.append({
                'timestamp': datetime.now(),
                'user_message': message,
                'type': 'user'
            })

            lower_message = message.lower()

            if any(cmd in lower_message for cmd in ['evaluar riesgo', 'test riesgo', 'cuestionario', 'evaluación']):
                risk_start = self.start_risk_assessment()
                response = {
                    "risk_assessment_active": True,
                    "bot_response": f"🔍 {risk_start['question']}",
                    "options": risk_start['options'],
                    "progress": risk_start['progress'],
                    "question_id": risk_start['question_id']
                }
                if risk_start.get('explicacion'):
                    response["bot_response"] += f"\n\n💡 {risk_start['explicacion']}"

                return response

            elif self.risk_assessment_active:
                risk_response = self.process_risk_response(message)

                if risk_response.get('type') == 'validation_error':
                    return {
                        "risk_assessment_active": True,
                        "bot_response": risk_response['message'],
                        "options": risk_response.get('options', []),
                        "question_id": risk_response.get('question_id')
                    }

                elif 'question' in risk_response:
                    response = {
                        "risk_assessment_active": True,
                        "bot_response": f"🔍 {risk_response['question']}",
                        "options": risk_response['options'],
                        "progress": risk_response['progress'],
                        "question_id": risk_response['question_id']
                    }
                    if risk_response.get('explicacion'):
                        response["bot_response"] += f"\n\n💡 {risk_response['explicacion']}"

                    return response

                else:
                    if isinstance(risk_response, dict) and 'risk_level' in risk_response:
                        result = risk_response

                        if result['risk_level'] == "ALTO":
                            risk_emoji = "🔴"
                            risk_title = "🔴 ALTO RIESGO"
                        elif result['risk_level'] == "MEDIO":
                            risk_emoji = "🟡"
                            risk_title = "🟡 RIESGO MEDIO"
                        else:  # BAJO
                            risk_emoji = "🟢"
                            risk_title = "🟢 BAJO RIESGO"

                        response_text = f"""{risk_emoji} EVALUACIÓN COMPLETADA - RESULTADOS

📊 Nivel de riesgo: {risk_title}
🎯 Confianza del modelo: {result['confidence']}"""
                        response_text += "\n\n⚠️ Factores de riesgo identificados:"

                        if result.get('risk_factors'):
                            for factor in result['risk_factors']:
                                if factor['severity'] == 'MÁXIMO':
                                    severity_emoji = "🔴"
                                elif factor['severity'] == 'ALTO':
                                    severity_emoji = "🟠"
                                elif factor['severity'] == 'MODERADO-ALTO':
                                    severity_emoji = "🟡"
                                else:
                                    severity_emoji = "🟢"
                                response_text += f"\n• {severity_emoji} **{factor['factor']}** (Nivel {factor['level']}/7 - {factor['severity']})"
                        else:
                            response_text += "\n• ✅ No se identificaron factores de riesgo significativos"

                        response_text += f"\n\n📋 **RECOMENDACIONES:**"
                        for rec in result['recommendations']:
                            response_text += f"\n• {rec}"

                        response_text += "\n\n💡 _Esta evaluación es informativa. Consulta siempre con un profesional de la salud._"

                        if result.get('based_on_ml'):
                            response_text += f"\n\n🤖 _Evaluación generada por modelo {result.get('model_type', 'Random Forest')}_"
                        else:
                            response_text += f"\n\n⚠️ _Evaluación de respaldo ({result.get('model_type', 'N/A')})_"

                        return {"bot_response": response_text, "risk_assessment_active": False}
                    else:
                        return {"bot_response": "✅ Evaluación completada.", "risk_assessment_active": False}

            elif any(cmd in lower_message for cmd in ['hola', 'hi', 'buenos días', 'buenas']):
                return {"bot_response": self.get_welcome_message()}
            elif any(cmd in lower_message for cmd in ['gracias', 'grax', 'thank you', 'muchas gracias']):
                return {
                    "bot_response": '🙏 De nada, estoy para responder cualquier otra consulta sobre cáncer de pulmón'}
            elif any(cmd in lower_message for cmd in ['adiós', 'bye', 'hasta luego', 'nos vemos']):
                return {"bot_response": '👋 Hasta la próxima consulta'}
            elif any(cmd in lower_message for cmd in ['ayuda', 'comandos', 'qué puedes hacer']):
                return {"bot_response": self.get_help_message()}
            else:
                # Búsqueda inteligente
                match = self.find_best_match(message)
                if match:
                    return {"bot_response": match['respuesta']}
                else:
                    return {"bot_response": self._get_default_response()}

        except Exception as e:
            logging.error(f"❌ Error en process_message: {e}")
            error_msg = f"😟 Lo siento, hubo un error procesando tu mensaje. Por favor intenta de nuevo o escribe 'evaluar riesgo' para comenzar una nueva evaluación."
            return {'bot_response': error_msg}

    def _get_default_response(self):
        """Respuesta por defecto mejorada"""
        default_responses = [
            "🤔 No encontré información específica sobre tu consulta en mi base de datos. ¿Te gustaría realizar una evaluación de riesgo completa? Escribe 'evaluar riesgo' para comenzar.",
            "📚 Sobre ese tema específico no tengo información detallada. Puedo ayudarte con una evaluación de riesgo personalizada o puedes intentar reformular tu pregunta.",
            "🎯 Mi especialidad es el cáncer de pulmón y la evaluación de riesgo. ¿Te interesa saber más sobre algún aspecto específico o prefieres una evaluación personalizada?"
        ]
        return random.choice(default_responses)

    def get_welcome_message(self):
        model_loaded = hasattr(self, 'model_data') and self.model_data is not None
        neural_loaded = hasattr(self, 'neural_model') and self.neural_model is not None

        model_status = "✅ Con evaluación de riesgo Random Forest" if model_loaded else "⚠️ Evaluación básica disponible"
        neural_status = "🧠 Con inteligencia neuronal avanzada" if neural_loaded else ""

        return f"""
👋 ¡Hola! Soy tu asistente especializado en Cáncer de Pulmón.

{model_status}
{neural_status}

🏥 CUESTIONARIO DISPONIBLE:
• 23 preguntas completas
• Evaluación con modelo Random Forest
• Resultados con 3 niveles de riesgo
💬 Puedo ayudarte con:
• Información sobre cáncer de pulmón
• Evaluación de riesgo personalizada
• Respuestas a preguntas médicas

¡Escribe tu pregunta o 'evaluar riesgo' para comenzar! 😊
"""

    def get_help_message(self):
        return """
🤖 CÓMO PUEDO AYUDARTE

🔍 EVALUACIÓN DE RIESGO:
• "evaluar riesgo" - Cuestionario completo de 23 preguntas
• Evaluación personalizada con modelo Random Forest
• Resultados detallados con 3 niveles de precisión

📊 Para obtener diferentes niveles de riesgo:
• Para riesgo BAJO: Respuestas bajas (1-3) en factores clave
• Para riesgo MEDIO: Algunas respuestas moderadas (4-5)
• Para riesgo ALTO: Respuestas altas (6-7) en múltiples factores

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

    def _extract_entities(self, query):
        try:
            query_lower = query.lower().strip()
            found_entities = set()

            all_entities = set()
            for entities in self.qa_data['entidades_lista']:
                if isinstance(entities, list):
                    for entity in entities:
                        if entity and isinstance(entity, str):
                            all_entities.add(entity.strip().lower())

            for entity in all_entities:
                if entity in query_lower:
                    found_entities.add(entity)
                else:
                    entity_words = entity.split()
                    if len(entity_words) > 1:
                        for word in entity_words:
                            if len(word) > 3 and word in query_lower:
                                found_entities.add(entity)
                                break

            return list(found_entities)

        except Exception as e:
            logging.error(f"❌ Error extrayendo entidades: {e}")
            return []

    def _improve_similarity_search(self, query, question):
        
        query_lower = query.lower()
        question_lower = question.lower()
        basic_similarity = difflib.SequenceMatcher(None, query_lower, question_lower).ratio()
        medical_keywords = ['cáncer', 'pulmón', 'síntoma', 'diagnóstico', 'tratamiento',
                            'tabaco', 'fumar', 'tos', 'dolor', 'respirar']

        keyword_bonus = 0
        for keyword in medical_keywords:
            if keyword in query_lower and keyword in question_lower:
                keyword_bonus += 0.1

        return min(1.0, basic_similarity + keyword_bonus)

    def find_best_match(self, query):
        try:
            query_lower = query.lower().strip()

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

            for i, question in enumerate(self.qa_data['pregunta']):
                if query_lower == question.lower():
                    return {
                        "pregunta": question,
                        "respuesta": self.qa_data.iloc[i]['respuesta'],
                        "score": 1.0,
                        "tipo": "exacta"
                    }

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

            entities = self._extract_entities(query)
            if entities:
                entity_matches = []
                for entity in entities:
                    if entity in self.entity_to_questions:
                        for idx in self.entity_to_questions[entity]:
                            entity_matches.append({
                                "pregunta": self.qa_data.iloc[idx]["pregunta"],
                                "respuesta": self.qa_data.iloc[idx]["respuesta"],
                                "score": 0.7,
                                "tipo": f"entidad_{entity}"
                            })

                if entity_matches:
                    unique_matches = {}
                    for match in entity_matches:
                        key = match['pregunta']
                        if key not in unique_matches:
                            unique_matches[key] = match

                    matches_list = list(unique_matches.values())
                    if matches_list:
                        return matches_list[0] 

            return None

        except Exception as e:
            logging.error(f"❌ Error en find_best_match: {e}")
            return None


try:
    chatbot = LungHealthChatbot('datasetchatbot_referencias.csv', 'saved_models')
    logging.info("🤖 Chatbot inicializado exitosamente")
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
    logging.info("⚠️ Chatbot básico inicializado como fallback")


@app.route('/')
def home():
    if chatbot:
        welcome_msg = chatbot.get_welcome_message()
    else:
        welcome_msg = "⚠️ El chatbot no está disponible en este momento."
    return render_template('chat.html',
                           welcome_message=welcome_msg,
                           current_time=datetime.now().strftime("%H:%M"))


@app.route('/send_message', methods=['POST'])
def send_message():
    if not chatbot:
        return jsonify({'error': '❌ Chatbot no disponible'}), 500

    if not request.json or 'message' not in request.json:
        return jsonify({'error': '❌ Mensaje no proporcionado'}), 400

    try:
        user_message = request.json['message']
        response = chatbot.process_message(user_message)

        return jsonify({
            'user_message': user_message,
            'bot_response': response.get('bot_response', ''),
            'risk_assessment_active': chatbot.risk_assessment_active,
            'options': response.get('options'),
            'progress': response.get('progress'),
            'timestamp': datetime.now().strftime("%H:%M")
        })

    except Exception as e:
        logging.error(f"❌ Error en send_message: {e}")
        return jsonify({
            'user_message': user_message,
            'bot_response': "😟 Lo siento, hubo un error procesando tu mensaje.",
            'timestamp': datetime.now().strftime("%H:%M")
        })


@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    if chatbot:
        chatbot.reset_conversation_state()
        return jsonify({'status': 'success', 'message': '🔄 Conversación reiniciada'})
    return jsonify({'error': '❌ Chatbot no disponible'}), 500


@app.route('/health')
def health_check():
    model_loaded = hasattr(chatbot, 'model_data') and chatbot.model_data is not None
    neural_loaded = hasattr(chatbot, 'neural_model') and chatbot.neural_model is not None
    dataset_size = len(chatbot.qa_data) if hasattr(chatbot, 'qa_data') and chatbot.qa_data is not None else 0

    return jsonify({
        'status': '✅ healthy' if chatbot else '❌ error',
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

