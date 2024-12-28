from flask import Flask, request, jsonify, render_template, send_file, Response
from flask_cors import CORS
from flask_pymongo import PyMongo
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer

import whisper
import os
import re
import spacy
import nltk
import language_tool_python
import numpy as np
import librosa
from pydub import AudioSegment
from bson.objectid import ObjectId
import bcrypt
import jwt
import datetime
from datetime import timezone
import urllib.parse
import tensorflow as tf
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import  T5Tokenizer, T5ForConditionalGeneration
import wave
from mega import Mega
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


# Mega account credentials
MEGA_EMAIL = 'akhatri1600@gmail.com'
MEGA_PASSWORD = '@Kash12345'

# Initialize Mega instance
mega = Mega()
m = mega.login(MEGA_EMAIL, MEGA_PASSWORD)

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))
filler_words = ['uh', 'um', 'like', 'you know', 'so', 'actually', 'basically']

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize LanguageTool
tool = language_tool_python.LanguageTool('en-US')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Initialize rephrasing model
rephrase_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
rephrase_model = T5ForConditionalGeneration.from_pretrained('t5-small')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = 'your_secret_key'
username = urllib.parse.quote_plus('akash')
password = urllib.parse.quote_plus('VocalAi@Database')
app.config["MONGO_URI"] = f'mongodb+srv://{username}:{password}@vocalai.mongocluster.cosmos.azure.com/vocal_ai_db?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
mongo = PyMongo(app)

# Load Whisper Tiny model
speech_to_text_model = whisper.load_model("tiny")

# Define paths
UPLOAD_FOLDER = '/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
model = tf.keras.models.load_model(r'bilstm_speech_model_with_overall_score.h5', custom_objects={'MeanSquaredError': tf.keras.losses.MeanSquaredError()})
model.compile(optimizer='adam', loss='mse')

ffmpeg_path = r'D:\AIMT\Sem 3\AML 3406 - AI and ML Capstone Project\ffmpeg-2024-06-21-git-d45e20c37b-full_build\bin'  # Change this to the actual path of your ffmpeg bin directory
os.environ["PATH"] += os.pathsep + ffmpeg_path

# Function to extract key points from the original text
def extract_key_points(text, num_sentences=5):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    scores = X.sum(axis=1).A1
    ranked_sentences = [sentences[i] for i in np.argsort(scores, axis=0)[::-1]]
    return ranked_sentences[:num_sentences]

# Function to extract important entities from the original text
def extract_entities(text):
    doc = nlp(text)
    entities = set(ent.text for ent in doc.ents)
    return entities

# Function to create a guided summary input
def create_guided_input(transcript, key_points, entities):
    guided_input = "summarize: "
    guided_input += "Key points: "
    guided_input += " ".join(key_points)
    guided_input += " Named entities: "
    guided_input += ", ".join(entities)
    guided_input += " Full transcript: "
    guided_input += transcript
    return guided_input


def clean_transcript(transcript):
    transcript = re.sub(r'{NOISE}|<sil>|{UH}|{COUGH}|{BREATH}|{SMACK}', '', transcript)
    transcript = re.sub(r'\s+', ' ', transcript).strip()
    return transcript

def extract_audio_features(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr).mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        silence = librosa.effects.split(y, top_db=20)
        stopping_time = sum((end - start) for start, end in silence) / sr

        pitches, _ = librosa.piptrack(y=y, sr=sr)
        pitch = np.mean(pitches[pitches > 0])  # Average pitch

        return {
            'sample_rate': sr,
            'mean_mfcc': mfcc.tolist(),
            'mean_chroma': chroma.tolist(),
            'mean_zcr': zcr,
            'stopping_time': stopping_time,
            'mean_pitch': pitch
        }
    except Exception as e:
        print(f"Error processing audio file {wav_path}: {e}")
        return None

def calculate_conciseness(transcript):
    words = transcript.split()
    content_words = [word for word in words if word.lower() not in stop_words]
    conciseness_ratio = len(content_words) / len(words) if len(words) > 0 else 0
    excess_words_percentage = (1 - conciseness_ratio) * 100
    return conciseness_ratio, excess_words_percentage

def get_rephrasing_suggestions(transcript):
    inputs = rephrase_tokenizer.encode("paraphrase: " + transcript, return_tensors="pt", max_length=512, truncation=True)
    outputs = rephrase_model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    rephrased_text = rephrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rephrased_text


def summarize_text(text):
    # Extract key points from the transcript
    key_points = extract_key_points(text)

    # Extract named entities from the transcript
    entities = extract_entities(text)

    # Create a guided input for the summary generation
    guided_input = create_guided_input(text, key_points, entities)
    inputs = rephrase_tokenizer.encode(guided_input, return_tensors="pt", max_length=1024, truncation=True)
    outputs = rephrase_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = rephrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def extract_text_features(transcript):
    transcript = clean_transcript(transcript)
    words = transcript.split()
    word_count = len(words)
    filler_word_count = sum(1 for word in words if word.lower() in filler_words)
    stopword_count = sum(1 for word in words if word.lower() in stop_words)

    sentences = re.split(r'[.!?]', transcript)
    sentences = [s for s in sentences if s]
    sentence_count = len(sentences)
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0

    doc = nlp(transcript)
    sentence_depths = [len([token for token in sent]) for sent in doc.sents]
    avg_sentence_depth = np.mean(sentence_depths) if sentence_depths else 0

    grammar_score = max(0, 100 - (avg_words_per_sentence - 20) ** 2)
    matches = tool.check(transcript)
    corrected_transcript = language_tool_python.utils.correct(transcript, matches)
    grammar_errors = len(matches)
    suggestions = []
    for match in matches:
        suggestions.append({
            "offset": match.offset,
            "length": match.errorLength,
            "message": match.message,
            "replacement": match.replacements[0] if match.replacements else ""
        })

    # sentiment = sentiment_analysis(transcript)
    sentiment = analyzer.polarity_scores(transcript)
    conciseness_ratio, excess_words_percentage = calculate_conciseness(transcript)
    rephrasing_suggestions = get_rephrasing_suggestions(transcript)

    return {
        'word_count': word_count,
        'filler_word_count': filler_word_count,
        'stopword_count': stopword_count,
        'avg_words_per_sentence': avg_words_per_sentence,
        'avg_sentence_depth': avg_sentence_depth,
        'grammar_score': grammar_score,
        'corrected_transcript': corrected_transcript,
        'grammar_errors': grammar_errors,
        'suggestions': suggestions,
        'sentiment': sentiment,
        'conciseness_ratio': conciseness_ratio,
        'excess_words_percentage': excess_words_percentage,
        'rephrasing_suggestions': rephrasing_suggestions
    }

def convert_to_wav(file_path):
    if not file_path.lower().endswith('.wav'):
        audio = AudioSegment.from_file(file_path)
        new_file_path = os.path.splitext(file_path)[0] + '.wav'
        audio.export(new_file_path, format='wav')
        return new_file_path
    return file_path

def speech_to_text(audio_file_path):
    result = speech_to_text_model.transcribe(audio_file_path, word_timestamps=True)
    segments = result["segments"]
    
    word_timestamps = []
    for segment in segments:
        for word in segment["words"]:
            word_timestamps.append({
                'word': word['word'],
                'start_time': word['start'],
                'end_time': word['end']
            })
    return result["text"], word_timestamps

def get_audio_duration(audio_path):
    with wave.open(audio_path, 'r') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
    return duration

def calculate_pacing(transcript, duration_seconds):
    word_count = len(transcript.split())
    pacing = word_count / (duration_seconds / 60)
    return pacing

def generate_feedback(predictions, corrected_transcript, suggestions, text_features):
    feedback = {
        'Quality Score': f"{predictions[0]:.2f}",
        'Clarity': f"{predictions[1]:.2f}",
        'Engagement': f"{predictions[2]:.2f}",
        'Professionalism': f"{predictions[3]:.2f}",
        'Overall Score': f"{predictions[4]:.2f}",
        'Corrected Transcript': corrected_transcript,
        'Suggestions': suggestions,
        'Conciseness': {
            'Ratio': f"{text_features['conciseness_ratio']:.2f}",
            'Excess Words Percentage': f"{text_features['excess_words_percentage']:.2f}%",
            'Rephrasing Suggestions': text_features['rephrasing_suggestions']
        }
    }
    return feedback


@app.route('/check_mongo_connection', methods=['GET'])
def check_mongo_connection():
    try:
        # The ismaster command is cheap and does not require auth.
        mongo.cx.admin.command('ismaster')
        return jsonify({"message": "MongoDB is connected!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({'status': 'success', 'message': 'Backend is connected!'})

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    # Check if mongo.db is None
    if mongo.db is None:
        return jsonify({"error": "Database connection error"}), 500


    if mongo.db.users.find_one({"email": data['email']}):
        return jsonify({"error": "Email already exists"}), 400

    hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
    user_id = str(ObjectId())
    mongo.db.users.insert_one({
        "_id": user_id,
        "name":data['username'],
        "email": data['email'],
        "password": hashed_password
    })
    return jsonify({"message": "User registered successfully", "user_id": user_id}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = mongo.db.users.find_one({"email": data['email']})
    if user and bcrypt.checkpw(data['password'].encode('utf-8'), user['password']):
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.datetime.now(timezone.utc) + datetime.timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        return jsonify({
            "token": token,
            "user_id": str(user['_id']),
            "name": user['name']  # Include the user's name in the response
        }), 200
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/')
def home():
    return render_template('index.html')

def convert_np_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def find_folder(parent_id, folder_name):
    files = m.get_files()
    for file_id, file_info in files.items():
        if file_info['a'].get('n') == folder_name and file_info['t'] == 1 and file_info.get('p') == parent_id:
            return file_id
    return None
def get_root_id():
    # Retrieve all files and folders
    print("2.1")
    files = m.get_files()
    print("2.2")
    # Find the root folder
    for file_id, file_info in files.items():
        if file_info['t'] == 2:  # type 2 is root
            print("2.3")
            return file_id
        
    print("2.4")
    return None

def get_folder_id(parent_folder_id, folder_name):
    # Retrieve all files and folders
    files = m.get_files()
    print("1.1")
    # Iterate through the files and folders to find the folder with the specified name under the given parent folder
    for file_id, file_info in files.items():
        if file_info['a'].get('n') == folder_name and file_info['t'] == 1 and file_info.get('p') == parent_folder_id:
            print("1.3")
            return file_id
    print("1.2")
    return None

def find_nested_folder(parent_folder_name, child_folder_name):
    # Find the parent folder ID
    print("find root folder id")
    root_id = get_root_id()  # Root folder ID
    print("1")
    if not root_id:
        print("Root folder not found.")
        return None
    parent_folder_id = get_folder_id(root_id, parent_folder_name)
    print("2")
    if parent_folder_id:
        # Find the child folder ID within the parent folder
        child_folder_id = get_folder_id(parent_folder_id, child_folder_name)
        print("3")
        if child_folder_id:
            print("4")
            return child_folder_id
        else:
            print("5")
            print(f'Child folder "{child_folder_name}" not found in parent folder "{parent_folder_name}".')
            child_folder_id = create_folder_if_not_exists(parent_folder_id, child_folder_name)
            print("8")
            return child_folder_id
    else:
        print("6")
        print(f'Parent folder "{parent_folder_name}" not found.')
    print("7")
    return None

def create_folder_if_not_exists(parent_folder_id, folder_name):
    # folder= m.find(folder_name, exclude_deleted=True)
    # if folder:
    #     return folder[0]
    # else:
    print("creating folder")
    folder = m.create_folder(folder_name, parent_folder_id)
    print("folder created",folder)
    return folder[0]

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files or 'user_id' not in request.form:
            return jsonify({"error": "No file or user_id provided"}), 400

        file = request.files['file']
        transcript = ''
        if 'transcript' in request.form:
            transcript = request.form['transcript']
        user_id = request.form['user_id']
        if not user_id:
            return jsonify({"error": "No user_id provided"}), 400
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_path = convert_to_wav(file_path)

        if 'transcript' not in request.form:
            transcript, word_info = speech_to_text(file_path)
        if not transcript:
            return jsonify({"error": "Failed to convert speech to text"}), 500

        audio_features = extract_audio_features(file_path)
        if not audio_features:
            return jsonify({"error": "Failed to extract audio features"}), 500

        text_features = extract_text_features(transcript)
        corrected_transcript = text_features.pop('corrected_transcript')
        features = {**audio_features, **text_features}

        mean_mfcc = np.array(features['mean_mfcc'])
        mean_chroma = np.array(features['mean_chroma'])
        other_features = np.array([
            features['mean_zcr'],
            features['stopping_time'],
            features['word_count'],
            features['filler_word_count'],
            features['stopword_count'],
            features['avg_sentence_depth'],
            features['grammar_score'],
            features['grammar_errors']
        ], dtype=np.float32)
        if mean_mfcc.ndim == 0:
            mean_mfcc = np.array([mean_mfcc])
        if mean_chroma.ndim == 0:
            mean_chroma = np.array([mean_chroma])
        if other_features.ndim == 0:
            other_features = np.array([other_features])
        
        combined_features = np.concatenate((mean_mfcc, mean_chroma, other_features))
        if combined_features.size < 10:
            combined_features = np.pad(combined_features, (0, 10 - combined_features.size), mode='constant')
        elif combined_features.size > 10:
            combined_features = combined_features[:10]

        scaler = StandardScaler()
        combined_features = scaler.fit_transform(combined_features.reshape(-1, 1)).flatten()
        X_input = combined_features.reshape(1, 1, 10)

        predictions = model.predict(X_input)[0]

        duration_seconds = get_audio_duration(file_path)
        pacing = calculate_pacing(transcript, duration_seconds)
        
        # Get root node
        # root_id = m.get_files()['0']['h']
        
        # # Ensure the main folder 'vocalAI' exists
        # vocalai_folder_id = m.find('vocalAI', exclude_deleted=True)
        
        # Ensure the user folder exists within 'vocalAI'
        print('start mega')
        user_folder_id = find_nested_folder('vocalAI', user_id)
        print("user_folder_id",user_folder_id)
        # Upload file to the user folder in Mega
        uploaded_file = m.upload(file_path, dest=user_folder_id)
        print("audio uploaded",uploaded_file)
        # Get the file link
        # Ensure the file is uploaded successfully
        if 'f' not in uploaded_file or not uploaded_file['f']:
            return jsonify({'error': 'Failed to upload file'}), 500

        # Extract the file ID from the upload response
        file_id = uploaded_file['f'][0]['h']
        print("fileID",file_id)
        # Get the file link
        file_link = m.get_upload_link(uploaded_file)
        # file_link = m.get_link(uploaded_file)
        print("file_link",file_link)
        transcript_summary = summarize_text(transcript)


        feedback = generate_feedback(predictions, corrected_transcript, features['suggestions'], text_features)
        result = {
            "user_id": user_id,
            "file_path": file_link,
            "file_id":file_id,
            "file_name":file.filename,
            "transcript": transcript,
            "word_timestamps": word_info,
            "transcript_summary": transcript_summary,
            'input_features': {
                'mean_mfcc': convert_np_to_python(features['mean_mfcc']),
                'mean_zcr': convert_np_to_python(features['mean_zcr']),
                'filler_word_count': convert_np_to_python(features['filler_word_count']),
                'grammar_errors': convert_np_to_python(features['grammar_errors']),
                'mean_chroma': convert_np_to_python(features['mean_chroma']),
                'stopword_count': convert_np_to_python(features['stopword_count']),
                'avg_words_per_sentence': convert_np_to_python(features['avg_words_per_sentence']),
                'avg_sentence_depth': convert_np_to_python(features['avg_sentence_depth']),
                'grammar_score': convert_np_to_python(features['grammar_score']),
                'sentiment': convert_np_to_python(features['sentiment']),
                'pacing': convert_np_to_python(pacing),
                'mean_pitch': convert_np_to_python(features['mean_pitch']),
                'stopping_time': convert_np_to_python(features['stopping_time']),
                'duration_seconds': convert_np_to_python(duration_seconds),
                'conciseness_ratio': convert_np_to_python(features['conciseness_ratio']),
                'excess_words_percentage': convert_np_to_python(features['excess_words_percentage'])
            },
            "feedback": feedback,
            'timestamp': datetime.datetime.now(timezone.utc)
        }

        mongo.db.results.insert_one(result)
        results_cursor = mongo.db.results.find({"user_id": user_id})
        results = []
        for result in results_cursor:
            result['_id'] = str(result['_id'])
            result['user_id'] = str(result['user_id'])
            result['input_features'] = {k: v for k, v in result.get('input_features', {}).items()}
            result['feedback'] = {k: v for k, v in result.get('feedback', {}).items()}
            results.append(result)

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results', methods=['GET'])
def get_results():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        results_cursor = mongo.db.results.find({"user_id": user_id})
        results = []
        for result in results_cursor:
            result['_id'] = str(result['_id'])
            result['user_id'] = str(result['user_id'])
            result['input_features'] = {k: v for k, v in result.get('input_features', {}).items()}
            result['feedback'] = {k: v for k, v in result.get('feedback', {}).items()}
            results.append(result)

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<result_id>', methods=['GET'])
def get_result(result_id):
    try:
        result = mongo.db.results.find_one({"_id": ObjectId(result_id)})
        if not result:
            return jsonify({"error": "Result not found"}), 404

        result['_id'] = str(result['_id'])
        result['user_id'] = str(result['user_id'])
        result['input_features'] = {k: v for k, v in result.get('input_features', {}).items()}
        result['feedback'] = {k: v for k, v in result.get('feedback', {}).items()}

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def download_file(file_url, dest_path):
    try:
        # Ensure the destination path exists
        dest_path = os.path.join(os.getcwd(), dest_path)
        if not os.path.exists(dest_path):
            print("in mkdir")
            os.makedirs(dest_path)
        
        # Download the file using the file ID
        m.download_url(file_url, dest_path)
        # print('file_path11:',file_path)
        # Ensure the file is properly closed after download
        # if isinstance(file, list) and len(file) > 0:
        #     print('file:',file)
        #     file = file[0]
        # return file_path
        return None

    except Exception as e:
        print(f"Failed to download file: {e}")
        return None
    
@app.route('/audio/<result_id>', methods=['GET'])
def get_audio(result_id):
    try:
        print("init",result_id)
        result = mongo.db.results.find_one({"_id": ObjectId(result_id)})
        if not result:
            return jsonify({"error": "Result not found"}), 404
        print("result",result['file_id'])
        download_file(result['file_path'], f"downloads/{result['user_id']}")
        # print("file_path",file_path)
        # if not file_path:
        #     return jsonify({"error": "Failed to download file"}), 500
        # file_full_path = f'downloads/{result['file_name']}'
        file_full_path = os.path.join(os.path.join(os.getcwd(), f"downloads/{result['user_id']}"), result['file_name'])
        if not os.path.exists(file_full_path):
            return jsonify({"error": "File not found"}), 404
        print("full_file_path",file_full_path)
        # response = send_file(file_full_path, as_attachment=True)
        
        # Serve the file using a custom response
        def generate():
            with open(file_full_path, 'rb') as f:
                yield from f
            # Delete the file after sending
            os.remove(file_full_path)
            print("Deleted local file:", file_full_path)

        response = Response(generate(), mimetype='audio/mpeg')
        response.headers.set('Content-Disposition', 'attachment')
        return response
    except Exception as e:
        print("in exception")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
