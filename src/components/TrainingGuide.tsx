import React, { useState } from 'react';
import { Code, Database, Brain, CheckCircle, AlertCircle, Play, FileText, Cpu, Target } from 'lucide-react';

const TrainingGuide = () => {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      id: 'setup',
      title: 'Environment Setup',
      icon: Code,
      description: 'Install required libraries and set up your development environment'
    },
    {
      id: 'preprocessing',
      title: 'Data Preprocessing',
      icon: Database,
      description: 'Clean and prepare your dataset for training'
    },
    {
      id: 'text-model',
      title: 'Text Classification Model',
      icon: FileText,
      description: 'Train models for SMS/Email scam detection'
    },
    {
      id: 'audio-model',
      title: 'Audio Classification Model',
      icon: Cpu,
      description: 'Train models for phone call scam detection'
    },
    {
      id: 'video-model',
      title: 'Video Analysis Model',
      icon: Target,
      description: 'Train models for deepfake video detection'
    },
    {
      id: 'deployment',
      title: 'Model Deployment',
      icon: Play,
      description: 'Deploy your trained models for real-time detection'
    }
  ];

  const codeExamples = {
    setup: `# Install required libraries
pip install pandas numpy scikit-learn
pip install transformers torch
pip install librosa soundfile  # For audio processing
pip install opencv-python face-recognition  # For video processing
pip install flask fastapi  # For API deployment

# Create project structure
mkdir scam-detection-project
cd scam-detection-project
mkdir data models notebooks src
mkdir data/text data/audio data/video`,

    preprocessing: `import pandas as pd
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your generated dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

# Load and preprocess datasets
text_df = load_dataset('text-dataset.json')
audio_df = load_dataset('audio-dataset.json')
video_df = load_dataset('video-dataset.json')

# Preprocess text content
text_df['content_clean'] = text_df['content'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
text_df['label_encoded'] = label_encoder.fit_transform(text_df['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    text_df['content_clean'], 
    text_df['label_encoded'], 
    test_size=0.2, 
    random_state=42,
    stratify=text_df['label_encoded']
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")`,

    textModel: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Method 1: Traditional ML with TF-IDF
def train_traditional_model():
    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test_vec)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred))
    
    return rf_model, vectorizer

# Method 2: BERT-based model
def train_bert_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Tokenize data
    train_encodings = tokenizer(
        list(X_train), 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors='pt'
    )
    
    test_encodings = tokenizer(
        list(X_test), 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors='pt'
    )
    
    # Create dataset class
    class ScamDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels.iloc[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    train_dataset = ScamDataset(train_encodings, y_train)
    test_dataset = ScamDataset(test_encodings, y_test)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained('./models/bert-scam-detector')
    tokenizer.save_pretrained('./models/bert-scam-detector')
    
    return model, tokenizer

# Train both models
traditional_model, vectorizer = train_traditional_model()
bert_model, bert_tokenizer = train_bert_model()`,

    audioModel: `import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from gtts import gTTS
import os

# Step 1: Convert text to speech (for your generated audio dataset)
def text_to_speech(text, filename, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    return filename

# Generate audio files from your dataset
def generate_audio_files(audio_df):
    audio_files = []
    for idx, row in audio_df.iterrows():
        filename = f"audio_{idx}_{row['label']}.mp3"
        text_to_speech(row['content'], filename)
        audio_files.append({
            'file': filename,
            'label': row['label'],
            'content': row['content']
        })
    return audio_files

# Step 2: Extract audio features
def extract_audio_features(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        
        # Extract features
        features = {}
        
        # Spectral features
        features['mfcc'] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitch_mean'] = np.mean(pitches[pitches > 0])
        
        # Combine all features
        feature_vector = np.concatenate([
            features['mfcc'],
            [features['spectral_centroid']],
            [features['spectral_rolloff']],
            [features['zero_crossing_rate']],
            [features['tempo']],
            [features['pitch_mean']]
        ])
        
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 3: Train audio classification model
def train_audio_model(audio_files):
    features = []
    labels = []
    
    for audio_file in audio_files:
        feature_vector = extract_audio_features(audio_file['file'])
        if feature_vector is not None:
            features.append(feature_vector)
            labels.append(1 if audio_file['label'] == 'scam' else 0)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    audio_model = RandomForestClassifier(n_estimators=100, random_state=42)
    audio_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = audio_model.predict(X_test)
    print("Audio Model Results:")
    print(classification_report(y_test, y_pred))
    
    return audio_model, scaler

# Generate audio files and train model
audio_files = generate_audio_files(audio_df)
audio_model, audio_scaler = train_audio_model(audio_files)`,

    videoModel: `import cv2
import face_recognition
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import dlib

# Step 1: Video feature extraction
def extract_video_features(video_path):
    """Extract features from video for deepfake detection"""
    cap = cv2.VideoCapture(video_path)
    features = []
    
    frame_count = 0
    face_consistency_scores = []
    
    # Initialize face detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    while cap.isOpened() and frame_count < 100:  # Process first 100 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_recognition.face_locations(rgb_frame)
        
        if faces:
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, faces)
            
            for face_encoding in face_encodings:
                # Face consistency check (for deepfake detection)
                if frame_count > 0:
                    # Compare with previous frame's face
                    consistency = face_recognition.face_distance([prev_encoding], face_encoding)[0]
                    face_consistency_scores.append(consistency)
                
                prev_encoding = face_encoding
        
        frame_count += 1
    
    cap.release()
    
    # Calculate features
    video_features = {
        'avg_face_consistency': np.mean(face_consistency_scores) if face_consistency_scores else 0,
        'face_consistency_std': np.std(face_consistency_scores) if face_consistency_scores else 0,
        'frames_with_faces': len(face_consistency_scores),
        'total_frames': frame_count
    }
    
    return list(video_features.values())

# Step 2: Create synthetic video data (since we have scripts)
def create_video_dataset_features(video_df):
    """Create features based on video scripts and scene descriptions"""
    features = []
    labels = []
    
    for idx, row in video_df.iterrows():
        # Extract features from scene description and dialogue
        scene_features = extract_scene_features(row['scene'])
        dialogue_features = extract_dialogue_features(row['dialogue'])
        
        # Combine features
        combined_features = scene_features + dialogue_features
        features.append(combined_features)
        labels.append(1 if row['label'] == 'scam' else 0)
    
    return np.array(features), np.array(labels)

def extract_scene_features(scene_description):
    """Extract features from scene description"""
    features = []
    
    # Check for suspicious elements
    suspicious_words = ['fake', 'uniform', 'badge', 'official', 'government', 'police']
    legit_words = ['office', 'desk', 'professional', 'bank', 'branch']
    
    features.append(sum(1 for word in suspicious_words if word in scene_description.lower()))
    features.append(sum(1 for word in legit_words if word in scene_description.lower()))
    features.append(len(scene_description.split()))  # Description length
    
    return features

def extract_dialogue_features(dialogue):
    """Extract features from dialogue content"""
    features = []
    
    # Threat indicators
    threat_words = ['arrest', 'jail', 'fine', 'penalty', 'immediately', 'urgent', 'transfer']
    legit_words = ['appointment', 'verification', 'documents', 'visit', 'branch', 'help']
    
    features.append(sum(1 for word in threat_words if word in dialogue.lower()))
    features.append(sum(1 for word in legit_words if word in dialogue.lower()))
    features.append(len(dialogue.split()))  # Dialogue length
    features.append(dialogue.count('!'))  # Exclamation marks (urgency indicator)
    
    return features

# Step 3: Train video classification model
def train_video_model(video_df):
    X, y = create_video_dataset_features(video_df)
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train model
    video_model = RandomForestClassifier(n_estimators=100, random_state=42)
    video_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = video_model.predict(X_test)
    print("Video Model Results:")
    print(classification_report(y_test, y_pred))
    
    return video_model, scaler

# Train video model
video_model, video_scaler = train_video_model(video_df)`,

    deployment: `from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained models
text_model = joblib.load('models/text_model.pkl')
text_vectorizer = joblib.load('models/text_vectorizer.pkl')
audio_model = joblib.load('models/audio_model.pkl')
audio_scaler = joblib.load('models/audio_scaler.pkl')
video_model = joblib.load('models/video_model.pkl')
video_scaler = joblib.load('models/video_scaler.pkl')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    try:
        data = request.json
        text = data['content']
        channel = data.get('channel', 'unknown')
        
        # Preprocess text
        text_clean = preprocess_text(text)
        
        # Vectorize
        text_vec = text_vectorizer.transform([text_clean])
        
        # Predict
        prediction = text_model.predict(text_vec)[0]
        probability = text_model.predict_proba(text_vec)[0]
        
        result = {
            'prediction': 'scam' if prediction == 1 else 'legit',
            'confidence': float(max(probability)),
            'scam_probability': float(probability[1]),
            'channel': channel
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    try:
        # Handle audio file upload
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Save temporarily
        temp_path = f"temp_{audio_file.filename}"
        audio_file.save(temp_path)
        
        # Extract features
        features = extract_audio_features(temp_path)
        
        if features is not None:
            # Scale features
            features_scaled = audio_scaler.transform([features])
            
            # Predict
            prediction = audio_model.predict(features_scaled)[0]
            probability = audio_model.predict_proba(features_scaled)[0]
            
            result = {
                'prediction': 'scam' if prediction == 1 else 'legit',
                'confidence': float(max(probability)),
                'scam_probability': float(probability[1])
            }
        else:
            result = {'error': 'Could not process audio file'}
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        data = request.json
        scene = data['scene']
        dialogue = data['dialogue']
        
        # Extract features
        scene_features = extract_scene_features(scene)
        dialogue_features = extract_dialogue_features(dialogue)
        features = scene_features + dialogue_features
        
        # Scale features
        features_scaled = video_scaler.transform([features])
        
        # Predict
        prediction = video_model.predict(features_scaled)[0]
        probability = video_model.predict_proba(features_scaled)[0]
        
        result = {
            'prediction': 'scam' if prediction == 1 else 'legit',
            'confidence': float(max(probability)),
            'scam_probability': float(probability[1])
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict/unified', methods=['POST'])
def predict_unified():
    """Unified endpoint that handles all communication types"""
    try:
        data = request.json
        channel = data['channel']
        
        if channel in ['sms', 'email']:
            return predict_text()
        elif channel == 'call':
            return predict_audio()
        elif channel == 'video':
            return predict_video()
        else:
            return jsonify({'error': 'Unsupported channel type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Save models
def save_models():
    joblib.dump(traditional_model, 'models/text_model.pkl')
    joblib.dump(vectorizer, 'models/text_vectorizer.pkl')
    joblib.dump(audio_model, 'models/audio_model.pkl')
    joblib.dump(audio_scaler, 'models/audio_scaler.pkl')
    joblib.dump(video_model, 'models/video_model.pkl')
    joblib.dump(video_scaler, 'models/video_scaler.pkl')

if __name__ == '__main__':
    save_models()
    app.run(debug=True, host='0.0.0.0', port=5000)`
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">ML Training Guide</h2>
        <p className="text-gray-600">
          Complete step-by-step guide to train machine learning models using your generated datasets
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          {steps.map((step, index) => {
            const Icon = step.icon;
            const isActive = index === activeStep;
            const isCompleted = index < activeStep;
            
            return (
              <div key={step.id} className="flex items-center">
                <button
                  onClick={() => setActiveStep(index)}
                  className={`flex items-center justify-center w-10 h-10 rounded-full border-2 transition-colors ${
                    isActive
                      ? 'bg-blue-600 border-blue-600 text-white'
                      : isCompleted
                      ? 'bg-green-600 border-green-600 text-white'
                      : 'bg-white border-gray-300 text-gray-400 hover:border-gray-400'
                  }`}
                >
                  {isCompleted ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <Icon className="w-5 h-5" />
                  )}
                </button>
                {index < steps.length - 1 && (
                  <div className={`w-16 h-0.5 mx-2 ${
                    isCompleted ? 'bg-green-600' : 'bg-gray-300'
                  }`} />
                )}
              </div>
            );
          })}
        </div>
        
        <div className="text-center">
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            {steps[activeStep].title}
          </h3>
          <p className="text-gray-600">
            {steps[activeStep].description}
          </p>
        </div>
      </div>

      {/* Content Area */}
      <div className="bg-white rounded-xl shadow-sm border">
        <div className="p-6">
          <div className="mb-6">
            <div className="flex items-center space-x-2 mb-4">
              <div className="bg-blue-100 p-2 rounded-lg">
                {React.createElement(steps[activeStep].icon, { className: "w-5 h-5 text-blue-600" })}
              </div>
              <h4 className="text-lg font-semibold text-gray-900">
                Step {activeStep + 1}: {steps[activeStep].title}
              </h4>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg p-4 overflow-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
              {codeExamples[steps[activeStep].id]}
            </pre>
          </div>

          {/* Navigation */}
          <div className="flex justify-between mt-6">
            <button
              onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
              disabled={activeStep === 0}
              className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              onClick={() => setActiveStep(Math.min(steps.length - 1, activeStep + 1))}
              disabled={activeStep === steps.length - 1}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Additional Resources */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-3">
            <Brain className="w-5 h-5 text-blue-600" />
            <h5 className="font-semibold text-blue-900">Model Performance Tips</h5>
          </div>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use cross-validation for better model evaluation</li>
            <li>• Implement ensemble methods for higher accuracy</li>
            <li>• Regular model retraining with new data</li>
            <li>• Monitor for concept drift in production</li>
          </ul>
        </div>

        <div className="bg-amber-50 border border-amber-200 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-3">
            <AlertCircle className="w-5 h-5 text-amber-600" />
            <h5 className="font-semibold text-amber-900">Important Considerations</h5>
          </div>
          <ul className="text-sm text-amber-800 space-y-1">
            <li>• Test thoroughly to avoid false positives</li>
            <li>• Consider privacy and data protection laws</li>
            <li>• Implement human review for edge cases</li>
            <li>• Regular security audits of deployed models</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default TrainingGuide;