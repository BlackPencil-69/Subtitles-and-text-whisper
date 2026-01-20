import os
import sys
import whisper
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import threading
import time
import subprocess
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mkv', 'flac', 'm4a', 'ogg', 'webm', 'mov'}
MAX_FILE_SIZE = 500 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['MAX_CONTENT_PATH'] = None
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

progress_data = {}
models_cache = {}

def allowed_file(filename):
    """Checks allowed file formats"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    """Extracts audio from video using FFmpeg"""
    try:
        logger.info(f"Video conversion: {video_path} -> {audio_path}")
        subprocess.run([
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', 
            audio_path, '-y'
        ], check=True, capture_output=True, text=True)
        logger.info("Conversion successful")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Install FFmpeg: sudo apt-get install ffmpeg")
        return False

def load_whisper_model(model_size="base"):
    """Loads Whisper model with caching"""
    try:
        if model_size not in models_cache:
            logger.info(f"Loading Whisper model: {model_size}")
            models_cache[model_size] = whisper.load_model(model_size)
            logger.info(f"Model {model_size} loaded successfully")
        return models_cache[model_size]
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise

def split_segments_by_length(segments, max_words=5, max_chars=35):
    """
    Splits long Whisper segments into smaller chunks based on word timestamps.
    """
    new_segments = []
    
    for segment in segments:
        if 'words' not in segment or not segment['words']:
            new_segments.append(segment)
            continue
            
        words = segment['words']
        current_chunk = []
        current_chars = 0
        
        for i, word_data in enumerate(words):
            word_text = word_data['word']
            current_chunk.append(word_data)
            current_chars += len(word_text)
            
            if len(current_chunk) >= max_words or current_chars >= max_chars or i == len(words) - 1:
                new_segments.append({
                    'start': current_chunk[0]['start'],
                    'end': current_chunk[-1]['end'],
                    'text': ''.join([w['word'] for w in current_chunk]).strip()
                })
                current_chunk = []
                current_chars = 0
                
    return new_segments

def detect_speech_boundaries(audio_path, segments):
    """Refines speech boundaries using audio energy analysis"""
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000)
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(range(len(energy)), sr=sr, hop_length=hop_length)
        threshold = np.percentile(energy, 20)
        
        refined_segments = []
        for segment in segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            if not text: continue
            
            search_window_start = max(0, start - 0.5)
            search_window_end = min(len(audio) / sr, end + 0.5)
            start_idx = np.searchsorted(times, search_window_start)
            end_idx = np.searchsorted(times, search_window_end)
            
            segment_energy = energy[start_idx:end_idx]
            segment_times = times[start_idx:end_idx]
            
            if len(segment_energy) == 0:
                refined_segments.append({'start': start, 'end': end, 'text': text})
                continue
            
            speech_mask = segment_energy > threshold
            speech_indices = np.where(speech_mask)[0]
            
            if len(speech_indices) > 0:
                actual_start = segment_times[speech_indices[0]]
                actual_end = segment_times[speech_indices[-1]]
                actual_start = max(start - 0.3, actual_start)
                actual_end = min(end + 0.2, actual_end + 0.3)
            else:
                actual_start, actual_end = start, end
            
            if actual_end - actual_start < 0.2:
                actual_end = actual_start + 0.2
            
            refined_segments.append({'start': actual_start, 'end': actual_end, 'text': text})
        
        for i in range(len(refined_segments) - 1):
            if refined_segments[i]['end'] > refined_segments[i + 1]['start']:
                gap = (refined_segments[i + 1]['start'] + refined_segments[i]['end']) / 2
                refined_segments[i]['end'] = gap - 0.05
                refined_segments[i + 1]['start'] = gap + 0.05
        
        return refined_segments
    except Exception as e:
        logger.error(f"Speech boundary detection error: {e}")
        return basic_speech_detection(segments)

def basic_speech_detection(segments):
    """Basic timestamp correction algorithm"""
    refined_segments = []
    for segment in segments:
        start, end, text = segment.get('start', 0), segment.get('end', 0), segment.get('text', '').strip()
        if not text: continue
        
        adjusted_start = start + 0.1
        adjusted_end = max(adjusted_start + 0.2, end - 0.05)
        
        refined_segments.append({'start': adjusted_start, 'end': adjusted_end, 'text': text})
    return refined_segments

def generate_srt(segments):
    """Generates SRT format"""
    srt_lines = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp_srt(segment['start'])
        end = format_timestamp_srt(segment['end'])
        srt_lines.append(f"{i}\n{start} --> {end}\n{segment['text']}\n")
    return '\n'.join(srt_lines)

def format_timestamp_srt(seconds):
    """Format: HH:MM:SS,mmm"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((secs % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{millis:03d}"

def transcribe_audio(file_path, language, model_size, task_id, max_words):
    """Main transcription process with dynamic segment length"""
    try:
        progress_data[task_id] = {'status': 'loading_model', 'progress': 10, 'message': 'Loading model...'}
        model = load_whisper_model(model_size)
        
        should_split = int(max_words) < 25
        
        progress_data[task_id] = {'status': 'processing', 'progress': 30, 'message': 'Processing audio...'}
        
        transcribe_options = {
            'verbose': False,
            'task': 'transcribe',
            'word_timestamps': True
        }
        if language and language != 'auto':
            transcribe_options['language'] = language
        
        result = model.transcribe(file_path, **transcribe_options)
        
        raw_segments = result.get('segments', [])

        if should_split:
            limit = int(max_words)
            progress_data[task_id] = {'status': 'refining', 'progress': 70, 'message': f'Splitting into {limit} words...'}
            processed_segments = split_segments_by_length(raw_segments, max_words=limit, max_chars=limit*7)
        else:
            progress_data[task_id] = {'status': 'refining', 'progress': 70, 'message': 'Refining original segments...'}
            processed_segments = raw_segments
        
        refined_segments = detect_speech_boundaries(file_path, processed_segments)
        
        progress_data[task_id] = {'status': 'completed', 'progress': 100, 'message': 'Done!', 'result': {
            'full_text': result['text'].strip(),
            'timestamped_text': '\n'.join([f"[{format_timestamp(s['start'])} → {format_timestamp(s['end'])}] {s['text']}" for s in refined_segments]),
            'srt_subtitles': generate_srt(refined_segments),
            'detected_language': result.get('language', 'unknown'),
            'segments_count': len(refined_segments)
        }}
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        progress_data[task_id] = {'status': 'error', 'progress': 0, 'message': f'Error: {str(e)}'}
    finally:
        if os.path.exists(file_path): 
            os.remove(file_path)

def format_timestamp(seconds):
    """Format: MM:SS.mmm"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((secs % 1) * 1000)
    return f"{int(minutes):02d}:{int(secs):02d}.{millis:03d}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info(f"Upload request received. Files: {request.files.keys()}")
        logger.info(f"Form data: {request.form}")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'File not found'}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Unsupported format. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
        
        language = request.form.get('language', 'auto')
        model_size = request.form.get('model_size', 'base')
        max_words = request.form.get('max_words', 5)
        
        task_id = str(int(time.time() * 1000))
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{task_id}_{filename}")
        
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        if file_ext in {'mp4', 'avi', 'mkv', 'webm', 'mov'}:
            audio_path = file_path.rsplit('.', 1)[0] + '.wav'
            logger.info(f"Converting video to audio: {file_ext} -> wav")
            if extract_audio(file_path, audio_path):
                os.remove(file_path)
                file_path = audio_path
                logger.info("Video converted successfully")
            else:
                logger.error("FFmpeg conversion failed")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': 'FFmpeg error. Please install FFmpeg or use audio file directly'}), 500
        
        threading.Thread(target=transcribe_audio, args=(file_path, language, model_size, task_id, max_words), daemon=True).start()
        logger.info(f"Transcription started. Task ID: {task_id}")
        return jsonify({'task_id': task_id})
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/progress/<task_id>')
def get_progress(task_id):
    return jsonify(progress_data.get(task_id, {'status': 'not_found'}))

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f} MB'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)