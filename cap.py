from flask import Flask, render_template, request, jsonify
import os
import tempfile
import cv2
import pytesseract
import yt_dlp
import threading
import uuid
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

processing_jobs = {}

def check_dependencies():
    try:
        import cv2, pytesseract, yt_dlp
    except ImportError as e:
        raise Exception(f"Missing dependency: {e.name}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_video():
    url = request.json['url']
    job_id = str(uuid.uuid4())
    
    processing_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Starting analysis...',
        'results': []
    }
    
    thread = threading.Thread(target=process_background, args=(url, job_id))
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def job_status(job_id):
    return jsonify(processing_jobs.get(job_id, {'error': 'Invalid job ID'}))

def process_background(url, job_id):
    try:
        update_job(job_id, 10, 'Downloading video...', 'downloading')
        video_path = download_video(url)
        
        update_job(job_id, 30, 'Processing frames...', 'processing')
        detected_texts = process_frames(video_path, job_id)
        
        update_job(job_id, 90, 'Finalizing results...', 'processing')
        os.remove(video_path)
        
        detected_locations = '\n- ' + '\n- '.join(detected_texts) if detected_texts else 'No location detected'
        update_job(job_id, 100, f'Analysis complete!\nDetected locations:{detected_locations}', 'completed', detected_texts)
        
    except Exception as e:
        update_job(job_id, 100, f'Error: {str(e)}', 'error')

def update_job(job_id, progress, message, status, results=None):
    job = processing_jobs[job_id]
    job.update({
        'progress': progress,
        'message': message,
        'status': status
    })
    if results is not None:
        job['results'] = results

def download_video(url: str) -> str:
    ydl_opts = {
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9, ]', '', text)
    words = text.split()
    filtered_words = [word for word in words if len(word) > 2]
    return ' '.join(filtered_words)

def process_frames(video_path: str, job_id: str) -> list:
    detected_text = set()
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_rate == 0:
            processed_frame = preprocess_frame(frame)
            text = pytesseract.image_to_string(processed_frame)
            clean_result = clean_text(text)
            
            if clean_result:
                detected_text.add(clean_result)
            
            processed_frames += 1
            progress = 30 + (processed_frames / (total_frames/frame_rate)) * 60
            update_job(job_id, int(progress), f'Processed {processed_frames} frames', 'processing')
        
        frame_count += 1
    
    cap.release()
    return sorted(detected_text)

if __name__ == '__main__':
    try:
        check_dependencies()
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
