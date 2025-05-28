from flask import Flask, request, render_template, send_file
import os
from RunPipeline_GetSegments import predict_seizure_ranges, process_video_with_keypoints

UPLOAD_FOLDER    = 'uploads'
PROCESSED_FOLDER = 'processed_videos'
MODEL_DIR        = 'models'

app = Flask(__name__)
app.config['UPLOAD_FOLDER']     = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video file provided.', 400

    video = request.files['video']
    if video.filename == '':
        return 'Empty filename.', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(filepath)

    processed_video_filename = "processed_" + video.filename
    processed_filepath       = os.path.join(app.config['PROCESSED_FOLDER'], processed_video_filename)

    try:
        segments = predict_seizure_ranges(filepath, MODEL_DIR)
        
        process_video_with_keypoints(filepath, processed_filepath, seizure_segments=segments)

        if not segments:
            return render_template('upload.html',
                                   message="No seizures detected.",
                                   processed_video_url=f'/{app.config["PROCESSED_FOLDER"]}/{processed_video_filename}')

        results = [f"Seizure {i+1}: Start = {start:.2f}s, End = {end:.2f}s"
                   for i, (start, end) in enumerate(segments)]
        return render_template('upload.html',
                               results=results,
                               processed_video_url=f'/{app.config["PROCESSED_FOLDER"]}/{processed_video_filename}')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error during processing: {e}", 500

@app.route('/processed_videos/<filename>')
def serve_processed_video(filename):
    file_full_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)

    if not os.path.exists(file_full_path):
        return "File not found.", 404

    return send_file(file_full_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)