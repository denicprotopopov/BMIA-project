from flask import Flask, request, render_template
import os
from RunPipeline_GetSegments import predict_seizure_ranges

UPLOAD_FOLDER = 'uploads'
MODEL_DIR    = 'models'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    # Run prediction inside try/except
    try:
        segments = predict_seizure_ranges(filepath, MODEL_DIR)
        if not segments:
            return "No seizures detected.", 200

        results = [f"Seizure {i+1}: Start = {start:.2f}s, End = {end:.2f}s"
                   for i, (start, end) in enumerate(segments)]
        return "<br>".join(results)

    except Exception as e:
        # Print the full traceback to console
        import traceback
        traceback.print_exc()
        # Return the exception message to the user (for debugging)
        return f"Error during prediction: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
