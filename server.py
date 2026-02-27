import subprocess
import sys
import os
import uuid
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import zipfile
import tempfile

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_digitize_script(data_folder, output_folder, model_folder=None, verbose=True, show_image=False, allow_failures=False):
    """Launch the digitize.py script with specified arguments."""
    
    # Build the command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "/Users/alinawaf/Desktop/Research/ECG-VECG/ecgdigit/ECG-Digitiser/src/run/digitize.py",
        "-d", data_folder,
        "-o", output_folder
    ]
    
    # Add optional arguments
    if model_folder:
        cmd.extend(["-m", model_folder])
    
    if verbose:
        cmd.append("-v")
    
    if show_image:
        cmd.append("--show_image")
    
    if allow_failures:
        cmd.extend(["-f"])
    
    try:
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Script completed successfully!")
        print("STDOUT:", result.stdout)
        return True, result.stdout, ""
    except subprocess.CalledProcessError as e:
        print(f"Script failed with return code {e.returncode}")
        print("STDERR:", e.stderr)
        return False, "", e.stderr

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/digitize', methods=['POST'])
def digitize_image():
    """Process uploaded image with digitize.py script."""
    try:
        # Check if file is in request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Create unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job-specific folders
        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_output_folder = os.path.join(OUTPUT_FOLDER, job_id)
        os.makedirs(job_upload_folder, exist_ok=True)
        os.makedirs(job_output_folder, exist_ok=True)
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(job_upload_folder, filename)
        file.save(file_path)
        
        # Get optional parameters from request
        model_folder = request.form.get('model_folder', 'models/M3/')
        verbose = request.form.get('verbose', 'true').lower() == 'true'
        show_image = request.form.get('show_image', 'false').lower() == 'true'
        allow_failures = request.form.get('allow_failures', 'false').lower() == 'true'
        
        # Run digitize script
        success, stdout, stderr = run_digitize_script(
            data_folder=job_upload_folder,
            output_folder=job_output_folder,
            model_folder=model_folder,
            verbose=verbose,
            show_image=show_image,
            allow_failures=allow_failures
        )
        
        if success:
            # List output files
            output_files = []
            if os.path.exists(job_output_folder):
                for file in os.listdir(job_output_folder):
                    output_files.append(file)
            
            response = {
                "status": "success",
                "job_id": job_id,
                "message": "Image processed successfully",
                "output_files": output_files,
                "stdout": stdout
            }
            return jsonify(response), 200
        else:
            # Clean up on failure
            shutil.rmtree(job_upload_folder, ignore_errors=True)
            shutil.rmtree(job_output_folder, ignore_errors=True)
            
            return jsonify({
                "status": "error",
                "job_id": job_id,
                "message": "Processing failed",
                "stderr": stderr
            }), 500
            
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/download/<job_id>', methods=['GET'])
def download_results(job_id):
    """Download results for a specific job."""
    try:
        job_output_folder = os.path.join(OUTPUT_FOLDER, job_id)
        
        if not os.path.exists(job_output_folder):
            return jsonify({"error": "Job not found"}), 404
        
        # Create a zip file with all results
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w') as zip_file:
                for root, dirs, files in os.walk(job_output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, job_output_folder)
                        zip_file.write(file_path, arcname)
            
            return send_file(tmp_file.name, as_attachment=True, download_name=f'results_{job_id}.zip')
            
    except Exception as e:
        return jsonify({"error": f"Download error: {str(e)}"}), 500

@app.route('/status/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get status of a specific job."""
    try:
        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_output_folder = os.path.join(OUTPUT_FOLDER, job_id)
        
        upload_exists = os.path.exists(job_upload_folder)
        output_exists = os.path.exists(job_output_folder)
        
        if not upload_exists:
            return jsonify({"status": "not_found", "job_id": job_id}), 404
        
        output_files = []
        if output_exists:
            output_files = os.listdir(job_output_folder)
        
        status = "completed" if output_files else "processing"
        
        return jsonify({
            "status": status,
            "job_id": job_id,
            "output_files": output_files
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Status check error: {str(e)}"}), 500

@app.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """Clean up files for a specific job."""
    try:
        job_upload_folder = os.path.join(UPLOAD_FOLDER, job_id)
        job_output_folder = os.path.join(OUTPUT_FOLDER, job_id)
        
        removed = []
        if os.path.exists(job_upload_folder):
            shutil.rmtree(job_upload_folder)
            removed.append("upload_folder")
        
        if os.path.exists(job_output_folder):
            shutil.rmtree(job_output_folder)
            removed.append("output_folder")
        
        if not removed:
            return jsonify({"message": "Job not found or already cleaned"}), 404
        
        return jsonify({
            "status": "cleaned",
            "job_id": job_id,
            "removed": removed
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Cleanup error: {str(e)}"}), 500

if __name__ == '__main__':
    port =  5123
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    print(f"Starting ECG Digitization Server on port {port}")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"Output folder: {os.path.abspath(OUTPUT_FOLDER)}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)