### 1. Navigate to your project folder:

(powershell)

in my case is:

    cd "C:\Users\kikal\OneDrive\Documentos\WorkspaceCarolina\AIB\Module2\Aitana"

### 2. Activate the virtual environment :

(powershell)

    .\.venv\Scripts\Activate

### 3. Set the environment variables:

(powershell)

    $env:FLASK_APP = "app.py"
    $env:FLASK_ENV = "development"  

### 4. Start the Flask server:

(powershell)

    flask run

### yey :) Then open your browser and go to:

    http://127.0.0.1:5000