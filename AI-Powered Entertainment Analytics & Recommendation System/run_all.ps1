Write-Host "Installing requirements..."
python -m pip install -r requirements.txt

Write-Host "Retrieving genuine Internet datasets..."
python data\download_data.py

Write-Host "Training AI Models securely..."
python src\recommender.py
python src\sentiment.py
python src\engagement.py


Write-Host "Everything complete!"
Write-Host "Run 'streamlit run app.py' to open the app."
Read-Host -Prompt "Press Enter to continue"
