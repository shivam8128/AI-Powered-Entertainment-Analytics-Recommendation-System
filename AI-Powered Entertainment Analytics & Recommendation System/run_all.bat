@echo off
echo Installing requirements...
python -m pip install -r requirements.txt

echo Retrieving genuine Internet datasets...
python data\download_data.py

echo Training AI Models securely...
python src\recommender.py
python src\sentiment.py
python src\engagement.py


echo Everything complete! 
echo Run 'streamlit run app.py' to open the app.
pause
