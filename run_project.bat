@echo off
echo Installing requirements...
python -m pip install -r requirements.txt
echo.
echo Training LSTM model...
python train_model.py
echo.
echo Starting Streamlit...
streamlit run app.py
pause
