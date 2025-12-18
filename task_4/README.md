

## Extract Dataset from Kaggle
```bash
kaggle export API_KEY = "your kaggle api key here"
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv.zip
```

## Run the application
```bash
pip install -r requirements.txt
cd task_4
streamlit run main_4.py
```