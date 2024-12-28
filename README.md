# Vocal AI Backend

This is the backend for the Vocal AI project. It is a Flask-based application that performs various speech and text processing tasks, including speech-to-text conversion, audio feature extraction, and text analysis.

## Features

- User authentication (registration and login)
- Speech-to-text conversion using Whisper
- Audio feature extraction using librosa
- Text feature extraction using spaCy and other NLP tools
- Grammar checking and correction using LanguageTool
- Sentiment analysis using VADER
- Audio file upload to Mega cloud storage
- Result storage and retrieval from MongoDB

## Requirements

- Python 3.8 or later
- Flask
- Flask-CORS
- Flask-PyMongo
- PyMongo
- Whisper
- spacy
- nltk
- language_tool_python
- librosa
- pydub
- bcrypt
- jwt
- tensorflow
- vaderSentiment
- transformers
- mega.py
- MongoDB (e.g., MongoDB Atlas)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/vocal-ai-backend.git
    cd vocal-ai-backend
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Download the required spaCy model:

    ```sh
    python -m spacy download en_core_web_sm
    ```

## Configuration

1. Set up your MongoDB URI in the `app.py` file:

    ```python
    username = urllib.parse.quote_plus('your_mongo_username')
    password = urllib.parse.quote_plus('your_mongo_password')
    app.config["MONGO_URI"] = f'mongodb+srv://{username}:{password}@your_mongo_cluster/vocal_ai_db?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
    ```

2. Set up your Mega account credentials in the `app.py` file:

    ```python
    MEGA_EMAIL = 'your_mega_email'
    MEGA_PASSWORD = 'your_mega_password'
    ```

## Running the Application

1. Run the Flask application:

    ```sh
    python app.py
    ```

2. The application should be running on `http://127.0.0.1:5000/`.

## API Endpoints

- `/check_mongo_connection`: Check MongoDB connection
- `/api/test`: Test API connection
- `/register`: Register a new user
- `/login`: User login
- `/upload`: Upload an audio file for processing
- `/results`: Get results for a specific user
- `/result/<result_id>`: Get a specific result by ID
- `/audio/<result_id>`: Download processed audio file

## Deployment

To deploy the application to Azure, follow these steps:

1. Ensure your project structure includes `requirements.txt`, `runtime.txt`, and `web.config` files.
2. Create a GitHub repository and push your project.
3. Set up an Azure App Service and configure it to deploy from your GitHub repository.
4. Monitor the deployment process and logs in the Azure Portal.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [Whisper](https://github.com/openai/whisper)
- [spaCy](https://spacy.io/)
- [nltk](https://www.nltk.org/)
- [LanguageTool](https://github.com/languagetool-org/languagetool-python)
- [librosa](https://librosa.org/)
- [VADER](https://github.com/cjhutto/vaderSentiment)
- [Transformers](https://huggingface.co/transformers/)
- [Mega.py](https://github.com/odwyersoftware/mega.py)
