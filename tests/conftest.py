import os
from dotenv import load_dotenv

# load the .env file before any test or sdk initialization
load_dotenv()

# (optional) print to confirm env vars are loaded
print("loaded environment variable ATHINA_API_KEY:", os.getenv("ATHINA_API_KEY"))
print("loaded environment variable API_BASE_URL:", os.getenv("API_BASE_URL"))
