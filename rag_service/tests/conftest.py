# Disable vector DB preload during tests (avoids real API calls and slow startup)
import os
os.environ["PRELOAD_VECTOR_DB"] = "0"
