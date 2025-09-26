# Initial Configuration:

- $ cd car-insurance-claim-utility
- $ python3 -m venv .venv
- $ source .venv/bin/activate
- $ pip install -r requirements.txt
- $ python3 -c "from app.db import init_db; init_db()"
- $ uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
- $ python3 ml/train_multi_input.py
- $ cd frontend
- $ npm start
