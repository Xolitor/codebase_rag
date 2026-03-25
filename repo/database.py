class Database:
    def __init__(self):
        self.connection = None

    def connect(self):
        self.connection = "connected"
        return "Database connected"

    def disconnect(self):
        self.connection = None
        return "Database disconnected"

    def fetch_user(self, user_id):
        return {"id": user_id, "name": "Test User"}