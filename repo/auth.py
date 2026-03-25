def login(username, password):
    if username == "admin" and password == "1234":
        return {"status": "success", "token": "abc123"}
    return {"status": "error", "message": "Invalid credentials"}


def logout(user_id):
    return f"User {user_id} logged out"