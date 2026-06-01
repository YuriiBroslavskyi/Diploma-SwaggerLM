from fastapi import FastAPI, HTTPException, Depends
from typing import Optional

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int, include_posts: Optional[bool] = False):
    """Get a user by their ID."""
    if not user_id:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user_id}

@app.post("/users")
async def create_user(name: str, email: str):
    """Create a new user."""
    return {"name": name, "email": email}

@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user by ID."""
    return {"deleted": user_id}

@app.get("/users")
async def list_users(limit: int = 10, offset: int = 0):
    """List all users with pagination."""
    return []
