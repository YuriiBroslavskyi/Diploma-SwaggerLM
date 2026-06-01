# repos/bookstore/routers/authors.py
from fastapi import APIRouter, HTTPException

router = APIRouter()

authors_db = {
    1: {"id": 1, "name": "George Orwell", "country": "UK"},
    2: {"id": 2, "name": "Frank Herbert", "country": "USA"},
}

@router.get("/authors")
async def list_authors():
    return list(authors_db.values())

@router.get("/authors/{author_id}")
async def get_author(author_id: int):
    if author_id not in authors_db:
        raise HTTPException(status_code=404, detail="Author not found")
    return authors_db[author_id]