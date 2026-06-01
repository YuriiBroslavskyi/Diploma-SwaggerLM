# repos/bookstore/routers/books.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class BookCreate(BaseModel):
    title: str
    author: str
    year: int

books_db = {
    1: {"id": 1, "title": "1984", "author": "George Orwell", "year": 1949},
    2: {"id": 2, "title": "Dune", "author": "Frank Herbert", "year": 1965},
}

@router.get("/books")
async def list_books(limit: int = 10):
    return list(books_db.values())[:limit]

@router.get("/books/{book_id}")
async def get_book(book_id: int):
    if book_id not in books_db:
        raise HTTPException(status_code=404, detail="Book not found")
    return books_db[book_id]

@router.post("/books")
async def create_book(book: BookCreate):
    new_id = max(books_db.keys()) + 1
    books_db[new_id] = {"id": new_id, **book.dict()}
    return books_db[new_id]

@router.delete("/books/{book_id}")
async def delete_book(book_id: int):
    if book_id not in books_db:
        raise HTTPException(status_code=404, detail="Book not found")
    del books_db[book_id]
    return {"deleted": book_id}