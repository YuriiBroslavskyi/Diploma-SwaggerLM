# repos/bookstore/routers/reviews.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class ReviewCreate(BaseModel):
    book_id: int
    rating: int
    comment: str

reviews_db = []

@router.get("/reviews")
async def list_reviews(book_id: int = None):
    if book_id:
        return [r for r in reviews_db if r["book_id"] == book_id]
    return reviews_db

@router.post("/reviews")
async def create_review(review: ReviewCreate):
    entry = {"id": len(reviews_db) + 1, **review.dict()}
    reviews_db.append(entry)
    return entry

@router.get("/reviews/{review_id}")
async def get_review(review_id: int):
    for r in reviews_db:
        if r["id"] == review_id:
            return r
    raise HTTPException(status_code=404, detail="Review not found")