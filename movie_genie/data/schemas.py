from pydantic import BaseModel, Field
from datetime import datetime

class MovieLensRating(BaseModel):
    user_id: int
    movie_id: int
    rating: float = Field(..., ge=0.5, le=5.0)
    timestamp: int