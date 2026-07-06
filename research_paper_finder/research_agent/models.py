from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class Author(BaseModel):
    name: str
    author_id: Optional[str] = None


class Paper(BaseModel):

    title: str

    abstract: Optional[str] = None

    authors: List[Author] = Field(default_factory=list)

    year: Optional[int] = None

    published: Optional[datetime] = None

    venue: Optional[str] = None

    doi: Optional[str] = None

    citation_count: Optional[int] = None

    influential_citation_count: Optional[int] = None

    pdf_url: Optional[HttpUrl] = None

    source: str

    paper_id: Optional[str] = None

    fields_of_study: List[str] = Field(default_factory=list)

    url: Optional[HttpUrl] = None
