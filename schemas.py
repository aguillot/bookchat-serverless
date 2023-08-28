from typing import List

from pydantic import UUID4, BaseModel, constr


class EmbedInput(BaseModel):
    book_id: UUID4
    openai_api_key: constr(min_length=1)


class QueryInput(BaseModel):
    book_id: UUID4
    openai_api_key: constr(min_length=1)
    chat_query: constr(min_length=1)
    chat_history: List[str]
