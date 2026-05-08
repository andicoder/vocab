import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import User
from ..operations import import_kindle_entries

router = APIRouter(tags=["imports"])


class ImportResult(BaseModel):
    added: int
    skipped: int


@router.post("/import/kindle", response_model=ImportResult)
async def import_kindle(
    file: Annotated[UploadFile, File()],
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ImportResult:
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        tf.write(data)
        tmp_path = Path(tf.name)
    try:
        added, skipped = await import_kindle_entries(session=session, user=user, db_path=tmp_path)
        await session.commit()
    finally:
        tmp_path.unlink(missing_ok=True)
    return ImportResult(added=added, skipped=skipped)
