from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import User
from ..schemas import SettingsRead, SettingsUpdate

router = APIRouter(prefix="/me/settings", tags=["settings"])


@router.get("", response_model=SettingsRead)
async def read_settings(user: Annotated[User, Depends(current_user)]) -> User:
    return user


@router.patch("", response_model=SettingsRead)
async def update_settings(
    payload: SettingsUpdate,
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    if payload.card_direction is not None:
        user.card_direction = payload.card_direction
    await session.commit()
    await session.refresh(user)
    return user
