import secrets
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_user
from ..db import get_session
from ..models import User
from ..schemas import SettingsRead, SettingsUpdate, TokenRead

router = APIRouter(tags=["settings"])


@router.get("/me/settings", response_model=SettingsRead)
async def read_settings(user: Annotated[User, Depends(current_user)]) -> User:
    return user


@router.patch("/me/settings", response_model=SettingsRead)
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


@router.post("/me/token", response_model=TokenRead)
async def rotate_token(
    user: Annotated[User, Depends(current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> TokenRead:
    user.api_token = secrets.token_urlsafe(32)
    await session.commit()
    await session.refresh(user)
    assert user.api_token is not None
    return TokenRead(token=user.api_token)
