from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import get_session
from .models import User


async def current_user(
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    x_authentik_username: Annotated[str | None, Header()] = None,
) -> User:
    if not x_authentik_username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"missing {settings.auth_user_header} header",
        )

    result = await session.execute(select(User).where(User.username == x_authentik_username))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(username=x_authentik_username)
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user
