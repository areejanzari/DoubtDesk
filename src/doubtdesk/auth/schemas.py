from pydantic import BaseModel, EmailStr, Field, field_validator


class SignupRequest(BaseModel):
    username: str = Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9_]+$",
    )
    email: EmailStr
    password: str = Field(
        min_length=8,
        max_length=128,
    )

    @field_validator("username", mode="before")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

class SignupResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    email_verified: bool

class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=1)

class ResendVerificationRequest(BaseModel):
    email: EmailStr

class LoginRequest(BaseModel):
    identifier: str = Field(min_length=1)
    password: str = Field(min_length=1)

class LoginResponse(BaseModel):
    user_id: str
    username: str
    email: EmailStr
    csrf_token: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)
