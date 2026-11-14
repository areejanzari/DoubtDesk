import os
import logging
import smtplib
from fastapi import (
    APIRouter,
    HTTPException, 
    status,
    Request,
    Response,
    Depends,
)
from doubtdesk.auth.mailer import send_verification_email, send_password_reset_email
from doubtdesk.auth.schemas import (
    SignupRequest,
    SignupResponse,
    VerifyEmailRequest,
    ResendVerificationRequest,
    LoginRequest,
    LoginResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
)
from doubtdesk.auth.service import register_user, authenticate_user
from doubtdesk.auth.dependencies import require_current_user, get_client_ip
from doubtdesk.auth.store import (
    create_verification_token,
    verify_email_token,
    resend_verification_token,
    create_session,
    revoke_session,
    create_password_reset_token,
    reset_password,
    is_login_rate_limited,
    record_failed_login,
    consume_auth_attempt,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)


def _limit_auth_action(action, request, email):
    if not consume_auth_attempt(action, get_client_ip(request), email):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")


@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
)
def signup(payload: SignupRequest, request: Request):
    _limit_auth_action("signup", request, str(payload.email))
    try:
        user = register_user(payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    token = create_verification_token(user["user_id"])

    try:
        send_verification_email(
            to_email=user["email"],
            token=token,
        )
    except (smtplib.SMTPException, OSError) as exc:
        logger.exception("Verification email delivery failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account created, but verification email could not be sent.",
        ) from exc

    return user


@router.post("/verify-email")
def verify_email(payload: VerifyEmailRequest):
    verified = verify_email_token(payload.token)

    if not verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already used verification token.",
        )

    return {"message": "Email verified successfully."}


@router.post("/resend-verification")
def resend_verification(payload: ResendVerificationRequest, request: Request):
    _limit_auth_action("resend-verification", request, str(payload.email))
    message = {
        "message": (
            "If this email belongs to an unverified account "
            "and a new link is allowed, a verification email "
            "will be sent."
        )
    }

    token = resend_verification_token(str(payload.email))

    if token is None:
        return message

    try:
        send_verification_email(
            to_email=str(payload.email),
            token=token,
        )
    except (smtplib.SMTPException, OSError):
        logger.exception("Failed to resend verification email")

    return message



@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    request: Request,
    response: Response,
):
    identifier = payload.identifier.strip()
    ip_address = get_client_ip(request)

    if is_login_rate_limited(identifier, ip_address):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again later.",
        )

    user = authenticate_user(
        identifier=identifier,
        password=payload.password,
    )

    if user is None:
        record_failed_login(identifier, ip_address)

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or unverified account.",
        )

    session_token, csrf_token = create_session(
        user_id=user["user_id"],
    )

    response.set_cookie(
        key="doubtdesk_session",
        value=session_token,
        max_age=4 * 60 * 60,
        httponly=True,
        secure=os.getenv("APP_ENV") == "production",
        samesite="lax",
        path="/",
    )

    return {
        **user,
        "csrf_token": csrf_token,
    }


@router.get("/me")
def get_my_profile(
    user: dict = Depends(require_current_user),
):
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user["email"],
        "csrf_token": user["csrf_token"],
    }


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    user: dict = Depends(require_current_user),
):
    session_token = request.cookies["doubtdesk_session"]

    revoke_session(session_token)

    response.delete_cookie(
        key="doubtdesk_session",
        path="/",
    )

    return {"message": "Logged out successfully."}

@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest, request: Request):
    _limit_auth_action("forgot-password", request, str(payload.email))
    message = {
        "message": (
            "If the email belongs to a verified account, "
            "a password reset link will be sent."
        )
    }

    token = create_password_reset_token(str(payload.email))

    if token is None:
        return message

    try:
        send_password_reset_email(
            to_email=str(payload.email),
            token=token,
        )
    except (smtplib.SMTPException, OSError):
        logger.exception("Password reset email delivery failed")

    return message


@router.post("/reset-password")
def reset_account_password(payload: ResetPasswordRequest):
    success = reset_password(
        token=payload.token,
        new_password=payload.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid, expired, or already used reset token.",
        )

    return {"message": "Password reset successfully."}
