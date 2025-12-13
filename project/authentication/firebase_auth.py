"""
Firebase Admin SDK initialization and authentication utilities.
"""
import os
import firebase_admin
from firebase_admin import credentials, auth
from django.conf import settings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Initialize Firebase Admin SDK
def initialize_firebase():
    """
    Initialize Firebase Admin SDK with project credentials.
    Uses environment variables for configuration.
    """
    if not firebase_admin._apps:
        # Use environment variables for Firebase initialization
        cred = credentials.Certificate({
            "type": "service_account",
            "project_id": os.getenv('FIREBASE_PROJECT_ID'),
            "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
            "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.getenv('FIREBASE_CLIENT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
        })
        firebase_admin.initialize_app(cred)


def verify_firebase_token(id_token):
    """
    Verify a Firebase ID token and return the decoded token.

    Args:
        id_token (str): The Firebase ID token to verify

    Returns:
        dict: Decoded token containing user information
        None: If token is invalid

    Raises:
        firebase_admin.auth.InvalidIdTokenError: If token is invalid
        firebase_admin.auth.ExpiredIdTokenError: If token is expired
        firebase_admin.auth.RevokedIdTokenError: If token is revoked
    """
    try:
        initialize_firebase()
        decoded_token = auth.verify_id_token(id_token)
        return decoded_token
    except Exception as e:
        print(f"Firebase token verification error: {str(e)}")
        return None


def get_user_info_from_token(decoded_token):
    """
    Extract user information from decoded Firebase token.

    Args:
        decoded_token (dict): Decoded Firebase token

    Returns:
        dict: User information including uid, email, display_name, etc.
    """
    if not decoded_token:
        return None

    return {
        'firebase_uid': decoded_token.get('uid'),
        'email': decoded_token.get('email'),
        'email_verified': decoded_token.get('email_verified', False),
        'display_name': decoded_token.get('name'),
        'avatar_url': decoded_token.get('picture'),
        'auth_provider': decoded_token.get('firebase', {}).get('sign_in_provider', 'unknown'),
    }
