# Oracle Database Configuration Example
# Add this to your settings.py when ready to use Oracle DB

import os

# Oracle Database Configuration (Autonomous Database with Wallet)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': os.getenv('ORACLE_DSN', 'visionforgedb_medium'),  # Connection string from tnsnames.ora
        'USER': os.getenv('ORACLE_USER', 'ADMIN'),
        'PASSWORD': os.getenv('ORACLE_PASSWORD'),
        'OPTIONS': {
            # Wallet configuration for mTLS connection
            'config_dir': os.getenv('TNS_ADMIN', '/opt/oracle/wallet'),
            'wallet_location': os.getenv('WALLET_LOCATION', '/opt/oracle/wallet'),
            'wallet_password': os.getenv('ORACLE_WALLET_PASSWORD'),
        },
    }
}

# Alternative: Using python-oracledb directly (recommended for new projects)
"""
import oracledb

# For Thick mode with wallet
oracledb.init_oracle_client(config_dir=os.getenv('TNS_ADMIN', '/opt/oracle/wallet'))

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.oracle',
        'NAME': os.getenv('ORACLE_DSN'),
        'USER': os.getenv('ORACLE_USER'),
        'PASSWORD': os.getenv('ORACLE_PASSWORD'),
    }
}
"""

# NOTE: For production, ensure these environment variables are set on Render:
# - ORACLE_USER
# - ORACLE_PASSWORD
# - ORACLE_DSN
# - ORACLE_WALLET_PASSWORD
# - TNS_ADMIN=/opt/oracle/wallet
# - WALLET_LOCATION=/opt/oracle/wallet
