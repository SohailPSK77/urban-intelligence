"""
SIH26124: Role-Based Authentication & Mobile OTP Engine (Phase 7)
Local prototype role-based authentication using PBKDF2-HMAC-SHA256 password hashing
and Mobile Number OTP password reset verification.
"""

import hashlib
import secrets
import sqlite3
import os
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_bytes(32)
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return hash_bytes.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """
    Verifies a plain password against stored hex hash and salt.
    """
    salt_bytes = bytes.fromhex(stored_salt)
    computed_hash, _ = hash_password(password, salt_bytes)
    return secrets.compare_digest(computed_hash, stored_hash)


def normalize_mobile(mobile: str) -> str:
    """Extracts last 10 digits of a mobile number."""
    digits = re.sub(r"\D", "", mobile)
    return digits[-10:] if len(digits) >= 10 else digits


def init_user_db(db_path: str = DB_PATH):
    """
    Initializes local SQLite user database and seeds prototype accounts with mobile numbers.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL,
            bus_id TEXT,
            route_id TEXT,
            operator_id TEXT,
            mobile_number TEXT NOT NULL DEFAULT '9876543210',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Ensure mobile_number column exists if upgrading existing table
    cursor.execute("PRAGMA table_info(users)")
    cols = [row[1] for row in cursor.fetchall()]
    if "mobile_number" not in cols:
        cursor.execute("ALTER TABLE users ADD COLUMN mobile_number TEXT NOT NULL DEFAULT '9876543210'")

    conn.commit()

    # Ensure all 18 buses + official accounts exist with updated mobile numbers
    mobile_pool = ["9491591473", "7842835677", "7995974455", "6303133198"]
    routes = ["ROUTE-101", "ROUTE-202", "ROUTE-303", "ROUTE-404"]

    seed_users = []
    # Seed 18 Bus accounts (BUS-01 to BUS-18)
    for i in range(1, 19):
        b_id = f"BUS-{i:02d}"
        r_id = routes[(i - 1) % 4]
        mob = mobile_pool[(i - 1) % len(mobile_pool)]
        # Map explicit requested mobiles & routes
        if b_id == "BUS-07":
            mob = "9491591473"
            r_id = "ROUTE-101"
        elif b_id == "BUS-08":
            mob = "7842835677"
            r_id = "ROUTE-202"
        
        seed_users.append((b_id, f"Onboard Bus {i:02d}", "bus123", "BUS", b_id, r_id, "APSRTC-VIZAG", mob))

    # Seed Official accounts
    seed_users.extend([
        ("OFFICIAL-001", "Chief Traffic Commander", "admin123", "OFFICIAL", None, None, "APSRTC-HQ", "7995974455"),
        ("OFFICIAL-002", "Fleet Operations Manager", "admin123", "OFFICIAL", None, None, "APSRTC-HQ", "6303133198"),
        ("OFFICIAL-003", "Transit Infrastructure Inspector", "admin123", "OFFICIAL", None, None, "APSRTC-HQ", "9491591473"),
        ("OFFICIAL-004", "Smart City Traffic Controller", "admin123", "OFFICIAL", None, None, "APSRTC-HQ", "7842835677"),
    ])

    for u_id, u_name, raw_pwd, role, b_id, r_id, op_id, mob in seed_users:
        p_hash, p_salt = hash_password(raw_pwd)
        cursor.execute("""
            INSERT OR IGNORE INTO users (user_id, username, password_hash, salt, role, bus_id, route_id, operator_id, mobile_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (u_id, u_name, p_hash, p_salt, role, b_id, r_id, op_id, mob))
    
    conn.commit()
    conn.close()


def authenticate_user(user_id: str, password: str, selected_role: str, db_path: str = DB_PATH) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticates user ID, password, and expected role.
    Returns (success, user_dict, error_message).
    """
    user_id = user_id.strip().upper() if user_id else ""
    if not user_id:
        return False, None, "User ID is required."
    if not password:
        return False, None, "Password is required."

    init_user_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, None, f"Unknown ID '{user_id}'. Please check your account credentials."

    user = dict(row)

    if not user.get("is_active"):
        return False, None, f"Account '{user_id}' is currently deactivated."

    # Verify password hash
    if not verify_password(password, user["password_hash"], user["salt"]):
        return False, None, "Invalid password. Access denied."

    # Verify role match
    if user["role"] != selected_role.upper():
        return False, None, f"Role mismatch: Account '{user_id}' has role '{user['role']}', but '{selected_role.upper()}' was selected."

    # Remove hash and salt before returning user context
    del user["password_hash"]
    del user["salt"]

    return True, user, "Authentication successful."


def get_user_info(user_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """
    Retrieves user information by ID.
    """
    init_user_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT user_id, username, role, bus_id, route_id, operator_id, mobile_number, is_active, created_at FROM users WHERE user_id = ?", (user_id.strip().upper(),))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def update_user_password(user_id: str, new_password: str, db_path: str = DB_PATH) -> bool:
    """
    Updates the password for a specific user ID with a newly generated salt and PBKDF2 hash.
    """
    user_id = user_id.strip().upper()
    p_hash, p_salt = hash_password(new_password)
    init_user_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET password_hash = ?, salt = ? WHERE user_id = ?", (p_hash, p_salt, user_id))
    rows = cursor.rowcount
    conn.commit()
    conn.close()
    return rows > 0


def generate_mobile_otp(user_id: str, mobile_number: str, db_path: str = DB_PATH) -> Tuple[bool, Optional[str], Optional[str], str]:
    """
    Verifies user ID and mobile number, generates 6-digit OTP string.
    Returns (success, otp_code, masked_mobile, message).
    """
    user_id = user_id.strip().upper() if user_id else ""
    if not user_id:
        return False, None, None, "Account ID is required."
    if not mobile_number:
        return False, None, None, "Registered Mobile Number is required."

    init_user_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, mobile_number FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False, None, None, f"Unknown Account ID '{user_id}'."

    stored_mob = normalize_mobile(row["mobile_number"])
    input_mob = normalize_mobile(mobile_number)

    if stored_mob != input_mob:
        return False, None, None, f"Mobile number mismatch for Account ID '{user_id}'."

    # Generate 6-digit numeric OTP
    otp_code = f"{secrets.randbelow(900000) + 100000}"
    masked_mobile = f"+91 ******{stored_mob[-4:]}"

    return True, otp_code, masked_mobile, f"OTP dispatched to registered mobile {masked_mobile}."


def verify_otp_and_reset_password(user_id: str, input_otp: str, valid_otp: str, new_password: str, db_path: str = DB_PATH) -> Tuple[bool, str]:
    """
    Verifies OTP and resets user password with PBKDF2 hash.
    """
    user_id = user_id.strip().upper()
    if not input_otp or not valid_otp:
        return False, "OTP verification required."
    if input_otp.strip() != valid_otp.strip():
        return False, "Invalid 6-digit OTP entered. Verification failed."
    if not new_password or len(new_password) < 4:
        return False, "New password must be at least 4 characters long."

    success = update_user_password(user_id, new_password, db_path=db_path)
    if success:
        return True, f"Password successfully updated for '{user_id}'. You can now log in with your new password."
    else:
        return False, f"Failed to update password for account '{user_id}'."


# Ensure database is initialized upon import
init_user_db()
