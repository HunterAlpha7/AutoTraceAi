import bcrypt
from database import User
from sqlalchemy import or_

def verify_password(plain_password, hashed_password):
    # bcrypt expects bytes
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)

def get_password_hash(password):
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    return hashed_bytes.decode('utf-8')

def authenticate_user(db, identifier, password):
    user = db.query(User).filter(
        or_(User.username == identifier, User.email == identifier)
    ).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

def create_user(db, username, email, password):
    existing_user = db.query(User).filter(
        or_(User.username == username, User.email == email)
    ).first()
    if existing_user:
        return None
    
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, password_hash=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
