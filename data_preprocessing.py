import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

# Define column names based on NSL-KDD documentation
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label", "difficulty_level"
]

# Mapping attacks to 5 specific classes
attack_mapping = {
    'normal': 'Normal',
    'neptune': 'DoS', 'smurf': 'DoS', 'pod': 'DoS', 'teardrop': 'DoS', 
    'land': 'DoS', 'back': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS', 
    'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe', 
    'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L', 
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L', 
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L', 
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 
    'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, names=col_names)
    df.drop('difficulty_level', axis=1, inplace=True) # Drop difficulty column
    
    # Map label to 5 categories
    df['label'] = df['label'].map(lambda x: attack_mapping.get(x, 'Unknown'))
    
    # If any unknown, filter them out or keep
    df = df[df['label'] != 'Unknown']
    
    return df

def preprocess_features(train_df, test_df):
    print("Preprocessing data (Encoding and Scaling)...")
    
    # Identify categorical columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    # We combine train and test specifically to fit the encoder on all possibilities
    combined_df = pd.concat([train_df, test_df])
    
    # Label Encoders
    encoders = {}
    for col in categorical_columns:
        encoders[col] = LabelEncoder()
        combined_df[col] = encoders[col].fit_transform(combined_df[col])
        
    # Split back to train and test
    train_encoded = combined_df[:len(train_df)].copy()
    test_encoded = combined_df[len(train_df):].copy()
    
    # Extract features and targets
    X_train = train_encoded.drop('label', axis=1)
    y_train = train_encoded['label']
    
    X_test = test_encoded.drop('label', axis=1)
    y_test = test_encoded['label']
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Encode Target Labels
    target_encoder = LabelEncoder()
    y_train_encoded = pd.Series(target_encoder.fit_transform(y_train), name='label')
    y_test_encoded = pd.Series(target_encoder.transform(y_test), name='label')
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, target_encoder

def apply_smote(X_train, y_train):
    print("Applying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled
