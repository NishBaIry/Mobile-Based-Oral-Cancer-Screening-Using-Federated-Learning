#!/usr/bin/env python3
"""
Cleanup Old Training Files
Keeps only the N most recent files in client_weights/ folder
"""

import os
import glob
from pathlib import Path
from datetime import datetime

# Configuration
KEEP_LAST_N = 3  # Keep only last 3 training sessions
CLIENT_WEIGHTS_DIR = Path(__file__).parent / 'client_weights'

def parse_timestamp_from_filename(filename):
    """Extract timestamp from filename like mobilenetv2_20250122_153045_final.h5"""
    try:
        # Format: modelname_YYYYMMDD_HHMMSS_suffix.ext
        parts = filename.split('_')
        if len(parts) >= 3:
            date_str = parts[-3]  # YYYYMMDD
            time_str = parts[-2]  # HHMMSS
            timestamp_str = f"{date_str}_{time_str}"
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except:
        pass
    return None

def get_training_sessions():
    """Group files by training session (same timestamp)"""
    if not CLIENT_WEIGHTS_DIR.exists():
        print(f"❌ Directory not found: {CLIENT_WEIGHTS_DIR}")
        return {}
    
    sessions = {}
    
    # Find all files in client_weights
    for file_path in CLIENT_WEIGHTS_DIR.glob('*'):
        if file_path.is_file():
            timestamp = parse_timestamp_from_filename(file_path.name)
            if timestamp:
                timestamp_key = timestamp.strftime("%Y%m%d_%H%M%S")
                if timestamp_key not in sessions:
                    sessions[timestamp_key] = {
                        'timestamp': timestamp,
                        'files': []
                    }
                sessions[timestamp_key]['files'].append(file_path)
    
    return sessions

def cleanup_old_files():
    """Remove old training sessions, keeping only KEEP_LAST_N most recent"""
    print(f"🧹 Cleaning up old training files...")
    print(f"   Keeping last {KEEP_LAST_N} training sessions")
    print()
    
    sessions = get_training_sessions()
    
    if not sessions:
        print("✅ No training files found to clean up")
        return
    
    # Sort sessions by timestamp (newest first)
    sorted_sessions = sorted(
        sessions.items(),
        key=lambda x: x[1]['timestamp'],
        reverse=True
    )
    
    print(f"📊 Found {len(sorted_sessions)} training session(s):")
    for session_key, session_data in sorted_sessions:
        timestamp = session_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        file_count = len(session_data['files'])
        print(f"   {timestamp}: {file_count} file(s)")
    print()
    
    # Keep the N most recent, delete the rest
    sessions_to_keep = sorted_sessions[:KEEP_LAST_N]
    sessions_to_delete = sorted_sessions[KEEP_LAST_N:]
    
    if not sessions_to_delete:
        print("✅ All sessions are recent, nothing to delete")
        return
    
    # Delete old sessions
    deleted_count = 0
    for session_key, session_data in sessions_to_delete:
        timestamp = session_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        print(f"🗑️  Deleting session: {timestamp}")
        
        for file_path in session_data['files']:
            try:
                file_path.unlink()
                print(f"   ✓ Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"   ✗ Failed to delete {file_path.name}: {e}")
    
    print()
    print(f"✅ Cleanup complete!")
    print(f"   Deleted: {deleted_count} file(s) from {len(sessions_to_delete)} session(s)")
    print(f"   Kept: {len(sessions_to_keep)} most recent session(s)")

if __name__ == '__main__':
    cleanup_old_files()
