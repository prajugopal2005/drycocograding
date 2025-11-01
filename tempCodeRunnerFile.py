def log(message):
    """Write to both console and log file"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg, flush=True)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')