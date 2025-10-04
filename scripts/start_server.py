#!/usr/bin/env python3
"""
Movie Genie Server Startup Script with Process Management
"""

import os
import sys
import signal
import subprocess
from pathlib import Path

# Global variable to store server process
server_process = None

def signal_handler(signum, frame):
    """Handle termination signals and cleanup processes"""
    print(f"\n\n👋 Shutting down Movie Genie backend...")

    if server_process:
        # Send SIGTERM to subprocess
        server_process.terminate()

        # Wait briefly for graceful shutdown
        try:
            server_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            server_process.kill()
            server_process.wait()

    # Quick cleanup of any remaining processes on the port
    port = os.environ.get('FLASK_PORT', '5001')
    try:
        result = subprocess.run(['lsof', '-ti', f':{port}'],
                              capture_output=True, text=True, check=False)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(['kill', '-9', pid], check=False, capture_output=True)
    except:
        pass

    print("✅ Shutdown complete\n")
    sys.exit(0)

def main():
    """Main function to start the server with proper signal handling"""
    global server_process

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination

    # Set default port
    port = os.environ.get('FLASK_PORT', '5001')
    os.environ['FLASK_PORT'] = port

    print(f"""
🎬 Movie Genie Server Manager
🔧 Port: {port}
📡 Starting backend server...
💡 Press Ctrl+C to stop the server and cleanup all processes
    """)

    # Change to backend directory
    backend_dir = Path(__file__).parent.parent / 'movie_genie' / 'backend'
    os.chdir(backend_dir)

    try:
        # Start the Flask server
        server_process = subprocess.Popen(
            [sys.executable, 'app.py'],
            env=os.environ.copy()
        )

        # Wait for the server process
        server_process.wait()

    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        signal_handler(signal.SIGTERM, None)

if __name__ == '__main__':
    main()