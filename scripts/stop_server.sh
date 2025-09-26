#!/bin/bash
# Movie Genie Server Cleanup Script

echo "🧹 Cleaning up Movie Genie processes..."

# Kill Flask processes
pkill -f "python.*app.py" 2>/dev/null && echo "✅ Killed Flask processes"

# Kill processes on common ports
for port in 5000 5001 8080; do
    pids=$(lsof -ti :$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "🔪 Killing processes on port $port: $pids"
        kill -9 $pids 2>/dev/null
    fi
done

# Clean up any remaining Movie Genie processes
pkill -f "movie.genie" 2>/dev/null
pkill -f "Movie Genie" 2>/dev/null

# Remove DVC locks
rm -rf .dvc/tmp/rwlock 2>/dev/null && echo "🔓 Removed DVC locks"

echo "✅ Cleanup complete!"