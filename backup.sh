#!/bin/bash
set -e

echo "🔄 Starting backup procedure..."

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Database backup
echo "📊 Backing up database..."
docker exec postgres pg_dump -U postgres agi_eval > "$BACKUP_DIR/database.sql"

# Redis backup
echo "📦 Backing up Redis..."
docker exec redis redis-cli --rdb > "$BACKUP_DIR/redis.rdb"

# Configuration backup
echo "⚙️  Backing up configurations..."
cp -r /root/repo/.env.* "$BACKUP_DIR/"
cp -r /root/repo/docker-compose*.yml "$BACKUP_DIR/"

# Application data backup
echo "💾 Backing up application data..."
docker exec agi-eval-api tar -czf - /app/data > "$BACKUP_DIR/app_data.tar.gz"

# Upload to cloud storage (example with AWS S3)
# aws s3 cp "$BACKUP_DIR" s3://your-backup-bucket/agi-eval-sandbox/ --recursive

echo "✅ Backup completed: $BACKUP_DIR"

# Cleanup old backups (keep last 30 days)
find /backups -type d -mtime +30 -exec rm -rf {} +
