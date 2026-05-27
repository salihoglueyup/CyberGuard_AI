"""
Notifications API - REAL DATA VERSION
Uses file-based persistence for notifications
"""

import json
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.routes.auth import require_auth

router = APIRouter()
logger = logging.getLogger(__name__)

# Path setup
from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)
data_dir = os.path.join(project_root, "data")
notifications_file = os.path.join(data_dir, "notifications.json")
preferences_file = os.path.join(data_dir, "notification_preferences.json")

# In-memory store with persistence
NOTIFICATIONS = []
PREFERENCES = {
    "email_enabled": True,
    "push_enabled": True,
    "slack_enabled": False,
    "alert_types": {
        "critical": True,
        "high": True,
        "medium": True,
        "low": False,
    },
    "quiet_hours": {"enabled": False, "start": "22:00", "end": "08:00"},
}


class NotificationCreate(BaseModel):
    title: str
    message: str
    type: str = "info"  # info, warning, error, success, security
    priority: str = "medium"  # low, medium, high, critical
    link: str | None = None
    metadata: dict | None = None


class PreferencesUpdate(BaseModel):
    email_enabled: bool | None = None
    push_enabled: bool | None = None
    slack_enabled: bool | None = None
    alert_types: dict[str, bool] | None = None
    quiet_hours: dict | None = None


def load_data():
    """Load notifications and preferences from files"""
    global NOTIFICATIONS, PREFERENCES

    if os.path.exists(notifications_file):
        try:
            with open(notifications_file, encoding="utf-8") as f:
                NOTIFICATIONS = json.load(f)
        except Exception as e:
            logger.warning("Failed to load notifications: %s", e)
            NOTIFICATIONS = []

    if os.path.exists(preferences_file):
        try:
            with open(preferences_file, encoding="utf-8") as f:
                loaded = json.load(f)
                PREFERENCES.update(loaded)
        except Exception as e:
            logger.warning("Failed to load notification preferences: %s", e)


def save_notifications():
    """Save notifications to file"""
    os.makedirs(os.path.dirname(notifications_file), exist_ok=True)
    with open(notifications_file, "w", encoding="utf-8") as f:
        json.dump(NOTIFICATIONS, f, indent=2, default=str)


def save_preferences():
    """Save preferences to file"""
    os.makedirs(os.path.dirname(preferences_file), exist_ok=True)
    with open(preferences_file, "w", encoding="utf-8") as f:
        json.dump(PREFERENCES, f, indent=2)


# Initialize on module load
load_data()


@router.get("")
async def get_notifications(
    unread_only: bool = False, notification_type: str | None = None, limit: int = 50,
    user: dict = Depends(require_auth),
):
    """Get notifications"""
    filtered = NOTIFICATIONS.copy()

    if unread_only:
        filtered = [n for n in filtered if not n.get("read", False)]

    if notification_type:
        filtered = [n for n in filtered if n.get("type") == notification_type]

    # Sort by created_at descending
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Limit
    filtered = filtered[:limit]

    # Count unread
    unread_count = len([n for n in NOTIFICATIONS if not n.get("read", False)])

    return {
        "success": True,
        "data": {
            "notifications": filtered,
            "total": len(filtered),
            "unread_count": unread_count,
        },
    }


@router.get("/unread-count")
async def get_unread_count(user: dict = Depends(require_auth)):
    """Get count of unread notifications"""
    unread = len([n for n in NOTIFICATIONS if not n.get("read", False)])

    # Count by priority
    by_priority = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for n in NOTIFICATIONS:
        if not n.get("read", False):
            priority = n.get("priority", "medium")
            if priority in by_priority:
                by_priority[priority] += 1

    return {
        "success": True,
        "data": {
            "unread_count": unread,
            "by_priority": by_priority,
            "has_critical": by_priority["critical"] > 0,
        },
    }


@router.post("")
async def create_notification(notification: NotificationCreate, user: dict = Depends(require_auth)):
    """Create a new notification"""
    new_notification = {
        "id": str(uuid.uuid4())[:8],
        "title": notification.title,
        "message": notification.message,
        "type": notification.type,
        "priority": notification.priority,
        "link": notification.link,
        "metadata": notification.metadata or {},
        "read": False,
        "created_at": datetime.now().isoformat(),
    }

    NOTIFICATIONS.append(new_notification)

    # Keep only last 500 notifications
    if len(NOTIFICATIONS) > 500:
        NOTIFICATIONS[:] = NOTIFICATIONS[-500:]

    save_notifications()

    return {
        "success": True,
        "data": new_notification,
        "message": "Notification created",
    }


@router.put("/{notification_id}/read")
async def mark_as_read(notification_id: str, user: dict = Depends(require_auth)):
    """Mark a notification as read"""
    for n in NOTIFICATIONS:
        if n.get("id") == notification_id:
            n["read"] = True
            n["read_at"] = datetime.now().isoformat()
            save_notifications()
            return {"success": True, "message": "Marked as read"}

    raise HTTPException(status_code=404, detail="Notification not found")


@router.put("/read-all")
async def mark_all_as_read(user: dict = Depends(require_auth)):
    """Mark all notifications as read"""
    count = 0
    for n in NOTIFICATIONS:
        if not n.get("read", False):
            n["read"] = True
            n["read_at"] = datetime.now().isoformat()
            count += 1

    save_notifications()

    return {"success": True, "message": f"Marked {count} notifications as read"}


@router.delete("/{notification_id}")
async def delete_notification(notification_id: str, user: dict = Depends(require_auth)):
    """Delete a notification"""
    global NOTIFICATIONS

    for i, n in enumerate(NOTIFICATIONS):
        if n.get("id") == notification_id:
            deleted = NOTIFICATIONS.pop(i)
            save_notifications()
            return {"success": True, "message": "Deleted", "data": deleted}

    raise HTTPException(status_code=404, detail="Notification not found")


@router.delete("/clear-all")
async def clear_all_notifications(user: dict = Depends(require_auth)):
    """Clear all notifications"""
    global NOTIFICATIONS
    count = len(NOTIFICATIONS)
    NOTIFICATIONS = []
    save_notifications()

    return {"success": True, "message": f"Cleared {count} notifications"}


@router.get("/preferences")
async def get_preferences(user: dict = Depends(require_auth)):
    """Get notification preferences"""
    return {"success": True, "data": PREFERENCES}


@router.put("/preferences")
async def update_preferences(update: PreferencesUpdate, user: dict = Depends(require_auth)):
    """Update notification preferences"""
    if update.email_enabled is not None:
        PREFERENCES["email_enabled"] = update.email_enabled

    if update.push_enabled is not None:
        PREFERENCES["push_enabled"] = update.push_enabled

    if update.slack_enabled is not None:
        PREFERENCES["slack_enabled"] = update.slack_enabled

    if update.alert_types:
        PREFERENCES["alert_types"].update(update.alert_types)

    if update.quiet_hours:
        PREFERENCES["quiet_hours"].update(update.quiet_hours)

    save_preferences()

    return {"success": True, "data": PREFERENCES, "message": "Preferences updated"}


@router.post("/send")
async def send_notification(
    title: str,
    message: str,
    notification_type: str = "info",
    priority: str = "medium",
    channels: list[str] | None = None,
    user: dict = Depends(require_auth),
):
    """Send a notification through specified channels"""
    if channels is None:
        channels = ["push"]
    # Check preferences
    should_send = True

    if priority in PREFERENCES.get("alert_types", {}):
        should_send = PREFERENCES["alert_types"][priority]

    if not should_send:
        return {
            "success": True,
            "message": "Notification skipped (disabled by preferences)",
            "sent": False,
        }

    # Create the notification
    notification = NotificationCreate(
        title=title, message=message, type=notification_type, priority=priority
    )
    result = await create_notification(notification)

    # In a real implementation, send to external channels here
    sent_to = []
    if "email" in channels and PREFERENCES.get("email_enabled"):
        sent_to.append("email")
    if "push" in channels and PREFERENCES.get("push_enabled"):
        sent_to.append("push")
    if "slack" in channels and PREFERENCES.get("slack_enabled"):
        sent_to.append("slack")

    return {
        "success": True,
        "data": result["data"],
        "sent_to": sent_to,
        "message": f"Notification sent to: {', '.join(sent_to) or 'internal only'}",
    }
