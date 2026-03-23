import time
import tomllib
import os
import threading

class AlertManager:
    def __init__(self, slouch_duration_threshold=3.0, cooldown_duration=10.0):
        """
        Args:
            slouch_duration_threshold (float): Seconds of continuous slouching before alerting
            cooldown_duration (float): Minimum seconds between consecutive alerts
        """
        self.slouch_duration_threshold = slouch_duration_threshold
        self.cooldown_duration = cooldown_duration

        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.toml")
        try:
            with open(settings_path, "rb") as f:
                settings = tomllib.load(f)
            self.notification_types = settings.get("notifications", {}).get("types", ["banner"])
        except FileNotFoundError:
            self.notification_types = ["banner"]

        self.slouch_start_time = None
        self.last_alert_time = 0.0

    def run_check(self, is_slouching):
        """
        Checks if the user has been slouching long enough to trigger an alert.
        Returns True if an alert was triggered in this cycle.
        """
        current_time = time.time()
        
        if is_slouching:
            if self.slouch_start_time is None:
                self.slouch_start_time = current_time
            
            elapsed = current_time - self.slouch_start_time
            if elapsed >= self.slouch_duration_threshold:
                if current_time - self.last_alert_time >= self.cooldown_duration:
                    self.trigger_alert()
                    self.last_alert_time = current_time
                    return True
        else:
            # User stopped slouching, reset the continuous slouch timer
            self.slouch_start_time = None
            
        return False
        
    def trigger_alert(self):
        """Triggers a desktop notification asynchronously so UI doesn't freeze."""
        def send_notification():
            import subprocess
            for ntype in self.notification_types:
                try:
                    if ntype == "banner":
                        subprocess.Popen([
                            'osascript', '-e',
                            'display notification "You have been slouching. Sit up straight!" with title "Posture Alert"'
                        ])
                    elif ntype == "speech":
                        subprocess.Popen(['say', 'Sit up straight!'])
                    elif ntype == "sound":
                        subprocess.Popen(['afplay', '/System/Library/Sounds/Sosumi.aiff'])
                    elif ntype == "dialog":
                        subprocess.Popen([
                            'osascript', '-e',
                            'display dialog "Sit up straight!" buttons {"OK"} giving up after 10'
                        ])
                except Exception as e:
                    print(f"Failed to send '{ntype}' notification: {e}")
            print("Alert notification sent.")

        threading.Thread(target=send_notification, daemon=True).start()
