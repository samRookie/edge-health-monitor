"""Alert Manager implementing fall and emergency logic with debouncing and cooldown."""

import time
import yaml
from src.fall_rule import FallRuleDetector

class AlertManager:
    """
    Manages emergency alert state machine.
    Modes:
      - 'fall_rule': Heuristic acceleration spike confirmed by LAYING posture
      - 'keyword_plus_activity': Emergency keyword ('stop') + abnormal posture/inactivity
      - 'both': Trigger if either condition is met
    """
    def __init__(self, config=None):
        if config is None:
            with open("config.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            config = cfg.get("alerts", {})
            self.emergency_keyword = cfg.get("kws", {}).get("emergency_keyword", "stop")
        else:
            self.emergency_keyword = config.get("emergency_keyword", "stop")

        self.mode = config.get("mode", "both")
        self.cooldown_seconds = config.get("cooldown_seconds", 5.0)
        self.keyword_conf_thresh = config.get("keyword_confidence_threshold", 0.80)
        self.inactivity_seconds_thresh = config.get("inactivity_seconds_threshold", 5.0)
        
        self.fall_detector = FallRuleDetector(
            spike_thresh_g=config.get("fall_accel_threshold_g", 2.5),
            stillness_thresh=config.get("stillness_variance_threshold", 0.05),
            posture_thresh=config.get("posture_confidence_threshold", 0.70)
        )
        
        self.last_alert_time = 0.0
        self.stationary_start_time = None
        self.alert_history = []

    def set_mode(self, mode):
        if mode in ["fall_rule", "keyword_plus_activity", "both"]:
            self.mode = mode

    def process_step(self, imu_window, activity_label, activity_conf, keyword_label, keyword_conf, timestamp=None):
        """
        Processes a live inference step.
        Returns: alert_dict or None
        """
        now = time.time() if timestamp is None else timestamp
        
        # Track stationary inactivity
        if activity_label in ["SITTING", "STANDING", "LAYING"]:
            if self.stationary_start_time is None:
                self.stationary_start_time = now
        else:
            self.stationary_start_time = None
            
        stationary_duration = (now - self.stationary_start_time) if self.stationary_start_time else 0.0
        
        # Check Rule 1: Fall rule
        fall_triggered, fall_details = self.fall_detector.evaluate(imu_window, activity_label, activity_conf)
        
        # Check Rule 2: Keyword plus activity
        keyword_matched = (keyword_label == self.emergency_keyword) and (keyword_conf >= self.keyword_conf_thresh)
        abnormal_posture = (activity_label == "LAYING") or (stationary_duration >= self.inactivity_seconds_thresh)
        kwd_triggered = keyword_matched and abnormal_posture
        
        # Determine firing based on mode
        fired = False
        rule_name = None
        details = {}
        
        if self.mode == "fall_rule":
            if fall_triggered:
                fired = True
                rule_name = "Rule-assisted fall alert (impact spike + stillness + LAYING posture confirmation)"
                details = fall_details
        elif self.mode == "keyword_plus_activity":
            if kwd_triggered:
                fired = True
                rule_name = f"Keyword plus activity alert (Keyword '{keyword_label}' + abnormal state: {activity_label})"
                details = {"keyword": keyword_label, "keyword_confidence": keyword_conf, "activity": activity_label, "stationary_s": stationary_duration}
        elif self.mode == "both":
            if fall_triggered and kwd_triggered:
                fired = True
                rule_name = "Combined alert (Fall impact rule AND emergency keyword triggered)"
                details = {"fall": fall_details, "keyword": keyword_label, "activity": activity_label}
            elif fall_triggered:
                fired = True
                rule_name = "Rule-assisted fall alert (impact spike + stillness + LAYING posture confirmation)"
                details = fall_details
            elif kwd_triggered:
                fired = True
                rule_name = f"Keyword plus activity alert (Keyword '{keyword_label}' + abnormal state: {activity_label})"
                details = {"keyword": keyword_label, "keyword_confidence": keyword_conf, "activity": activity_label, "stationary_s": stationary_duration}

        # Apply debouncing / cooldown
        if fired:
            if (now - self.last_alert_time) >= self.cooldown_seconds:
                self.last_alert_time = now
                alert_entry = {
                    "timestamp": now,
                    "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                    "rule": rule_name,
                    "activity": activity_label,
                    "activity_confidence": activity_conf,
                    "keyword": keyword_label,
                    "keyword_confidence": keyword_conf,
                    "details": details
                }
                self.alert_history.append(alert_entry)
                return alert_entry
            else:
                # Suppressed due to cooldown
                return None
                
        return None
