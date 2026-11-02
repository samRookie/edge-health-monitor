"""Rule-based fall detection heuristic using IMU acceleration magnitude and posture confirmation."""

import numpy as np

class FallRuleDetector:
    """
    Flags potential falls when:
    1. A sharp acceleration-magnitude spike occurs (> spike_thresh_g, default 2.5 g).
    2. Followed within a window by stillness (acceleration variance < stillness_thresh).
    3. Confirmed by HAR model predicting LAYING posture (confidence > posture_thresh).
    """
    def __init__(self, spike_thresh_g=2.5, stillness_thresh=0.05, posture_thresh=0.70):
        self.spike_thresh_g = spike_thresh_g
        self.stillness_thresh = stillness_thresh
        self.posture_thresh = posture_thresh
        self.spike_detected = False
        self.samples_since_spike = 0
        self.max_samples_after_spike = 150  # ~3 seconds at 50 Hz

    def compute_magnitude(self, imu_window):
        """
        Computes acceleration magnitude array for an IMU window.
        imu_window: (N, 6) where cols 0, 1, 2 are total_acc_x/y/z.
        """
        acc = imu_window[:, :3]
        return np.sqrt(np.sum(acc ** 2, axis=1))

    def evaluate(self, imu_window, predicted_activity, activity_confidence):
        """
        Evaluates a 128-sample window (2.56s).
        Returns (is_fall, reason_dict)
        """
        mag = self.compute_magnitude(imu_window)
        max_mag = float(np.max(mag))
        tail_len = min(64, len(mag))
        stillness_var = float(np.var(mag[-tail_len:]))
        
        has_spike = max_mag >= self.spike_thresh_g
        has_stillness = stillness_var <= self.stillness_thresh
        is_laying = (predicted_activity == "LAYING") and (activity_confidence >= self.posture_thresh)
        
        triggered = False
        details = {
            "max_magnitude_g": max_mag,
            "stillness_variance": stillness_var,
            "predicted_activity": predicted_activity,
            "activity_confidence": activity_confidence,
            "has_spike": has_spike,
            "has_stillness": has_stillness,
            "is_laying": is_laying
        }
        
        if has_spike:
            self.spike_detected = True
            self.samples_since_spike = 0
        elif self.spike_detected:
            self.samples_since_spike += len(imu_window)
            if self.samples_since_spike > self.max_samples_after_spike:
                self.spike_detected = False

        if (self.spike_detected or has_spike) and has_stillness and is_laying:
            triggered = True
            self.spike_detected = False # Reset on trigger
            
        return triggered, details

def create_synthetic_fall_sequence(sample_rate=50, duration_s=6):
    """
    Creates a synthetic test trace:
    Normal walking (1.0g with oscillation) -> Impact spike (3.2g) -> Stillness (1.0g flat).
    """
    n_samples = sample_rate * duration_s
    t = np.linspace(0, duration_s, n_samples)
    
    # 6 channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    data = np.zeros((n_samples, 6), dtype=np.float32)
    data[:, 2] = 1.0  # gravity along Z
    
    # Phase 1: Walking (0 to 2s)
    walk_mask = t < 2.0
    data[walk_mask, 0] = 0.3 * np.sin(2 * np.pi * 2.0 * t[walk_mask])
    data[walk_mask, 2] += 0.4 * np.cos(2 * np.pi * 2.0 * t[walk_mask])
    
    # Phase 2: Impact spike at t = 2.2s (width 100ms)
    spike_mask = (t >= 2.15) & (t <= 2.25)
    data[spike_mask, 0] = 2.0
    data[spike_mask, 2] = 2.5  # Magnitude ~ 3.2g
    
    # Phase 3: Stillness on ground (2.3s to 6.0s)
    # Laying posture: gravity along X or Y instead of Z
    still_mask = t > 2.25
    data[still_mask, 0] = 0.98 + np.random.normal(0, 0.01, np.sum(still_mask))
    data[still_mask, 2] = 0.05 + np.random.normal(0, 0.01, np.sum(still_mask))
    
    return data
