import math
import numpy as np
import cv2
import cv2.aruco as aruco


class CameraSystem:
    def __init__(self, focal_length_px=613.0, marker_physical_width_mm=50.0):
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.params)
        self.target_x = None   # alignment crosshair; None = use frame center
        self.target_y = None
        self.focal_length_px = focal_length_px
        self.marker_physical_width_mm = marker_physical_width_mm
        self.last_corners = []      # [(marker_id, corners_4x2), ...] — all detected this frame
        self.last_orientation = None  # {'roll': deg, 'pitch': deg, 'yaw': deg} for TARGET marker

    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y

    def _detect_all(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            self.last_corners = [(int(ids[i][0]), corners[i][0]) for i in range(len(ids))]
        else:
            self.last_corners = []

    @staticmethod
    def _marker_angle_2d(corners):
        """Fallback: in-plane tilt from TL→TR edge only."""
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        return math.degrees(math.atan2(dy, dx))

    def _estimate_pose(self, corners, frame_shape):
        """Full 3D pose via solvePnP → {'roll', 'pitch', 'yaw'} in degrees, or None on failure.

        Conventions (all 0 when marker is perfectly flat-on to camera):
          roll  – in-plane rotation around optical axis (+ve = clockwise tilt in image)
          pitch – vertical tilt (+ve = top of marker tilts away from camera)
          yaw   – horizontal turn (+ve = marker normal tilts toward camera-left / right edge goes back)
        """
        h, w = frame_shape[:2]
        half = self.marker_physical_width_mm / 2.0
        # ArUco corner order: TL, TR, BR, BL — in marker-local frame X-right, Y-up
        obj_pts = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float32)
        img_pts = np.array(corners, dtype=np.float32)
        K = np.array([
            [self.focal_length_px, 0.0,                   w / 2.0],
            [0.0,                  self.focal_length_px,  h / 2.0],
            [0.0,                  0.0,                   1.0    ],
        ], dtype=np.float64)
        dist = np.zeros((4,), dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            return None

        R, _ = cv2.Rodrigues(rvec)
        # For flat-on marker: R[:,0]=[1,0,0], R[:,1]=[0,-1,0], R[:,2]=[0,0,-1]
        roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        pitch = math.degrees(math.atan2(R[1, 2], -R[2, 2]))
        yaw   = math.degrees(math.atan2(R[0, 2], -R[2, 2]))
        return {'roll': roll, 'pitch': pitch, 'yaw': yaw}

    def _calculate_marker_center(self, corners):
        center_x = int((corners[0][0] + corners[2][0]) / 2)
        center_y = int((corners[0][1] + corners[2][1]) / 2)
        return center_x, center_y

    @staticmethod
    def _marker_size(corners):
        """Orientation-robust marker size in px = sqrt of the quadrilateral area (shoelace).
        For a square-on marker this equals the edge width, but unlike a single edge's horizontal
        span it is invariant to in-plane roll and averages out pitch/yaw foreshortening — so it's
        an honest distance proxy for the Y-approach regardless of how the marker is tilted."""
        c = corners
        n = len(c)
        area2 = 0.0
        for i in range(n):
            x1, y1 = c[i][0], c[i][1]
            x2, y2 = c[(i + 1) % n][0], c[(i + 1) % n][1]
            area2 += x1 * y2 - x2 * y1
        return int(round((abs(area2) / 2.0) ** 0.5))

    def get_target_error(self, frame, target_id):
        """Returns (found, x_err, z_err, marker_width, mx, my).
        Errors are relative to the configurable crosshair (default: frame center).
        Screen Y maps to robot Z. Also populates last_orientation for the target marker."""
        h, w, _ = frame.shape
        tx = self.target_x if self.target_x is not None else w // 2
        ty = self.target_y if self.target_y is not None else h // 2

        self._detect_all(frame)
        self.last_orientation = None

        for mid, c in self.last_corners:
            if mid == target_id:
                mx, my = self._calculate_marker_center(c)
                marker_width = self._marker_size(c)
                orient = self._estimate_pose(c, frame.shape)
                if orient is None:
                    roll_2d = self._marker_angle_2d(c)
                    orient = {'roll': roll_2d, 'pitch': 0.0, 'yaw': 0.0}
                self.last_orientation = orient
                return True, mx - tx, my - ty, marker_width, mx, my

        return False, 0, 0, 0, 0, 0

    def get_nearest_error(self, frame, near_x, near_y):
        """Like get_target_error but ID-agnostic: returns the error for the detected marker whose
        center is nearest (near_x, near_y). Used for our-slot refine, where we don't care which
        battery (ID) is in the slot, just that we align to it. Errors are relative to the crosshair."""
        h, w, _ = frame.shape
        tx = self.target_x if self.target_x is not None else w // 2
        ty = self.target_y if self.target_y is not None else h // 2

        self._detect_all(frame)
        self.last_orientation = None

        best = None
        best_d = None
        for mid, c in self.last_corners:
            mx, my = self._calculate_marker_center(c)
            d = (mx - near_x) ** 2 + (my - near_y) ** 2
            if best_d is None or d < best_d:
                best_d, best = d, c
        if best is None:
            return False, 0, 0, 0, 0, 0

        mx, my = self._calculate_marker_center(best)
        marker_width = self._marker_size(best)
        orient = self._estimate_pose(best, frame.shape)
        if orient is None:
            orient = {'roll': self._marker_angle_2d(best), 'pitch': 0.0, 'yaw': 0.0}
        self.last_orientation = orient
        return True, mx - tx, my - ty, marker_width, mx, my

    def locate_marker(self, frame, marker_id):
        """Self-contained detection of a single marker for CALIBRATION capture. Runs its own
        detectMarkers and returns results locally WITHOUT touching the shared last_corners /
        last_orientation — so it's safe to call from the Flask thread while the camera loop is
        also detecting, and it must be given a CLEAN (un-annotated) frame so the on-feed overlays
        don't obscure the marker. Returns (found, mx, my, width_px, orient, corners_list)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return False, 0, 0, 0, None, None
        for i in range(len(ids)):
            if int(ids[i][0]) == marker_id:
                c = corners[i][0]
                mx, my = self._calculate_marker_center(c)
                width = self._marker_size(c)
                orient = self._estimate_pose(c, frame.shape)
                if orient is None:
                    orient = {'roll': self._marker_angle_2d(c), 'pitch': 0.0, 'yaw': 0.0}
                return True, mx, my, width, orient, c.tolist()
        return False, 0, 0, 0, None, None
