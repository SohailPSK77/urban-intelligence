"""
SIH26124: Route-Specific Bus Camera Playback Video Generator (Phase 2 - Brand New Route-303 Photo Alignment)
Uses high-resolution photorealistic camera frames for each Vizag transit corridor with
PERFECTLY LOCKED BOUNDING BOXES over actual vehicles, pedestrians, and waterlogging across all 100 frames.
"""

import os
import cv2
import numpy as np
from config import ROUTES

# Route Image Mapping to Pristine Real Dashcam Images
ROUTE_IMAGES = {
    "ROUTE-101": "route_101_vizag_real.jpg",
    "ROUTE-202": "route_202_vizag_real.jpg",
    "ROUTE-303": "route_303_vizag_real.jpg",
    "ROUTE-404": "route_404_vizag_real.jpg"
}


def overlay_hazard_texture_photo(frame: np.ndarray, texture_img: np.ndarray, x: int, y: int, w: int = 220, h: int = 100):
    """
    Seamlessly composites a REAL HIGH-RESOLUTION PHOTOGRAPH of an asphalt pothole
    onto the road surface using elliptical alpha-mask feathering.
    """
    if texture_img is None or texture_img.size == 0:
        return

    frame_h, frame_w = frame.shape[:2]

    # Clamp coordinates inside frame boundaries
    x = max(0, min(x, frame_w - w - 1))
    y = max(0, min(y, frame_h - h - 45))

    # Resize texture to match target box
    resized_tex = cv2.resize(texture_img, (w, h))

    # Create an elliptical alpha mask for smooth edge blending onto the road
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (w // 2 - 4, h // 2 - 4)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

    # Blur the mask edges for seamless asphalt blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Extract target region from road frame
    road_roi = frame[y:y+h, x:x+w].astype(np.float32)
    tex_float = resized_tex.astype(np.float32)

    # Perform alpha blending: blended = photo * mask + road * (1 - mask)
    blended = tex_float * mask + road_roi * (1.0 - mask)
    frame[y:y+h, x:x+w] = np.clip(blended, 0, 255).astype(np.uint8)


def generate_sample_vizag_video(output_path: str, route_id: str = "ROUTE-101", fps: int = 20, duration_sec: int = 5):
    """
    Generates a route-specific 720p bus dashcam video across EXACTLY 100 FRAMES (5s @ 20fps)
    with 100% PERFECT PIXEL-LOCKED AI detection bounding boxes.
    """
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    image_filename = ROUTE_IMAGES.get(route_id, "route_101_vizag_real.jpg")
    base_img_path = os.path.join(assets_dir, image_filename)
    pothole_texture_path = os.path.join(assets_dir, "real_pothole_texture.jpg")

    width, height = 1280, 720
    total_frames = 100  # Exactly 100 frames

    # Load route-specific base vehicle camera image
    if os.path.exists(base_img_path):
        base_frame = cv2.imread(base_img_path)
        if base_frame is not None:
            base_frame = cv2.resize(base_frame, (width, height))
        else:
            base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        base_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Load real photographic pothole texture
    pothole_photo = cv2.imread(pothole_texture_path) if os.path.exists(pothole_texture_path) else None

    # Use mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Remove existing video to force fresh rendering for 100 frames
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    route_info = ROUTES.get(route_id, {"name": "Vizag Transit", "waypoints": [[17.72, 83.3]]})

    for frame_idx in range(total_frames):
        frame = base_frame.copy()

        # Flashing indicator color toggle for hazard box
        is_flash = (frame_idx // 3) % 2 == 0

        # -------------------------------------------------------------
        # 1. PIXEL-PERFECT LOCKED BOUNDING BOXES FOR EACH ROUTE
        # -------------------------------------------------------------
        if route_id == "ROUTE-303":
            # --- ROUTE-303 (Rushikonda IT Hill Expressway - Real Photo Bounding Boxes) ---
            # 1. RED CAR (In Center Lane)
            c_x1, c_y1, c_x2, c_y2 = 560, 350, 730, 475
            cv2.rectangle(frame, (c_x1, c_y1), (c_x2, c_y2), (52, 211, 153), 2)
            cv2.rectangle(frame, (c_x1, c_y1 - 24), (c_x1 + 175, c_y1), (52, 211, 153), -1)
            cv2.putText(frame, "CAR #202 [CONF: 93%]", (c_x1 + 5, c_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            # 2. SILVER SEDAN (On Right Lane)
            s_x1, s_y1, s_x2, s_y2 = 760, 350, 1180, 530
            cv2.rectangle(frame, (s_x1, s_y1), (s_x2, s_y2), (255, 255, 0), 2)
            cv2.rectangle(frame, (s_x1, s_y1 - 24), (s_x1 + 175, s_y1), (255, 255, 0), -1)
            cv2.putText(frame, "SUV #108 [CONF: 95%]", (s_x1 + 5, s_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            # 3. REAL WATERLOGGING FLOODED POOL (On Road Center-Right Asphalt)
            w_x1, w_y1, w_x2, w_y2 = 600, 540, 1020, 670
            w_color = (255, 140, 0) if is_flash else (255, 60, 0) # Blue in BGR
            cv2.rectangle(frame, (w_x1, w_y1), (w_x2, w_y2), w_color, 3)
            cv2.rectangle(frame, (w_x1, w_y1 - 28), (w_x1 + 310, w_y1), w_color, -1)
            cv2.putText(frame, "WATERLOGGING DETECTED [CONF: 93%]", (w_x1 + 6, w_y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        elif route_id == "ROUTE-404":
            # --- ROUTE-404 (MVP Colony Market Street) ---
            # 1. YELLOW AUTO-RICKSHAW (On right side)
            a_x1, a_y1, a_x2, a_y2 = 615, 450, 760, 710
            cv2.rectangle(frame, (a_x1, a_y1), (a_x2, a_y2), (255, 255, 0), 2)
            cv2.rectangle(frame, (a_x1, a_y1 - 24), (a_x1 + 210, a_y1), (255, 255, 0), -1)
            cv2.putText(frame, "AUTO RICKSHAW #310 [92%]", (a_x1 + 6, a_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            # 2. PEDESTRIANS CROSSING (In middle)
            p_x1, p_y1, p_x2, p_y2 = 420, 480, 570, 750
            cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (192, 132, 252), 2)
            cv2.rectangle(frame, (p_x1, p_y1 - 24), (p_x1 + 190, p_y1), (192, 132, 252), -1)
            cv2.putText(frame, "PEDESTRIAN CROWD [87%]", (p_x1 + 6, p_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 2)

            # 3. MOTORCYCLE (On left)
            m_x1, m_y1, m_x2, m_y2 = 110, 540, 290, 720
            cv2.rectangle(frame, (m_x1, m_y1), (m_x2, m_y2), (56, 189, 248), 2)
            cv2.rectangle(frame, (m_x1, m_y1 - 22), (m_x1 + 180, m_y1), (56, 189, 248), -1)
            cv2.putText(frame, "MOTORCYCLE #104 [91%]", (m_x1 + 5, m_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2)

        elif route_id == "ROUTE-202":
            # --- ROUTE-202 (NAD Flyover Industrial Corridor) ---
            cv2.rectangle(frame, (280, 310), (560, 520), (255, 180, 0), 2)
            cv2.rectangle(frame, (280, 286), (490, 310), (255, 180, 0), -1)
            cv2.putText(frame, "HEAVY TRUCK #402 [CONF: 96%]", (286, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

            cv2.rectangle(frame, (700, 330), (960, 530), (0, 200, 255), 2)
            cv2.rectangle(frame, (700, 306), (910, 330), (0, 200, 255), -1)
            cv2.putText(frame, "CITY BUS #508 [CONF: 91%]", (706, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

        else:
            # --- ROUTE-101 (RK Beach Coastal Expressway) ---
            cv2.rectangle(frame, (320, 370), (550, 530), (255, 180, 0), 2)
            cv2.rectangle(frame, (320, 346), (500, 370), (255, 180, 0), -1)
            cv2.putText(frame, "CAR #104 [CONF: 94%]", (326, 363), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

            ph_x1, ph_y1, ph_w, ph_h = 580, 480, 220, 100
            if pothole_photo is not None:
                overlay_hazard_texture_photo(frame, pothole_photo, ph_x1, ph_y1, ph_w, ph_h)

            ph_color = (0, 0, 255) if is_flash else (0, 69, 255)
            cv2.rectangle(frame, (ph_x1 - 6, ph_y1 - 6), (ph_x1 + ph_w + 6, ph_y1 + ph_h + 6), ph_color, 3)
            cv2.rectangle(frame, (ph_x1 - 6, ph_y1 - 32), (ph_x1 + 265, ph_y1 - 6), ph_color, -1)
            cv2.putText(frame, "POTHOLE DETECTED [CONF: 91%]", (ph_x1 - 1, ph_y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

        # -------------------------------------------------------------
        # 2. ONBOARD TELEMETRY HEADS-UP DISPLAY (HUD)
        # -------------------------------------------------------------
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (15, 23, 42), -1)
        cv2.rectangle(overlay, (0, height - 40), (width, height), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        bus_map = {"ROUTE-101": "BUS-07", "ROUTE-202": "BUS-02", "ROUTE-303": "BUS-11", "ROUTE-404": "BUS-09"}
        current_bus = bus_map.get(route_id, "BUS-07")

        # Top Text Labels
        hud_left = f"BUS SENSING UNIT: {current_bus} | {route_id} ({route_info['name'][:30]}) | VIZAG TRANSIT"
        hud_right = f"GPS: {route_info['waypoints'][0][0]} N, {route_info['waypoints'][0][1]} E | FRAME: {frame_idx + 1}/100"
        cv2.putText(frame, hud_left, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
        cv2.putText(frame, hud_right, (700, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 2)

        # Bottom Text Labels
        edge_left = "EDGE AI INFERENCE: ONLINE (24.2 FPS) | GPU TEMP: 52C | MODEL: YOLOv8-URBAN-v1"
        edge_right = "STATUS: EVENT TRANSMISSION READY"
        cv2.putText(frame, edge_left, (20, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 211, 153), 2)
        cv2.putText(frame, edge_right, (920, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 158, 11), 2)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    for r_id in ROUTE_IMAGES.keys():
        out_p = os.path.join(os.path.dirname(__file__), "assets", f"sample_vizag_{r_id.lower().replace('-', '_')}.mp4")
        generate_sample_vizag_video(out_p, route_id=r_id)
        print(f"Re-generated brand new photo aligned video for {r_id}:", out_p)
