import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()

def create_anpr_image(base_img_path, output_path, alert_title, obj_label, plate_text, ocr_conf, speed_info, loc_info, bbox_coords):
    if not os.path.exists(base_img_path):
        # Create a synthetic realistic dark asphalt dashcam background if base doesn't exist
        img = Image.new('RGB', (1024, 576), color=(20, 25, 35))
    else:
        img = Image.open(base_img_path).convert('RGB')
        img = img.resize((1024, 576))

    draw = ImageDraw.Draw(img)

    # Top HUD Bar
    draw.rectangle([(0, 0), (1024, 50)], fill=(15, 23, 42))
    font_header = get_font(18)
    font_sub = get_font(13)

    # Red icon indicator
    draw.rectangle([(15, 12), (25, 38)], fill=(239, 68, 68))
    draw.text((35, 13), f"AI EDGE BUS ANPR | {alert_title}", fill=(248, 250, 252), font=font_header)
    draw.text((700, 15), f"CAM-FRONT-01 | {loc_info}", fill=(148, 163, 184), font=font_sub)

    # Bounding Box
    x1, y1, x2, y2 = bbox_coords
    # Draw thick box
    for i in range(4):
        draw.rectangle([(x1-i, y1-i), (x2+i, y2+i)], outline=(239, 68, 68))

    # Object Label Header above box
    draw.rectangle([(x1, y1 - 28), (x1 + 260, y1)], fill=(239, 68, 68))
    draw.text((x1 + 6, y1 - 24), obj_label, fill=(255, 255, 255), font=get_font(14))

    # Yellow License Plate Badge Overlay
    plate_w, plate_h = 240, 54
    px1 = min(x1 + 20, 1024 - plate_w - 20)
    py1 = min(y2 + 10, 576 - plate_h - 60)

    # Yellow Plate Frame
    draw.rectangle([(px1, py1), (px1 + plate_w, py1 + plate_h)], fill=(250, 204, 21), outline=(0, 0, 0), width=3)
    # Blue IND strip on left of license plate
    draw.rectangle([(px1, py1), (px1 + 25, py1 + plate_h)], fill=(29, 78, 216))
    draw.text((px1 + 4, py1 + 8), "IND", fill=(255, 255, 255), font=get_font(10))

    # License Plate Text
    draw.text((px1 + 35, py1 + 10), plate_text, fill=(15, 23, 42), font=get_font(20))

    # OCR Confidence Badge attached to plate
    draw.rectangle([(px1 + plate_w - 95, py1 + plate_h - 18), (px1 + plate_w - 4, py1 + plate_h - 2)], fill=(15, 23, 42))
    draw.text((px1 + plate_w - 90, py1 + plate_h - 17), f"OCR {ocr_conf}%", fill=(52, 211, 153), font=get_font(10))

    # Bottom Telemetry Overlay Bar
    draw.rectangle([(0, 526), (1024, 576)], fill=(15, 23, 42, 220))
    draw.text((20, 538), f"⚠️ SPEED INFRACTION: {speed_info}", fill=(248, 113, 113), font=get_font(15))
    draw.text((550, 538), f"GPS: 17.7325° N, 83.2515° E | TELEMETRY VERIFIED", fill=(148, 163, 184), font=get_font(13))

    img.save(output_path, quality=95)
    print(f"Created ANPR evidence image: {output_path}")

assets_dir = r"c:\Users\basha\OneDrive\Documents\anti\sih26124_mobile_urban_intelligence\assets"

# Image 5
create_anpr_image(
    base_img_path=os.path.join(assets_dir, "vizag_traffic_night_202.jpg"),
    output_path=os.path.join(assets_dir, "anpr_hit_run_ap35.jpg"),
    alert_title="HIT-AND-RUN EVASION | HIGH SPEED DRIFT",
    obj_label="TRUCK #409 | DET CONF: 0.94",
    plate_text="AP 35 TH 8831",
    ocr_conf="96.4",
    speed_info="88 km/h (+38 km/h over limit)",
    loc_info="Gajuwaka Industrial Corridor",
    bbox_coords=(280, 160, 680, 420)
)

# Image 6
create_anpr_image(
    base_img_path=os.path.join(assets_dir, "vizag_traffic_heavy_303.jpg"),
    output_path=os.path.join(assets_dir, "anpr_rash_driving_night.jpg"),
    alert_title="RASH DRIVING | RECKLESS OVERTAKE & WHEELIE",
    obj_label="MOTORCYCLE #214 | DET CONF: 0.91",
    plate_text="AP 39 MW 5021",
    ocr_conf="92.8",
    speed_info="76 km/h (+31 km/h over limit)",
    loc_info="Siripuram Junction Circle",
    bbox_coords=(340, 200, 620, 450)
)

# Image 7
create_anpr_image(
    base_img_path=os.path.join(assets_dir, "route_202_nad_flyover.jpg"),
    output_path=os.path.join(assets_dir, "anpr_hit_run_ap31_car.jpg"),
    alert_title="RED LIGHT VIOLATION & HIT-AND-RUN EVASION",
    obj_label="CAR #188 | DET CONF: 0.95",
    plate_text="AP 31 EA 1109",
    ocr_conf="95.5",
    speed_info="84 km/h (+34 km/h over limit)",
    loc_info="NAD Junction Flyover Ramp",
    bbox_coords=(220, 180, 640, 430)
)
