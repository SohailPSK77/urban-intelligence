# 🚍 SIH26124 — AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet

**Smart India Hackathon (SIH) 2026 Project Implementation**  
*Problem Statement ID: SIH26124*  
*Deployment Target: Visakhapatnam (Vizag) Smart City Area*

---

## 📌 Executive Summary

Modern smart cities rely heavily on fixed CCTV cameras, citizen complaints, or periodic manual surveys to track road hazards, traffic bottlenecks, and safety incidents. Fixed cameras have blind spots, manual surveys are slow, and citizen reports arrive after significant delay.

This project transforms **existing public transport bus fleets (APSRTC / Vizag City Transit)** into a **distributed network of Mobile AI Sensing Units**. As buses navigate city transit corridors, front/side cameras and edge AI compute units continuously detect, understand, geotag, and transmit urban event metadata to a central GIS platform.

The primary differentiator of this platform is **Multi-Bus Event Fusion**: independent visual observations of persistent road hazards (e.g., potholes on RK Beach Road seen by Bus-07, Bus-12, and Bus-15) are merged spatially and temporally into single high-confidence **Persistent Urban Issues** with cumulative probabilistic verification.

---

## 🚀 Key Features (Phase 1 Prototype - Visakhapatnam)

1. **GIS Urban Intelligence Command Map**:
   - Centered on Visakhapatnam (`17.7200, 83.3000`).
   - 4 Major Vizag Transit Corridors (RK Beach Road, NAD Flyover Corridor, Siripuram - Rushikonda IT Hill, MVP Colony - Bheemli).
   - 18 Active APSRTC Public Bus Units with speed, driver ID, camera status, and edge GPU health.
   - Prominent **Fused Persistent Issue Clusters** (RK Beach Pothole, NAD Flyover Waterlogging).

2. **Multi-Bus Event Fusion Interactive Lab**:
   - Live demonstration workspace for mentors/judges.
   - Haversine spatial proximity distance thresholding ($R_{\text{threshold}} = 20.0\text{ meters}$).
   - Probabilistic joint confidence formula visualization:
     $$C_{\text{fused}} = 1 - \prod_{i=1}^{k} (1 - c_i)$$

3. **Traffic Intelligence & Route Delay Analytics**:
   - Vehicle classification breakdown (Cars, Two-Wheelers, Rickshaws, Buses, Port Freight Trucks).
   - Real-time corridor bottleneck identification (NAD Junction Flyover & Jagadamba Junction).

4. **Hit-and-Run / ANPR Rash Driving Portal**:
   - Automatic License Plate Recognition (ANPR) OCR output for AP-39 registered vehicles (`AP 39 TV 7219`).
   - Technical honesty safeguard: Explicit `REQUIRES HUMAN REVIEW` badge with officer approval buttons.

5. **Edge-to-Cloud Compact JSON Inspector**:
   - Inspect raw lightweight edge telemetry payloads transmitted over MQTT/REST instead of heavy raw video streams.

---

## ⚡ Quick Start & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 🧪 Demonstration Scenario for SIH Mentors (Vizag Setup)

1. **Open Command Center**: View the 18 active public buses operating across Visakhapatnam coastal and industrial corridors.
2. **Observe Single Detections**: Bus-07 detects a pothole on RK Beach Road near Submarine Museum ($91\%$ confidence).
3. **Simulate Multi-Bus Pass**: Bus-12 passes the same pothole 20 minutes later ($88\%$ confidence).
4. **Trigger Event Fusion**: The system detects spatial proximity ($< 15\text{m}$) and fuses the two detections into **Persistent Road Issue #PR-0001** with **$98.9\%$ combined confidence**.
5. **Demonstrate 3rd Bus Verification**: In the *Multi-Bus Fusion Lab* tab, toggle Bus-15. The fused confidence jumps to **$99.9\%$**, auto-escalating priority to **URGENT DISPATCH**.

---

## 🏷️ Disclaimer
*Data in Phase 1 is clearly labeled as **[SIMULATED ENGINE (VIZAG)]** pending real-time video pipeline integration in Phase 2/3.*
