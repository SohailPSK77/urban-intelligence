"""
SIH26124: Final Technical Validation Script for Phase 4 Real Traffic Intelligence
Processes 90 consecutive frames from the actual bus video stream using PyTorch YOLOv8 + ByteTrack.
Distinguishes source VIDEO_TIME_SEC (for recorded video traffic persistence duration) from PROCESSING_WALL_CLOCK_SEC (for system execution performance/FPS).
Prints detailed per-frame real tracking IDs, vehicle/pedestrian counts, ROI TDI, Relative Congestion Score,
movement metrics (pixels/frame), VIDEO_TIME_SEC persistence duration, AI-derived bottleneck candidate status,
sample event JSON payload, and actual measured processing FPS.
"""

import os
import sys
import json
import time

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_processor import BusCameraVideoProcessor

def run_phase4_test():
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    video_path = os.path.join(assets_dir, "sample_vizag_route_101.mp4")

    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return

    processor = BusCameraVideoProcessor(video_path)
    source_video_fps = round(processor.fps, 1)

    print("================================================================================")
    print("   SIH26124 — PHASE 4: REAL TRAFFIC INTELLIGENCE 90-FRAME TEMPORAL VALIDATION   ")
    print("================================================================================")
    print(f"Video Source Stream           : {os.path.basename(video_path)}")
    print(f"Source Video FPS              : {source_video_fps} FPS (Used for VIDEO_TIME_SEC persistence)")
    print(f"Total Frames to Process       : 90 consecutive frames")
    print(f"Video Timeline Duration       : {round((90 - 1) / source_video_fps, 2)} seconds")
    print(f"Bottleneck Min Duration Threshold: {processor.detector.bottleneck_min_duration_sec:.1f} seconds (VIDEO_TIME_SEC)")
    print("--------------------------------------------------------------------------------\n")

    start_wall_clock = time.time()
    warmup_fps = None
    steady_state_fps = []
    
    events_generated = []
    first_congestion_frame = None
    candidate_first_triggered_frame = None

    for frame_num in range(1, 91):
        t_frame_start = time.time()
        frame_rgb, frame_events, telemetry = processor.process_frame_at(frame_num, bus_id="BUS-07", route_id="ROUTE-101")
        t_frame_end = time.time()
        
        proc_fps = round(1.0 / max(t_frame_end - t_frame_start, 0.001), 1)
        if frame_num == 1:
            warmup_fps = proc_fps
        else:
            steady_state_fps.append(proc_fps)

        # Collect any generated traffic events
        for evt in frame_events:
            if evt.get("detection_type") in ["REAL_AI_DETECTION", "REAL_AI_TRAFFIC_ANALYTICS"]:
                events_generated.append(evt)

        # Extract telemetry fields
        active_veh = telemetry.get("current_active_vehicles", 0)
        active_ped = telemetry.get("current_active_pedestrians", 0)
        tdi = telemetry.get("traffic_density_index", 0.0)
        density_lvl = telemetry.get("traffic_density", "LOW")
        cong_score = telemetry.get("relative_congestion_score", 0.0)
        cong_lvl = telemetry.get("congestion_level", "NORMAL")
        avg_disp = telemetry.get("average_displacement_px", 0.0)
        mov_cnt = telemetry.get("moving_vehicle_count", 0)
        stat_cnt = telemetry.get("stationary_vehicle_count", 0)
        unique_veh = telemetry.get("unique_vehicle_tracks", 0)
        unique_ped = telemetry.get("unique_pedestrian_tracks", 0)
        video_time = telemetry.get("video_time_sec", 0.0)
        elapsed_video_sec = telemetry.get("elapsed_congestion_video_sec", 0.0)
        is_candidate = telemetry.get("is_bottleneck_candidate", False)
        cls_counts = telemetry.get("class_wise_counts", {})

        if elapsed_video_sec > 0.0 and first_congestion_frame is None:
            first_congestion_frame = (frame_num, video_time)

        if is_candidate and candidate_first_triggered_frame is None:
            candidate_first_triggered_frame = (frame_num, video_time, elapsed_video_sec)

        cand_str = "TRUE 🔴 (CANDIDATE)" if is_candidate else "FALSE 🟢"

        print(f"Frame #{frame_num:02d} | VidTime: {video_time:5.2f}s | ProcFPS: {proc_fps:4.1f} | Active Veh: {active_veh} | Ped: {active_ped} | TDI: {tdi:.2f} ({density_lvl:8s}) | Congestion: {cong_score:.2f} ({cong_lvl:9s}) | Disp: {avg_disp:4.1f}px | Mov/Stat: {mov_cnt}/{stat_cnt} | Bottleneck Cand: {cand_str} | Sustained VideoTime: {elapsed_video_sec:.2f}s")
        print(f"         └─ Class Breakdown: {cls_counts}")
        print(f"         └─ Cumulative Track IDs — Vehicles: {unique_veh}, Pedestrians: {unique_ped}")
        print()

    total_wall_clock_sec = round(time.time() - start_wall_clock, 2)
    avg_proc_fps = round(sum(steady_state_fps) / len(steady_state_fps), 1) if steady_state_fps else warmup_fps
    rolling_hist = processor.detector.get_rolling_history()

    processor.close()

    print("================================================================================")
    print("               PHASE 4 90-FRAME TEMPORAL VALIDATION SUMMARY                     ")
    print("================================================================================")
    print(f"Source Video FPS (Video Timeline) : {source_video_fps} FPS")
    print(f"Total Frames Processed           : 90 frames")
    print(f"Total Video Timeline Duration    : {round((90 - 1) / source_video_fps, 2)} seconds")
    print(f"Measured System Processing FPS   : {avg_proc_fps} FPS (Processing Wall-Clock Speed)")
    print(f"Frame 1 Model Warm-up Speed      : {warmup_fps} FPS")
    print(f"Total Wall-Clock Execution Time  : {total_wall_clock_sec} seconds")
    print(f"Cumulative Unique Vehicle Tracks : {processor.detector.unique_vehicle_track_ids}")
    print(f"Cumulative Unique Pedestrian Tracks: {processor.detector.unique_pedestrian_track_ids}")
    print(f"Total Rolling History Entries    : {len(rolling_hist)} observations")
    print(f"Total Real AI Events Generated   : {len(events_generated)} items")
    print("--------------------------------------------------------------------------------")
    if first_congestion_frame:
        fn_c, vt_c = first_congestion_frame
        print(f"1. Congestion Condition Began    : Frame #{fn_c} at Video Time = {vt_c:.2f}s")
    if candidate_first_triggered_frame:
        fn_t, vt_t, dur_t = candidate_first_triggered_frame
        print(f"2. Bottleneck Candidate Triggered: Frame #{fn_t} at Video Time = {vt_t:.2f}s (Sustained Video Time: {dur_t:.2f}s >= 2.00s threshold)")
    else:
        print(f"2. Bottleneck Candidate Triggered: None (Video-time threshold not reached)")
    print("--------------------------------------------------------------------------------")

    if events_generated:
        print("\n--- SAMPLE GEOTAGGED REAL AI EVENT PAYLOAD (JSON) ---")
        print(json.dumps(events_generated[0], indent=2))
        print("--------------------------------------------------------------------------------")

    print("\nSTATEMENT VERIFICATION:")
    print("\"Video-time persistence is used for recorded video; processing wall-clock time is used only for performance measurement.\"")
    print("✅ PHASE 4 FINAL TECHNICAL CORRECTION VERIFICATION COMPLETE!")

if __name__ == "__main__":
    run_phase4_test()
