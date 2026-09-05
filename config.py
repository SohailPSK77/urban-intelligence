"""
SIH26124: AI-Powered Mobile Urban Intelligence Platform Using Public Transport Fleet
Configuration & Core Parameters - Visakhapatnam (Vizag) Deployment
"""

CITY_NAME = "Visakhapatnam (Vizag) Smart City Area"
CITY_CENTER = [17.7200, 83.3000]
DEFAULT_ZOOM = 13

# Fleet Configuration
ACTIVE_BUS_COUNT = 18
TOTAL_ROUTES = 4

ROUTES = {
    "ROUTE-101": {
        "name": "RTC Complex ↔ RK Beach Coastal Expressway",
        "color": "#3B82F6",
        "waypoints": [
            [17.7170, 83.3005], # Dwaraka Bus Station (RTC Complex)
            [17.7115, 83.3030], # Jagadamba Junction
            [17.7100, 83.3180], # RK Beach / Submarine Museum
            [17.7220, 83.3250], # Lawsons Bay Colony
            [17.7320, 83.3320]  # Tenneti Park
        ]
    },
    "ROUTE-202": {
        "name": "Gajuwaka ↔ NAD Flyover ↔ RTC Complex Trunk Corridor",
        "color": "#8B5CF6",
        "waypoints": [
            [17.6890, 83.2080], # Gajuwaka Main Junction
            [17.7150, 83.2350], # Sheela Nagar
            [17.7320, 83.2510], # NAD Flyover Junction
            [17.7280, 83.2800], # Kancharapalem
            [17.7170, 83.3005]  # RTC Complex
        ]
    },
    "ROUTE-303": {
        "name": "Visakhapatnam Station ↔ Siripuram ↔ Rushikonda IT Hill",
        "color": "#10B981",
        "waypoints": [
            [17.7225, 83.2905], # Vizag Railway Station
            [17.7210, 83.3150], # Siripuram Circle
            [17.7350, 83.3280], # Kailasagiri Foothills
            [17.7820, 83.3850], # Rushikonda IT Park Hill 1
            [17.7950, 83.3920]  # GITAM University Gate
        ]
    },
    "ROUTE-404": {
        "name": "Maddilapalem ↔ MVP Colony ↔ Bheemli Beach Road",
        "color": "#F59E0B",
        "waypoints": [
            [17.7380, 83.3180], # Maddilapalem Bus Depot
            [17.7450, 83.3310], # MVP Colony Circle
            [17.7850, 83.3880], # Rushikonda Beach
            [17.8400, 83.4150], # INS Kalinga
            [17.8900, 83.4550]  # Bheemunipatnam (Bheemli)
        ]
    }
}

# Multi-Bus Fusion Algorithm Thresholds
FUSION_DISTANCE_THRESHOLD_METERS = 20.0  # Spatial proximity radius
FUSION_TIME_WINDOW_HOURS = 24.0          # Observation temporal window

# System Simulation Flags
SIMULATION_MODE = True
SIMULATION_LABEL = "[PHASE-1 SIMULATED ENGINE]"

# Visual Theme Colors (Dark Theme Palette)
COLORS = {
    "background": "#0F172A",
    "card_bg": "#1E293B",
    "card_border": "#334155",
    "text_primary": "#F8FAFC",
    "text_secondary": "#94A3B8",
    "accent_blue": "#38BDF8",
    "accent_emerald": "#34D399",
    "accent_amber": "#FBBF24",
    "accent_red": "#F87171",
    "accent_purple": "#C084FC"
}
