"""
SIH26124: Login Screen UI & Mobile OTP Password Reset Component (Phase 7)
Renders dark glassmorphic login card and Mobile OTP password reset workflow.
"""

import streamlit as st
from datetime import datetime
from auth import authenticate_user, generate_mobile_otp, verify_otp_and_reset_password, register_user


def render_login_screen():
    """
    Renders the platform login page with Mobile OTP password reset tab and User Registration tab.
    Prevents access to the operational dashboard until valid authentication occurs.
    """
    st.markdown(
        """<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-radius: 16px; padding: 24px 32px; margin: 15px auto 25px auto; max-width: 720px; box-shadow: 0 20px 40px -15px rgba(0,0,0,0.7); text-align: center;">
<span style="background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;">SIH26124 • SECURE PORTAL</span>
<h1 style="color: #F8FAFC; font-size: 2.1rem; font-weight: 800; margin: 10px 0 4px 0; letter-spacing: -0.5px; background: linear-gradient(90deg, #F8FAFC 0%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AI-Powered Mobile Urban Intelligence Platform</h1>
<p style="color: #94A3B8; margin: 0 0 10px 0; font-size: 1.05rem; font-weight: 500;">Secure Fleet & Urban Intelligence Access</p>
</div>""",
        unsafe_allow_html=True
    )

    col_l, col_center, col_r = st.columns([1, 2.8, 1])

    with col_center:
        tab_login, tab_reset, tab_register = st.tabs([
            "🔐 Account Login",
            "📱 Forgot / Reset Password",
            "📝 Create New Account"
        ])

        # -------------------------------------------------------------
        # TAB 1: STANDARD ACCOUNT LOGIN
        # -------------------------------------------------------------
        with tab_login:
            st.markdown("#### 🔑 Enter Credentials")
            
            with st.form("login_form", clear_on_submit=False):
                selected_role_label = st.selectbox(
                    "Select Operational Role",
                    ["ONBOARD BUS", "OFFICIAL COMMAND CENTER"],
                    index=0,
                    help="Choose ONBOARD BUS for single-bus edge POV or OFFICIAL for cross-fleet command center access."
                )
                
                role_code = "BUS" if selected_role_label == "ONBOARD BUS" else "OFFICIAL"
                
                user_id = st.text_input(
                    "Account ID / User ID",
                    placeholder="e.g. BUS-07 or OFFICIAL-001",
                    help="Enter your assigned account ID."
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter account password",
                    help="Passwords are never stored in plain text."
                )
                
                submit_btn = st.form_submit_button("🔓 Authenticate & Launch Dashboard", use_container_width=True)

            if submit_btn:
                success, user_dict, err_msg = authenticate_user(user_id, password, role_code)
                if success and user_dict:
                    st.session_state.authenticated = True
                    st.session_state.user_role = user_dict["role"]
                    st.session_state.user_id = user_dict["user_id"]
                    st.session_state.username = user_dict["username"]
                    st.session_state.bus_id = user_dict.get("bus_id")
                    st.session_state.route_id = user_dict.get("route_id")
                    st.session_state.operator_id = user_dict.get("operator_id")
                    st.session_state.login_time = datetime.now().strftime("%H:%M:%S")
                    
                    st.success(f"✅ Authenticated as {user_dict['username']} ({user_dict['role']}). Redirecting...")
                    st.rerun()
                else:
                    st.error(f"❌ {err_msg}")

        # -------------------------------------------------------------
        # TAB 2: MOBILE OTP PASSWORD RESET SYSTEM
        # -------------------------------------------------------------
        with tab_reset:
            st.markdown("#### 📱 Mobile Number OTP Password Reset")
            st.caption("Verify your registered mobile number to receive a 6-digit OTP code and update your account password.")

            if "reset_stage" not in st.session_state:
                st.session_state.reset_stage = "request"
                st.session_state.active_otp = None
                st.session_state.otp_user_id = None
                st.session_state.otp_masked_mobile = None

            # STAGE 1: Request Mobile OTP
            with st.form("request_otp_form"):
                req_user_id = st.text_input(
                    "Account ID",
                    placeholder="e.g. BUS-07 or OFFICIAL-001",
                    value=st.session_state.get("otp_user_id", "")
                )
                req_mobile = st.text_input(
                    "Registered Mobile Number",
                    placeholder="e.g. 9491591473 (BUS-07), 7842835677 (BUS-08), 7995974455 (OFFICIAL-001)",
                    help="Registered mobile numbers: BUS-07: 9491591473 | BUS-08: 7842835677 | OFFICIAL-001: 7995974455 | OFFICIAL-002: 6303133198"
                )
                btn_req_otp = st.form_submit_button("📩 Send 6-Digit Mobile OTP", use_container_width=True)

            if btn_req_otp:
                ok, code, masked_mob, msg = generate_mobile_otp(req_user_id, req_mobile)
                if ok:
                    st.session_state.reset_stage = "verify"
                    st.session_state.active_otp = code
                    st.session_state.otp_user_id = req_user_id.strip().upper()
                    st.session_state.otp_masked_mobile = masked_mob
                    st.success(f"✅ {msg}")
                else:
                    st.error(f"❌ {msg}")

            # Display Simulated SMS Dispatch Box if OTP active
            if st.session_state.get("active_otp"):
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(56, 189, 248, 0.1);
                        border: 1px solid #38BDF8;
                        border-radius: 10px;
                        padding: 14px 18px;
                        margin: 16px 0;
                    ">
                        <div style="color: #38BDF8; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.5px;">
                            📱 SIMULATED SMS GATEWAY NOTIFICATION
                        </div>
                        <div style="color: #F8FAFC; font-size: 0.95rem; margin-top: 4px;">
                            SMS sent to <b>{st.session_state.otp_masked_mobile}</b>:
                        </div>
                        <div style="
                            background: #0F172A;
                            border: 1px dashed #38BDF8;
                            padding: 8px 14px;
                            border-radius: 6px;
                            color: #FBBF24;
                            font-size: 1.25rem;
                            font-weight: 800;
                            letter-spacing: 4px;
                            display: inline-block;
                            margin-top: 6px;
                        ">
                            {st.session_state.active_otp}
                        </div>
                        <div style="color: #94A3B8; font-size: 0.75rem; margin-top: 4px;">
                            Use this 6-digit OTP code below to verify mobile ownership and reset password.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # STAGE 2: Verify OTP & Enter New Password
                with st.form("verify_otp_form"):
                    st.markdown(f"**Verify OTP for Account:** `{st.session_state.otp_user_id}`")
                    entered_otp = st.text_input("Enter 6-Digit OTP Code", placeholder="e.g. 482910")
                    new_pwd = st.text_input("New Password", type="password", placeholder="Enter new password (min 4 chars)")
                    btn_reset = st.form_submit_button("🔑 Verify OTP & Update Password", use_container_width=True)

                if btn_reset:
                    v_ok, v_msg = verify_otp_and_reset_password(
                        st.session_state.otp_user_id,
                        entered_otp,
                        st.session_state.active_otp,
                        new_pwd
                    )
                    if v_ok:
                        st.success(f"✅ {v_msg}")
                        # Clear OTP session state
                        st.session_state.active_otp = None
                        st.session_state.reset_stage = "request"
                    else:
                        st.error(f"❌ {v_msg}")

        # -------------------------------------------------------------
        # TAB 3: ACCOUNT CREATION / USER REGISTRATION
        # -------------------------------------------------------------
        with tab_register:
            st.markdown("#### 📝 Create New Account")
            st.caption("Register a new ONBOARD BUS or OFFICIAL COMMAND CENTER operator account.")

            with st.form("register_form", clear_on_submit=False):
                reg_role_label = st.selectbox(
                    "Select Operational Role",
                    ["OFFICIAL COMMAND CENTER", "ONBOARD BUS"],
                    index=0,
                    help="Choose OFFICIAL for command center officers or ONBOARD BUS for fleet vehicle operators."
                )
                reg_role_code = "BUS" if reg_role_label == "ONBOARD BUS" else "OFFICIAL"

                reg_user_id = st.text_input(
                    "Desired Account ID / User ID *",
                    placeholder="e.g. OFFICIAL-005 or BUS-19",
                    help="Unique identifier for login (e.g. OFFICIAL-005 or BUS-19)."
                )

                reg_username = st.text_input(
                    "Full Name / Display Name *",
                    placeholder="e.g. Officer Rajesh Varma",
                    help="Name of the account owner."
                )

                reg_mobile = st.text_input(
                    "10-Digit Mobile Number *",
                    placeholder="e.g. 9876543210",
                    help="Required for OTP password recovery."
                )

                reg_pwd = st.text_input(
                    "Password (Min 4 chars) *",
                    type="password",
                    placeholder="Enter account password"
                )

                reg_pwd_confirm = st.text_input(
                    "Confirm Password *",
                    type="password",
                    placeholder="Re-enter password"
                )

                # Optional Bus & Route info for Bus operators
                reg_bus_id = None
                reg_route_id = None
                if reg_role_code == "BUS":
                    c_b1, c_b2 = st.columns(2)
                    with c_b1:
                        reg_bus_id = st.text_input("Assigned Bus ID", placeholder="e.g. BUS-19")
                    with c_b2:
                        reg_route_id = st.selectbox("Assigned Transit Route", ["ROUTE-101", "ROUTE-202", "ROUTE-303", "ROUTE-404"])

                btn_register = st.form_submit_button("📝 Register Account & Save", use_container_width=True)

            if btn_register:
                if reg_pwd != reg_pwd_confirm:
                    st.error("❌ Passwords do not match. Please ensure both password fields match.")
                else:
                    reg_ok, reg_msg = register_user(
                        user_id=reg_user_id,
                        username=reg_username,
                        password=reg_pwd,
                        role=reg_role_code,
                        mobile_number=reg_mobile,
                        bus_id=reg_bus_id,
                        route_id=reg_route_id
                    )
                    if reg_ok:
                        st.success(f"✅ {reg_msg}")
                    else:
                        st.error(f"❌ {reg_msg}")

        st.markdown("---")
        
        st.info(
            "ℹ️ **SECURITY & LOGIN GUIDE**\n\n"
            "• **Role-Based Authentication**: Secure password verification for ONBOARD BUS operators and OFFICIAL COMMAND CENTER officers.\n"
            "• **Account Registration**: Create new official commander or bus fleet operator accounts under the 'Create New Account' tab.\n"
            "• **Registered Mobile OTP Verification**: For demonstration, test OTP verification codes are generated for registered numbers (`BUS-07`: `9491591473` | `BUS-08`: `7842835677` | `OFFICIAL-001`: `7995974455` | `OFFICIAL-002`: `6303133198`)."
        )
