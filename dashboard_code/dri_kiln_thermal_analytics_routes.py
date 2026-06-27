import os
import json
import sqlite3
import threading
import time
from datetime import datetime
from flask import Blueprint, render_template, jsonify, request
from collections import defaultdict
import numpy as np
import requests
from sklearn.cluster import KMeans
from rbac_auth.rbac import require 
from rbac_auth.audit import audit_log
from openpyxl import Workbook


# =========================================================
# GLOBAL CACHE & CONFIG
# =========================================================
KILN_ROWS = 250
KILN_COLS = 1200
KILN_LENGTH_M = 70.0

PATCH_H_ROWS = 2
PATCH_W_COLS = 40  

PATCH_GRID_ROWS = KILN_ROWS // PATCH_H_ROWS  
PATCH_GRID_COLS = KILN_COLS // PATCH_W_COLS  

CACHE_LOCK = threading.Lock()
POLLER_STARTED = False

LATEST_DATA_CACHE = {
    "timestamp": None,
    "kpis": {
        "max": 0.0, "min": 0.0, "avg": 0.0, "std": 0.0,
        "metric_ranges": {"max": {"min": 0.0, "max": 0.0}, "mean": {"min": 0.0, "max": 0.0}, "std": {"min": 0.0, "max": 0.0}, "grad": {"min": 0.0, "max": 0.0}}
    },
    "meta": {
        "kiln_rows": KILN_ROWS, "kiln_cols": KILN_COLS, "kiln_length_m": KILN_LENGTH_M,
        "patch_h_rows": PATCH_H_ROWS, "patch_w_cols": PATCH_W_COLS, "grid_rows": PATCH_GRID_ROWS, "grid_cols": PATCH_GRID_COLS,
    },
    "lengths": [], "angles": [], "patches": []
}

PATCH_LUT = {}  
POLL_INTERVAL_SEC = 120  

def _generate_patch_lut():
    lut = {}
    patch_id = 0
    for r in range(0, KILN_ROWS, PATCH_H_ROWS):          
        for c in range(0, KILN_COLS, PATCH_W_COLS):      
            angle_deg = round((r / float(KILN_ROWS)) * 360.0, 2)
            length_m = round((c / float(KILN_COLS)) * KILN_LENGTH_M, 2)
            lut[patch_id] = (length_m, angle_deg)
            patch_id += 1
    return lut

def _get_db_path(project_root):
    return os.path.join(project_root, 'dri_kiln_thermal_data', 'database', 'dri_kiln_thermal.db')
    
def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table_name,))
    return cur.fetchone() is not None

def _refresh_cache_from_db(db_path: str) -> bool:
    global LATEST_DATA_CACHE
    if not os.path.exists(db_path): return False

    conn = sqlite3.connect(db_path)
    conn.execute('pragma journal_mode=wal') # Prevents "database locked" errors
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if not _table_exists(cur, "thermal_patches"):
        conn.close()
        return False

    cur.execute("SELECT timestamp, patches_json FROM thermal_patches ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()

    if not row: return False
    try: raw_patches = json.loads(row["patches_json"] or "[]")
    except Exception: raw_patches = []
    if not isinstance(raw_patches, list) or not raw_patches: return False

    try: raw_patches.sort(key=lambda p: int(p[0]) if isinstance(p, (list, tuple)) and len(p) else 0)
    except Exception: pass

    enriched = []
    means, maxes, stds, grads = [], [], [], []
    lut = PATCH_LUT  

    for p in raw_patches:
        if not isinstance(p, (list, tuple)) or len(p) < 6: continue
        try: pid = int(p[0])
        except Exception: continue

        length_m, angle_deg = lut.get(pid, (0.0, 0.0))
        p_list = list(p) 
        enriched.append(p_list + [length_m, angle_deg])

        try: means.append(float(p_list[1]))
        except Exception: pass
        try: maxes.append(float(p_list[2]))
        except Exception: pass
        try: stds.append(float(p_list[3]))
        except Exception: pass
        try: grads.append(float(p_list[4]))
        except Exception: pass

    def _rng(arr):
        if not arr: return {"min": 0.0, "max": 0.0}
        return {"min": float(min(arr)), "max": float(max(arr))}

    if means:
        avg = sum(means) / len(means)
        var = sum((x - avg) * (x - avg) for x in means) / len(means)
        std_mean = var ** 0.5
        kpis = {
            "max": float(max(maxes) if maxes else max(means)), "min": float(min(means)), "avg": round(avg, 2), "std": round(std_mean, 2),
            "metric_ranges": {"max": _rng(maxes), "mean": _rng(means), "std": _rng(stds), "grad": _rng(grads)}
        }
    else:
        kpis = {
            "max": 0.0, "min": 0.0, "avg": 0.0, "std": 0.0,
            "metric_ranges": {"max": {"min": 0.0, "max": 0.0}, "mean": {"min": 0.0, "max": 0.0}, "std": {"min": 0.0, "max": 0.0}, "grad": {"min": 0.0, "max": 0.0}}
        }

    with CACHE_LOCK:
        LATEST_DATA_CACHE["timestamp"] = row["timestamp"]
        LATEST_DATA_CACHE["patches"] = enriched
        LATEST_DATA_CACHE["kpis"] = kpis
        if not LATEST_DATA_CACHE.get("lengths") and lut: LATEST_DATA_CACHE["lengths"] = [lut[i][0] for i in range(PATCH_GRID_COLS)]
        if not LATEST_DATA_CACHE.get("angles") and lut: LATEST_DATA_CACHE["angles"] = [lut[i * PATCH_GRID_COLS][1] for i in range(PATCH_GRID_ROWS)]
    return True

def _poll_latest_data(db_path):
    while True:
        try: _refresh_cache_from_db(db_path)
        except Exception as e: print(f"[Analytics Poller Error]: {e}")
        time.sleep(POLL_INTERVAL_SEC)

# =========================================================
# PROFESSIONAL ACCRETION MODEL & ML (ZONAL K-MEANS)
# =========================================================
def compute_accretion_model(history_payload, target_pid=None, patch_lut=None):
    daily_patch_data = defaultdict(list)
    daily_ref_data = defaultdict(list) 

    target_zone = None
    if target_pid is not None and patch_lut is not None:
        length_m, _ = patch_lut.get(target_pid, (0.0, 0.0))
        if length_m <= 23.3: target_zone = "feed"
        elif length_m <= 46.6: target_zone = "mid"
        else: target_zone = "burning"

    for frame in history_payload:
        ts = frame.get("timestamp")
        if not ts: continue
        date = ts.split(" ")[0]

        # Professional Filter: Ignore data when Plant is shut down (< 80C)
        g_mean = frame.get("frame_global_mean", 0)
        if g_mean < 80.0:
            continue 

        if target_zone:
            ref_temp = frame.get("zonal_means", {}).get(target_zone, g_mean)
        else:
            ref_temp = g_mean
            
        daily_ref_data[date].append(ref_temp)

        patch_temp = None
        for p in frame.get("patches", []):
            if isinstance(p, list) and len(p) > 1:
                try:
                    t = float(p[1])
                    if target_pid is not None and int(p[0]) == target_pid:
                        patch_temp = t
                    elif target_pid is None and int(p[0]) == -1:
                        patch_temp = t
                except: pass
        
        if patch_temp is not None:
            daily_patch_data[date].append(patch_temp)

    daily_stats = []
    for date in sorted(daily_ref_data.keys()):
        r_vals = daily_ref_data[date]
        p_vals = daily_patch_data.get(date, r_vals)

        r_avg = sum(r_vals) / len(r_vals) if r_vals else 0
        p_avg = sum(p_vals) / len(p_vals) if p_vals else r_avg

        metric_val = (p_avg - r_avg) if target_pid is not None else r_avg

        daily_stats.append({
            "date": date,
            "patch_avg": round(p_avg, 2),
            "ref_avg": round(r_avg, 2),
            "metric_val": round(metric_val, 2)
        })
    
    return daily_stats

def compute_slopes_and_clusters(daily_stats, target_pid=None):
    if not daily_stats: return [], []

    slopes = []
    for i in range(1, len(daily_stats)):
        y2 = daily_stats[i]["metric_val"]
        y1 = daily_stats[i - 1]["metric_val"]
        slope = round(y2 - y1, 3)
        slopes.append({"date": daily_stats[i]["date"], "slope": slope})

    clusters_result = []
    if len(slopes) >= 3:
        X = []
        valid_dates = []
        for i in range(1, len(daily_stats)):
            try:
                val = daily_stats[i]["metric_val"]
                slp = slopes[i-1]["slope"]
                X.append([val, slp])
                valid_dates.append(daily_stats[i]["date"])
            except: continue

        if len(X) >= 3:
            X_np = np.array(X)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_np)

            centers = kmeans.cluster_centers_
            sorted_indices = np.argsort(centers[:, 0])
            
            if target_pid is not None:
                zone_mapping = { sorted_indices[0]: "accretion_risk", sorted_indices[1]: "normal", sorted_indices[2]: "hotspot" }
            else:
                zone_mapping = { sorted_indices[0]: "cold_anomaly", sorted_indices[1]: "normal", sorted_indices[2]: "hotspot" }

            for i, label in enumerate(labels):
                val, slp = X_np[i]
                clusters_result.append({
                    "date": valid_dates[i], "metric_val": float(val), "slope": float(slp), "cluster": int(label), "zone": zone_mapping[label]
                })

    return slopes, clusters_result
    
# =========================================================
# ANOMALY DETECTION (ISOLATION FOREST)
# =========================================================
from sklearn.ensemble import IsolationForest

def detect_anomalies(patches):
    features = []
    pids = []

    for p in patches:
        if not isinstance(p, (list, tuple)) or len(p) < 5:
            continue
        try:
            pid = int(p[0])
            mean = float(p[1])
            max_t = float(p[2])
            std = float(p[3])
            grad = float(p[4])

            features.append([mean, max_t, std, grad])
            pids.append(pid)
        except:
            continue

    if len(features) < 20:
        return []

    X = np.array(features)

    model = IsolationForest(
        n_estimators=120,
        contamination=0.05,
        random_state=42
    )

    preds = model.fit_predict(X)

    anomalies = []
    for i, pred in enumerate(preds):
        if pred == -1:
            anomalies.append({
                "pid": pids[i],
                "mean": features[i][0],
                "max": features[i][1],
                "std": features[i][2],
                "grad": features[i][3]
            })

    return anomalies

# =========================================================
# BLUEPRINT & ROUTES
# =========================================================
def create_blueprint(project_root):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    template_dir = os.path.join(root_dir, 'templates')
    static_dir = os.path.join(root_dir, 'static')

    bp = Blueprint(
        'dri_kiln_thermal_analytics', __name__, url_prefix='/apps/dri-kiln-thermal-analytics',
        template_folder=template_dir, static_folder=static_dir, static_url_path='static'
    )

    global PATCH_LUT
    PATCH_LUT = _generate_patch_lut()
    with CACHE_LOCK:
        if not LATEST_DATA_CACHE.get("lengths") and PATCH_LUT: LATEST_DATA_CACHE["lengths"] = [PATCH_LUT[i][0] for i in range(PATCH_GRID_COLS)]
        if not LATEST_DATA_CACHE.get("angles") and PATCH_LUT: LATEST_DATA_CACHE["angles"] = [PATCH_LUT[i * PATCH_GRID_COLS][1] for i in range(PATCH_GRID_ROWS)]

    db_path = _get_db_path(project_root)
    
    global POLLER_STARTED
    if not POLLER_STARTED:
        POLLER_STARTED = True
        poller_thread = threading.Thread(target=_poll_latest_data, args=(db_path,), daemon=True)
        poller_thread.start()
    
    try: _refresh_cache_from_db(db_path)
    except: pass

    @bp.route('')
    @bp.route('')
    @bp.route('/')
    @bp.route('/interactive-dashboard')
    @require()
    @audit_log()
    def interactive_dashboard():
        return render_template('interactive-dashboard.html', app_title='Interactive Dashboard')

    @bp.route('/api/latest', methods=['GET'])
    @require()
    def api_get_latest():
        if not LATEST_DATA_CACHE.get("timestamp"):
            try: _refresh_cache_from_db(db_path)
            except: pass
        
        with CACHE_LOCK:
            data = dict(LATEST_DATA_CACHE)
                
        try:
            anomalies = detect_anomalies(data.get("patches", []))
        except Exception:
            anomalies = []
        data["anomalies"] = anomalies
        return jsonify(data)

    @bp.route('/api/trendline', methods=['POST'])
    @require()
    def api_get_trendline():
        data = request.json
        if not data or 'start' not in data or 'end' not in data:
            return jsonify({"error": "Missing start or end date"}), 400

        start_date = data['start']
        end_date = data['end']
        
        target_pid = data.get('pid')
        start_len = data.get('start_len')
        end_len = data.get('end_len')
        
        metric_key = data.get('metric', 'avg') 
        idx_map = {"avg": 1, "max": 2, "std": 3, "grad": 4}
        val_idx = idx_map.get(metric_key, 1)

        # ROBUST PARSING FOR ZONE PLOTTING
        try: target_pid = int(target_pid) if target_pid is not None and str(target_pid).strip() != "" else None
        except: target_pid = None
        try: start_len = float(start_len) if start_len is not None and str(start_len).strip() != "" else None
        except: start_len = None
        try: end_len = float(end_len) if end_len is not None and str(end_len).strip() != "" else None
        except: end_len = None

        is_patch_mode = (target_pid is not None)
        is_zone_mode = (start_len is not None and end_len is not None and not is_patch_mode)

        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT timestamp, patches_json
                FROM thermal_patches
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """, (start_date, end_date))
            rows = cur.fetchall()
            conn.close()

            history_payload = []
            hottest_temp, hottest_date, hottest_loc = 0, "", ""

            # --- PRE-COMPUTE LOOKUPS OUTSIDE THE LOOP ---
            valid_zone_pids = set()
            if is_zone_mode:
                for pid_key, (l, a) in PATCH_LUT.items():
                    if start_len <= l <= end_len:
                        valid_zone_pids.add(pid_key)

            # --- PREVENT TIMEOUTS: DOWNSAMPLING LOGIC ---
            from datetime import datetime
            is_long_range = False
            try:
                dt_start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
                dt_end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
                if (dt_end - dt_start).days > 2:
                    is_long_range = True
            except: pass
            
            last_processed_period = None

            for r in rows:
                ts = r['timestamp']
                
                if is_long_range:
                    try:
                        date, time_str = ts.split(" ", 1)
                        hour = int(time_str.split(":")[0])
                        period_key = f"{date}-{hour}"
                        if period_key == last_processed_period:
                            continue
                        last_processed_period = period_key
                    except: pass
                import logging
                try: 
                    raw_patches = json.loads(r['patches_json'] or "[]")
                except json.JSONDecodeError as e: 
                    logging.error(f"JSON parsing failed for thermal data: {e}")
                    continue

                optimized_patches = []
                zone_means, zone_maxes, g_means, g_maxes = [], [], [], []

                # --- OPTIMIZED TARGETING ---
                if is_patch_mode:
                    for p in raw_patches:
                        if not isinstance(p, list) or len(p) < 6: continue
                        if int(p[0]) == target_pid:
                            max_val = float(p[2])
                            optimized_patches.append(p)
                            
                            if max_val >= 350.0 and hottest_date == "":
                                hottest_date = ts
                                l_m, a_deg = PATCH_LUT.get(target_pid, (0.0, 0.0))
                                hottest_loc = f"{l_m}m, {a_deg}°"
                                hottest_temp = max_val
                            elif max_val > hottest_temp:
                                hottest_temp = max_val
                                
                            break  # CRITICAL: Stop searching once the target patch is found!

                elif is_zone_mode:
                    for p in raw_patches:
                        if not isinstance(p, list) or len(p) < 6: continue
                        pid = int(p[0])
                        if pid in valid_zone_pids:  # Fast O(1) set lookup
                            cur_val = float(p[val_idx])
                            max_val = float(p[2])
                            zone_means.append(cur_val)
                            zone_maxes.append(max_val)
                            
                            if max_val >= 350.0 and hottest_date == "":
                                hottest_date = ts
                                hottest_loc = f"Zone {start_len}m - {end_len}m"
                                hottest_temp = max_val
                            elif max_val > hottest_temp:
                                hottest_temp = max_val

                else: # Global Mode
                    for p in raw_patches:
                        if not isinstance(p, list) or len(p) < 6: continue
                        cur_val = float(p[val_idx])
                        max_val = float(p[2])
                        g_means.append(cur_val)
                        g_maxes.append(max_val)
                        
                        if max_val >= 350.0 and hottest_date == "":
                            hottest_date = ts
                            l_m, a_deg = PATCH_LUT.get(int(p[0]), (0.0, 0.0))
                            hottest_loc = f"{l_m}m, {a_deg}°"
                            hottest_temp = max_val
                        elif max_val > hottest_temp:
                            hottest_temp = max_val

                # --- AGGREGATION ---
                if is_zone_mode and zone_means:
                    z_avg = sum(zone_means) / len(zone_means)
                    z_max = max(zone_maxes)
                    optimized_patches = [[-1, z_avg, z_max, 0, 0, 0, 0]]
                elif not is_patch_mode and not is_zone_mode and g_means:
                    g_avg = sum(g_means) / len(g_means)
                    g_max = max(g_maxes)
                    optimized_patches = [[-1, g_avg, g_max, 0, 0, 0, 0]]

                if not optimized_patches: continue
                history_payload.append({"timestamp": ts, "patches": optimized_patches})

            return jsonify({
                "status": "success", 
                "data": history_payload,
                "hotspot": {"temp": hottest_temp, "date": hottest_date, "loc": hottest_loc}
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    from flask import send_file
    from io import BytesIO
    import pandas as pd
    
    @bp.route('/api/ai-predict', methods=['POST'])
    @require()
    def api_ai_predict():
        try:
            payload = request.json or {}

            patch_id = payload.get("patch_id")

            if patch_id is None:
                return jsonify({
                    "status": "error",
                    "message": "No patch_id supplied"
                }), 400
                
            import os
            # Tunnel mapping targeting the live FastAPI app on your laptop via Ngrok
            LSTM_URL = os.getenv("LSTM_API_URL", "https://audience-unfiled-mayday.ngrok-free.dev/predict-patch")
            response = requests.post(
                LSTM_URL, json={
                    "patch_id": patch_id
                },
                timeout=30
            )

            response.raise_for_status()
            return jsonify(response.json())

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    
    @bp.route('/api/export-excel', methods=['POST'])
    @require()
    def export_excel():
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # 90 Days ka data stream karein
            cur.execute("""
                SELECT timestamp, patches_json
                FROM thermal_patches
                WHERE timestamp >= datetime('now', '-90 days')
                ORDER BY timestamp ASC
            """)

            TOTAL_PATCHES = PATCH_GRID_ROWS * PATCH_GRID_COLS
            
            # Direct mapping ke liye structure optimize kiya hai
            patch_day_data = {pid: {} for pid in range(TOTAL_PATCHES)}
            last_processed_date = None
            
            for r in cur:
                try:
                    ts = r['timestamp']
                    if not ts or " " not in ts: continue
                    date, time_str = ts.split(" ", 1)
                    hour = int(time_str.split(":")[0])
                    
                    # --- 90-DAY SUPER OPTIMIZATION ---
                    # Har din ke 7 ghante ke bajaye mid-day window (11AM-1PM) ka sirf 1 representational frame uthein
                    if hour < 11 or hour > 13: continue
                    if date == last_processed_date: continue
                    last_processed_date = date
                    
                except:
                    continue
                
                try:
                    patches = json.loads(r['patches_json'])
                except:
                    continue
                  
                for p in patches:
                    if not isinstance(p, list) or len(p) < 2: continue
                    try:
                        pid = int(p[0])
                        temp = float(p[1])
                        # 1 frame per day hai, toh direct store kar sakte hain (no extra loops needed)
                        patch_day_data[pid][date] = temp
                      
                    except: continue

            conn.close()

            all_dates = sorted({d for pid in patch_day_data for d in patch_day_data[pid]})
            final_rows = []

            for pid in range(TOTAL_PATCHES):
                row = {"Patch ID": pid}
                
                dates_for_patch = sorted(patch_day_data[pid].keys())
                x_vals_for_slope = []
                y_vals_for_slope = []
                
                for i, d in enumerate(dates_for_patch):
                    val = patch_day_data[pid][d]
                    y_vals_for_slope.append(val)
                    x_vals_for_slope.append(i + 1)
                
                # 3,750 patches ke liye rapid linear regression
                if len(y_vals_for_slope) > 1:
                    slope = np.polyfit(x_vals_for_slope, y_vals_for_slope, 1)[0]
                else:
                    slope = 0.0

                for d in all_dates:
                    row[d] = patch_day_data[pid].get(d, None)
                
                row["90-Day Slope"] = round(float(slope), 3)
                final_rows.append(row)
                
            from io import BytesIO
            import pandas as pd
            from flask import send_file
            
            df = pd.DataFrame(final_rows)
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            
            return send_file(
                output,
                as_attachment=True,
                download_name="kiln_patch_90day_avg.xlsx",
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            return str(e), 500


    @bp.route('/api/critical-patches', methods=['GET'])
    @require()
    def get_critical_patches():
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT timestamp, patches_json
                FROM thermal_patches
                WHERE timestamp >= datetime('now', '-90 days')
                ORDER BY timestamp ASC
            """)
            rows = cur.fetchall()
            conn.close()

            patch_daily = defaultdict(lambda: defaultdict(list))

            # --- 90-DAY SUPER OPTIMIZATION (1 Frame Per Day) ---
            last_processed_date = None

            for r in rows:
                try:
                    ts = r['timestamp']
                    if not ts or " " not in ts: continue
                    date, time_str = ts.split(" ", 1)
                    hour = int(time_str.split(":")[0])
                    
                    # Grab one representative frame per day (e.g., mid-day window)
                    if hour < 11 or hour > 13: continue
                    if date == last_processed_date: continue
                    last_processed_date = date

                except: continue

                try: patches = json.loads(r['patches_json'])
                except: continue

                for p in patches:
                    if not isinstance(p, list) or len(p) < 3: continue
                    try:
                        pid = int(p[0])
                        t_mean = float(p[1])
                        t_max = float(p[2])
                        patch_daily[pid][date].append((t_mean, t_max))
                    except: continue

            patch_stats = []
            hotspots = []
            
            for pid, dates in patch_daily.items():
                sorted_dates = sorted(dates.keys())
                
                # 1. REAL HISTORICAL HOTSPOT ENGINE (Works with 1 day of data)
                latest_day = sorted_dates[-1]
                latest_max = max([v[1] for v in dates[latest_day]])
                
                # Operator Feedback: Normal is 225-275. 
                # Track as hotspot if it exceeds 325 (Warning) or 400 (Critical)
                if latest_max >= 325.0:
                    start_date = latest_day
                    for d in sorted_dates:
                        if max([v[1] for v in dates[d]]) >= 325.0:
                            start_date = d
                            break 
                    
                    if latest_max >= 400.0:
                        severity = "critical"
                    else:
                        severity = "warning"
                        
                    # Calculate short-term trend direction
                    trend_dir = "→ Stable"
                    if len(sorted_dates) >= 2:
                        prev_max = max([v[1] for v in dates[sorted_dates[-2]]])
                        diff = latest_max - prev_max
                        if diff >= 2.0: trend_dir = "↑ Rising"
                        elif diff <= -2.0: trend_dir = "↓ Cooling"

                    l_m, a_deg = PATCH_LUT.get(pid, (0.0, 0.0))
                    hotspots.append({
                        "pid": pid, 
                        "loc": f"{l_m}m, {a_deg}°", 
                        "temp": round(latest_max, 1), 
                        "start_date": start_date,
                        "severity": severity,
                        "trend": trend_dir
                    })
                
                if len(dates) < 2: 
                    continue
                
                # 2. REGULAR ACCRETION SLOPE
                x_vals = []
                y_vals = []
                for i, d in enumerate(sorted_dates):
                    avg_t = sum([v[0] for v in dates[d]]) / len(dates[d])
                    x_vals.append(i + 1) 
                    y_vals.append(avg_t)
                
                # Fault-tolerant polyfit to prevent 500 errors on NaN
                try:
                    slope = float(np.polyfit(x_vals, y_vals, 1)[0])
                except:
                    slope = 0.0
                    
                min_temp = min(y_vals)
                max_temp = max(y_vals)

                l_m, a_deg = PATCH_LUT.get(pid, (0.0, 0.0))
                patch_stats.append({
                    "pid": pid, "length": l_m, "angle": a_deg,
                    "slope": round(slope, 3),
                    "min_temp": round(min_temp, 1),
                    "max_temp": round(max_temp, 1)
                })

            accretion_candidates = [p for p in patch_stats if p["slope"] <= -0.5]
            insulation_candidates = [p for p in patch_stats if p["slope"] >= 0.5]

            accretion_candidates.sort(key=lambda x: x["slope"])
            insulation_candidates.sort(key=lambda x: x["slope"], reverse=True)

            def get_spatially_diverse(candidates, max_count=20, min_len_dist=2.0, min_ang_dist=20.0):
                selected = []
                for p in candidates:
                    if len(selected) >= max_count:
                        break
                    is_distinct = True
                    for s in selected:
                        l_dist = abs(p["length"] - s["length"])
                        a_dist = min(abs(p["angle"] - s["angle"]), 360 - abs(p["angle"] - s["angle"]))
                        if l_dist < min_len_dist and a_dist < min_ang_dist:
                            is_distinct = False
                            break
                    if is_distinct:
                        selected.append(p)
                return selected

            accretion_risk = get_spatially_diverse(accretion_candidates)
            insulation_risk = get_spatially_diverse(insulation_candidates)

            accretion_formatted = [{"pid": p["pid"], "length": p["length"], "angle": p["angle"], "slope": p["slope"], "temp": p["min_temp"]} for p in accretion_risk]
            insulation_formatted = [{"pid": p["pid"], "length": p["length"], "angle": p["angle"], "slope": p["slope"], "temp": p["max_temp"]} for p in insulation_risk]

            return jsonify({
                "status": "success",
                "accretion": accretion_formatted,
                "insulation": insulation_formatted,
                "hotspots": hotspots
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @bp.route('/api/recommendation/<int:pid>', methods=['GET'])
    @require()
    def get_smart_recommendation(pid):
        with CACHE_LOCK:
            patches = LATEST_DATA_CACHE.get("patches", [])
            
        patch_data = next((p for p in patches if int(p[0]) == pid), None)
        if not patch_data:
            return jsonify({"error": "Patch not found"}), 404

        t_mean = float(patch_data[1])
        t_max = float(patch_data[2])
        t_grad = float(patch_data[4])
        length_m = float(patch_data[6])

        zone = "Feed" if length_m <= 23.3 else "Mid" if length_m <= 46.6 else "Burning"

        risk_score = 0
        explanations = []

        # --- FIX 1: Strict Operator-Aligned Temperature Assessment ---
        temp_status = "NORMAL"
        if t_max < 225:
            risk_score += 10
            temp_status = "LOW"
            explanations.append(f"Temperature ({t_max:.1f}°C) is cooling below baseline.")
        elif 225 <= t_max <= 275:
            risk_score += 0
            temp_status = "NORMAL"
        elif 275 < t_max <= 325:
            risk_score += 15
            temp_status = "ELEVATED"
            explanations.append(f"Temperature ({t_max:.1f}°C) is above normal range, monitor closely.")
        elif 325 < t_max <= 400:
            risk_score += 40
            temp_status = "HIGH"
            explanations.append(f"Elevated hotspot temperature ({t_max:.1f}°C).")
        else:
            risk_score += 70
            temp_status = "CRITICAL"
            explanations.append(f"Critical hotspot detected ({t_max:.1f}°C).")

        # --- FIX 2: Tuned Gradient Logic (Separated from Temp Status) ---
        if abs(t_grad) > 5:
            risk_score += 30
            explanations.append(f"Severe thermal gradient ({t_grad:.2f}) indicates structural stress.")
        elif abs(t_grad) > 2:
            risk_score += 15
            explanations.append(f"Rapid temperature shift detected (Gradient: {t_grad:.2f}).")
            
        if len(explanations) == 0: 
            explanations.append("All parameters are within safe operational limits.")

        # --- FIX 3: 4-Tier Overall Status Classification ---
        if risk_score < 20:
            status = "NORMAL"
        elif risk_score < 50:
            status = "WARNING"
        elif risk_score < 80:
            status = "HIGH"
        else:
            status = "CRITICAL"
            
        # Ensure operators aren't confused if Temp is Normal but Gradient makes Status=WARNING
        if status != "NORMAL" and temp_status == "NORMAL":
            explanations.insert(0, "Note: Temperature is Normal, but structural condition triggered a Warning.")

        # --- FIX 2: Tuned Gradient Logic ---
        if abs(t_grad) > 5:
            risk_score += 25
            explanations.append(f"Severe thermal gradient ({t_grad:.2f}).")
        elif abs(t_grad) > 2:
            risk_score += 15
            explanations.append(f"Rapid temperature change detected ({t_grad:.2f}).")

        # Default normal reason
        if len(explanations) == 0: 
            explanations.append("Parameters are within normal operational limits.")

        # --- FIX 3: 4-Tier Status Classification ---
        if risk_score < 20:
            status = "NORMAL"
        elif risk_score < 50:
            status = "WARNING"
        elif risk_score < 80:
            status = "HIGH"
        else:
            status = "CRITICAL"

        # --- FIX 4: Operator-Specific Action Texts ---
        if status == "NORMAL":
            action = "Continue current operation. Thermal profile is within acceptable limits."
        elif status == "WARNING":
            action = "Increase monitoring frequency. Inspect hotspot growth during next operating cycle."
        elif status == "HIGH":
            action = "Investigate potential accretion formation. Review burner settings and shell condition."
        else:
            action = "Immediate engineering review recommended. Potential refractory damage or severe accretion growth."

        # Calculate dynamic deviations and stats for the industrial dashboard
        baseline_temp = 250.0 # Midpoint of normal 225-275 range
        deviation = t_max - baseline_temp
        
        # Estimate active days based on severity
        active_days = 14 if status in ["HIGH", "CRITICAL"] else (5 if status == "WARNING" else 0)

        # Refractory risk mapping
        refractory_risk = "LOW"
        if t_max > 325: refractory_risk = "MEDIUM"
        if t_max > 380: refractory_risk = "HIGH"
        if t_max >= 400: refractory_risk = "CRITICAL"

        # Pass 'status' as 'risk_level' so the frontend JS maps it correctly
        return jsonify({
            "pid": pid, "zone": zone, "risk_score": risk_score, "risk_level": status,
            "xai": explanations, "recommendation": action,
            "current_temp": t_max, "deviation": deviation, 
            "gradient": t_grad, "active_days": active_days,
            "refractory_risk": refractory_risk
        })

    @bp.route('/api/predict-trend/<int:pid>', methods=['POST'])
    @require()
    def predict_trend(pid):
        try:
            # Send the patch_id directly to the new FastAPI service
            payload = {
                "patch_id": pid
            }

            # This runs server-to-server, avoiding browser CORS issues entirely
            print("=" * 80)
            print("PREDICTION REQUEST")
            print("PID:", pid)
            print("Payload:", payload)

            import os
            # Tunnel mapping targeting the live FastAPI app on your laptop via Ngrok
            LSTM_URL = os.getenv("LSTM_API_URL", "https://audience-unfiled-mayday.ngrok-free.dev/predict-patch")
            
            response = requests.post(
                LSTM_URL,
                json=payload,
                timeout=15
            )

            print("STATUS:", response.status_code)
            print("BODY:", response.text[:1000])
            print("=" * 80)

            response.raise_for_status() 
        
            return jsonify(response.json())
        
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"AI Server (Port 8001 via Ngrok) is unreachable. Details: {str(e)}"}), 503
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
            
    return bp
    
    