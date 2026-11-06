import json
import os
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

memory_paths = [
    os.path.join(os.path.dirname(__file__), "memory.json"),
    os.path.join(os.path.dirname(__file__), "dist", "TeoAssistant", "memory.json"),
    os.path.join(os.path.dirname(__file__), "dist", "TeoAssistant", "_internal", "memory.json")
]

for p in memory_paths:
    if not os.path.exists(p):
        print(f"Skipping (not found): {p}")
        continue
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {p}: {e}")
        continue

    initial_count = len(data)
    cleaned_data = {}

    for k, v in data.items():
        # Check if key or action contains multi-command artifacts
        k_lower = k.lower().strip()
        should_remove = False

        if "mở powerpoint mở excel" in k_lower or "mở powerpoint và mở excel" in k_lower:
            should_remove = True
        
        # Check actions
        if isinstance(v, list):
            for act in v:
                if isinstance(act, dict) and act.get("action") == "open_app":
                    app_name = act.get("name", "").lower()
                    if any(term in app_name for term in [" mở ", " bật ", " và ", "vở", "powerpoint mở"]):
                        should_remove = True
                        break
        
        # Check verb conflicts (e.g. key has "tắt" but action has "open_app")
        has_close_key = any(w in k_lower for w in ["tắt", "đóng", "thoát", "dừng"])
        has_open_key = any(w in k_lower for w in ["mở", "bật", "khởi động"])
        
        if isinstance(v, list):
            act_types = [act.get("action") for act in v if isinstance(act, dict)]
            if has_close_key and not has_open_key and "open_app" in act_types:
                should_remove = True
            if has_open_key and not has_close_key and "close_app" in act_types:
                should_remove = True

            # Remove measurement actions with numbers/percentages
            for act in v:
                if isinstance(act, dict) and act.get("action") in ("volume", "brightness"):
                    if "percentage" in act or "step" in act:
                        should_remove = True
                        break

        # Check measurement commands in key with digits or %
        if any(c.isdigit() or c == '%' for c in k_lower) and any(term in k_lower for term in ["âm lượng", "độ sáng", "ánh sáng", "tiếng", "volume", "brightness"]):
            should_remove = True

        if not should_remove:
            cleaned_data[k] = v
        else:
            print(f"[{os.path.basename(p)}] Removed bad entry: '{k}' -> {v}")

    with open(p, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    print(f"Cleaned {p}: {initial_count} -> {len(cleaned_data)} entries.")

if __name__ == "__main__":
    pass
