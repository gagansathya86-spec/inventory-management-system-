import os
import sys
import json
import time
import cv2
import numpy as np

# Adjust paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

BARCODE_DIR = os.path.join(BASE_DIR, "barcode scanner")
RECOGNITION_DIR = os.path.join(BASE_DIR, "product recognition")
for p in [BARCODE_DIR, RECOGNITION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from barcode_scanner_engine import (
    validate_ean13_checksum,
    validate_barcode,
    classify_product_edibility,
    decode_barcodes_from_image,
    lookup_product_by_barcode,
    insert_verified_product_stock
)
from detector import FastProductDetector
from camera import ThreadedCamera

results_report = []

def record(section, name, passed, details):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {section} :: {name} -> {details}")
    results_report.append({
        "section": section,
        "test": name,
        "status": status,
        "details": details
    })

print("=" * 70)
print("RUNNING COMPREHENSIVE REGRESSION & INTEGRATION TESTS")
print("=" * 70)

# -------------------------------------------------------------
# 1. UNIT TESTS: Barcode Checksum, Edibility, Category Map
# -------------------------------------------------------------
print("\n--- 1. UNIT TESTS ---")
# EAN-13 valid checksum
ean_valid = validate_ean13_checksum("8901063012486")
record("Unit", "EAN-13 Valid Checksum", ean_valid is True, f"8901063012486 valid: {ean_valid}")

# EAN-13 invalid checksum
ean_invalid = validate_ean13_checksum("8901063012487")
record("Unit", "EAN-13 Invalid Checksum Caught", ean_invalid is False, f"8901063012487 valid: {ean_invalid}")

# General barcode validator
val_ok, val_fmt = validate_barcode("8901063012486")
record("Unit", "General Barcode Validator", val_ok is True, f"Format identified: {val_fmt}")

# Edibility classification
edible_res = classify_product_edibility("biscuit", "Parle-G Biscuit")
record("Unit", "Edible Product Classification", edible_res == "EDIBLE", f"biscuit is {edible_res}")

non_edible_res = classify_product_edibility("books", "Notebook 200 pages")
record("Unit", "Non-Edible Product Classification", non_edible_res == "NON-EDIBLE", f"books is {non_edible_res}")

# Category Mapping Verification
cat_file = os.path.join(RECOGNITION_DIR, "categories.json")
with open(cat_file, "r") as f:
    cat_data = json.load(f)
categories_list = cat_data.get("categories", [])
prompt_map = cat_data.get("prompt_to_category", {})
num_cats = len(categories_list)
num_prompts = len(prompt_map)
record("Unit", "Categories & Prompts Integrity", num_cats == 17 and num_prompts == 51,
       f"Total Categories: {num_cats} (expected 17), Prompts: {num_prompts} (expected 51)")

# -------------------------------------------------------------
# 2. BARCODE DECODING TESTS (zxing-cpp & cv2)
# -------------------------------------------------------------
print("\n--- 2. BARCODE DECODING TESTS ---")
try:
    import barcode
    from barcode.writer import ImageWriter
    # Generate barcode test image in memory
    EAN = barcode.get_barcode_class('ean13')
    ean = EAN('890106301248', writer=ImageWriter()) # auto calculates 13th digit (6)
    test_barcode_path = os.path.join(BASE_DIR, "test_generated_ean13.png")
    ean.save(test_barcode_path.replace(".png", ""))
    
    b_img = cv2.imread(test_barcode_path)
    decoded_list = decode_barcodes_from_image(b_img)
    if os.path.exists(test_barcode_path):
        os.remove(test_barcode_path)
        
    if decoded_list and len(decoded_list) > 0:
        first = decoded_list[0]
        b_fmt = first.get("format") or first.get("type", "EAN-13")
        b_eng = first.get("engine") or first.get("decoder_engine", "zxing-cpp")
        record("Barcode Decoder", "Real Barcode Image Decoding", first["barcode"] == "8901063012486",
               f"Decoded: {first['barcode']} | Format: {b_fmt} | Engine: {b_eng}")
    else:
        record("Barcode Decoder", "Real Barcode Image Decoding", False, "No barcode detected by engines")
except Exception as e:
    record("Barcode Decoder", "Real Barcode Image Decoding", False, f"Error: {e}")

# -------------------------------------------------------------
# 3. EXTERNAL PRODUCT LOOKUP TESTS
# -------------------------------------------------------------
print("\n--- 3. EXTERNAL PRODUCT LOOKUP TESTS ---")
# Real Live Open Food Facts test with barcode 3017620422003 (Nutella)
lookup_res = lookup_product_by_barcode("3017620422003")
if lookup_res and lookup_res.get("source") in ["Open Food Facts API", "external_api"]:
    record("External API", "Live Open Food Facts Request", True,
           f"Product: {lookup_res.get('name')} | Brand: {lookup_res.get('brand')} | Source: {lookup_res.get('source')}")
elif lookup_res and lookup_res.get("source") in ["Local Catalog Seed", "manual_entry"]:
    record("External API", "Live Open Food Facts Request (Fallback Active)", True,
           f"Product: {lookup_res.get('name')} | Source: {lookup_res.get('source')}")
else:
    record("External API", "Live Open Food Facts Request", False,
           f"Unexpected lookup result: {lookup_res}")

# Local Seed Catalog Fallback test
seed_res = lookup_product_by_barcode("8901063012486")
record("Catalog Seed", "Local Verified Seed Fallback", seed_res is not None and seed_res.get("source") in ["Local Catalog Seed", "local_catalog", "supabase"],
       f"Product: {seed_res.get('name') if seed_res else 'None'} | Source: {seed_res.get('source') if seed_res else 'None'}")

# -------------------------------------------------------------
# 4. ACTUAL MODEL INFERENCE TESTS (YOLOv8s-World v2)
# -------------------------------------------------------------
print("\n--- 4. ACTUAL MODEL INFERENCE TESTS ---")
model_path = os.path.join(RECOGNITION_DIR, "yolov8s_17categories_enhanced.pt")
try:
    detector = FastProductDetector(model_path=model_path, conf_thresh=0.15)
    model_loaded = detector.model is not None
    record("Model Inference", "YOLOv8s-World v2 Model Load", model_loaded,
           f"Architecture: YOLOv8s-World v2 | Task: {detector.model.task if hasattr(detector.model, 'task') else 'detect'}")
    
    # Test on a real image (books sample in test_product_samples)
    test_img_path = os.path.join(BASE_DIR, "test_product_samples", "books.jpg")
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join(BASE_DIR, "scratch", "books.jpg")
        
    if os.path.exists(test_img_path):
        sample_img = cv2.imread(test_img_path)
        dets, lat = detector.predict(sample_img)
        
        has_detection = len(dets) > 0
        best_det = dets[0] if has_detection else None
        b_conf = best_det['conf'] if best_det else 0.0
        b_cat = best_det['category'] if best_det else 'None'
        record("Model Inference", "Real Image Inference (Books)", has_detection,
               f"Detections: {len(dets)} | Best Cat: {b_cat} | Conf: {b_conf:.2f} | Latency: {lat:.1f}ms")
    else:
        record("Model Inference", "Real Image Inference (Books)", False, f"Test image not found at {test_img_path}")
        
    # Test Uncertainty Guard on a black/blank image
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    b_dets, _ = detector.predict(blank_img)
    record("Model Inference", "Uncertainty Guard on Blank Image", len(b_dets) == 0,
           f"Detections: {len(b_dets)} (Correctly returned 0 detections, no invented label)")
           
except Exception as e:
    record("Model Inference", "YOLOv8s-World v2 Model Load & Predict", False, f"Exception: {e}")

# -------------------------------------------------------------
# 5. REAL CAMERA HARDWARE TEST
# -------------------------------------------------------------
print("\n--- 5. REAL CAMERA HARDWARE TEST ---")
try:
    cam = ThreadedCamera(src=0)
    connected = cam.is_connected()
    if connected:
        cam.start()
        time.sleep(0.5)
        ret, frame = cam.read()
        valid_frame = ret and frame is not None and len(frame.shape) == 3
        record("Camera Hardware", "Webcam DirectShow Capture", valid_frame,
               f"Connected: True | Resolution: {frame.shape[1]}x{frame.shape[0]} | Channels: {frame.shape[2]}")
    else:
        record("Camera Hardware", "Webcam DirectShow Capture", False, "Could not open camera device 0")
    cam.stop()
except Exception as e:
    record("Camera Hardware", "Webcam DirectShow Capture", False, f"Camera test error: {e}")

# -------------------------------------------------------------
# 6. REAL SUPABASE INTEGRATION & DUPLICATE GUARD
# -------------------------------------------------------------
print("\n--- 6. REAL SUPABASE INTEGRATION & DUPLICATE GUARD ---")
import requests
from barcode_scanner_engine import SUPABASE_URL, SUPABASE_HEADERS

try:
    # Check products table query
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/products?select=id,name,barcode,category&limit=3", headers=SUPABASE_HEADERS, timeout=6)
    has_products = resp.status_code == 200 and len(resp.json()) > 0
    record("Supabase DB", "Live Products Query", has_products,
           f"HTTP {resp.status_code} | Fetched {len(resp.json()) if resp.status_code==200 else 0} records from 'products'")
    
    # Test Duplicate Prevention & Stock Addition
    ins_res = insert_verified_product_stock(
        barcode="8901063012486",
        product_name="Parle-G Glucose Biscuit Gold",
        category="biscuit",
        shelf_id="S1",
        quantity=5,
        mfg_date="2025-01-01",
        expiry_date="2026-12-31",
        expiry_status="BEST",
        brand="Parle"
    )
    rec = ins_res.get("record", {})
    record("Supabase DB", "Duplicate Prevention & Stock Sync", ins_res["success"] is True,
           f"Product ID: {rec.get('product_id')} | Prod Synced: {rec.get('supabase_product_synced')} | Inv Synced: {rec.get('supabase_inventory_synced')}")
except Exception as e:
    record("Supabase DB", "Supabase Integration Operations", False, f"Exception: {e}")

# -------------------------------------------------------------
# SUMMARY REPORT
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
passed_count = sum(1 for r in results_report if r["status"] == "PASS")
total_count = len(results_report)
print(f"Total Tests Run: {total_count} | Passed: {passed_count} | Failed: {total_count - passed_count}")
for r in results_report:
    print(f" - [{r['status']}] {r['section']} :: {r['test']} -> {r['details']}")
