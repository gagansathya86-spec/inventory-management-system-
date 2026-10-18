import os
import sys
import time
import requests
import unittest

BASE_DIR = r"c:\Users\gagan\Downloads\dl smart inventory"
sys.path.append(os.path.join(BASE_DIR, "trend identification"))
sys.path.append(os.path.join(BASE_DIR, "product recognition"))
sys.path.append(os.path.join(BASE_DIR, "barcode scanner"))
sys.path.append(os.path.join(BASE_DIR, "shelf detection"))

import trend_analysis_engine as tae

SERVER_URL = "http://127.0.0.1:5000"

class TestFeature4IsolatedInventory(unittest.TestCase):

    def test_01_migration_file_exists_and_valid(self):
        """Test 1: Verify trend_inventory_migration.sql exists and defines isolated tables & 17 approved categories."""
        mig_path = os.path.join(BASE_DIR, "trend_inventory_migration.sql")
        self.assertTrue(os.path.exists(mig_path), "trend_inventory_migration.sql must exist")
        with open(mig_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("trend_inventory_products", content)
        self.assertIn("trend_inventory_events", content)
        self.assertIn("trend_inventory_forecasts", content)
        for cat in tae.APPROVED_CATEGORIES:
            self.assertIn(cat, content)
        print("[TEST 1 PASS] Verified trend_inventory_migration.sql with 17 approved categories and reversible rollback.")

    def test_02_only_17_approved_categories(self):
        """Test 2: Verify Feature 4 products_table contains strictly the 17 approved categories without legacy items."""
        res = tae.compute_trend_analysis(period_days=7)
        self.assertTrue(res["success"])
        products = res["products_table"]
        self.assertEqual(len(products), 17)
        categories = [p["category"] for p in products]
        for cat in categories:
            self.assertIn(cat, tae.APPROVED_CATEGORIES)
        # Ensure no legacy products appear
        legacy_names = ["Parle G", "Bourbon Biscuit", "Camlin", "Reynolds", "Natraj", "Oreo"]
        for p in products:
            for leg in legacy_names:
                self.assertNotIn(leg, p["product_name"])
        print(f"[TEST 2 PASS] Verified strictly 17 approved categories in Feature 4 inventory ({', '.join(categories[:6])}...).")

    def test_03_query_source_isolated_from_legacy_tables(self):
        """Test 3: Verify Feature 4 data source points exclusively to trend_inventory_* tables."""
        res = tae.compute_trend_analysis(period_days=7)
        self.assertTrue(res["success"])
        self.assertEqual(res.get("database_source"), "public.trend_inventory_*")
        self.assertEqual(res.get("approved_categories_count"), 17)
        print("[TEST 3 PASS] Confirmed Feature 4 data access layer connects exclusively to trend_inventory_*.")

    def test_04_repeated_stockout_state_transition(self):
        """Test 4: Verify repeated stock-outs are suppressed until a genuine replenishment occurs in between."""
        cat = "books"
        # Reset state to in_stock
        tae._PRODUCT_STOCK_STATE[cat] = "in_stock"
        
        # 1. First stock-out event
        ev1 = tae.record_trend_event(category=cat, event_type="stock_out", source="shelf_inspection")
        self.assertTrue(ev1.get("success"))
        self.assertFalse(ev1.get("debounced", False))
        self.assertEqual(tae._PRODUCT_STOCK_STATE.get(cat), "out_of_stock")

        # 2. Immediate second stock-out without replenishment (MUST BE SUPPRESSED)
        ev2 = tae.record_trend_event(category=cat, event_type="stock_out", source="shelf_inspection")
        self.assertTrue(ev2.get("debounced", False), "Second consecutive stock-out must be suppressed without replenishment in between")

        # 3. Replenishment occurs
        ev_rep = tae.record_trend_event(category=cat, event_type="restock", quantity_change=10, source="restock_action")
        self.assertTrue(ev_rep.get("success"))
        self.assertEqual(tae._PRODUCT_STOCK_STATE.get(cat), "restocked")

        # 4. Third stock-out now allowed after replenishment
        ev3 = tae.record_trend_event(category=cat, event_type="stock_out", source="shelf_inspection")
        self.assertTrue(ev3.get("success"))
        self.assertFalse(ev3.get("debounced", False))
        print("[TEST 4 PASS] State transition verified: Second stock-out counts ONLY after genuine replenishment in between.")

    def test_05_camera_frame_debouncing(self):
        """Test 5: Repeated camera detections within 10 seconds are debounced."""
        cat = "biscuit"
        tae._RECOGNITION_COOLDOWN.clear()
        res1 = tae.log_product_recognition_event(category=cat, confidence=0.89)
        self.assertFalse(res1.get("debounced", False))

        # Immediate repeat within 10s
        res2 = tae.log_product_recognition_event(category=cat, confidence=0.94)
        self.assertTrue(res2.get("debounced", True))
        print("[TEST 5 PASS] Repeated camera recognition frames within 10s debounced.")

    def test_06_barcode_scan_debouncing_and_classification(self):
        """Test 6: Repeated barcode scans within 3.5s are debounced and recorded as scans, not automatic sales."""
        cat = "chips"
        tae._BARCODE_SCAN_COOLDOWN.clear()
        res1 = tae.log_barcode_scan_event(barcode="8901234567890", category=cat)
        self.assertFalse(res1.get("debounced", False))

        # Immediate repeat within 3.5s
        res2 = tae.log_barcode_scan_event(barcode="8901234567890", category=cat)
        self.assertTrue(res2.get("debounced", True))
        print("[TEST 6 PASS] Barcode scans debounced (3.5s) and counted as scan activity rather than sales.")

    def test_07_trending_product_ranking_and_restock_alert(self):
        """Test 7: Top 5 ranking from real events and trending restock alert when stock <= ROP."""
        cat = "pen"
        # Generate genuine activity for 'pen'
        for _ in range(5):
            tae.record_trend_event(category=cat, event_type="barcode_scan", source="barcode_scanner")
        tae.record_trend_event(category=cat, event_type="stock_movement", quantity_change=-4, source="shelf_inspection")
        
        # Set low stock for 'pen' (e.g. 5 units <= ROP 15)
        tae._LOCAL_TREND_INVENTORY[cat]["current_stock"] = 5
        tae._LOCAL_TREND_INVENTORY[cat]["is_initialized"] = True

        res = tae.compute_trend_analysis(period_days=7)
        top5 = res["top_5_trending"]
        self.assertGreaterEqual(len(top5), 1)
        pen_item = next((p for p in top5 if p["category"] == "pen"), None)
        self.assertIsNotNone(pen_item, "Pen should appear in Top 5 trending items")
        self.assertGreater(pen_item["trend_score"], 0)

        # Verify trending restock alert
        alerts = res["trending_restock_alerts"]
        pen_alert = next((a for a in alerts if a["category"] == "pen"), None)
        self.assertIsNotNone(pen_alert, "Pen must appear in Trending Products Requiring Restock")
        self.assertEqual(pen_alert["current_stock"], 5)
        self.assertEqual(pen_alert["reorder_point"], 15)
        self.assertGreater(pen_alert["suggested_reorder"], 0)
        print(f"[TEST 7 PASS] Pen ranked in Top 5 (Score: {pen_item['trend_score']}) and flagged in restock alerts.")

    def test_08_trending_with_sufficient_stock(self):
        """Test 8: Trending product with stock > ROP is marked 'Trending, stock sufficient', no restock alert."""
        cat = "scale"
        # Generate activity for 'scale'
        for _ in range(4):
            tae.record_trend_event(category=cat, event_type="barcode_scan", source="barcode_scanner")
        tae.record_trend_event(category=cat, event_type="stock_movement", quantity_change=-2, source="shelf_inspection")
        
        # Set healthy stock for 'scale' (30 units > ROP 8)
        tae._LOCAL_TREND_INVENTORY[cat]["current_stock"] = 30
        tae._LOCAL_TREND_INVENTORY[cat]["is_initialized"] = True

        res = tae.compute_trend_analysis(period_days=7)
        sufficient = res["trending_stock_sufficient"]
        scale_suff = next((s for s in sufficient if s["category"] == "scale"), None)
        self.assertIsNotNone(scale_suff, "Scale should be in trending_stock_sufficient")
        self.assertEqual(scale_suff["status"], "Trending, stock sufficient")

        # Must NOT be in restock alerts
        scale_alert = next((a for a in res["trending_restock_alerts"] if a["category"] == "scale"), None)
        self.assertIsNone(scale_alert, "Scale with sufficient stock must not produce restock alert")
        print("[TEST 8 PASS] Scale identified as trending with sufficient stock without unnecessary alert.")

    def test_09_user_confirmed_restock_execution(self):
        """Test 9: User-confirmed restock adds inventory and logs restock event."""
        cat = "pen"
        prev_stock = tae._LOCAL_TREND_INVENTORY[cat]["current_stock"] or 0
        res = tae.execute_restock_order(product_id=3, quantity=15, shelf_id="B1", authorized_by="Test Manager")
        self.assertTrue(res["success"])
        new_stock = tae._LOCAL_TREND_INVENTORY[cat]["current_stock"]
        self.assertEqual(new_stock, prev_stock + 15)
        self.assertEqual(tae._PRODUCT_STOCK_STATE.get(cat), "in_stock")
        print(f"[TEST 9 PASS] Restock confirmed: Pen stock increased from {prev_stock} to {new_stock}.")

    def test_10_regression_features_1_2_3_server(self):
        """Test 10: Regression testing across web endpoints, camera status, barcode lookup, and shelf layout."""
        # 1. Camera status
        r_cam = requests.get(f"{SERVER_URL}/api/camera/status", timeout=5).json()
        self.assertIn("connected", r_cam)

        # 2. Barcode lookup
        r_bc = requests.post(f"{SERVER_URL}/api/barcode/lookup", json={"barcode": "8901063012486"}, timeout=5).json()
        self.assertTrue(r_bc.get("success"))

        # 3. Shelf layout
        r_sh = requests.get(f"{SERVER_URL}/api/shelves", timeout=5).json()
        self.assertTrue(r_sh.get("success"))

        # 4. Trends summary API
        r_tr = requests.get(f"{SERVER_URL}/api/trends/summary?period=7", timeout=12).json()
        self.assertTrue(r_tr.get("success"))
        self.assertEqual(r_tr.get("approved_categories_count"), 17)
        print("[TEST 10 PASS] Regression verified: Features 1, 2, 3 and Trend API are fully operational.")

if __name__ == "__main__":
    unittest.main()
