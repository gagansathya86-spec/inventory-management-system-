# Smart Inventory Management System (DL + CV + Supabase)

An enterprise-grade Smart Inventory Management System integrating Deep Learning (YOLOv8s 17-category recognition, PyTorch LSTM demand forecasting), Optical Character Recognition (OCR), Computer Vision damage inspection, and Cloud Database synchronization (Supabase).

---

## 🚀 Application Navigation Architecture (4 Main Tabs + Supabase Inspector)

The application features a modern, responsive web dashboard with four primary operational tabs and an authorized Supabase database viewer:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│  ⚡ SMART INVENTORY SYSTEM                                              🟢 Supabase Connected    │
├────────────────────┬────────────────────┬────────────────────┬──────────────────┬────────────────┤
│ 📷 Camera Settings │ 🔍 Recognition     │ 🏬 Shelf & Layout  │ 📈 Trend Analysis│ 🗄️ Supabase DB │
└────────────────────┴────────────────────┴────────────────────┴──────────────────┴────────────────┘
```

1. **📷 Camera Settings**:
   - Switch between **Laptop Built-in Webcam** (Device 0) and **Phone Camera** (Gradio/IP network stream URL or USB virtual webcam).
   - Real-time connection testing (`⚡ Test Connection`), active frame verification, and resource release (`cap.release()`) on previous streams.
   - Live camera preview with resolution and FPS telemetry.
   - Selected camera automatically propagates across Product Recognition and Shelf & Damage Detection.

2. **🔍 Product Recognition**:
   - **Mode A: Live CNN Product Recognition**:
     - Real-time video preview from selected camera.
     - Start Recognition, Stop Recognition, and Capture & Classify controls.
     - Powered by the 17-category trained CNN model (`yolov8s_17categories_enhanced.pt`).
     - Returns predicted category, confidence score, inference time, and captured snapshot preview.
     - Displays **Uncertain** when confidence is low (< 0.35) without inventing labels.
     - Duplicate suppression cache avoids reprocessing identical items in consecutive frames.
   - **Mode B: Barcode & Expiry Scanner**:
     - Scans barcode from selected camera feed with EAN-13/UPC checksum validation.
     - Supabase catalog lookup.
     - **Automatic Edible vs Non-Edible Classification**:
       - `EDIBLE` (Biscuits, Chips, Snacks): Enforces Manufacturing Date (`MFG`) and Expiry Date (`EXP`) entry. Expired items rejected from available stock.
       - `NON-EDIBLE` (Books, Stationery, Electronics, Cables): Skips expiry forms entirely.
     - Review quantity and destination shelf (`A1`–`C3`) before saving to Supabase.

3. **🏬 Shelf & Layout Inspection (Camera-Based)**:
   - Live camera preview using the selected Laptop Webcam or Phone Camera (with alternative photo upload).
   - Real CNN product package detection across shelves with bounding boxes, labels, and per-category counts.
   - **Misplaced-Product Detection**: Checks detected items against the expected shelf planogram layout and warns with the correct location (e.g., *"Chips detected in Shelf A3. Expected location: Shelf A2. Please return this product to Shelf A2."*).
   - **Empty Shelf & Out-of-Stock Detection**: Differentiates between *Shelf position empty* (warehouse inventory available for restocking) and *Product out of stock* (total inventory is 0).
   - **Observed vs. Recorded Layout Comparison Table**: Shows product, detected shelf, expected shelf, observed count, database count, discrepancy, and status.
   - **Review & Confirm Stock Adjustment**: Manual user confirmation before updating Supabase (never automatic).

4. **📈 Trend Analysis**:
   - Preserves `trend identification/trend_identification.ipynb` completely intact without altering LSTM model/eval/forecasting.
   - 6 Calculated Written Insights: Top Trending Product, Highest Predicted Demand, Low-Stock Alert, Urgent Reorder, Falling Demand, Insufficient Data.
   - Interactive Forecast Table: Product, Category, Current Stock, Predicted Demand, Growth %, Reorder Point (ROP), Safety Stock, Suggested Reorder Qty, and Priority.

5. **🗄️ Supabase Cloud Database Inspector (Utility Tab)**:
   - Direct live table inspector for: `products`, `stock`, `inventory`, `shelf_counts`, `barcode_scans`, `demand_predictions`.
   - Table selection dropdown, search filtering, and pagination controls (‹ Prev, Page X of Y, Next ›).
   - Secure proxy execution through backend—service-role key is never exposed to browser clients.

---

## 🏃 How to Run the Application

### 1. Launch Web Dashboard (All 5 Features Integrated)

#### One-Click Launcher:
Double-click `start_dashboard.bat` in the project root directory.

#### Or via Terminal:
```powershell
cd "c:\Users\gagan\Downloads\dl smart inventory\product recognition"
python run.py --mode web
```
- Open browser at: **`http://localhost:5000`**

### 2. Standalone Downloadable Jupyter Notebooks (`.ipynb`)
- **Feature 1**: [`product recognition/fast_custom_cnn_live_product_recognition.ipynb`](file:///c:/Users/gagan/Downloads/dl%20smart%20inventory/product%20recognition/fast_custom_cnn_live_product_recognition.ipynb)
- **Feature 2**: [`barcode scanner/barcode_scanner_expiry_inventory.ipynb`](file:///c:/Users/gagan/Downloads/dl%20smart%20inventory/barcode%20scanner/barcode_scanner_expiry_inventory.ipynb)
- **Feature 3**: [`shelf detection/shelf_detection.ipynb`](file:///c:/Users/gagan/Downloads/dl%20smart%20inventory/shelf%20detection/shelf_detection.ipynb)
- **Feature 4**: [`trend identification/trend_identification.ipynb`](file:///c:/Users/gagan/Downloads/dl%20smart%20inventory/trend%20identification/trend_identification.ipynb)

---

## 📱 Connecting a Phone Webcam
1. Install **DroidCam** or **Iriun Webcam** on your phone (Android/iOS) and Windows PC.
2. Connect via USB or same local Wi-Fi network.
3. Open the **Camera Settings** tab in the web application and click **"🔄 Refresh Camera List"**.
4. Select the phone webcam from the dropdown and click **"✅ Use Selected Camera for Barcode Scanner & App"**.

# inventory-management-system-
