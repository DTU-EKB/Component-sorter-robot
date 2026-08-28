"""
Usage:
    python3 vision-scanner.py --image path/to/chip.jpg
    python3 vision-scanner.py --image path/to/chip.jpg --qty 5
    python3 vision-scanner.py --use-cache
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import ollama
from PIL import Image
from dotenv import load_dotenv

try:
    import cv2
except ImportError:
    cv2 = None

from inventree.api import InvenTreeAPI
from inventree.part import Part
from inventree.stock import StockItem


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

INVENTREE_URL = os.environ.get("INVENTREE_URL", "http://192.168.1.26:8000")
INVENTREE_TOKEN = os.environ.get("INVENTREE_TOKEN")
INVENTREE_IC_CATEGORY = os.environ.get("INVENTREE_IC_CATEGORY")

OLLAMA_MODEL = "qwen2.5vl"
ORIENTATIONS = (0, 180)     # degrees; each one is read READS_PER_ORIENTATION times
READS_PER_ORIENTATION = 3
CONFIDENCE_WEIGHTS = {"high": 3, "medium": 2, "low": 1}

SCAN_LOG_FILE = Path("./Scans/scan_log.json")
SCANS_DIR = Path("./Scans")
SCANS_DIR.mkdir(exist_ok=True)
TEMP_CAPTURE_FILE = Path("./_capture_tmp.jpg")  # overwritten each --device capture, then archived into Scans/

PROMPT = (
    "You are looking at a close-up photo of an integrated circuit (IC) chip. "
    "Read every line of text printed/etched on the chip package. "
    "Identify which line is the manufacturer part number (top line, usually "
    "alphanumeric, e.g. 'ATMEGA328P-PU' or 'LM358N') and ignore date codes, "
    "lot codes, and country-of-origin markings (those are usually short "
    "numeric codes like '2049' or 'YWWL').\n\n"
    "Respond with ONLY valid JSON, no other text, in this exact format:\n"
    '{"part_number": "<best guess part number>", '
    '"manufacturer_guess": "<manufacturer name or empty string>", '
    '"confidence": "high|medium|low", '
    '"raw_text": "<everything you read on the chip, all lines>"}'
)


# --------------------------------------------------------------------------
# Optional: capture a frame from a camera device instead of a file
# --------------------------------------------------------------------------

def capture_image(device: str, save_path: Path = TEMP_CAPTURE_FILE) -> str:
    """
    Grabs a single frame from a camera device (e.g. '/dev/video0' or a plain
    index like '0') and saves it to save_path. Returns the path as a string
    so it can be used just like any --image path.
    """
    if cv2 is None:
        sys.exit("ERROR: opencv-python is not installed. Run: pip install opencv-python")

    # cv2 wants an int for numeric indices ("0") but a string for device paths
    device_arg = int(device) if device.isdigit() else device

    cap = cv2.VideoCapture(device_arg)
    if not cap.isOpened():
        sys.exit(f"ERROR: could not open camera device '{device}'.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        sys.exit(f"ERROR: failed to read a frame from '{device}'.")

    cv2.imwrite(str(save_path), frame)
    print(f"📸 Captured image from '{device}' -> '{save_path}'")
    return str(save_path)


# --------------------------------------------------------------------------
# Step 1: read the chip with the local vision model
# --------------------------------------------------------------------------

def image_to_b64(img: Image.Image) -> str:
    """Encode an image as-is (no resizing or quality changes) for the model."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask_model(img: Image.Image) -> dict:
    """Send one image to the local vision model and parse its JSON reply."""
    client = ollama.Client(host="http://ollama:11434")
    
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT,
            "images": [image_to_b64(img)],
        }],
        options={"num_ctx": 8192},
    )
    content = resp["message"]["content"].strip()
    content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()
    return json.loads(content)


def normalize_part_number(pn: str) -> str:
    return re.sub(r"[\s\-_.]", "", pn or "").upper()


def scan_chip(image_path: str) -> dict:
    """
    Reads the chip photo multiple times (see ORIENTATIONS / READS_PER_ORIENTATION)
    and returns the most likely result, chosen by a confidence-weighted vote.
    """
    base_img = Image.open(image_path)
    attempts = []

    for angle in ORIENTATIONS:
        rotated = base_img if angle == 0 else base_img.rotate(-angle, expand=True)

        for i in range(READS_PER_ORIENTATION):
            try:
                result = ask_model(rotated)
            except (json.JSONDecodeError, ollama.ResponseError) as e:
                print(f"⚠️  Read failed ({angle}°, attempt {i + 1}): {e}")
                continue

            attempts.append(result)
            print(f"  [{angle:>3}° #{i + 1}] part_number={result.get('part_number')!r} "
                  f"confidence={result.get('confidence')}")

    if not attempts:
        raise RuntimeError("All read attempts failed — no usable results.")

    votes = defaultdict(float)
    best_attempt_for_key = {}

    for attempt in attempts:
        key = normalize_part_number(attempt.get("part_number", ""))
        if not key:
            continue
        weight = CONFIDENCE_WEIGHTS.get(attempt.get("confidence", "low"), 1)
        votes[key] += weight

        current = best_attempt_for_key.get(key)
        if current is None or weight > CONFIDENCE_WEIGHTS.get(current.get("confidence", "low"), 1):
            best_attempt_for_key[key] = attempt

    if not votes:
        raise RuntimeError("No attempt returned a usable part number.")

    print("\n--- Vote tally (normalized part number : weighted score) ---")
    for key, score in sorted(votes.items(), key=lambda kv: -kv[1]):
        print(f"  {key}: {score}")

    winning_key = max(votes, key=votes.get)
    return best_attempt_for_key[winning_key]


# --------------------------------------------------------------------------
# Scans directory: keep a copy of every chip photo, named by part number
# --------------------------------------------------------------------------

def archive_scan_image(image_path: str, part_number: str) -> str:
    """
    Copies image_path into Scans/<part_number>.jpg (adding a numeric suffix
    if that name is already taken, e.g. from a previous scan of the same
    part). Returns the new path. The original file at image_path is left
    untouched.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", part_number) or "unknown"
    ext = Path(image_path).suffix or ".jpg"

    target = SCANS_DIR / f"{safe_name}{ext}"
    counter = 2
    while target.exists():
        target = SCANS_DIR / f"{safe_name}_{counter}{ext}"
        counter += 1

    shutil.copy2(image_path, target)
    print(f"🗂️  Archived chip image as '{target}'")
    return str(target)


# --------------------------------------------------------------------------
# Scan log (append-only)
# --------------------------------------------------------------------------

def append_scan_result(result: dict):
    log = json.loads(SCAN_LOG_FILE.read_text()) if SCAN_LOG_FILE.exists() else []
    log.append(result)
    SCAN_LOG_FILE.write_text(json.dumps(log, indent=2))
    print(f"💾 Appended scan result to '{SCAN_LOG_FILE}' ({len(log)} total entries)")


def load_newest_scan_result() -> dict:
    if not SCAN_LOG_FILE.exists():
        sys.exit(f"No scan log found at '{SCAN_LOG_FILE}'. Run once without --use-cache first.")
    log = json.loads(SCAN_LOG_FILE.read_text())
    if not log:
        sys.exit(f"'{SCAN_LOG_FILE}' exists but is empty.")
    return log[-1]


# --------------------------------------------------------------------------
# Step 2/3: InvenTree
# --------------------------------------------------------------------------

def connect_inventree() -> InvenTreeAPI:
    if not INVENTREE_TOKEN:
        sys.exit("ERROR: INVENTREE_TOKEN is not set (env var or .env file).")
    return InvenTreeAPI(INVENTREE_URL, token=INVENTREE_TOKEN)


def find_existing_part(api: InvenTreeAPI, part_number: str):
    """Search InvenTree by name and IPN for a matching part."""
    matches = Part.list(api, search=part_number)
    for p in matches:
        if p.name.strip().upper() == part_number.strip().upper():
            return p
        if getattr(p, "IPN", None) and p.IPN.strip().upper() == part_number.strip().upper():
            return p
    return matches[0] if matches else None


def increment_stock(api: InvenTreeAPI, part: Part, qty: int):
    stock_items = StockItem.list(api, part=part.pk)

    if stock_items:
        target = stock_items[0]
        api.post("stock/add/", data={"items": [{"pk": target.pk, "quantity": qty}]})
        print(f"✅ Added {qty} to existing stock item #{target.pk} for '{part.name}'")
    else:
        StockItem.create(api, {"part": part.pk, "quantity": qty})
        print(f"✅ Created new stock entry with qty {qty} for '{part.name}' (no prior stock found)")


def set_parameter(api: InvenTreeAPI, part: Part, name: str, value: str):
    """
    Find-or-create a parameter template, then attach the value to the part.

    Uses raw REST calls instead of the PartParameterTemplate / PartParameter
    client classes, because on newer InvenTree servers (apiVersion 511+):
      - those classes refuse to run (they cap out at API version 428)
      - the endpoint paths changed to 'parameter/template/' and 'parameter/'
        (no 'part/' prefix), and parameter instances are now addressed via
        'model_type' + 'model_id' instead of a 'part' field.
    """
    existing = api.get("parameter/template/", params={"search": name})
    templates = existing["results"] if isinstance(existing, dict) else (existing or [])
    template = next((t for t in templates if t["name"].strip().lower() == name.strip().lower()), None)

    if template is None:
        template = api.post("parameter/template/", data={"name": name})

    api.post("parameter/", data={
        "template": template["pk"],
        "model_type": "part.part",
        "model_id": part.pk,
        "data": value,
    })


def create_new_part(api: InvenTreeAPI, part_number: str, scan_result: dict, qty: int, image_path: str) -> Part:
    if not INVENTREE_IC_CATEGORY:
        sys.exit("ERROR: INVENTREE_IC_CATEGORY is not set (env var or .env file).")

    part = Part.create(api, {
        "name": part_number,
        "IPN": part_number,
        "description": "",
        "category": int(INVENTREE_IC_CATEGORY),
        "active": True,
        "component": True,
        "purchaseable": True,
    })

    manufacturer = scan_result.get("manufacturer_guess", "").strip() or "???"
    set_parameter(api, part, "Manufacturer", manufacturer)

    raw_text = re.sub(r"\s+", " ", scan_result.get("raw_text", "")).strip()
    if raw_text:
        set_parameter(api, part, "Raw Text", raw_text)

    if image_path:
        try:
            part.uploadImage(image_path)
            print(f"🖼️  Uploaded '{image_path}' as part image")
        except Exception as e:
            print(f"⚠️  Could not upload part image: {e}")

    StockItem.create(api, {"part": part.pk, "quantity": qty})

    print(f"✅ Created new part '{part_number}' (pk={part.pk}) with qty {qty}")
    return part


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def scan_and_store(image_path: str, qty: int = 1, use_cache: bool = False):
    if use_cache:
        print(f"📂 Using newest entry from '{SCAN_LOG_FILE}' (skipping AI scan)...")
        scan_result = load_newest_scan_result()
    else:
        print(f"🔍 Reading chip from '{image_path}'...")
        scan_result = scan_chip(image_path)
        append_scan_result(scan_result)

    print("\n--- Best result ---")
    print(json.dumps(scan_result, indent=2))

    part_number = scan_result.get("part_number", "").strip()
    if not part_number:
        sys.exit("No usable part number identified. Aborting.")

    if image_path:
        image_path = archive_scan_image(image_path, part_number)

    api = connect_inventree()

    print(f"\n🔎 Checking InvenTree for existing part '{part_number}'...")
    existing_part = find_existing_part(api, part_number)

    if existing_part:
        print(f"Found existing part: '{existing_part.name}' (pk={existing_part.pk})")
        increment_stock(api, existing_part, qty)
    else:
        create_new_part(api, part_number, scan_result, qty, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan an IC chip and log it into InvenTree.")
    parser.add_argument("--image", help="Path to an existing chip photo")
    parser.add_argument("--device", help="Camera device to capture from instead of --image, "
                                          "e.g. /dev/video0 or 0")
    parser.add_argument("--qty", type=int, default=1, help="Quantity to add/create (default: 1)")
    parser.add_argument("--use-cache", action="store_true",
                         help="Skip the AI scan and reuse the newest entry from scan_log.json")
    args = parser.parse_args()

    if args.image and args.device:
        parser.error("Use either --image or --device, not both")

    if not args.image and not args.device and not args.use_cache:
        parser.error("--image or --device is required unless --use-cache is set")

    image_path = capture_image(args.device) if args.device else args.image

    scan_and_store(image_path, qty=args.qty, use_cache=args.use_cache)
