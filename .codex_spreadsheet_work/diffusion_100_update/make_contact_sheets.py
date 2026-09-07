from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WORK_DIR = Path("/Users/yimingzang/Documents/Project/benchmark2/.codex_spreadsheet_work/diffusion_100_update")
ROOT = WORK_DIR / "previews_after"
THUMB_W = 320
THUMB_H = 210
LABEL_H = 34
COLS = 4


for folder in sorted(path for path in ROOT.iterdir() if path.is_dir()):
    images = sorted(folder.glob("*.png"))
    rows = (len(images) + COLS - 1) // COLS
    canvas = Image.new("RGB", (COLS * THUMB_W, rows * (THUMB_H + LABEL_H)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((THUMB_W - 12, THUMB_H - 12))
        cell_x = (index % COLS) * THUMB_W
        cell_y = (index // COLS) * (THUMB_H + LABEL_H)
        x = cell_x + (THUMB_W - image.width) // 2
        y = cell_y + (THUMB_H - image.height) // 2
        canvas.paste(image, (x, y))
        label = image_path.stem
        draw.text((cell_x + 6, cell_y + THUMB_H + 6), label, fill="#1F2937", font=font)
        draw.rectangle(
            (cell_x, cell_y, cell_x + THUMB_W - 1, cell_y + THUMB_H + LABEL_H - 1),
            outline="#CBD5E1",
            width=1,
        )
    output = WORK_DIR / f"{folder.name}__contact_sheet.png"
    canvas.save(output)
    print(output)
