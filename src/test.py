import os
import glob


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

png_lst = glob.glob(os.path.join(IMAGES_PATH,"*.png"))
print(png_lst)