import tomllib
from pathlib import Path

from theme import CSS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
HEX = {"primaryColor": "#0B7F77", "backgroundColor": "#F6F6F3", "secondaryBackgroundColor": "#E6F3F0",
       "textColor": "#0F2A36"}


def test_the_theme_uses_the_agreed_light_palette():
    theme = CONFIG["theme"]
    assert theme["base"] == "light"
    for key, value in HEX.items():
        assert theme[key].upper() == value, key


def test_every_font_file_the_theme_names_exists_and_is_served():
    assert CONFIG["server"]["enableStaticServing"] is True
    faces = CONFIG["theme"]["fontFaces"]
    assert {f["family"] for f in faces} == {"Source Serif 4", "Inter"}
    for face in faces:
        path = ROOT / face["url"]                       # "app/static/<file>" relative to the repo root
        assert path.exists() and path.stat().st_size > 1000, face["url"]


def test_the_fonts_carry_their_licence_files():
    for name in ("OFL-SourceSerif4.txt", "OFL-Inter.txt"):
        assert (ROOT / "app" / "static" / name).exists()

def test_the_css_does_not_paint_every_block_white():
    assert '[data-testid="stVerticalBlockBorderWrapper"]' not in CSS
    assert '[class*="st-key-card"]' in CSS