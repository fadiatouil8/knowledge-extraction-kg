"""Convertit report/rapport.md en report/rapport.pdf (Markdown -> HTML -> PDF via Edge/Chrome headless).

Usage : python report/build_report.py
"""
import shutil
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CSS = """
@page { size: A4; margin: 15mm 16mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 9.6pt; line-height: 1.32; color: #222; }
h1 { font-size: 15pt; margin: 0 0 2px 0; color: #1f3b63; }
h2 { font-size: 11.5pt; color: #1f3b63; border-bottom: 1px solid #c9d3e0; margin: 9px 0 4px 0; }
h3 { font-size: 10pt; margin: 6px 0 2px 0; }
p, li { margin: 2px 0; text-align: justify; }
ul { margin: 2px 0 2px 16px; padding: 0; }
table { border-collapse: collapse; margin: 4px 0; font-size: 8.4pt; }
th, td { border: 1px solid #c9d3e0; padding: 1.5px 5px; }
th { background: #eef2f7; }
img { display: block; margin: 4px auto; }
.fig { text-align: center; font-size: 8pt; color: #555; margin-bottom: 4px; }
.meta { color: #555; font-size: 9pt; margin-bottom: 6px; }
code { font-size: 8.4pt; background: #f3f3f3; padding: 0 2px; }
"""


def find_browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("msedge")


def main():
    body = markdown.markdown((HERE / "rapport.md").read_text(encoding="utf-8"), extensions=["tables", "attr_list"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = HERE / "rapport.html"
    html_path.write_text(html, encoding="utf-8")
    browser = find_browser()
    if not browser:
        print("Aucun navigateur Chromium trouvé : ouvrir report/rapport.html et l'imprimer en PDF.")
        return
    pdf = HERE / "rapport.pdf"
    subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=True, timeout=120)
    print("PDF généré :", pdf)


if __name__ == "__main__":
    main()
