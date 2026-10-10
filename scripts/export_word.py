import markdown
import codecs
import re
import base64
import os
import argparse

parser = argparse.ArgumentParser(description="Converte um relatorio Markdown em .doc (HTML do Word) com imagens embutidas")
parser.add_argument("md_file", help="Relatorio em Markdown")
parser.add_argument("doc_file", help="Arquivo .doc de saida")
args = parser.parse_args()
md_file = args.md_file
doc_file = args.doc_file

with codecs.open(md_file, "r", encoding="utf-8") as f:
    text = f.read()

# Convert markdown to html
html = markdown.markdown(text, extensions=['tables'])

# Function to replace img src with base64 data
def image_to_base64(match):
    src = match.group(1)
    # Remove leading slash if it exists and clean the path
    if src.startswith('/C:'):
        img_path = src[1:]
    elif src.startswith('file:///C:'):
        img_path = src[8:]
    else:
        img_path = src
    
    img_path = img_path.replace('/', '\\')
    
    try:
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return f'src="data:image/png;base64,{encoded_string}"'
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return f'src="{src}"'

# Regex to find all img src attributes
html = re.sub(r'src="([^"]+)"', image_to_base64, html)

word_html = f"""<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<style>
body {{ font-family: 'Calibri', 'Arial', sans-serif; font-size: 11pt; }}
h1 {{ font-size: 16pt; color: #2E74B5; }}
h2 {{ font-size: 14pt; color: #2E74B5; border-bottom: 1px solid #ccc; }}
h3 {{ font-size: 12pt; color: #1F4D78; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; }}
th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{html}
</body>
</html>"""

with codecs.open(doc_file, "w", encoding="utf-8") as f:
    f.write(word_html)

print("Images embedded successfully!")
