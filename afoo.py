with open("tts/text/template.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()
for text in texts:
    text = text.strip()
    print(f"Processing: {text}")