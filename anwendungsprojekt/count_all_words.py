import os

# Optional: alle Dateien zählen, die typische Code-Erweiterungen haben
EXTENSIONS = ('.py', '.html', '.js', '.css', '.json', '.txt', '.md', '.xml')

def count_all_words(directory):
    word_count = 0
    file_count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(EXTENSIONS):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        words = len(content.split())
                        word_count += words
                        file_count += 1
                except Exception as e:
                    print(f"Fehler bei Datei {file}: {e}")
    return word_count, file_count

# Ergebnis anzeigen
words, files = count_all_words('.')
print(f"Gesamtwörter (inkl. venv & libs): {words:,}")
print(f"Dateien gezählt: {files}")
