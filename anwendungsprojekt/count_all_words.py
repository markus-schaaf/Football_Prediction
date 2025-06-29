import os
import subprocess

EXTENSIONS = ('.py', '.html', '.js', '.css', '.json', '.txt', '.md', '.xml')
SITE_PACKAGES = os.path.join("venv", "Lib", "site-packages")

def get_used_packages():
    """Liest alle installierten Pakete aus pip freeze und normalisiert die Namen."""
    result = subprocess.run(
        [os.path.join('venv', 'Scripts', 'pip.exe'), 'freeze'],
        capture_output=True, text=True
    )
    packages = set()
    for line in result.stdout.splitlines():
        if '==' in line:
            pkg = line.split('==')[0].strip().lower().replace('-', '_')
            packages.add(pkg)
    return packages

def count_words_in_used_packages(site_packages_dir, package_names):
    word_count = 0
    file_count = 0
    for entry in os.listdir(site_packages_dir):
        entry_path = os.path.join(site_packages_dir, entry)

        norm_name = entry.lower().split('-')[0].split('.')[0].replace('-', '_')
        if norm_name in package_names and os.path.isdir(entry_path):
            for root, _, files in os.walk(entry_path):
                for file in files:
                    if file.endswith(EXTENSIONS):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                                words = len(f.read().split())
                                word_count += words
                                file_count += 1
                        except Exception as e:
                            print(f"Fehler bei Datei {file}: {e}")
    return word_count, file_count

used_packages = get_used_packages()
words, files = count_words_in_used_packages(SITE_PACKAGES, used_packages)

print(f"Verwendete Bibliothekswörter: {words:,}")
print(f"Dateien gezählt: {files}")
