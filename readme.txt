Das Projekt wurde als Django-Projekt aufgebaut.

Zur Installation müssen folgende Schritte durchgeführt werden:

Zu beginn muss eine virtuelle Umgebung erstellt werden. Daraufhin müssen mit dem Befehl "pip install -r requirements.txt" alle notwendigen Libraries installiert werden.

Als weitere Software muss zunächst eine PostgreSQL installiert werden. In dieser muss eine Tabelle erstellt werde, die die Spalten der bundesliga_gesamt_2020_2024.csv datei enthält.

Daraufhin muss im Hauptordner eine .env Datei erstellt werden und mit den gewählten Namen und Daten der Postrge Datenbank ausgefüllt werden nach diesem Schema:

DB_NAME=dein_datenbankname
DB_USER=dein_benutzername
DB_PASSWORD=dein_neues_passwort
DB_HOST=localhost
DB_PORT=5432

Zum Import der Daten in die Datenbank muss der Befehl "python manage.py import_matches" in der virtellen Umgebung ausgeführt werden.
Im Anschluss muss die Datei predict_all_matchups.py im Unterordner prediction außerhalb der virtuellen Umgebung ausgeführt werden.