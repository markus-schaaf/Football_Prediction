import os
import sys
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "anwendungsprojekt.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django..."
        ) from exc
    execute_from_command_line(sys.argv)

print("DB:", os.getenv("DB_NAME"))

