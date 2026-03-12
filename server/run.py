#!/usr/bin/env python3

import os
from app import create_app


def main():
    config_name = os.environ.get("FLASK_ENV", "development")
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")

    app = create_app(config_name)

    if config_name == "development":
        app.run(host=host, port=port, debug=True, threaded=True)
    else:
        app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
