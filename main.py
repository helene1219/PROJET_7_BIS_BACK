""" 
main.py
"""

import uvicorn
from src.api import app


def main():

    # uvicorn.run(api, host='127.0.0.1', port=8000)
    # version KO du 18/02/24: uvicorn.run(app, host='127.0.0.1', port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
