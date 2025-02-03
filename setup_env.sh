#!bash
if [ ! -d "./ip" ]; then
	echo "Python virtual environment not found, creating now"
	python3 -m venv ip
else
	echo "Python virtual enviornment found!"
fi
source ip/bin/activate
pip install -r requirements.txt
