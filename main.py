from flask import Flask, render_template, redirect
import subprocess
import threading
import webbrowser

app = Flask(__name__)

# Hàm chạy Flask app con
def run_script(script_name, port):
    subprocess.Popen(["python", script_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    webbrowser.open_new(f"http://127.0.0.1:{port}")

@app.route("/")
def index():
    return render_template("main_menu.html")

@app.route("/extract")
def extract():
    threading.Thread(target=run_script, args=("tx.py", 5003)).start()
    return "OK"

@app.route("/classify")
def classify():
    threading.Thread(target=run_script, args=("run2.py", 5001)).start()
    return "OK"

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
