set -e

echo " Creating virtual environment…"
python3 -m venv .venv
source .venv/bin/activate

echo "Installing Python dependencies…"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo " Installing Playwright browsers…"
playwright install chromium

echo ""
echo "Setup complete."
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  python -m game_qa.main --episodes 5"
echo ""
echo "Options:"
echo "  --episodes N      number of game episodes (default: 10)"
echo "  --headless        run without a visible browser window"
echo "  --no-dashboard    skip the web dashboard"
echo "  --warmup-only     just train the anomaly detector, don't run QA"
