#!/bin/bash
set -euo pipefail

# --------------------------
# auto_run.sh installer (self-locating)
# --------------------------

# Determine the directory where THIS script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_DIR/venv"
AUTO_RUNNER="$PROJECT_DIR/auto_runner.sh"
SERVICE_PATH="/etc/systemd/system/Thread.service"

# Determine target user
TARGET_USER="${SUDO_USER:-$USER}"
HOME_DIR="$(eval echo "~$TARGET_USER")"

echo "Installer running as: $(whoami)"
echo "Target user: $TARGET_USER"
echo "Project dir (from script location): $PROJECT_DIR"

# --------------------------
# 1) Install system packages
# --------------------------
echo
echo "==> Installing required system packages (python3-venv, pip, acpid ,git)..."
sudo apt update
sudo apt install -y python3-venv python3-pip acpid git


# --------------------------
# 2) Add user to dialout
# --------------------------
echo
echo "==> Adding user '$TARGET_USER' to group 'dialout' for serial/USB access..."
sudo usermod -a -G dialout "$TARGET_USER" || true
echo "Note: user must log out and log back in (or reboot) for group change to take effect."

# --------------------------
# 3) Ensure project dir exists
# --------------------------
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Project directory $PROJECT_DIR not found. Creating it."
  mkdir -p "$PROJECT_DIR"
fi

# --------------------------
# 3.1) Clone or update GitHub repo
# --------------------------
echo
echo "==> Cloning/updating GitHub repository..."

if [ -d "$PROJECT_DIR/.git" ]; then
    echo "Repository already exists, pulling latest changes from main..."
    git -C "$PROJECT_DIR" fetch origin main
    git -C "$PROJECT_DIR" reset --hard origin/main
else
    echo "Cloning repository from GitHub..."
    git clone --branch main https://github.com/RishWijewardhena/vision-textile-inspection.git "$PROJECT_DIR"
fi


# --------------------------
# 4) Create virtualenv & install requirements
# --------------------------
echo
echo "==> Creating virtual environment (if missing) and installing requirements..."
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  echo "Virtual environment created at $VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "Installing pip packages from requirements.txt..."
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
  echo "Dependencies installed into venv."
else
  echo "No requirements.txt found — skipping pip install."
fi

# --------------------------
# 5) Configure ACPI power button
# --------------------------
echo
echo "==> Configuring ACPI power button to shut down the system..."
sudo tee /etc/acpi/events/powerbtn > /dev/null <<'ACPI_RULE'
event=button/power
action=/usr/sbin/poweroff
ACPI_RULE

sudo systemctl restart acpid
sudo systemctl enable acpid
echo "ACPI configured and acpid restarted."

# --------------------------
# 6) Create auto_runner.sh
# --------------------------
echo
echo "==> Creating helper runner script: $AUTO_RUNNER"

sudo tee "$AUTO_RUNNER" > /dev/null <<EOF
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$PROJECT_DIR"
VENV_DIR="$VENV_DIR"

cd "\$PROJECT_DIR" || exit 1

if [ -x "\$VENV_DIR/bin/python" ]; then
    exec "\$VENV_DIR/bin/python" "\$PROJECT_DIR/main.py"
else
    exec /usr/bin/python3 "\$PROJECT_DIR/main.py"
fi
EOF

sudo chown "$TARGET_USER":"$TARGET_USER" "$AUTO_RUNNER"
sudo chmod +x "$AUTO_RUNNER"
echo "auto_runner.sh created and set executable."

# --------------------------
# 7) Create systemd service file
# --------------------------
echo
echo "==> Creating systemd service: $SERVICE_PATH"

sudo tee "$SERVICE_PATH" > /dev/null <<SERVICE_UNIT
[Unit]
Description=Run Thread main script at boot
After=network.target 
Requires=dev-video0.device
StartLimitIntervalSec=0

[Service]
User=$TARGET_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$AUTO_RUNNER
Restart=on-failure
RestartSec=1
# For GUI (NOT recommended for system services):
# Environment=DISPLAY=:0
# Environment=XAUTHORITY=$HOME_DIR/.Xauthority

[Install]
WantedBy=multi-user.target
SERVICE_UNIT

sudo chmod 644 "$SERVICE_PATH"
echo "Service file written."

# --------------------------
# 8) Reload systemd and start service
# --------------------------
echo
echo "==> Reloading systemd, enabling and starting Thread.service..."
sudo systemctl daemon-reload
sudo systemctl enable Thread.service
# sudo systemctl restart Thread.service ket user to modify the .env file 

echo
echo "==> Current service status:"
sudo systemctl status Thread.service --no-pager

echo
echo "Setup complete."
echo "Notes:"
echo " - Reboot or log out/in for dialout permissions to apply."
echo " - cv2.imshow() will NOT display from a systemd service."
echo "   Run manually or use desktop autostart if GUI is required."

echo "modify the .env file  before restart ........"

