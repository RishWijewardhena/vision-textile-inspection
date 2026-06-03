#!/usr/bin/env python3
"""
A&C Textile Inspection System — Service Control Panel
Controls Thread.service via systemctl
"""

import tkinter as tk
import subprocess
import threading
import time

# ── Colour palette ────────────────────────────────────────────────────────
BG_DARK      = "#0D0F14"
BG_CARD      = "#161A23"
BG_STRIP     = "#1C2130"
ACCENT_BLUE  = "#3A7BF7"
GREEN_RUN    = "#00C96E"
GREEN_HOVER  = "#00E87E"
RED_STOP     = "#E03A3A"
RED_HOVER    = "#FF4F4F"
TEXT_PRIMARY = "#F0F4FF"
TEXT_MUTED   = "#6B7A99"
TEXT_STATUS  = "#A8B4CC"
BORDER_DIM   = "#252D40"


def run_systemctl(action: str):
    """Use sudo -n (requires passwordless sudoers rule for Thread.service)."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "systemctl", action, "Thread.service"],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            return True, ""

        err = result.stderr.strip()
        if "password" in err.lower() or "sudoers" in err.lower():
            return False, "No sudo rule found. Run: sudo visudo -f /etc/sudoers.d/thread-control"
        return False, err

    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_service_status() -> str:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "Thread.service"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class PulsingDot(tk.Canvas):
    COLORS = {
        "active":   ("#00C96E", "#00FF8A"),
        "inactive": ("#6B7A99", "#8A9BB8"),
        "failed":   ("#E03A3A", "#FF5555"),
        "unknown":  ("#F0A500", "#FFC840"),
    }

    def __init__(self, parent):
        super().__init__(parent, width=20, height=20, bg=BG_STRIP, highlightthickness=0)
        self._state = "unknown"
        self._radius = 7.0
        self._growing = False
        self._animate()

    def set_state(self, state: str):
        self._state = state if state in self.COLORS else "unknown"

    def _animate(self):
        glow, core = self.COLORS.get(self._state, self.COLORS["unknown"])
        r = self._radius
        cx, cy = 10, 10
        self.delete("all")
        self.create_oval(cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2,
                         fill="", outline=glow, width=1)
        self.create_oval(cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1,
                         fill=core, outline="")

        if self._growing:
            self._radius = min(8.0, self._radius + 0.25)
            if self._radius >= 8.0:
                self._growing = False
        else:
            self._radius = max(5.5, self._radius - 0.25)
            if self._radius <= 5.5:
                self._growing = True

        self.after(60, self._animate)


class StyledButton(tk.Button):
    def __init__(self, parent, text, color, hover_color, command=None):
        super().__init__(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=TEXT_PRIMARY,
            activebackground=hover_color,
            activeforeground=TEXT_PRIMARY,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Courier New", 12, "bold"),
            cursor="hand2",
            padx=18,
            pady=14
        )
        self._base_color = color
        self._hover_color = hover_color
        self._enabled = True

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)

    def _on_enter(self, _):
        if self._enabled:
            self.configure(bg=self._hover_color)

    def _on_leave(self, _):
        if self._enabled:
            self.configure(bg=self._base_color)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if enabled:
            self.configure(
                state="normal",
                bg=self._base_color,
                fg=TEXT_PRIMARY,
                cursor="hand2"
            )
        else:
            self.configure(
                state="disabled",
                bg="#2A3040",
                fg=TEXT_MUTED,
                cursor="watch"
            )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("A&C Textile Inspection System")
        self.configure(bg=BG_DARK)

        self.resizable(True, True)
        self.minsize(620, 560)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        W = max(620, int(sw * 0.34))
        H = max(560, int(sh * 0.52))
        x = (sw - W) // 2
        y = (sh - H) // 2
        self.geometry(f"{W}x{H}+{x}+{y}")

        self._build_ui()
        self._refresh_status()
        self._start_status_poll()

        self.bind("<Configure>", self._on_resize)

    def _build_ui(self):
        # Top accent bar
        tk.Frame(self, bg=ACCENT_BLUE, height=4).pack(fill="x", side="top")

        wrapper = tk.Frame(self, bg=BG_DARK)
        wrapper.pack(fill="both", expand=True, padx=16, pady=(10, 16))

        card = tk.Frame(
            wrapper,
            bg=BG_CARD,
            highlightthickness=1,
            highlightbackground=BORDER_DIM
        )
        card.pack(fill="both", expand=True)

        card.columnconfigure(0, weight=1)
        card.rowconfigure(2, weight=0)
        card.rowconfigure(3, weight=0)
        card.rowconfigure(4, weight=1)

        # Header
        header = tk.Frame(card, bg=BG_CARD)
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(20, 0))
        header.columnconfigure(1, weight=1)

        icon_frame = tk.Frame(header, bg=ACCENT_BLUE, width=46, height=46)
        icon_frame.grid(row=0, column=0, sticky="w")
        icon_frame.pack_propagate(False)
        tk.Label(
            icon_frame,
            text="⟳",
            fg=TEXT_PRIMARY,
            bg=ACCENT_BLUE,
            font=("Courier New", 22, "bold")
        ).place(relx=0.5, rely=0.5, anchor="center")

        title_block = tk.Frame(header, bg=BG_CARD)
        title_block.grid(row=0, column=1, sticky="w", padx=(14, 0))

        tk.Label(
            title_block,
            text="A&C Textile Inspection",
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
            font=("Georgia", 17, "bold")
        ).pack(anchor="w")

        tk.Label(
            title_block,
            text="Service Control Panel  ·  Thread.service",
            fg=TEXT_MUTED,
            bg=BG_CARD,
            font=("Courier New", 9)
        ).pack(anchor="w")

        # Divider
        tk.Frame(card, bg=BORDER_DIM, height=1).grid(
            row=1, column=0, sticky="ew", padx=24, pady=(16, 0)
        )

        # Status strip
        strip = tk.Frame(
            card,
            bg=BG_STRIP,
            highlightthickness=1,
            highlightbackground=BORDER_DIM
        )
        strip.grid(row=2, column=0, sticky="ew", padx=24, pady=(14, 0))
        strip.columnconfigure(0, weight=1)

        status_inner = tk.Frame(strip, bg=BG_STRIP)
        status_inner.pack(fill="x", padx=18, pady=16)

        tk.Label(
            status_inner,
            text="SERVICE STATUS",
            fg=TEXT_MUTED,
            bg=BG_STRIP,
            font=("Courier New", 8, "bold")
        ).pack(anchor="w")

        dot_row = tk.Frame(status_inner, bg=BG_STRIP)
        dot_row.pack(anchor="w", pady=(8, 0))

        self._dot = PulsingDot(dot_row)
        self._dot.pack(side="left", padx=(0, 12))

        self._status_label = tk.Label(
            dot_row,
            text="Checking…",
            fg=TEXT_STATUS,
            bg=BG_STRIP,
            font=("Georgia", 16, "italic")
        )
        self._status_label.pack(side="left")

        self._detail_label = tk.Label(
            status_inner,
            text="",
            fg=TEXT_MUTED,
            bg=BG_STRIP,
            font=("Courier New", 8)
        )
        self._detail_label.pack(anchor="w", pady=(6, 0))

        # Buttons
        btn_frame = tk.Frame(card, bg=BG_CARD)
        btn_frame.grid(row=3, column=0, sticky="ew", padx=24, pady=(20, 0))
        btn_frame.columnconfigure(0, weight=1, uniform="btn")
        btn_frame.columnconfigure(1, weight=1, uniform="btn")

        self._run_btn = StyledButton(
            btn_frame,
            text="▶   START SERVICE",
            color=GREEN_RUN,
            hover_color=GREEN_HOVER,
            command=self._start_service
        )
        self._run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 10), ipady=6)

        self._stop_btn = StyledButton(
            btn_frame,
            text="■   STOP SERVICE",
            color=RED_STOP,
            hover_color=RED_HOVER,
            command=self._stop_service
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(10, 0), ipady=6)

        # Log area
        log_outer = tk.Frame(
            card,
            bg=BG_DARK,
            highlightthickness=1,
            highlightbackground=BORDER_DIM
        )
        log_outer.grid(row=4, column=0, sticky="nsew", padx=24, pady=(18, 0))
        log_outer.columnconfigure(0, weight=1)
        log_outer.rowconfigure(0, weight=1)

        self._log_label = tk.Label(
            log_outer,
            text="Ready. Use the buttons above to control the service.",
            fg=TEXT_MUTED,
            bg=BG_DARK,
            font=("Courier New", 9),
            wraplength=420,
            justify="left",
            anchor="w"
        )
        self._log_label.grid(row=0, column=0, sticky="ew", padx=16, pady=12)

        # Footer
        tk.Label(
            card,
            text="A&C Textile · Vision Inspection Platform",
            fg=BORDER_DIM,
            bg=BG_CARD,
            font=("Courier New", 8)
        ).grid(row=5, column=0, pady=(12, 14))

        self._card = card
        self._log_outer = log_outer

    def _on_resize(self, _event):
        # Keep the log text wrapped nicely as the window changes size.
        width = max(320, self.winfo_width() - 120)
        self._log_label.configure(wraplength=width)

    # ── Service actions ───────────────────────────────────────────────────

    def _set_buttons_busy(self, busy: bool):
        self._run_btn.set_enabled(not busy)
        self._stop_btn.set_enabled(not busy)

    def _start_service(self):
        self._set_buttons_busy(True)
        self._log("Sending START command to Thread.service…")
        threading.Thread(target=self._do_action, args=("start",), daemon=True).start()

    def _stop_service(self):
        self._set_buttons_busy(True)
        self._log("Sending STOP command to Thread.service…")
        threading.Thread(target=self._do_action, args=("stop",), daemon=True).start()

    def _do_action(self, action: str):
        ok, msg = run_systemctl(action)
        self.after(0, self._on_action_done, action, ok, msg)

    def _on_action_done(self, action: str, ok: bool, msg: str):
        if ok:
            if action == "start":
                self._log("✓ Service started successfully.")
            elif action == "stop":
                self._log("✓ Service stopped successfully.")
            else:
                self._log("✓ Service action completed successfully.")
        else:
            self._log(f"✗ Failed to {action}: {msg or 'unknown error'}")

        self._set_buttons_busy(False)
        self._refresh_status()

    # ── Status polling ────────────────────────────────────────────────────

    def _refresh_status(self):
        threading.Thread(target=self._poll_once, daemon=True).start()

    def _poll_once(self):
        status = get_service_status()
        self.after(0, self._update_status_ui, status)

    def _update_status_ui(self, status: str):
        self._dot.set_state(status)

        labels = {
            "active":   ("Running", TEXT_PRIMARY),
            "inactive": ("Stopped", TEXT_MUTED),
            "failed":   ("Failed", RED_STOP),
            "unknown":  ("Unknown", "#F0A500"),
        }
        text, color = labels.get(status, ("Unknown", "#F0A500"))
        self._status_label.config(text=text, fg=color)
        self._detail_label.config(
            text=f"systemctl is-active → {status}  ·  {time.strftime('%H:%M:%S')}"
        )

    def _start_status_poll(self):
        def loop():
            self._refresh_status()
            self.after(5000, loop)
        self.after(5000, loop)

    def _log(self, message: str):
        self._log_label.config(text=message)


if __name__ == "__main__":
    app = App()
    app.mainloop()
