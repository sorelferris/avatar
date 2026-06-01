from collections import deque
import math

import pygame
import pyjoycon


WIDTH = 1100
HEIGHT = 680
FPS = 60
BG = (14, 18, 28)
PANEL = (24, 30, 44)
PANEL_ALT = (30, 37, 55)
TEXT = (236, 242, 255)
MUTED = (160, 174, 197)
ACCENT = (102, 212, 255)
GOOD = (90, 221, 128)
WARN = (255, 196, 102)
BAD = (255, 104, 104)


class JoyCon(pyjoycon.GyroTrackingJoyCon, pyjoycon.ButtonEventJoyCon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.button_state = set()


def coerce_vector(value):
    if hasattr(value, "x") and hasattr(value, "y"):
        components = [float(value.x), float(value.y)]
        if hasattr(value, "z"):
            components.append(float(value.z))
        return tuple(components)
    if isinstance(value, (list, tuple)) and value:
        nested_vectors = [coerce_vector(item) for item in value]
        nested_vectors = [item for item in nested_vectors if item is not None]
        if nested_vectors and len(nested_vectors) == len(value):
            width = max(len(item) for item in nested_vectors)
            totals = [0.0] * width
            counts = [0] * width
            for vector in nested_vectors:
                for index, component in enumerate(vector):
                    totals[index] += component
                    counts[index] += 1
            return tuple(
                totals[index] / counts[index] if counts[index] else 0.0
                for index in range(width)
            )
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
    return None


def format_value(value):
    vector = coerce_vector(value)
    if vector is not None:
        return "(" + ", ".join(f"{item:.2f}" for item in vector) + ")"
    return str(value)


def first_two_components(value):
    vector = coerce_vector(value)
    if vector is None:
        return None
    x = vector[0]
    y = vector[1] if len(vector) > 1 else 0.0
    return x, y, vector


def normalize_pair(x, y):
    scale = max(1.0, abs(x), abs(y))
    return x / scale, y / scale


def status_is_pressed(status):
    if isinstance(status, bool):
        return status
    if isinstance(status, (int, float)):
        return status != 0
    if isinstance(status, str):
        lowered = status.lower()
        if lowered in {"press", "pressed", "down", "on", "true", "1"}:
            return True
        if lowered in {"release", "released", "up", "off", "false", "0"}:
            return False
    return bool(status)


def draw_text(surface, font, text, position, color=TEXT, background=None):
    rendered = font.render(text, True, color, background)
    surface.blit(rendered, position)


def draw_panel(surface, rect, title, font):
    pygame.draw.rect(surface, PANEL, rect, border_radius=18)
    pygame.draw.rect(surface, PANEL_ALT, rect, width=2, border_radius=18)
    draw_text(surface, font, title, (rect.x + 18, rect.y + 14), ACCENT)
    pygame.draw.line(
        surface,
        (45, 58, 82),
        (rect.x + 18, rect.y + 46),
        (rect.right - 18, rect.y + 46),
        1,
    )


def draw_status_chip(surface, rect, label, active, font):
    fill = GOOD if active else PANEL_ALT
    border = GOOD if active else (70, 80, 100)
    pygame.draw.rect(surface, fill, rect, border_radius=12)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=12)
    text_color = (12, 18, 26) if active else TEXT
    draw_text(surface, font, label, (rect.x + 10, rect.y + 8), text_color)


def draw_scope(surface, rect, title, value, font, small_font, color=ACCENT):
    pygame.draw.rect(surface, (18, 24, 35), rect, border_radius=16)
    pygame.draw.rect(surface, (58, 68, 88), rect, width=2, border_radius=16)
    draw_text(surface, font, title, (rect.x + 14, rect.y + 10), TEXT)

    vector_data = first_two_components(value)
    inner = rect.inflate(-30, -66)
    inner.top += 30
    center = inner.center
    radius = min(inner.width, inner.height) // 2 - 8

    pygame.draw.rect(surface, (20, 26, 38), inner, border_radius=16)
    for ring in (0.28, 0.55, 0.82, 1.0):
        pygame.draw.circle(surface, (42, 52, 68), center, int(radius * ring), 1)
    pygame.draw.line(
        surface,
        (70, 80, 100),
        (inner.left + 8, center[1]),
        (inner.right - 8, center[1]),
        1,
    )
    pygame.draw.line(
        surface,
        (70, 80, 100),
        (center[0], inner.top + 8),
        (center[0], inner.bottom - 8),
        1,
    )

    if vector_data is not None:
        x, y, vector = vector_data
        nx, ny = normalize_pair(x, y)
        point = (
            int(center[0] + nx * radius * 0.72),
            int(center[1] - ny * radius * 0.72),
        )
        pygame.draw.circle(surface, (35, 49, 70), center, int(radius * 0.92), 2)
        pygame.draw.line(surface, color, center, point, 5)
        pygame.draw.circle(surface, color, point, 10)
        pygame.draw.circle(surface, (255, 255, 255), point, 3)
        magnitude = math.sqrt(x * x + y * y)
        angle = math.degrees(math.atan2(y, x))
        draw_text(
            surface,
            small_font,
            f"mag={magnitude:.2f}  ang={angle:.0f}°",
            (rect.x + 14, rect.bottom - 54),
            MUTED,
        )
        draw_text(
            surface,
            small_font,
            f"x={x:.2f}  y={y:.2f}",
            (rect.x + 14, rect.bottom - 30),
            MUTED,
        )
    else:
        pygame.draw.circle(surface, (35, 49, 70), center, int(radius * 0.92), 2)
        draw_text(
            surface,
            small_font,
            "no vector data",
            (rect.x + 14, rect.bottom - 30),
            MUTED,
        )


def draw_centered_axis(
    surface,
    x,
    y,
    width,
    label,
    value,
    font,
    small_font,
    positive_color=ACCENT,
    negative_color=WARN,
):
    track = pygame.Rect(x, y, width, 16)
    pygame.draw.rect(surface, (44, 52, 68), track, border_radius=8)
    pygame.draw.line(
        surface,
        (92, 104, 126),
        (track.centerx, track.top),
        (track.centerx, track.bottom),
        1,
    )
    pygame.draw.line(
        surface,
        (92, 104, 126),
        (track.left + 6, track.centery),
        (track.right - 6, track.centery),
        1,
    )

    normalized = max(-1.0, min(1.0, value))
    if normalized >= 0:
        fill = pygame.Rect(
            track.centerx, track.y, int(track.width / 2 * normalized), track.height
        )
        color = positive_color
    else:
        fill = pygame.Rect(
            track.centerx + int(track.width / 2 * normalized),
            track.y,
            int(track.width / 2 * (-normalized)),
            track.height,
        )
        color = negative_color
    pygame.draw.rect(surface, color, fill, border_radius=8)
    pygame.draw.circle(surface, color, (track.centerx, track.centery), 5)
    draw_text(surface, small_font, f"{label}: {value:.2f}", (x, y - 20), MUTED)


def draw_tri_axis_gauges(
    surface,
    rect,
    title,
    value,
    font,
    small_font,
    positive_color=ACCENT,
    negative_color=WARN,
):
    pygame.draw.rect(surface, (18, 24, 35), rect, border_radius=16)
    pygame.draw.rect(surface, (58, 68, 88), rect, width=2, border_radius=16)
    draw_text(surface, font, title, (rect.x + 14, rect.y + 10), TEXT)

    vector = coerce_vector(value)
    labels = ("X", "Y", "Z")
    if vector is None:
        draw_text(
            surface, small_font, format_value(value), (rect.x + 14, rect.y + 42), MUTED
        )
        return

    top = rect.y + 58
    for index, axis in enumerate(vector[:3]):
        draw_centered_axis(
            surface,
            rect.x + 16,
            top + index * 42,
            rect.width - 32,
            labels[index],
            axis,
            font,
            small_font,
            positive_color=positive_color,
            negative_color=negative_color,
        )


def draw_rotor(surface, rect, value, color, label, font, small_font):
    pygame.draw.rect(surface, (18, 24, 35), rect, border_radius=16)
    pygame.draw.rect(surface, (58, 68, 88), rect, width=2, border_radius=16)
    draw_text(surface, font, label, (rect.x + 14, rect.y + 10), TEXT)

    inner = rect.inflate(-30, -58)
    inner.top += 26
    center = inner.center
    radius = min(inner.width, inner.height) // 2 - 10
    pygame.draw.circle(surface, (24, 32, 46), center, radius)
    pygame.draw.circle(surface, (45, 58, 82), center, radius, 2)
    for tick in range(0, 360, 30):
        angle = math.radians(tick - 90)
        inner_pt = (
            int(center[0] + math.cos(angle) * (radius - 10)),
            int(center[1] + math.sin(angle) * (radius - 10)),
        )
        outer_pt = (
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        )
        pygame.draw.line(surface, (76, 88, 108), inner_pt, outer_pt, 2)

    vector_data = first_two_components(value)
    if vector_data is None:
        draw_text(
            surface,
            small_font,
            format_value(value),
            (rect.x + 14, rect.bottom - 30),
            MUTED,
        )
        return

    x, y, vector = vector_data
    nx, ny = normalize_pair(x, y)
    angle = math.atan2(ny, nx)
    tip = (
        int(center[0] + math.cos(angle) * radius * 0.82),
        int(center[1] + math.sin(angle) * radius * 0.82),
    )
    tail = (
        int(center[0] - math.cos(angle) * radius * 0.20),
        int(center[1] - math.sin(angle) * radius * 0.20),
    )
    pygame.draw.line(surface, color, tail, tip, 6)
    pygame.draw.circle(surface, color, tip, 11)
    pygame.draw.circle(surface, (255, 255, 255), tip, 3)
    pygame.draw.circle(surface, (255, 255, 255), center, 5)

    heading = (math.degrees(angle) + 360.0) % 360.0
    magnitude = math.sqrt(x * x + y * y)
    draw_text(
        surface,
        small_font,
        f"heading={heading:.0f}°  mag={magnitude:.2f}",
        (rect.x + 14, rect.bottom - 54),
        MUTED,
    )
    draw_text(
        surface,
        small_font,
        f"x={x:.2f}  y={y:.2f}",
        (rect.x + 14, rect.bottom - 30),
        MUTED,
    )


def update_pressed_buttons(pressed_buttons, event_type, status):
    pressed = status_is_pressed(status)
    if pressed:
        pressed_buttons.add(str(event_type))
    else:
        pressed_buttons.discard(str(event_type))


def append_event_log(event_log, event_type, status):
    event_log.appendleft(f"{event_type}: {status}")


def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("JoyCon Dashboard")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("DejaVu Sans", 28, bold=True)
    panel_font = pygame.font.SysFont("DejaVu Sans", 22, bold=True)
    body_font = pygame.font.SysFont("DejaVu Sans Mono", 18)
    small_font = pygame.font.SysFont("DejaVu Sans Mono", 16)

    try:
        joycon = JoyCon(*pyjoycon.get_R_id())
        device_error = None
    except Exception as exc:  # pragma: no cover - hardware dependent
        joycon = None
        device_error = exc

    pressed_buttons = set()
    event_log = deque(maxlen=10)
    running = True

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(BG)

        header_rect = pygame.Rect(24, 20, WIDTH - 48, 74)
        pygame.draw.rect(screen, (18, 24, 35), header_rect, border_radius=20)
        pygame.draw.rect(screen, (58, 68, 88), header_rect, width=2, border_radius=20)
        draw_text(screen, title_font, "JoyCon Dashboard", (42, 34), TEXT)
        draw_text(
            screen,
            small_font,
            f"fps target: {FPS}   frame: {dt} ms   esc to quit",
            (42, 64),
            MUTED,
        )

        left_panel = pygame.Rect(24, 112, 470, 544)
        right_panel = pygame.Rect(514, 112, 562, 544)
        draw_panel(screen, left_panel, "Pose", panel_font)
        draw_panel(screen, right_panel, "Button Events", panel_font)

        if joycon is None:
            message = f"Joy-Con not available: {device_error}"
            draw_text(
                screen, body_font, message, (left_panel.x + 20, left_panel.y + 70), BAD
            )
            draw_text(
                screen,
                body_font,
                "Connect a Joy-Con and relaunch the script.",
                (left_panel.x + 20, left_panel.y + 100),
                MUTED,
            )
            draw_text(
                screen,
                body_font,
                message,
                (right_panel.x + 20, right_panel.y + 70),
                BAD,
            )
        else:
            try:
                pointer = joycon.pointer
                rotation = joycon.rotation
                direction = joycon.direction
                accel = joycon.accel
                gyro = joycon.gyro

                for event_type, status in joycon.events():
                    update_pressed_buttons(pressed_buttons, event_type, status)
                    append_event_log(event_log, event_type, status)
            except Exception as exc:  # pragma: no cover - hardware dependent
                device_error = exc
                joycon = None
                pointer = rotation = direction = accel = gyro = None

            if joycon is not None:
                draw_scope(
                    screen,
                    pygame.Rect(
                        left_panel.x + 18, left_panel.y + 62, left_panel.width - 36, 150
                    ),
                    "Pointer",
                    pointer,
                    panel_font,
                    small_font,
                    ACCENT,
                )
                draw_rotor(
                    screen,
                    pygame.Rect(
                        left_panel.x + 18,
                        left_panel.y + 228,
                        left_panel.width - 36,
                        150,
                    ),
                    direction if coerce_vector(direction) is not None else rotation,
                    GOOD,
                    "Direction / Rotation",
                    panel_font,
                    small_font,
                )
                draw_tri_axis_gauges(
                    screen,
                    pygame.Rect(
                        left_panel.x + 18,
                        left_panel.y + 394,
                        left_panel.width - 36,
                        126,
                    ),
                    "Accelerometer",
                    accel,
                    panel_font,
                    small_font,
                )
                draw_tri_axis_gauges(
                    screen,
                    pygame.Rect(
                        right_panel.x + 18,
                        right_panel.y + 62,
                        right_panel.width - 36,
                        126,
                    ),
                    "Gyroscope",
                    gyro,
                    panel_font,
                    small_font,
                    positive_color=ACCENT,
                    negative_color=WARN,
                )
                draw_text(
                    screen,
                    small_font,
                    "left: pointer scope / rotation dial   right: gyro history + buttons",
                    (left_panel.x + 18, left_panel.bottom - 32),
                    MUTED,
                )

        if joycon is not None:
            chip_area = pygame.Rect(
                right_panel.x + 18, right_panel.y + 206, right_panel.width - 36, 96
            )
            pygame.draw.rect(screen, (18, 24, 35), chip_area, border_radius=14)
            pygame.draw.rect(screen, (58, 68, 88), chip_area, width=2, border_radius=14)
            draw_text(
                screen,
                small_font,
                "Held / active buttons",
                (chip_area.x + 14, chip_area.y + 10),
                MUTED,
            )

            active_buttons = sorted(pressed_buttons)
            if active_buttons:
                x = chip_area.x + 12
                y = chip_area.y + 38
                for label in active_buttons:
                    chip_width = min(
                        max(90, 18 + len(label) * 10), chip_area.width - 24
                    )
                    if x + chip_width > chip_area.right - 12:
                        x = chip_area.x + 12
                        y += 36
                    draw_status_chip(
                        screen,
                        pygame.Rect(x, y, chip_width, 28),
                        label,
                        True,
                        small_font,
                    )
                    x += chip_width + 8
            else:
                draw_text(
                    screen,
                    small_font,
                    "No active buttons",
                    (chip_area.x + 14, chip_area.y + 40),
                    MUTED,
                )

            log_rect = pygame.Rect(
                right_panel.x + 18, right_panel.y + 318, right_panel.width - 36, 206
            )
            pygame.draw.rect(screen, (18, 24, 35), log_rect, border_radius=14)
            pygame.draw.rect(screen, (58, 68, 88), log_rect, width=2, border_radius=14)
            draw_text(
                screen,
                small_font,
                "Recent events",
                (log_rect.x + 14, log_rect.y + 10),
                MUTED,
            )

            if event_log:
                for index, line in enumerate(event_log):
                    color = TEXT if index == 0 else MUTED
                    draw_text(
                        screen,
                        small_font,
                        line,
                        (log_rect.x + 14, log_rect.y + 42 + index * 28),
                        color,
                    )
            else:
                draw_text(
                    screen,
                    small_font,
                    "No events yet",
                    (log_rect.x + 14, log_rect.y + 42),
                    MUTED,
                )

            footer_text = (
                f"pointer: {format_value(pointer)}   rotation: {format_value(rotation)}"
            )
            draw_text(
                screen,
                small_font,
                footer_text,
                (right_panel.x + 18, right_panel.bottom - 28),
                MUTED,
            )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
