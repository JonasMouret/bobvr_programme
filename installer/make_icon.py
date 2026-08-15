"""Draw the application icon.

Kept as a script rather than a stray binary so the icon can be changed without
a drawing program, and so anyone can see what it is made of. The result,
``bobvr.ico``, is committed next to it -- the Windows build should not need Qt
running just to have a tab icon.

    python installer/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

#: Windows picks the nearest size from the file; these cover everything from
#: the taskbar to the "extra large icons" view of Explorer.
SIZES = (16, 24, 32, 48, 64, 128, 256)

NIGHT = QColor("#101C2E")
NIGHT_LOW = QColor("#1D3350")
ICE = QColor("#BFE6FF")
ICE_DEEP = QColor("#4FA8E8")
BOB = QColor("#F2F5F8")


def draw(size: int) -> QImage:
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(Qt.transparent)

    p = QPainter(image)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(size / 256.0, size / 256.0)          # draw once, at 256

    # Background: a rounded square, lit from the top like a night sky.
    sky = QLinearGradient(QPointF(0, 0), QPointF(0, 256))
    sky.setColorAt(0.0, NIGHT_LOW)
    sky.setColorAt(1.0, NIGHT)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(sky))
    p.drawRoundedRect(QRectF(0, 0, 256, 256), 52, 52)

    # The track: a bank curving away, drawn as a thick ribbon that narrows
    # with distance. This is the shape of the whole app in one line.
    track = QPainterPath()
    track.moveTo(18, 232)
    track.cubicTo(86, 210, 92, 132, 148, 104)
    track.cubicTo(186, 85, 214, 84, 240, 88)

    ice = QLinearGradient(QPointF(0, 240), QPointF(240, 80))
    ice.setColorAt(0.0, ICE)
    ice.setColorAt(1.0, ICE_DEEP)

    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QBrush(ice), 34, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(track)
    # A brighter inner line: ice reads as ice when it has a highlight.
    p.setPen(QPen(QColor(255, 255, 255, 90), 8, Qt.SolidLine, Qt.RoundCap))
    p.drawPath(track)

    # The bob, nose-on, sitting in the near bank of the curve.
    p.setPen(Qt.NoPen)
    p.setBrush(BOB)
    body = QPainterPath()
    body.moveTo(58, 214)
    body.cubicTo(74, 176, 116, 168, 132, 190)
    body.cubicTo(140, 202, 128, 216, 108, 220)
    body.closeSubpath()
    p.drawPath(body)

    # The camera on top: the 360 eye, the reason this app exists.
    p.setBrush(QColor("#101C2E"))
    p.drawEllipse(QPointF(103, 180), 19, 19)
    p.setBrush(ICE)
    p.drawEllipse(QPointF(103, 180), 11, 11)
    p.setBrush(QColor(255, 255, 255, 220))
    p.drawEllipse(QPointF(98, 175), 4, 4)

    p.end()
    return image


def main() -> None:
    # Qt's raster painter wants a running application object, even offscreen.
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        QGuiApplication(["make_icon"])

    here = Path(__file__).resolve().parent
    images = [draw(size) for size in SIZES]

    # Qt's .ico writer keeps one frame per file, and Windows wants all the
    # sizes in one container -- so the container is assembled here, from PNG
    # payloads, which Windows has read inside .ico files since Vista.
    target = here / "bobvr.ico"
    _write_ico(target, images)

    # The window and the taskbar read this one at run time: the .ico is only
    # the icon of the .exe file itself, and Qt would otherwise show its own.
    runtime = here.parent / "bobvr" / "ui" / "icon.png"
    images[-1].save(str(runtime), "png")
    print(f"écrit {target} ({target.stat().st_size} octets) et {runtime}")


def _write_ico(target: Path, images: list[QImage]) -> None:
    import struct
    from PySide6.QtCore import QBuffer, QByteArray

    payloads = []
    for image in images:
        # The byte array has to outlive the buffer that writes into it; a
        # temporary here is freed under Qt's feet.
        store = QByteArray()
        buffer = QBuffer(store)
        buffer.open(QBuffer.WriteOnly)
        image.save(buffer, "png")
        buffer.close()
        payloads.append(bytes(store))

    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries, blob = b"", b""
    for image, payload in zip(images, payloads):
        side = 0 if image.width() >= 256 else image.width()
        entries += struct.pack(
            "<BBBBHHII", side, side, 0, 0, 1, 32, len(payload), offset
        )
        blob += payload
        offset += len(payload)
    target.write_bytes(header + entries + blob)


if __name__ == "__main__":
    main()
