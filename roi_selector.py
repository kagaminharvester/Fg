"""
roi_selector.py
================

This module defines ``ROISelector``, a reusable QWidget allowing users
to draw a rectangular region of interest (ROI) on an image.  It
exposes a PyQt signal ``roiSelected`` whenever the user finishes
drawing.  The widget can be reused in other contexts where an ROI
needs to be selected interactively (e.g. for object tracking).
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets


class ROISelector(QtWidgets.QGraphicsView):
    """A view on which the user can draw a rectangular ROI.

    Once the user releases the mouse button a ``roiSelected`` signal
    will be emitted with the rectangle coordinates (x1, y1, x2, y2).
    """
    roiSelected = QtCore.pyqtSignal(int, int, int, int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._rubber_band: QtWidgets.QGraphicsRectItem | None = None
        self._origin: QtCore.QPointF | None = None
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    def setImage(self, image: QtGui.QImage) -> None:
        """Set the base image on which to draw the ROI."""
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QtGui.QPixmap.fromImage(image))
        self.setSceneRect(self._pixmap_item.boundingRect())

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._origin = self.mapToScene(event.pos())
            if self._rubber_band:
                self._scene.removeItem(self._rubber_band)
                self._rubber_band = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._origin:
            current = self.mapToScene(event.pos())
            rect = QtCore.QRectF(self._origin, current).normalized()
            if self._rubber_band is None:
                pen = QtGui.QPen(QtCore.Qt.GlobalColor.red)
                pen.setWidth(2)
                self._rubber_band = self._scene.addRect(rect, pen)
            else:
                self._rubber_band.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._origin and self._rubber_band:
            rect: QtCore.QRectF = self._rubber_band.rect()
            x1 = int(rect.left())
            y1 = int(rect.top())
            x2 = int(rect.right())
            y2 = int(rect.bottom())
            self.roiSelected.emit(x1, y1, x2, y2)
            self._origin = None
        super().mouseReleaseEvent(event)
