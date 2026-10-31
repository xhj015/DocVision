"""image_utils.py —— DocVision 的图片处理模块。

职责：把 OCR 识别结果可视化 —— 在原始图片上画出每个
文字块的检测框，并标注置信度，方便直观查看识别效果。

依赖 OpenCV（已在项目环境中安装）。

注意：cv2.imread / cv2.imwrite 在 Windows 上无法处理含中文的路径，
因此本模块统一使用 np.fromfile + cv2.imdecode / cv2.imencode + tofile
的读写方式，兼容中文文件路径。
"""

import os
from typing import List, Dict, Optional

import cv2
import numpy as np


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """读取图片，兼容中文路径。

    cv2.imread 在 Windows 下遇到非 ASCII 路径会返回 None，
    这里先用 np.fromfile 读成字节流，再交给 cv2.imdecode 解码。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """保存图片，兼容中文路径。

    cv2.imwrite 同样不支持中文路径，改为先编码成字节流再写文件。
    """
    try:
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except (OSError, ValueError, cv2.error):
        return False


def draw_boxes(
    image_path: str,
    items: List[Dict],
    output_path: Optional[str] = None,
) -> str:
    """在图片上画出文字检测框，返回标注图保存路径。

    Args:
        image_path: 原始图片路径。
        items: OCREngine.recognize() 返回的文字块列表，
               每个元素含 "box"（[x1,y1,x2,y2]）和 "score"。
        output_path: 标注图保存路径；不传则在原图同目录生成 *_boxes.*。

    Returns:
        标注图的实际保存路径。
    """
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")

    for item in items:
        box = item.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = box
        score = item.get("score", 0.0)

        # 置信度越高，框越绿；越低越橙（BGR 格式）
        color = _score_to_color(score)
        # 画矩形框，线宽 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # 在框的左上角标注置信度
        label = f"{score:.2f}"
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_boxes{ext or '.png'}"

    if not imwrite_unicode(output_path, img):
        raise OSError(f"无法保存标注图：{output_path}")
    return output_path


def _score_to_color(score: float):
    """根据置信度返回框的颜色（BGR）。

    高置信度（接近 1）→ 绿色；低置信度（接近 0）→ 橙色。
    这样一眼就能看出哪些识别结果可信、哪些需要人工核对。
    """
    # 把 score 映射到 0~1 的插值：score 高 → 绿分量高
    green = int(255 * score)
    red = int(255 * (1.0 - score))
    return (0, green, red)  # BGR: 蓝=0，绿随置信度，红随不确定度
