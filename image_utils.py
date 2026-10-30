"""image_utils.py —— DocVision 的图片处理模块。

职责：把 OCR 识别结果可视化 —— 在原始图片上画出每个
文字块的检测框，并标注置信度，方便直观查看识别效果。

依赖 OpenCV（已在项目环境中安装）。
"""

from typing import List, Dict, Optional

import cv2


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
        output_path: 标注图保存路径；不传则自动生成。

    Returns:
        标注图的实际保存路径。
    """
    img = cv2.imread(image_path)
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
        base, ext = image_path.rsplit(".", 1)
        output_path = f"{base}_boxes.{ext}"

    cv2.imwrite(output_path, img)
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
