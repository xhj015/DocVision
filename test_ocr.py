"""test_ocr.py —— DocVision 的最小可运行测试。

功能：识别一张图片，在终端打印每个文字块的「文本 + 置信度 + 坐标」，
并把结果保存为 TXT 文件。

用法：
    python test_ocr.py <图片路径> [输出TXT路径]

示例：
    python test_ocr.py examples/sample.png
    python test_ocr.py examples/sample.png output/result.txt
"""

import os
import sys

from ocr_engine import OCREngine
from image_utils import draw_boxes


def main():
    if len(sys.argv) < 2:
        print("用法：python test_ocr.py <图片路径> [输出TXT路径]")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
        sys.exit(1)

    # 输出 TXT 路径：命令行指定，否则默认放到图片同目录下
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base, _ = os.path.splitext(image_path)
        out_path = base + "_result.txt"

    print("=" * 60)
    print(f"正在识别：{image_path}")
    print("首次运行会加载模型，CPU 上约需 5~10 秒，请稍候...")
    print("=" * 60)

    engine = OCREngine()
    items = engine.recognize(image_path)

    print(f"\n共识别到 {len(items)} 个文字块：\n")
    lines = []
    for i, item in enumerate(items, start=1):
        box = item["box"]
        box_str = (
            f"({box[0]},{box[1]})->({box[2]},{box[3]})" if box else "无坐标"
        )
        line = f"[{i:>2}] {item['text']}  置信度 {item['score']:.4f}  {box_str}"
        print(line)
        lines.append(f"{item['text']}\t{item['score']:.4f}")

    # 保存 TXT
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n结果已保存到：{out_path}")

    # 生成检测框可视化图
    box_img_path = draw_boxes(image_path, items)
    print(f"检测框标注图已保存到：{box_img_path}")

    # 汇总：按置信度从高到低排前 5 条
    top = sorted(items, key=lambda x: x["score"], reverse=True)[:5]
    print("\n置信度最高的 5 条：")
    for item in top:
        print(f"  {item['text']}  ({item['score']:.4f})")


if __name__ == "__main__":
    main()
