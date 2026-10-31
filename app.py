"""app.py —— DocVision 的 Web 界面（Gradio）。

功能：浏览器里上传图片 → 调用 PaddleOCR 识别 → 展示标注图、
结果表格、统计摘要，并支持导出 TXT。

启动：
    python app.py
然后浏览器打开终端提示的地址（默认 http://127.0.0.1:7860）。
"""

import os
import tempfile
import uuid

import gradio as gr

from ocr_engine import OCREngine
from image_utils import draw_boxes

# 全局只创建一个引擎实例：模块加载时不会真正加载模型，
# 首次点击「开始识别」时才加载（懒加载），之后一直复用。
ENGINE = OCREngine()

# 供界面展示的示例图片
EXAMPLES = ["examples/sample.png"] if os.path.exists("examples/sample.png") else None


def run_ocr(image_path: str, score_thresh: float):
    """处理一次识别请求，返回四样结果给界面。"""
    if not image_path:
        return None, [], "请先上传一张图片。", None

    # ① 识别
    items = ENGINE.recognize(image_path, score_thresh=score_thresh)
    if not items:
        return (
            None,
            [],
            "没有识别到文字。可以换一张图，或把置信度阈值调低再试。",
            None,
        )

    # ② 检测框可视化
    # 输出到系统临时目录并随机命名：Gradio 的上传临时目录可能被回收，
    # 且原文件名可能含中文，统一放到可控路径更稳。
    boxed_path = draw_boxes(
        image_path,
        items,
        output_path=os.path.join(
            tempfile.gettempdir(), f"docvision_boxes_{uuid.uuid4().hex[:8]}.png"
        ),
    )

    # ③ 整理成表格数据
    rows = [
        [i, it["text"], round(it["score"], 4), str(it["box"])]
        for i, it in enumerate(items, start=1)
    ]

    # ④ 导出 TXT（文字 + 置信度，制表符分隔）
    txt_path = tempfile.mktemp(suffix="_docvision.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{it['text']}\t{it['score']:.4f}" for it in items))

    avg_score = sum(it["score"] for it in items) / len(items)
    summary = (
        f"**共识别 {len(items)} 个文字块**　|　"
        f"平均置信度 {avg_score:.4f}　|　"
        f"最高 {max(it['score'] for it in items):.4f}"
    )

    return boxed_path, rows, summary, txt_path


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面。"""
    with gr.Blocks(title="DocVision - 图片文字识别") as demo:
        gr.Markdown(
            """
            # DocVision
            基于百度飞桨 **PaddleOCR** 的图片文字识别工具

            上传图片 → 自动检测并识别文字 → 查看检测框与置信度 → 导出 TXT
            """
        )

        with gr.Row():
            # 左栏：输入
            with gr.Column(scale=1):
                img_in = gr.Image(label="上传图片", type="filepath", height=360)
                thresh = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="置信度阈值（越低保留越多，但可能有噪声）",
                )
                btn = gr.Button("开始识别", variant="primary", size="lg")

            # 右栏：输出
            with gr.Column(scale=1):
                img_out = gr.Image(label="检测框可视化", height=360)
                summary = gr.Markdown("等待识别……")
                table = gr.Dataframe(
                    headers=["#", "文字", "置信度", "坐标 [x1,y1,x2,y2]"],
                    label="识别结果",
                    wrap=True,
                )
                file_out = gr.File(label="下载识别结果 TXT")

        btn.click(
            fn=run_ocr,
            inputs=[img_in, thresh],
            outputs=[img_out, table, summary, file_out],
        )

        if EXAMPLES:
            gr.Examples(examples=EXAMPLES, inputs=[img_in], label="示例图片")

        gr.Markdown(
            "> OCR 模型来自 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0）。"
            "本项目为模型调用与应用层封装，不含模型训练。"
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, share=False)
