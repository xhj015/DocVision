"""app.py —— DocVision 的 Web 界面（Gradio）。

功能：浏览器里上传图片 → 调用 PaddleOCR 识别 → 展示标注图、
结果表格、统计摘要，并支持导出 TXT。

交互设计：
    点「开始识别」执行一次模型推理，把**全量结果**（阈值 0）缓存到 State；
    之后拖动置信度滑块只是在缓存结果上做过滤与重画，不会重复推理。
    这样调阈值是实时的，也避免了重复调用模型。

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

DEFAULT_THRESH = 0.5

# 供界面展示的示例图片
EXAMPLES = ["examples/sample.png"] if os.path.exists("examples/sample.png") else None


def _render(image_path: str, items, thresh: float):
    """按阈值过滤缓存结果，重新生成图 / 表 / 摘要 / TXT（不调用模型）。"""
    kept = [it for it in items if it["score"] >= thresh]
    if not kept:
        return (
            None,
            [],
            f"当前阈值 **{thresh:.2f}** 下没有结果，试试把阈值调低。",
            None,
        )

    # 检测框可视化（输出路径放系统临时目录，避免中文路径与回收问题）
    boxed_path = draw_boxes(
        image_path,
        kept,
        output_path=os.path.join(
            tempfile.gettempdir(), f"docvision_boxes_{uuid.uuid4().hex[:8]}.png"
        ),
    )

    rows = [
        [i, it["text"], round(it["score"], 4), str(it["box"])]
        for i, it in enumerate(kept, start=1)
    ]

    txt_path = tempfile.mktemp(suffix="_docvision.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{it['text']}\t{it['score']:.4f}" for it in kept))

    avg_score = sum(it["score"] for it in kept) / len(kept)
    summary = (
        f"**共识别 {len(kept)} 个文字块**（全量 {len(items)} 个）　|　"
        f"平均置信度 {avg_score:.4f}　|　"
        f"最高 {max(it['score'] for it in kept):.4f}"
    )

    return boxed_path, rows, summary, txt_path


def do_recognize(image_path: str):
    """点击「开始识别」：跑一次模型推理，缓存全量结果并渲染。"""
    if not image_path:
        return [], None, [], "请先上传一张图片。", None

    # 阈值传 0，拿到全部结果，后续由滑块过滤
    items = ENGINE.recognize(image_path, score_thresh=0.0)
    if not items:
        return [], None, [], "没有识别到文字，可以换一张图片试试。", None

    img, rows, summary, txt = _render(image_path, items, DEFAULT_THRESH)
    return items, img, rows, summary, txt


def on_thresh_change(items, image_path: str, thresh: float):
    """拖动滑块：只过滤缓存结果，不重新推理。"""
    if not items or not image_path:
        return None, [], "请先上传图片并点击「开始识别」。", None
    return _render(image_path, items, thresh)


def build_ui() -> gr.Blocks:
    """构建 Gradio 界面。"""
    with gr.Blocks(title="DocVision - 图片文字识别") as demo:
        gr.Markdown(
            """
            # DocVision
            基于百度飞桨 **PaddleOCR** 的图片文字识别工具

            上传图片 → 点击「开始识别」→ 拖动阈值实时筛选 → 导出 TXT
            """
        )

        # 缓存本次识别的全量结果，供滑块过滤复用
        state = gr.State([])

        with gr.Row():
            # 左栏：输入
            with gr.Column(scale=1):
                img_in = gr.Image(label="上传图片", type="filepath", height=360)
                thresh = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_THRESH,
                    step=0.05,
                    label="置信度阈值（识别后拖动可实时筛选）",
                    info="低于该值的结果会被隐藏；调整无需重新识别",
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
            fn=do_recognize,
            inputs=[img_in],
            outputs=[state, img_out, table, summary, file_out],
            concurrency_limit=1,
        )

        # 松开滑块即触发过滤（不重新推理）
        thresh.release(
            fn=on_thresh_change,
            inputs=[state, img_in, thresh],
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
