"""ocr_engine.py —— DocVision 的 OCR 识别核心模块。

职责：封装 PaddleOCR 的调用，把「检测 + 识别」的结果整理成
统一、易用的数据结构，供界面层（Gradio）和命令行工具复用。

设计原则：
1. 只做「模型调用 + 结果整理」，不训练模型、不写神经网络。
2. OCR 模型来自 PaddleOCR 官方，本模块负责应用层的封装。
3. 结果统一为 list[dict]，每个 dict 对应一个识别出的文字块，
   包含：文本、置信度、检测框坐标（左上/右下）。
"""

from typing import List, Dict, Optional


class OCREngine:
    """PaddleOCR 推理引擎的轻量封装。

    内部维护一个单例模型，避免每次识别都重新加载模型
    （模型加载在 CPU 上需要 5~10 秒，只加载一次）。
    """

    def __init__(
        self,
        det_model_name: str = "PP-OCRv5_server_det",
        rec_model_name: str = "PP-OCRv5_server_rec",
        score_thresh: float = 0.0,
    ):
        """初始化 OCR 引擎。

        Args:
            det_model_name: 文字检测模型名。
            rec_model_name: 文字识别模型名。
            score_thresh: 置信度阈值，低于该值的识别结果会被丢弃。
        """
        self.score_thresh = score_thresh
        # 延迟导入 + 延迟初始化：真正的模型加载放到第一次识别时，
        # 这样 import 本模块不会拖慢启动，也能在无 PaddleOCR 环境下
        # 只报一次清晰错误。
        self._ocr = None
        self._det_model_name = det_model_name
        self._rec_model_name = rec_model_name

    def _load_model(self):
        """加载 PaddleOCR 模型（仅第一次调用时执行）。"""
        if self._ocr is not None:
            return

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "未找到 PaddleOCR。请先安装：pip install paddleocr"
            ) from e

        self._ocr = PaddleOCR(
            text_detection_model_name=self._det_model_name,
            text_recognition_model_name=self._rec_model_name,
            # 关闭文档方向分类 / 纠偏 / 文本行方向，加快速度且我们不需要
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image_path: str) -> List[Dict]:
        """识别一张图片，返回整理后的文字块列表。

        Args:
            image_path: 图片文件路径。

        Returns:
            list[dict]，每个元素结构：
            {
                "text":   str,   识别出的文字,
                "score":  float, 置信度 0~1,
                "box":    [x1, y1, x2, y2], 左上/右下角像素坐标,
            }
        """
        self._load_model()

        result = self._ocr.predict(image_path)

        items: List[Dict] = []
        # predict() 返回一个 list，每个元素对应一页/一张图。
        # 这里我们只处理单张图片，取第一个结果。
        for res in result:
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            rec_boxes = res.get("rec_boxes", [])

            for i, text in enumerate(rec_texts):
                text = (text or "").strip()
                # 过滤空文本块（检测到空白区域的常见情况）
                if not text:
                    continue

                score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                # 过滤低于阈值的低置信度结果
                if score < self.score_thresh:
                    continue

                box = self._box_to_xyxy(rec_boxes, i)
                items.append(
                    {
                        "text": text,
                        "score": round(score, 4),
                        "box": box,
                    }
                )

        return items

    @staticmethod
    def _box_to_xyxy(rec_boxes, index: int) -> Optional[List[int]]:
        """把 PaddleOCR 的检测框转成 [x1, y1, x2, y2] 格式。

        PaddleOCR 的 rec_boxes 是一个形状为 (N, 4) 的数组，
        每一行已经直接是 [x1, y1, x2, y2]（左上角、右下角），
        无需再做多边形转外接矩形的换算。
        """
        if rec_boxes is None:
            return None
        try:
            box = rec_boxes[index]
            return [int(v) for v in box[:4]]
        except (IndexError, TypeError, ValueError):
            return None


def recognize_image(image_path: str, score_thresh: float = 0.0) -> List[Dict]:
    """便捷函数：一行调用识别图片。

    适合在命令行脚本或测试中直接使用。
    """
    engine = OCREngine(score_thresh=score_thresh)
    return engine.recognize(image_path)
