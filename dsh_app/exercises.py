"""练习导入：把老师发的 xlsx / csv 题目接进小测。

为什么要这一层：模型出的题受幻觉限制（所以要查原文），但老师发的题本来就是
权威答案——一份 40 题的练习册比模型现编的 3 道题有用得多。
两条路都要留着，但**出题来源必须标出来**：模型题靠原文回查，导入题靠文件答案。

表格不要求严格表头，只要列能认出来（下面 COLUMN_ALIASES）。认不出来就报清楚是哪一列，
而不是"导入成功 0 条"——静默失败是这一层最不该有的行为。
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

MIN_CHOICES = 2
MAX_CHOICES = 6

# 表头别名（中文/英文/short）。key 是内部字段名。
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "kp_name": ("知识点", "考点", "术语", "概念", "kp", "knowledge", "knowledge_point",
                "term", "topic"),
    "prompt": ("题干", "题目", "问题", "question", "prompt", "q", "stem"),
    "choices": ("选项", "choices", "options", "options_all"),
    "answer": ("答案", "正确选项", "answer", "key", "correct", "correct_answer"),
    "explain": ("解析", "说明", "解释", "explain", "explanation", "note", "reason"),
    "kind": ("题型", "类型", "kind", "type"),
}

_LETTERS = "ABCDEFGH"


def _norm_header(h: Any) -> str:
    s = str(h or "").strip().lower()
    s = re.sub(r"[\s_\-（）()：:]+", "", s)
    return s


def _map_headers(header: Sequence[Any]) -> dict[str, int]:
    """把表头行映射成 {字段: 列号}。认不出的列忽略，认出的列按别名匹配。"""
    out: dict[str, int] = {}
    choice_cols: dict[int, str] = {}
    for idx, cell in enumerate(header):
        key = _norm_header(cell)
        if not key:
            continue
        # 选项A / 选项1 / choice_a / a
        m = re.fullmatch(r"(?:选项|choice|option|opt)?([a-h])", key)
        m2 = re.fullmatch(r"(?:选项|choice|option|opt)(\d)", key)
        if m:
            choice_cols[idx] = m.group(1).upper()
            continue
        if m2:
            n = int(m2.group(1))
            if 1 <= n <= len(_LETTERS):
                choice_cols[idx] = _LETTERS[n - 1]
            continue
        for fieldname, aliases in COLUMN_ALIASES.items():
            if fieldname == "choices":
                continue
            if key in {_norm_header(a) for a in aliases}:
                out.setdefault(fieldname, idx)
                break
    if choice_cols:
        out["_choice_cols"] = choice_cols          # type: ignore[assignment]
    return out


@dataclass
class ExerciseRow:
    prompt: str
    choices: list[str]
    answer_index: int
    kp_name: str = ""
    explain: str = ""
    kind: str = ""
    row_no: int = 0                           # 原始表格行号，报错时用

    def as_dict(self) -> dict:
        return {"prompt": self.prompt, "choices": self.choices,
                "answer_index": self.answer_index, "kp_name": self.kp_name,
                "explain": self.explain, "kind": self.kind, "row_no": self.row_no}


# ---------------------------------------------------------------- 读文件
def read_table(path: str | Path) -> list[list[Any]]:
    """读 xlsx 或 csv 成二维表。xlsx 需要 openpyxl；没有就明确报错。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到文件：{p}")
    suffix = p.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        try:
            from openpyxl import load_workbook
        except ModuleNotFoundError as e:      # pragma: no cover - 环境相关
            raise RuntimeError("读 xlsx 需要 openpyxl（pip install openpyxl）；"
                               "或者把表格另存为 csv 再导入") from e
        wb = load_workbook(str(p), data_only=True, read_only=True)
        ws = wb.active
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        wb.close()
        return rows
    if suffix in (".csv", ".txt"):
        raw = p.read_bytes()
        for enc in ("utf-8-sig", "gb18030", "utf-16"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:                                  # pragma: no cover
            text = raw.decode("utf-8", "replace")
        # 逗号/制表符/分号都见得到，按出现最多的那个当分隔符
        sample = text[:4000]
        delim = max([",", "\t", ";"], key=lambda d: sample.count(d))
        return [row for row in csv.reader(io.StringIO(text), delimiter=delim)]
    raise RuntimeError(f"不认识的表格格式：{p.suffix}（支持 .xlsx / .csv）")


def _answer_index(value: Any, choices: Sequence[str]) -> int:
    """答案列支持：A/B/C、1/2/3、以及直接写选项原文。"""
    s = str(value or "").strip()
    if not s:
        return -1
    up = s.upper()
    if len(up) == 1 and up in _LETTERS:
        idx = _LETTERS.index(up)
        return idx if idx < len(choices) else -1
    if s.isdigit():
        n = int(s)
        if 1 <= n <= len(choices):
            return n - 1
        if 0 <= n < len(choices):
            return n
        return -1
    # 写成选项原文：去前缀后比较
    body = re.sub(r"^\s*[A-Za-z]\s*[.:：、)）-]\s*", "", s)
    for i, c in enumerate(choices):
        if " ".join(body.split()) == " ".join(str(c).split()):
            return i
    return -1


def parse_exercises(rows: Sequence[Sequence[Any]]) -> tuple[list[ExerciseRow], list[dict]]:
    """把二维表解析成题目。返回 (题目, 错误清单)。错误带行号与原因。"""
    errors: list[dict] = []
    rows = [r for r in rows if any(str(c or "").strip() for c in r)]
    if not rows:
        return [], [{"row_no": 0, "reason": "表格是空的"}]

    # 找表头：第一行能认出「题干」就当表头，否则按无表头处理（题干在 A 列）
    head_map: dict[str, int] = {}
    body = rows
    for i in range(min(5, len(rows))):
        m = _map_headers(rows[i])
        if "prompt" in m:
            head_map = m
            body = rows[i + 1:]
            break
    if not head_map:
        head_map = {"prompt": 0, "_choice_cols": {1: "A", 2: "B", 3: "C", 4: "D"}}
        errors.append({"row_no": 0, "reason": "没认出表头，按「题干在第 1 列、选项在 2-5 列」解析；"
                                              "建议下载模板对照"})

    choice_cols: dict[int, str] = head_map.get("_choice_cols") or {}    # type: ignore[assignment]
    out: list[ExerciseRow] = []
    for n, row in enumerate(body, start=2 if body is not rows else 1):
        cells = ["" if c is None else str(c).strip() for c in row]
        prompt = cells[head_map["prompt"]] if head_map.get("prompt") is not None else ""
        if not prompt:
            continue
        pairs = sorted(((letters, cells[idx]) for idx, letters in choice_cols.items()
                        if idx < len(cells) and cells[idx])) 
        pairs.sort(key=lambda x: x[0])
        choices = [text for _, text in pairs]
        if len(choices) < MIN_CHOICES:
            errors.append({"row_no": n, "reason": f"选项不足 {MIN_CHOICES} 个", "prompt": prompt[:40]})
            continue
        if len(choices) > MAX_CHOICES:
            errors.append({"row_no": n, "reason": f"选项超过 {MAX_CHOICES} 个",
                           "prompt": prompt[:40]})
            continue
        ans_cell = cells[head_map["answer"]] if head_map.get("answer") is not None else ""
        idx = _answer_index(ans_cell, choices)
        if idx < 0:
            errors.append({"row_no": n,
                           "reason": f"答案「{ans_cell}」对不上选项（应为 A-D 或选项原文）",
                           "prompt": prompt[:40]})
            continue
        out.append(ExerciseRow(
            prompt=prompt, choices=choices, answer_index=idx,
            kp_name=cells[head_map["kp_name"]] if head_map.get("kp_name") is not None else "",
            explain=cells[head_map["explain"]] if head_map.get("explain") is not None else "",
            kind=cells[head_map["kind"]] if head_map.get("kind") is not None else "",
            row_no=n))
    if not out and not errors:
        errors.append({"row_no": 0, "reason": "一行题也没解析出来（题干列可能是空的）"})
    return out, errors


def load_file(path: str | Path) -> tuple[list[ExerciseRow], list[dict]]:
    return parse_exercises(read_table(path))


# ---------------------------------------------------------------- 模板
TEMPLATE_HEADER = ["知识点", "题干", "选项A", "选项B", "选项C", "选项D", "答案", "解析"]
TEMPLATE_ROWS = [
    ["分布函数", "分布函数 F(x) 的定义是什么？",
     "P(X 小于等于 x)", "P(X 等于 x)", "X 的期望", "X 落在区间内的概率", "A",
     "定义就是累积概率，右连续是它的性质，不是定义本身。"],
    ["条件概率", "条件概率 P(A|B) 成立的前提是什么？",
     "P(B) 不为零", "P(A) 不为零", "A 与 B 独立", "A 包含 B", "A",
     "条件概率以 B 已发生为前提，所以要求 P(B)>0。"],
]


def write_template(path: str | Path) -> Path:
    """写一份能直接填的模板。有 openpyxl 就写 xlsx，否则写 csv。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        try:
            from openpyxl import Workbook
        except ModuleNotFoundError:            # pragma: no cover
            p = p.with_suffix(".csv")
        else:
            wb = Workbook()
            ws = wb.active
            ws.title = "练习"
            ws.append(TEMPLATE_HEADER)
            for r in TEMPLATE_ROWS:
                ws.append(r)
            for col, width in zip("ABCDEFGH", (14, 46, 18, 18, 18, 18, 8, 40)):
                ws.column_dimensions[col].width = width
            wb.save(str(p))
            return p
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(TEMPLATE_HEADER)
        w.writerows(TEMPLATE_ROWS)
    return p