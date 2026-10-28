# DeepTutor 借鉴清单

来源：`HKUDS/DeepTutor`（43.4k ★，Apache-2.0），本地只做浅克隆 + 稀疏检出，
放在 `_ref/DeepTutor/`，**12 MB，没装它的依赖**。

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/HKUDS/DeepTutor.git
cd DeepTutor && git sparse-checkout set \
  deeptutor/learning deeptutor/knowledge deeptutor/video_learning \
  deeptutor/reading deeptutor/services deeptutor/tools docs-for-user
```

## 0. 值得看的目录（为什么是这几个）

| 目录 | 体积 | 内容 | 对我们的价值 |
|---|---|---|---|
| `deeptutor/learning/` | 1.1 MB / 13k 行 | 掌握度、门槛、遗忘调度、出题、评分、大纲生成 | ★★★ 我们缺的整条"知识点→掌握度→复习"链 |
| `deeptutor/services/rag/` | — | 5 种检索管线、查询改写、索引版本化、按页索引 | ★★★ 我们的 BM25 太薄 |
| `deeptutor/video_learning/` | 112 KB | 时间戳标记（marks）、笔记、AI 建议标记 | ★★ 和我们的回看点同构，能对照 |
| `deeptutor/reading/` | 352 KB | 文档阅读、quiz、references、study_guidance | ★ 同类但不重叠 |
| `deeptutor/knowledge/` | 233 KB | KB 生命周期（创建/加文档/索引/清单） | ★ 工程管理，非算法 |
| `deeptutor/tools/` | 666 KB | Level-1 工具实现 | ✗ 我们要的不是这个 |

**不值得看**（不是不好，是对端侧单机是负担）：`agents/`、`capabilities/`、
`runtime/`（双层插件 + turn replay + StreamBus）、`multi_user/`、`partners/`（IM 渠道）、
`book/`、`visualizers/`（Manim）。那是给多租户 SaaS 用的，我们是单机桌面程序。

---

## 1. 掌握度评分 `compute_mastery`（最该抄的一个）

`deeptutor/learning/mastery.py:17-37`，全文只有 40 行：

```python
_RECENCY_WEIGHTS = (0.5, 0.7, 0.85, 0.95, 1.0)   # 越近的尝试权重越高
_CONFIDENCE_CAP = {1: 0.5, 2: 0.8}               # 只答对 1 次最高 0.5，2 次最高 0.8

def compute_mastery(correctness):
    recent = correctness[-5:]
    weights = _RECENCY_WEIGHTS[-len(recent):]
    score = sum(w * clamp01(c) for c, w in zip(recent, weights)) / sum(weights)
    return min(score, _CONFIDENCE_CAP.get(len(recent), 1.0))
```

两个设计点值得直接搬：

1. **近因加权**：早期错、后来对的，不会被历史拖死（`0.5→1.0` 递增）。
2. **低置信上限**：答对一次不算会，这是防"蒙对"的最小代价手段。

### 我们的差距（核对过代码，比想象的更尴尬）

事实是：**知识点图已经建了，但没人读，也没有掌握度。**

- 建：`pipeline.py:625-648` 的 `_link_knowledge()` 在每次分析后跑，从回看点原文里抽可信术语
  → `upsert_knowledge_point` → `link_knot_point` → 同一回看点内的知识点两两 `add_edge(co_occur)`
- 表结构（`db.py:167-175`）：`name / first_seen_t / session_ids / occurrences` ——
  **没有 mastery、没有 last_reviewed、没有衰减**
- 读：`get_knowledge_points()` **在整个项目里零调用**；`knowledge_edges` 只在写入时出现一次

也就是说：我们花力气建了一张图，然后让它烂在库里。「漏洞地图」页实际走的是
`diagnosis()` 里的术语统计（另一条完全独立的路径），和这张图**没有任何关系**。

同时 `cognitive/scoring.py` 只回答"这一段内容有多难"（L/B/S），不回答"这个知识点你会不会"。
两者不是一回事：回看点告诉你"这里难"，掌握度告诉你"这里你已经补上了"。
缺了后者，漏洞地图上的回看点**永远不会熄灭**——只能一条条叠加，没有闭环。

### 落地方案

```
回看点(knot) --link--> 知识点(kp) --feedback(有用/本来就会/重讲后自测)--> mastery 0..1
```

- 「这个讲明白了（记为有用）」→ 相当于一次 correct
- 重讲后的自测答对/答错 → correct / incorrect
- 掌握度 ≥ 门槛 → 该知识点在漏洞地图上转绿，不再进复习队列

成本：一个 30 行的 `mastery.py` + 反馈写入路径 + **让 `get_knowledge_points()` 第一次被真正调用**。
**这是投入产出比最高的一块**——图已经建好了，只差读它。

---

## 2. 硬门槛 `QUANTITATIVE_GATE`：0.9 才放行

`deeptutor/learning/policy.py:33-46`：

```python
QUANTITATIVE_GATE = {MEMORY: 0.9, PROCEDURE: 0.9}     # 记忆/流程类：90% 才算过
QUALITATIVE_TYPES = {CONCEPT, DESIGN}                  # 概念/设计类：不讲分数
```

概念类走 **Feynman 式口述** 判断（"用你自己的话讲一遍，我判断你懂没懂"），因为这类没有唯一
标准答案，字符串评分没意义。模块级还有一个 `pass_threshold = 0.7`（`models.py:103`）。

### 我们的差距

我们的 `grade_strong = 0.65` / `grade_medium = 0.45` 是**回看点强度阈值**——回答"这里值不值得提醒你"。
它没有"放行 / 继续 / 你得先补这个"的概念。所以现在的产品只是个**报告生成器**，
不是学习路径。

### 落地方案（可选，P2）

给 `knowledge_points` 加 `type` 和 `gate`，先在「漏洞地图」页加一列"状态"：
未开始 / 在补 / 已通过。不要急着做路径锁——端侧用户是"补课"不是"闯关"，
但**颜色变化**（红→黄→绿）本身就是正反馈。

---

## 3. 遗忘调度：DSR 的简化版

`deeptutor/learning/scheduler.py`，550 行，核心是三个量与几个常数：

| 量 | 含义 | 更新规则（`scheduler.py:280-324`） |
|---|---|---|
| `stability` | 记忆稳定度（天） | 答对：`× (1 + 质量强度 × 可学性 × 提取努力 × 间隔奖励)`；答错：`× (0.55 - 0.25·上次可提取率) × (1 - 0.25·难度)` |
| `difficulty` | 这题对你多难 | `+0.12·实际误差 + 0.08·意外度`，夹在 `[0.05, 0.95]` |
| `retrievability` | 现在还记得的概率 | `exp(-经过天数 / stability)`，目标留存 `0.9` |

常数：`_MIN_STABILITY_DAYS = 0.5`、`_FAIL_QUALITY = 0.5`、`desired_retention ∈ [0.7, 0.99]`（默认 0.9）。
间隔换算：`interval = stability × (-ln(desired_retention))`（`:143-146`）。

它自己声明这是"**指数遗忘 + 类型先验，不是校准过的 FSRS**"（`:65-68`）——这种诚实的
自我限定值得学。

### 我们的差距

`diagnosis()` 跨节汇总只做**共现统计**（同一术语在几节里都卡 → root cause），
没有时间维度。今天卡的地方和三个月前卡的地方，权重一样。

### 落地方案（P2）

不需要全搬 DSR。最小版本：给每个知识点记 `last_seen_at` + `review_count`，
用 `exp(-天数/stability)` 算"现在还记得的概率"，低于 0.7 就进「该复习了」列表。
这样"漏洞地图"就有了时间轴，而不是一张静态照片。

---

## 4. 按页索引 + 混合检索（我们的 BM25 太薄）

`deeptutor/services/rag/source_pages.py:220-270` + `factory.py:6`：

- 默认管线是 **vector + BM25 融合**（hybrid），不是单一 BM25
- 分块用 `SentenceSplitter(chunk_size=512, chunk_overlap=50)`
- 关键：**按物理页建目录**（`catalog.json` 记每页的 `document_key` + 页内文本），
  检索命中后能回到**具体某一页**

### 我们的差距

`kb.py:19-20`：`MAX_CHUNK = 600, OVERLAP = 60`，纯 BM25（k1=1.5, b=0.75），
`kb_chunks` 表只有 `ord / title / text`——**没有页码、没有幻灯片号、没有版本**。

后果：引用只能写「老周笔记 第 1 段 score 1.0」，用户没法回到原文那一页。
而我们已经有「补看回看点画面」能把 B 站视频定位到某一帧，**KB 这边却定位不到页**，不对称。

### 落地方案（P0 的一部分）

`kb_chunks` 加三列：`page INT`、`section TEXT`、`source_ref TEXT`（存页码或 PPT 页号）。
导入 PDF/PPTX 时按页切，而不是全文按 600 字硬切。引用变成「老周笔记 p.12」。

顺带抄一条原则（`reading/knowledge_capture.py` 的思路）：**模型输出是证据，不是事务**。
引用必须能由代码校验到原文，模型不允许"看着像"就引用。

---

## 5. 查询改写（fail-open）

`deeptutor/services/rag/query_rewrite.py:29-34, 82-87`：

```
system: 把用户问题改写成一条自包含的检索查询，保持原语言，不要翻译，
        把代词和上下文消解成显式关键词，只输出查询本身
fail-open: 任何异常/超时/空输出 → 退回原始查询（min_query_chars 以下直接跳过）
```

### 我们的差距

`pipeline.ask()` 直接把用户原话交给 BM25。用户问「**这步**为什么成立」时，
"这步"在语料里一个词都匹配不上，检索必然空手而归。

### 落地方案（P1）

在 `chat.ask()` 前加一次改写：有历史上下文就把指代补全（"这步"→"协方差矩阵的特征值分解"），
5 秒超时，失败就用原句。**纯增益，无风险**（fail-open 是它最重要的设计）。

---

## 6. 索引版本化 + 签名

`deeptutor/services/rag/index_versioning.py:1-40`：`index_versions/version-N/` +
`meta.json{version, signature, model}`。换嵌入模型 → 签名变 → 自动重建，旧版本留着可回滚。
`preflight.py` 还会在检索前检查依赖（"BM25 hybrid retrieval: Installed / 未装则退化为纯向量"）。

### 我们的差距

`kb_chunks` 没有任何签名。改动分块参数或换检索模型，只能全删重建，且**无从知道该不该重建**。

### 落地方案（P1）

`kb_docs` 加一列 `sig TEXT`（= hash(分块参数 + 模型名 + 文件 mtime/size)）。
打开知识库时比对，不一致就在界面提示「索引已过期，点这里重建」，而不是静默用旧索引。

---

## 7. 建议标记：从转写生成候选（不给用户背锅）

`deeptutor/video_learning/marks.py:339-373`：

```
prompt: Extract up to 5 learning marks ... Return JSON only:
[{"kind":"key_point|question|review","start_seconds":N,"end_seconds":N,"quote":"原文片段","note":""}]
       Do not invent timestamps outside the supplied transcript.
```

三个细节：

1. **不许编时间戳**（明确写进 system prompt）——我们做「补看回看点画面」时踩过同类坑。
2. **有 heuristic 兜底**（`:296-325`）：没有模型时，含问号的句子直接归为 `question`，
   当前转写段归为 `key_point`。保证功能在离线时也不空白。
3. `_public_suggestion` 会把引文用 `quote_for_range()` 回查原文，**空引文的建议直接丢弃**。

### 我们的差距

我们的回看点是算法算出来的（L/B/S），没有"AI 建议你标这几个点"。而且
「问模型」和「回看点」是两套东西，没有互相喂。

### 落地方案（P1）

「时间轴」页加一个「让我看看该标哪」：把该会话的转写（前 40 段）丢给模型，
拿回 3–5 个候选标记，**引文必须能在转写里查到**否则丢弃，用户点一下就变成回看点。
这比全自动判断更符合你之前那条裁决——**算法不确定的时候让人来选**。

---

## 8. 优先级（按投入产出比）

| 级别 | 做的事 | 成本 | 收益 |
|---|---|---|---|
| **P0** ✅ | `compute_mastery` + 让已有的知识点图被读起来 | 半天 | 漏洞地图第一次能"熄灭"，从报告变工具 |
| **P0** ✅ | `kb_chunks` 加 page/section/source_ref，按页切 | 半天 | 引用能跳回原文，和"补看画面"对齐 |
| **P1** ✅ | 查询改写（fail-open + 5s 超时） | 2 小时 | 指代问句终于检得到东西 |
| **P1** ✅ | 索引签名 + 过期提示 | 2 小时 | 告别"改了参数不知道要不要重建" |
| **P1** | 「让我看看该标哪」（带引文校验） | 半天 | 回看点从"算法说"变"你选" |
| **P2** ✅ | 遗忘调度 + 复习队列 | 1–2 天 | 有了时间维度，才叫"终身"（见文末第三轮） |
| **P2** | 硬门槛 / 状态颜色 | 1 天 | 路径感 |
| **不做** | agents/capabilities 双层插件、turn replay、多租户、IM 渠道 | — | 端侧单机不需要 |

---

## 9. 一句话总结

它最值得抄的不是 RAG，也不是 agent 架构，而是**「证据 → 掌握度 → 门槛 → 复习」这条闭环**：
每个判断都要有出处的证据，每个证据都要更新一个可衰减的掌握度，掌握度不过门槛就要重来。

我们现在有证据（回看点带原文），甚至有一张知识图谱（只是建完没人读），
但**没有掌握度、没有衰减、没有门槛**——所以它只能告诉你哪里疼，不能告诉你**好了没有**。

---

# 第二轮：第一批做完之后又翻出来的东西

第一批的 P0 已经落地（`cognitive/mastery.py` + 掌握度闭环 + `kb_chunks` 按页定位）。
再往下翻，还剩五处值得抄，按「能不能立刻用上」排序。

## A. 出题 + 判分：给掌握度补上真正的证据源 ★★★

**问题**：刚做完的掌握度只有一个证据入口——你自己点「这个讲明白了」。
按钮是主观的，点三次就 100%，那掌握度就退化成"心情记录"。

DeepTutor 的做法是**用题当证据**：

`deeptutor/reading/quiz.py:21-35`（中英双语系统提示，直接可用）：

```
只能使用提供的阅读上下文，不得编造事实、引用、页码或外部答案。
只返回 JSON：{"questions":[{"prompt","choices":[4条],"correct_choice_index","evidence"}]
返回恰好三道题。每题四个不同选项，且只有一个最佳答案。
evidence 只用于服务端校验，展示前会被移除。
```

三个细节值得抄：

1. **`evidence` 是必填字段**（`:44`，8–600 字）：每道题必须给出"支持正确答案的那句原文"。
   这是防"模型编题"的最低成本手段——编的题找不到出处，直接丢掉。
2. **选项去重**（`:46-54`）：`_normalise` 做 casefold + 空白折叠后比较，四个选项不许重复。
   我们自己踩过类似坑：棋盘式选项里两个"以上都对"就废了。
3. **引文回查**（`_grounding.py:56-66`）：

```python
def evidence_key(value: str) -> str:
    return _SPACING.sub("", _LINE_NUMBER.sub(" ", value.casefold()))
```

比较"这句是不是原文"时，先去掉**行号行、空格、连字符**——因为排版会把一句话写断
（`"fall short in\n3\ndelivering"`、`"DeepTu-\n4\ntor"`），这不是内容差异。
我们要做"引文必须能回查"，这就是现成的归一化函数，直接搬。

**判分的诚实边界**（`learning/semantic_assessment.py:27-37`）：

```python
def receipt(self):
    final = self.result != "ungraded" and self.confidence >= 0.8
    result = self.result if final else "ungraded"
```

模型判分置信度 < 0.8 → 一律记 `ungraded`，**不给分、不写掌握度**。
这条比"判个大概"重要：宁可这一次不算，也不能拿噪声训练掌握度。

> 落到我们：回看点 → 生成 3 道 4 选 1 → 你答 → 对/错写进 `kp_attempts`。
> 现在 `record_kp_attempt()` 已经收 `correct/source` 两个参数，判分接上去就能用。

## B. 结构化调用的一次重试规则 ★★

`deeptutor/services/llm/structured_retry.py:1-29` 把一类失败讲透了：

> 推理模型把隐藏思考的 token 从**同一个 `max_tokens` 预算**里扣。请求整块 JSON 时，
> 思考可能先把预算花光，一个字都还没吐。**provider 不报错**——它返回截断的 body 或空 body。

它的解法只有一条：**同一个模型、同一个问题，把推理档位调低再问一次**，把预算让给答案。
文件开头就写明"这是本项目对这类失败的唯一规则，各步骤不许自己长一套"。

我们三处各写了一套，而且已经踩过同一个坑：

| 位置 | 现在的做法 |
|---|---|
| `llm.py:31` | 有 `reasoning_only` 标记（发现了症状），`llm.py:147` 只是把错误文本甩给用户 |
| `screen.py:96` | `json.loads` 失败 → 自己兜底返回空 |
| `review.py:148` | 模型空返回 → `fallback_outline()` 给结构化提纲 |

三处症状一样、处理不一样。收成一个 `complete_json()`：正常问 → 解析失败 → **降档重问一次** → 再失败才降级。

## C. 待答问题队列 ★★

`deeptutor/learning/pending.py:1-8`：

> 持久化的 `PendingQuestion` 里带着**只有服务端持有的期望答案**；这个模块把它投影成
> 一份可以安全交给模型和客户端的更小契约。

对我们的意义不是"安全"，是**时机**：我们现在的疑问只能当场问，而我们的 ADR-004 是
「静默运行、课后交付」。听课途中把问题攒进队列（带着时间戳和上下文），下课集中处理，
比要求用户当场停下来打字更符合这个产品的前提。

同一个文件 `:53-60` 还讲了一个具体 bug：选项前缀识别。`"x - 1 = 0"` 会被
`^([A-Z])[.:：、)）-]` 匹配成"选项 X，内容 1 = 0"——**于是数学题选项全错位**。
他们要的是"每个选项都带前缀 + 第一个是 A"两个条件同时成立才算带标签。
做自测题时一定会踩。

## D. 覆盖度审计：不许把"抽到了"说成"全覆盖" ★

`deeptutor/services/rag/visual_coverage.py:1`：

> Inspect extraction limits without equating emitted assets with whole-book coverage.

一句话原则：**产出的资产数 ≠ 覆盖率**。它单独记录"哪些页的图没抽出来"。

落在我们身上是「补看回看点画面」：现在只对少数回看点抓了帧，界面没说"这一段没抓到"。
用户会以为是这里本来就没画面。加一句「本段 4 个回看点，抓到 2 个画面」，比多抓十倍帧更有用。

## E. 标注 → 成文笔记 ★

`deeptutor/reading/knowledge_capture.py:31-56`：把阅读标注按
「工作区 → 材料 → 标注」组织成 markdown，并且校验**"这份笔记的来源属于这个工作区"**。

我们手上的原料比它多（回看点 + 屏幕笔记 + 重讲结果 + 对话），但**没有一件能导出**。
这条给的是一个组织形状，不是算法。

---

## 第二轮优先级

| 级别 | 做的事 | 成本 | 为什么不先做别的 |
|---|---|---|---|
| **P0** | 回看点小测：3 道 4 选 1 + `evidence` 回查 + 置信度 <0.8 记 ungraded | 1 天 | 掌握度现在只有一个主观按钮当证据，等于没测 |
| **P1** | `complete_json()` 统一"降档重问一次" | 2 小时 | 三处重复逻辑，且都是静默失效 |
| **P1** | 待答问题队列（听课攒、课后答） | 半天 | 符合 ADR-004；现在只能当场打断 |
| **P2** | 覆盖度诚实报告 | 2 小时 | 界面从"看起来没问题"变"明确告诉你缺什么" |
| **P2** | 笔记导出成 markdown | 半天 | 用户手里终于有个能带走的东西 |
| **不做** | practice 的 xlsx/csv 题库导入（212 行） | — | 那是题库平台的活，我们是单机助手 |
| **不做** | `services/session`（12k 行）、`partners`（6.4k 行）、`partner_groups` | — | 多租户与 IM 渠道，与端侧单机无关 |

## 第二轮的总结

第一批补的是**闭环**（回看点能不能熄灭）；第二批补的是**证据的质量**：

> 掌握度不能靠"你点了三次"算出来，得靠"你答对了三次"算出来。

前者是自评，后者是实测。中间差的这一层，就是出题 + 判分。
---

# 第二轮执行结果（已落地）

| 项 | 状态 | 落地位置 | 验证 |
|---|---|---|---|
| **A 出题 + 判分** | ✅ 已做 | `dsh_app/quiz.py`（190 行）+ `Engine.make_quiz / answer_quiz` + 「小测与待答」页 | `tests/test_quiz_queue_export.py` 9 条；`docs/screenshots/20-quiz.png` |
| **B 结构化重试** | ✅ 已做 | `llm.call_with_retry / complete_json / complete_text`；`screen.py`、`review.py` 改走它 | 3 条测试（重问一次、缺字段、未配置快速失败） |
| **C 待答问题队列** | ✅ 已做 | `pending_questions` 表 + `Engine.queue_question / answer_question / answer_all_questions` + 同页下半区 | 3 条测试；`21-queue-answered.png` |
| **D 覆盖度审计** | ✅ 已做 | `Engine.capture_coverage` + 时间轴页提示 + 导出里同一行 | 2 条测试；`22-coverage.png` |
| **E 笔记导出** | ✅ 已做 | `Engine.export_notes` + 工具栏「导出笔记」 | 2 条测试；UI 自检断言四个小节 |

## 与 DeepTutor 原实现的差别（以及为什么）

1. **判分不用模型**。它的 `semantic_assessment` 是开放式判分，所以要 `confidence >= 0.8 → ungraded`。
   我们做 4 选 1，判分是二值，就没有"模型判不准"这个失败面。
   **借用的是它的思路（不确定就不记分），不是它的实现（模型打分）。**
2. **作废不丢弃**。它把校验不过的题丢掉；我们把题留下并标黄：让人看见模型编过一道题。
   丢掉会让"出题质量"这个指标永远是满分。
3. **选项前缀剥离加了防护**。它记录过 `"x - 1 = 0"` 被误认成「选项 X」的 bug，
   我们的 `strip_option_prefixes` 要求**整组都带前缀且第一个是 A** 才剥。
4. **重试只降档、不换模型**。它是同一模型降 reasoning effort；我们对应地在提示词里
   加「别推理」并把 `max_tokens` 放宽 1.5 倍——OpenAI 兼容端点不保证支持 `reasoning_effort` 参数，
   加参数反而会 400。

## 顺手修掉的既有问题

- `tools_ui_check.py` / `tools_smoke.py` 的 `ROOT` 多退了一层父目录，**产物一直写在盘符根目录**
  （`F:\data\…`）。已改成 `Path(__file__).resolve().parent`，并把误写目录清掉。
- `_current_course_id` 用了 `QListWidget.currentData()`（那是 `QComboBox` 的 API），
  异常被 Qt 定时器回调吞掉，界面表现只是「知识库空了」。已修 + 加测试钉住。

## 还没做的（留着）

- ~~**遗忘曲线**~~：已做（第三轮），一维 FSRS 简化 + 复习队列 + 下一步建议。
- ~~**硬门槛路由**~~：已做（第四轮），门槛成了闸门，重讲策略由证据强制（见文末第四轮）。
- ~~**知识图谱喂学习顺序**~~：已做（第四轮），边判前置，下一步先补没有被挡住的。
- ~~**查询改写 fail-open**~~、~~**索引签名**~~：已做（第二轮 P1）。

## 第三轮（P2 收尾）：复习排程

| 项 | 状态 | 落地 | 验证 |
|---|---|---|---|
| 遗忘曲线（一维 FSRS 简化） | ✅ | `cognitive/mastery.py` 下半部分：`retrievability / next_stability / due_in_days` | 11 条测试（曲线、边界、增长/收缩、封顶） |
| 复习队列 | ✅ | `Engine.review_queue`，`knowledge_points` 加 `stability / due_at` | 「该复习」排序 + 时间旅行测试 |
| 下一步建议（硬门槛路由） | ✅ | `Engine.next_action`：该复习 → 没过门槛 → 开新节 | 三种分支各一条测试 |
| 界面 | ✅ | 漏洞地图加「记忆保持」列、该复习标黄并排最前；小测页加「下一步做什么」 | `23-review.png` / `24-next-action.png` |

与 DeepTutor `learning/mastery.py` 的 DSR 相比，我们只留了稳定性一个维度：
单用户单机没有难度分布可估，多留一个参数只会多一个没人调的旋钮。
常数沿用 FSRS 默认值，好处是"稳定度 S 天时留存正好 90%"，S 可以直接当间隔用。

## 第四轮：知识点这一层本来是空的

这一轮的起点不是"加功能"，是**发现已经做的东西是假的**：
掌握度、小测、复习、复习队列全都挂在"知识点"上，而抽出来的知识点是
`定义 / 必要条件 / 边界条件 / 充分条件 / 最优解`。
这不是抽错了词，是取错了层——那套词表的作用是点火，不是命名。

所以这一轮先修地基，再谈策略。

| 项 | 状态 | 落地 | 验证 |
|---|---|---|---|
| 术语抽取分层（元词不再当知识点） | ✅ | 新模块 `terms.py`：定义句式 + 术语后缀（左边界校验）+ 学科词表；`features.BUILTIN_TERMS` 只留作密度点火 | `tests/test_terms.py`（22 条：元词出局、窗口垃圾、单字术语、子串去重、模型提名验真） |
| 模型只提名、代码验真 | ✅ | `learn_terms()`：每个提名必须 `grounded()` 于原文，否则丢弃计数；任何异常退回规则路径；来源写 `knowledge_points.origin` | 假模型说谎的两条用例 |
| 不重听音频就能重建 | ✅ | `Engine.rebuild_knowledge()` + `prune_orphan_points()`；漏洞地图「重建知识点」按钮；`tools_migrate_demo.py --apply` 迁移真实库 | 迁移脚本前后对比；`test_analysis_writes_real_terms_and_prunes_old_junk` |
| 硬门槛锁死重讲策略 | ✅ | `gate_lock / forced_strategy / _gate_strategy_for_knot`；`next_action` 顺序变成 replay → quiz → 复习 → 新节；`reword()` 取值顺序：显式 > 闸门 > 推荐 | `tests/test_gate_order.py`（15 条：状态机、策略四档、闸门消失） |
| 图谱的边喂学习顺序 | ✅ | `learning_order()`：严格先后定前置、插入式修正、`prereqs / blocked_by`；首次出现时间按转写真实计算 | 互为前置的反例、被挡住标记、前置补齐后解锁 |
| 练习导入 xlsx/csv | ✅ | 新模块 `exercises.py` + `exercises` 表 + `quiz_items.source`；小测页「导入练习 / 用练习出 3 道 / 练习模板」 | `tests/test_exercises_import.py`（18 条：模板往返、表头别名、答案三种写法、GBK、错误带行号、来源标记） |

顺手修掉的既有问题：

- **演示课的内容是空的**：`DemoASR` 的合成文本是"这节课我们来看一个现象"这类填充句，
  和它的标题（概率论与数理统计）无关——所以术语抽取怎么调都只能抽到垃圾。
  换成真正的概率论内容（样本空间/随机变量/分布函数/期望/方差/条件概率/泊松分布），
  同一套规则现在抽出 13 个真术语。演示数据不自洽，会把所有下游指标一起带偏。
- **详情面板显示旧值**：点「我会了」之后整表重画，选中行掉回第一行，右边详情还是上一个知识点
  （看着像"点了没生效"）。`refresh_kp_table(keep_kp_id=…)` 现在把选中行落回去并刷新详情。
- **截图脚本直接开用户库**：`tools_shot_mastery.py` 写的是 `data/app.db`，截图会污染真实数据；
  改为独立 probe 库 + 离线模型。另外页签写死编号（`setCurrentIndex(2)`）在加页之后错位，
  改成按标题定位。
- **测试会出网**：`seed_demo()` 现在会走术语抽取，本机配了 key 时每个用例都真的打云端接口
  （104 → 160 个用例时，耗时从 14 秒涨到 158 秒，结果还随网络波动）。
  `conftest.py` 加了自动置空的离线 fixture，需要读真实配置的两个用例用 `real_settings` 标记豁免。
- **依赖下载不落 C 盘**：装 openpyxl 时把 `PIP_CACHE_DIR` 指到 `F:\pip-cache`。

后来又补的两处（都是"跑起来才发现"）：

- **ASR 文本让抽取规则失效**：真实转写没标点、有口头语，第一版抽出 `呃然后分布函数`
  `个方差` `Fx等于PX小于等于x`。修法：边界剥除交替做到不动、公式判死、后缀候选必须比后缀长。
  现在同一段 ASR 文本和干净文本抽出同一组 8 个词。
- **垃圾知识点造成沉默的死锁**：「必要条件」有历史证据所以删不掉，又没有回看点所以没法
  重讲/出题，却永远挡在门槛前。加 `retired=1`：证据保留、退出决策，界面明说「N 个已退役」。

## 第五轮：把"别忘了点录制"这件事去掉

DeepTutor 那边没有这一层（它是纯文件/纯对话），所以这轮是**从用户动作反推**出来的：
方案写"录屏 + 转写"，但真实场景里最先失效的是"点开始录制"这个动作本身。

| 项 | 状态 | 落地 | 验证 |
|---|---|---|---|
| 网页视频自动起停 | ✅ | `webvideo.py`：音频会话探针 + 迟滞状态机（2.5s 起、25s 停）+ 电平复核 | 29 条时序/判据用例（假探针 + 假录音机，不碰声卡） |
| 跨站点（不装插件） | ✅ | 判据在 OS 层（WASAPI 会话），浏览器进程白名单 | 机器实测：放音频时进程会话变 `state=1` |
| 会话归档 | ✅ | 按**站点**归课、页面标题当节名、`source='webvideo'`、记 `web_capture_stop` 事件 | 引擎胶水 3 条用例 |
| 自检入口 | ✅ | `run_app.py --probe-webvideo` | 输出「浏览器在放声音：msedge」即可确认可用 |

顺带修掉：**会话赖着 Active** 会录三小时空气 —— 停录改成"会话不活跃 **且** 电平安静"；
反过来会话短暂消失但还有声（切 tab）时继续录。

留下的不足（明说）：

1. 术语抽取对**口语化转写**的统一性已经解决（ASR 与干净文本同结果），但修剪字表是手工维护的：
   换一门课的口头习惯（"咱就是说""这个玩意儿叫"）可能要补表。
2. `learning_order` 只做一遍插入式修正，不是拓扑排序；边很密时（同一节里术语共现）
   前置关系会偏保守——宁可多拦一个，也不给错顺序。
3. 导入题没有难度估计，抽题只按"没做过"先来后到。
4. 网页视频自动抓只拿得到声音：同一浏览器里另一个出声页面会被一起录进去
   （音频会话是进程级的）。要 tab 级隔离就得回到浏览器扩展路线，暂时不做。
