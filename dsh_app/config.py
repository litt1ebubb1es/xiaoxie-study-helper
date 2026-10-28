"""配置层：设置模型、模型注册表、路径。

对应方案 4.3 技术选型 / 4.4 模型分工 / 4.6 统一接入层。

设计要点（对方案缺口的裁决，见 docs/算法说明.md）：
- 所有后端一律「OpenAI 兼容」，仅 base_url 与 model 名不同（方案 4.6 协议统一）。
- 本地与服务端协议等价，上层逻辑不感知差异（方案 4.6 本地与云端等价）。
- 能力探测与降级状态是配置的一等公民（方案 4.6 能力探测 / 降级策略）。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

APP_NAME = "小蟹学习助手"
APP_ID = "knot-driven-course-assistant"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DSHK_DATA", ROOT / "data"))
CONFIG_PATH = DATA_DIR / "config.json"
DB_PATH = DATA_DIR / "app.db"
AUDIO_DIR = DATA_DIR / "audio"
EXPORT_DIR = DATA_DIR / "export"
DEPLOY_DIR = DATA_DIR / "deploy"
LOG_DIR = DATA_DIR / "logs"


def ensure_dirs() -> None:
    for d in (DATA_DIR, AUDIO_DIR, EXPORT_DIR, DEPLOY_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# 模型注册表：供「检测 + 一键部署」使用
# --------------------------------------------------------------------------
# 每个条目描述一个方案里点名过的模型。注意：本文件不假设模型一定存在，
# 检测结果以「本机磁盘 / 本机端口 / 本机显存」为准，registry 只提供"去哪找、怎么起"。

LOCAL_MODEL_REGISTRY: list[dict[str, Any]] = [
    {
        "key": "qwen3-asr",
        "role": "realtime_asr",
        "name": "Qwen3-ASR-1.7B",
        "plan_ref": "4.4 实时转写",
        "size_gb": 2.0,
        "min_mem_gb": 4,
        "modelscope_id": "Qwen/Qwen3-ASR-1.7B",
        "serve": "vllm",
        "port": 8001,
        "spec_claims_license": "Apache 2.0",
    },
    {
        "key": "sensevoice-small",
        "role": "fallback_asr",
        "name": "SenseVoice-Small",
        "plan_ref": "4.4 轻量兜底转写",
        "size_gb": 0.5,
        "min_mem_gb": 2,
        "modelscope_id": "iic/SenseVoiceSmall",
        "serve": "funasr",
        "port": 8011,
        "spec_claims_license": "Apache 2.0",
    },
    {
        "key": "fireredasr2",
        "role": "precise_asr",
        "name": "FireRedASR2",
        "plan_ref": "4.4 课后精转",
        "size_gb": 5.0,
        "min_mem_gb": 8,
        "modelscope_id": "FireRedTeam/FireRedASR2-AED",
        "serve": "funasr",
        "port": 8012,
        "spec_claims_license": "Apache 2.0",
    },
    {
        "key": "qwen3-9b",
        "role": "batch_llm",
        "name": "Qwen3.5-9B（本地占位：按实际可得版本替换）",
        "plan_ref": "4.4 认知负荷辅助",
        "size_gb": 6.0,
        "min_mem_gb": 8,
        "modelscope_id": "Qwen/Qwen3-8B-GGUF",
        "serve": "llama.cpp",
        "port": 8080,
        "spec_claims_license": "Apache 2.0",
    },
    {
        "key": "qwen3-122b",
        "role": "main_brain",
        "name": "Qwen3.5-122B-A10B（按需加载主脑）",
        "plan_ref": "4.4 重讲与诊断主脑",
        "size_gb": 75.0,
        "min_mem_gb": 96,
        "modelscope_id": "Qwen/Qwen3-235B-A22B-GGUF",
        "serve": "vllm",
        "port": 8000,
        "spec_claims_license": "Apache 2.0",
    },
]

CLOUD_PRESETS: list[dict] = [
    # 每个预设只负责三件事：地址、候选模型、去哪拿 key。
    # 候选是"建议清单"，真正可用的一定以 /models 返回为准（界面上点「刷新列表」）。
    {
        "key": "dashscope",
        "label": "阿里云百炼 DashScope（通义千问）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "vision_model": "qwen-vl-max",
        "asr_model": "paraformer-v2",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long",
                   "qwen3-max", "deepseek-v3", "deepseek-r1"],
        "vision_models": ["qwen-vl-max", "qwen-vl-plus", "qwen2.5-vl-72b-instruct"],
        "key_url": "https://bailian.console.aliyun.com/",
        "needs_key": True,
        "hint": "控制台 → 右上角「API-KEY」→ 创建。key 只存在本机 data/config.json。",
    },
    {
        "key": "deepseek",
        "label": "DeepSeek 官方",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "vision_model": "",
        "asr_model": "",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "vision_models": [],
        "key_url": "https://platform.deepseek.com/api_keys",
        "needs_key": True,
        "hint": "官方目前只有文本模型，屏幕识别需要另配视觉模型（可换别家）。",
    },
    {
        "key": "zhipu",
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "vision_model": "glm-4v-plus",
        "asr_model": "",
        "models": ["glm-4-plus", "glm-4-air", "glm-4-flash", "glm-4.5", "glm-4.5-air"],
        "vision_models": ["glm-4v-plus", "glm-4v-flash", "glm-4.5v"],
        "key_url": "https://open.bigmodel.cn/usercenter/apikeys",
        "needs_key": True,
        "hint": "glm-4v 系列可以直接看图，课程截图识别用得上。",
    },
    {
        "key": "moonshot",
        "label": "月之暗面 Kimi",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-latest",
        "vision_model": "moonshot-v1-8k-vision-preview",
        "asr_model": "",
        "models": ["kimi-latest", "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "vision_models": ["moonshot-v1-8k-vision-preview", "moonshot-v1-32k-vision-preview"],
        "key_url": "https://platform.moonshot.cn/console/api-keys",
        "needs_key": True,
        "hint": "长文本课材料适合 128k 那档。",
    },
    {
        "key": "siliconflow",
        "label": "硅基流动 SiliconFlow（国产开源模型）",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "deepseek-ai/DeepSeek-V3",
        "vision_model": "Qwen/Qwen2-VL-72B-Instruct",
        "asr_model": "",
        "models": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
                   "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen3-235B-A22B"],
        "vision_models": ["Qwen/Qwen2-VL-72B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct"],
        "key_url": "https://cloud.siliconflow.cn/account/ak",
        "needs_key": True,
        "hint": "开源模型便宜，模型名带斜杠是正常写法。",
    },
    {
        "key": "openai",
        "label": "OpenAI 官方",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "vision_model": "gpt-4o",
        "asr_model": "whisper-1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o4-mini"],
        "vision_models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
        "key_url": "https://platform.openai.com/api-keys",
        "needs_key": True,
        "hint": "国内需要能连通的网络环境。",
    },
    {
        "key": "ollama",
        "label": "Ollama（本机跑，不用 key）",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "vision_model": "qwen2.5vl:7b",
        "asr_model": "",
        "models": ["qwen2.5:7b", "qwen2.5:14b", "llama3.2:3b", "gemma3:4b"],
        "vision_models": ["qwen2.5vl:7b", "llava:7b", "gemma3:4b"],
        "key_url": "",
        "needs_key": False,
        "hint": "先在命令行 ollama pull qwen2.5:7b，再点「刷新列表」看本机到底有哪些。",
    },
    {
        "key": "lmstudio",
        "label": "LM Studio（本机跑，不用 key）",
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "",
        "vision_model": "",
        "asr_model": "",
        "models": [],
        "vision_models": [],
        "key_url": "",
        "needs_key": False,
        "hint": "在 LM Studio 里打开本地服务器后点「刷新列表」，模型名由它自己报。",
    },
    {
        "key": "custom",
        "label": "自定义 / 其他 OpenAI 兼容端点",
        "base_url": "",
        "model": "",
        "vision_model": "",
        "asr_model": "",
        "models": [],
        "vision_models": [],
        "key_url": "",
        "needs_key": True,
        "hint": "填自己的地址（例如公司网关、云厂商专用域名），再点「刷新列表」。",
    },
]


def provider(key: str) -> dict:
    """按 key 取预设；取不到返回自定义那一项，绝不抛异常。"""
    for p in CLOUD_PRESETS:
        if p["key"] == key:
            return p
    return CLOUD_PRESETS[-1]


# 判断一个模型名是不是"能看图"：各家的命名习惯不一样，这里只做粗筛，
# 真正的结论由探测（能对话 + 能返回内容）决定，不靠名字硬猜。
_VISION_HINTS = ("vl", "vision", "omni", "multimodal", "4v", "2-vl", "llava", "gemma3")


def looks_vision(model_id: str) -> bool:
    low = (model_id or "").lower()
    return any(h in low for h in _VISION_HINTS)


def vision_of(models: list[str], preset_vision: list[str] | None = None) -> list[str]:
    """给视觉下拉挑候选：先放预设里明确的视觉模型，再放列表里看着像的。"""
    out: list[str] = []
    for m in list(preset_vision or []) + [x for x in models if looks_vision(x)]:
        if m and m not in out:
            out.append(m)
    return out


# --------------------------------------------------------------------------
# 设置
# --------------------------------------------------------------------------
@dataclass
@dataclass
class AutoCapSettings:
    """自动抓屏（算法决定时机，不需要快捷键）。默认关闭：屏幕内容涉及隐私。"""

    enabled: bool = False
    interval_s: float = 8.0
    change_threshold: float = 0.12
    min_gap_s: float = 30.0
    max_per_hour: int = 40


@dataclass
class LLMSettings:
    """生成层：统一 OpenAI 兼容接入（方案 4.6）。"""

    backend: str = "cloud"          # cloud | local | off
    preset: str = "dashscope"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = ""
    model: str = "qwen-plus"
    vision_model: str = "qwen-vl-max"
    temperature: float = 0.3
    timeout_s: float = 120.0
    preload_on_start: bool = True   # 用户裁决：启动时选择并预热，课上不冷启


@dataclass
class SourceSettings:
    """链接导入：B 站字幕需要登录 Cookie（SESSDATA）。只存在本机配置文件里。"""

    bili_sessdata: str = ""
    prefer: str = "auto"          # auto | subtitle | audio


@dataclass
class ASRSettings:
    """感知层：转写后端。first/second pass 分开配置（方案 5.1 双遍转写）。"""

    backend: str = "auto"           # auto | cloud | local_openaicompat | offline
    base_url: str = ""
    api_key: str = ""
    realtime_model: str = "paraformer-realtime-v2"
    precise_model: str = "paraformer-v2"
    language: str = "zh"
    offline_seed: int = 20261001    # 离线演示后端的确定性种子
    # 本地 faster-whisper：尺寸与设备（auto=有 CUDA 就试 GPU，失败回落 CPU）
    local_size: str = "small"
    local_device: str = "auto"      # auto | cuda | cpu
    # 默认关闭：没有转写后端时宁可报错，也不能用合成文本假装"转写成功"。
    allow_demo: bool = False


@dataclass
class CaptureSettings:
    """采集层：系统回环（方案 4.2）。"""

    device_name: str = ""           # 空 = 用默认 WASAPI 回环
    sample_rate: int = 16000
    channels: int = 2
    block_ms: int = 40
    save_raw_audio: bool = True     # 方案 7.3：原始音频可配置为课后删除
    keep_audio_days: int = 7        # 0 = 永不自动清理
    # 课后把 44.1k 立体声 WAV 压成 16k 单声道 FLAC：转写只要 16k，
    # 一节 90 分钟的课从 890 MB 降到 ~95 MB。压完仍然能重新转写、重新分析。
    audio_compress: bool = True
    audio_flac_rate: int = 16000
    # 不看屏幕也能抓的两类行为：声音重复=回退；声音断掉=暂停
    detect_pauses: bool = True
    pause_min_s: float = 3.0        # 短于这个的静音当换气，不算暂停
    pause_long_s: float = 20.0      # 超过这个算"停在那儿很久"
    # 网页视频自动抓取（不靠插件、不靠站点 API，按浏览器音频会话判定）
    auto_web_capture: bool = False  # 默认关：录制是敏感动作，要用户明确打开
    web_start_after_s: float = 2.5  # 连续出声这么久才开始录，躲开"点开就关"
    web_stop_after_s: float = 25.0  # 安静这么久才停：课中间暂停、缓冲都算
    web_browsers: str = ""          # 空 = 用内置浏览器进程名表
    web_min_keep_s: float = 8.0     # 自动抓取短于这个秒数就当误触发丢掉


@dataclass
class CognitiveSettings:
    """认知层参数。所有缺省值对应 docs/算法说明.md 里的裁决记录。"""

    window_s: float = 60.0
    step_s: float = 15.0
    # 五个负荷特征的权重（方案 5.2.1）
    w_term: float = 0.35            # TD 术语密度
    w_speed: float = 0.25           # SS 语速偏移
    w_sentence: float = 0.15        # SL 句长
    w_symbol: float = 0.15          # SD 符号密度
    w_pause: float = 0.10           # PS 停顿结构
    # 缺口裁决：sigmoid 直接吃 z-score 加权和会饱和，加缩放温度
    tau: float = 0.5
    bias: float = 0.0
    # 缺口裁决：象限阈值不用绝对 0.5，改用课程内分位数，避免"半数窗口都是高负荷"
    quadrant_percentile: float = 70.0
    # 行为信号时间衰减（方案 5.2.2 的"越靠近窗口中心权重越高"）
    decay_sigma_ratio: float = 1.0 / 3.0   # σ = window_s * ratio
    # 融合（方案 5.2.4）
    w_load: float = 0.60
    w_behavior: float = 0.40
    grade_strong: float = 0.65
    grade_medium: float = 0.45
    # 缺口裁决：回看点起止时间与合并规则
    merge_gap_s: float = 30.0
    knot_half_width_s: float = 25.0
    # 缺口裁决：校准的负样本与学习率
    eta: float = 0.05
    eta_half_below: int = 20
    eta_decay_above: int = 100
    hard_negative_weight: float = 0.3
    # 实时分析（用户裁决：听课中可以做分析）
    live_enabled: bool = True
    live_min_s: float = 20.0


@dataclass
class UISettings:
    theme: str = "dark"
    font_size: int = 13
    show_personal_quadrant: bool = False
    timeline_height: int = 190


@dataclass
class Settings:
    llm: LLMSettings = field(default_factory=LLMSettings)
    autocap: AutoCapSettings = field(default_factory=AutoCapSettings)
    source: SourceSettings = field(default_factory=SourceSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    capture: CaptureSettings = field(default_factory=CaptureSettings)
    cognitive: CognitiveSettings = field(default_factory=CognitiveSettings)
    ui: UISettings = field(default_factory=UISettings)

    # ---- 持久化 -------------------------------------------------------
    def __post_init__(self) -> None:
        # 不放进 dataclass 字段里，否则会被 asdict() 写进 JSON
        self._path = CONFIG_PATH

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        p = path or CONFIG_PATH
        s = cls()
        s._path = p
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
            for f in fields(cls):
                sub = raw.get(f.name)
                if isinstance(sub, dict):
                    target = getattr(s, f.name)
                    for k, v in sub.items():
                        if hasattr(target, k):
                            setattr(target, k, v)
        s.apply_env_overrides()
        return s

    def apply_env_overrides(self) -> None:
        """环境变量优先，便于不写盘切换后端（CI / 演示）。"""
        if v := os.environ.get("DSHK_LLM_BASE_URL"):
            self.llm.base_url = v
        if v := os.environ.get("DSHK_LLM_API_KEY"):
            self.llm.api_key = v
        if v := os.environ.get("DSHK_LLM_MODEL"):
            self.llm.model = v
        if v := os.environ.get("DSHK_ASR_BACKEND"):
            self.asr.backend = v
        if v := os.environ.get("DSHK_ASR_BASE_URL"):
            self.asr.base_url = v
        if v := os.environ.get("DSHK_ASR_API_KEY"):
            self.asr.api_key = v
        if v := os.environ.get("DSHK_LLM_BACKEND"):
            self.llm.backend = v
        if v := os.environ.get("DSHK_VISION_MODEL"):
            self.llm.vision_model = v
        # 常见供应商环境变量兜底：机器上已经配过就直接用，不必再粘一次 key。
        # 只在这几项为空时兜底，不覆盖用户显式写在配置里的值。
        if not self.llm.api_key:
            for name in ("QW_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY",
                         "DEEPSEEK_API_KEY"):
                if v := (os.environ.get(name) or "").strip():
                    self.llm.api_key = v
                    break
        if not self.llm.base_url:
            for name in ("QW_BASE_URL", "DASHSCOPE_BASE_URL", "OPENAI_BASE_URL"):
                if v := (os.environ.get(name) or "").strip():
                    self.llm.base_url = v
                    break

    def save(self, path: Path | None = None) -> None:
        p = path or getattr(self, "_path", None) or CONFIG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 便捷 ---------------------------------------------------------
    def has_cloud_llm(self) -> bool:
        return bool(self.llm.api_key and self.llm.base_url and self.llm.backend != "off")

    def effective_llm_base(self) -> str:
        if self.llm.backend == "local" and self.llm.preset == "custom":
            return self.llm.base_url
        return self.llm.base_url


_CJK = re.compile(r"[\u4e00-\u9fff]")


def has_cjk(text: str) -> bool:
    return bool(_CJK.search(text or ""))