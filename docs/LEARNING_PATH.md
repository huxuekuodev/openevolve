# OpenEvolve 学习路径

> 一份按依赖顺序组织的源码阅读指南。目标不是"读完所有文件"，而是**能自己改一个进化策略、加一个 LLM 后端、或者排查一次跑不出结果的进化**。
>
> 阅读顺序按**数据流**排，而不是按目录字母序 —— 这样每一步你都清楚"这个模块的输入从哪来、输出给谁"。

---

## 0. 先建立一张心智地图

### 这个项目在做什么（30 秒版）

OpenEvolve 是 Google DeepMind **AlphaEvolve** 的开源实现。它把"写代码"变成一个**进化搜索**问题：

```
初始程序 ──► LLM 改写 ──► 评测打分 ──► 存进"多样性档案" ──► 挑出下一批父本 ──► 循环
   ▲                                                                          │
   └──────────────────────────────────────────────────────────────────────────┘
```

三个关键设计让它比"让 LLM 反复重写同一个文件"强得多：

| 设计 | 解决的问题 | 代码位置 |
|---|---|---|
| **MAP-Elites 特征网格** | 只留最好的那个，会丢掉"不同思路但分数稍低"的解 | `database.py` |
| **岛屿模型（islands）** | 单一族群会过早收敛到局部最优 | `database.py` + `process_parallel.py` |
| **双选择（double selection）** | 给 LLM 看的例子 ≠ 被改写的父本，兼顾"稳妥"与"探索" | `prompt/sampler.py` + `database.py` |

### 一次迭代的完整数据流

这是**最该先搞懂的一张图**。建议对着 `process_parallel.py::_run_iteration_worker`（约 165 行起）逐行看一遍：

```
控制进程 (controller.py)
  │
  ├─ database.sample_from_island(island_id, num_inspirations)
  │     └─► (parent, inspirations)          ← 双选择的"父本"和"灵感"
  │
  ├─ _create_database_snapshot()            ← 把整个数据库序列化成 dict
  │
  └─ ProcessPoolExecutor.submit(_run_iteration_worker, snapshot, parent_id, inspiration_ids)
        │
        ▼  ──────────── 子进程（真正的并行单元）────────────
        │
        ├─ _lazy_init_worker_components()   ← 懒加载 Evaluator / LLMEnsemble / PromptSampler
        │
        ├─ prompt_sampler.build_prompt(...) ← 组装提示词（含历史尝试、指标、artifacts）
        │
        ├─ llm_ensemble.generate_with_context(...)   ← 按权重随机选一个模型调用
        │
        ├─ 解析 LLM 输出：
        │    ├─ diff_based_evolution=true  → extract_diffs / apply_diff_blocks
        │    └─ diff_based_evolution=false → parse_full_rewrite
        │
        ├─ evaluator.evaluate_program(child_code, child_id)   ← 级联评测 1→2→3 阶段
        │
        └─ 返回 SerializableResult(child_program_dict, prompt, llm_response, artifacts, ...)
        │
        ▼  ──────────── 回到控制进程 ────────────
        │
        └─ database.add(child_program)      ← 特征分箱 + 岛屿准入 + 迁移 + 存档
              └─► 下一次迭代
```

**读完这一张图，你就掌握了 80% 的骨架。** 剩下的都是这张图上的某个方框的内部细节。

---

## 1. 前置知识

按重要程度排序：

| 主题 | 需要到什么程度 | 为什么需要 |
|---|---|---|
| Python `dataclass` / `typing` | 熟练 | 整个项目重度使用 dataclass 传配置和状态 |
| `multiprocessing` + `ProcessPoolExecutor` | 会用 | 并行是性能关键，`process_parallel.py` 1113 行都在讲这件事 |
| `asyncio` | 会用 | LLM 调用是 async，但 worker 里用 `asyncio.run()` 包成同步 |
| MAP-Elites 算法 | 理解概念即可 | 不需要会推导，但要理解"特征网格 + 每个格子留最优" |
| 遗传算法基础 | 理解概念即可 | 变异、选择、迁移、精英保留 |
| LLM Prompt 工程 | 实践经验 | 这个项目的"质量上限"很大程度取决于提示词 |

**不需要**事先掌握的：AlphaEvolve 论文、Rust/其他语言、Flask（只有 `api.py` 用到一点）。

---

## 2. 分阶段学习路径

每个阶段都标注了**预计时间**、**要读的文件**、**要回答的问题**、**动手练习**。

---

### 阶段 0：先跑起来（约 1 小时）

**不要一上来就读源码。** 先看它动起来，建立直觉。

```bash
# 1. 安装（本仓库已配好 .venv，可直接用）
pip install -e ".[dev]"          # 或 uv sync

# 2. 配一个 LLM。最快的是 Google Gemini 免费额度：
export OPENAI_API_KEY="你的 Gemini API Key"

# 3. 跑最小的例子（10 次迭代，几分钟）
python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config examples/function_minimization/config.yaml \
  --iterations 10

# 4. 看进化树
python scripts/visualizer.py \
  --path examples/function_minimization/openevolve_output/checkpoints/checkpoint_5/
```

**要观察的**：终端里每次迭代输出的分数变化、`openevolve_output/` 下生成了什么、可视化页面里树是怎么分叉的。

**要回答的问题**：
- `initial_program.py` 里 `# EVOLVE-BLOCK-START/END` 是什么？为什么要有它？
- `evaluator.py` 返回值里 `combined_score` 为什么必须有？

**必读**：`README.md` 的 Quick Start + `examples/README.md`（特别是 "Common Configuration Mistakes" 一节）。

---

### 阶段 1：一次迭代的解剖（约 3–4 小时）⭐ 最重要

**这是投入产出比最高的阶段。**

**读的顺序**（严格按此顺序，每一步都建立在上一步之上）：

1. **`openevolve/config.py`（547 行）** —— 先看数据结构，再看逻辑。
   - 重点：`LLMModelConfig` / `LLMConfig` / `PromptConfig` / `DatabaseConfig` / `EvaluatorConfig` / `Config`
   - **关键机制**：`LLMConfig.__post_init__` 里的 `update_model_params(shared_config)` —— 顶层 `llm.xxx` 配置会**向下填充**到 `llm.models[]` 里每个模型（只在字段为 `None` 时才填）。这是理解"为什么 YAML 里配一次 api_base 就够"的关键。
   - **动手**：`python -c "from openevolve.config import Config; c=Config.from_yaml('configs/default_config.yaml'); print(c.llm.models)"`

2. **`openevolve/process_parallel.py::_run_iteration_worker`（1113 行里的核心 250 行）**
   - 这是 worker 子进程的主函数，上面那张数据流图的全部实现。
   - **注意**：文件顶部的 `_worker_config` / `_worker_evaluator` 等是**每个子进程独立**的全局变量，由 `ProcessPoolExecutor(initializer=_worker_init)` 在每个进程启动时填充。这是跨进程状态传递的核心技巧。
   - **动手**：给 `_run_iteration_worker` 里加 `logger.info(...)`，观察子进程日志怎么回到主进程。

3. **`openevolve/controller.py`（579 行）**
   - 看 `run()` → `_run_evolution_with_checkpoints()` 的调用链。
   - 重点：**初始程序的处理**（iteration 0 单独评测一次）、**文件后缀/语言的自动推断**、**检查点回调**。
   - **关键细节**：`run_evolution` 返回 `Optional[Program]`，`None` 表示一次都没评测成功。

4. **`openevolve/evaluator.py`（752 行）的 `evaluate_program`**
   - **级联评测（cascade evaluation）**：阶段 1 快速验证 → 阶段 2 基础性能 → 阶段 3 完整评测，每阶段有阈值，不达标就提前返回。
   - **这是省钱的机制**：坏程序在第 1 阶段就被淘汰，不会浪费第 3 阶段的算力。
   - 重点看 `_validate_cascade_configuration`：它会在启动时检查你的 evaluator 有没有定义 `evaluate_stage1/2/3`，避免跑了一半才发现配置错了。

**要回答的问题**：
- 为什么 worker 里要传 `db_snapshot`（数据库快照）而不是共享数据库对象？
- `evaluator` 和 `llm_evaluator_ensemble` 有什么区别？（一个是执行评测的，一个是"LLM 当评委"的）
- 如果 LLM 返回的内容完全无法解析，会发生什么？会不会让整个进化崩掉？

**练习**：把 `diff_based_evolution` 设成 `false` 再跑一次，对比 prompt 内容和结果质量。

---

### 阶段 2：数据库 / MAP-Elites（约 4–6 小时）

`database.py` 有 **2840 行**，是最大的文件。**不要通读**，按下面的顺序读。

**第一遍——只读这 4 个方法**：
1. `add(program)` —— 准入流程：特征分箱 → 找目标格子 → 比较 → 替换或丢弃
2. `sample_from_island(island_id, num_inspirations)` —— 采样：精英 + 探索
3. `_calculate_feature_coords(program)` —— 特征向量怎么变成网格坐标
4. `migrate_programs()` —— 岛屿间迁移

**第二遍——理解这些概念**：

| 概念 | 一句话解释 | 配置项 |
|---|---|---|
| **Feature grid** | 把程序按 N 个特征维度分箱，每箱留一个最优 | `feature_dimensions` |
| **Cell replacement** | 新程序要么替换格子里的旧程序，要么被丢弃 | `elite_selection_ratio` |
| **Archive** | 全局"历史最佳"档案，独立于岛屿 | `archive_size` |
| **Island** | 独立演化的子种群 | `num_islands` |
| **Migration** | 定期在岛屿间搬运程序，防止各自收敛 | `migration_interval` / `migration_rate` |
| **Lazy migration** | 按"代数"而不是"迭代数"触发迁移 | `migration_interval` |
| **Novelty** | 用 embedding 余弦相似度判断"是不是换汤不换药" | `similarity_threshold` |

**⚠️ 最常见的坑**：`feature_dimensions` 必须返回**原始连续值**（比如 `latency_ms: 234.5`），**不能**返回预先算好的箱号（比如 `bin: 7`）。数据库内部会做 min-max 缩放。

```python
# ✅ 对
return {"combined_score": 0.85, "prompt_length": 1247, "execution_time": 0.234}
# ❌ 错（会被数据库二次分箱，变得毫无意义）
return {"combined_score": 0.85, "prompt_length": 7, "execution_time": 3}
```

**要读的辅助文件**：
- `configs/README.md` —— 岛屿参数的调参指南
- `openevolve/population.py`（67 行，很短）—— **策略钩子**：`admit` / `replace_cell` / `archive` / `evict` / `migrate`，想改准入逻辑从这里入手，不用动 `database.py`

**练习**：
1. 写一个自定义 `PopulationStrategy`，让 `admit` 拒绝所有 `generation < 3` 的程序，观察种群变化。
2. 把 `num_islands` 从 3 改成 1，对比最终分数和收敛速度。

---

### 阶段 3：Prompt 工程（约 2–3 小时）

**`openevolve/prompt/sampler.py`（751 行）+ `templates.py`（236 行）**

这是决定"LLM 改得好不好"的地方。

**核心问题**：一个提示词里要塞进哪些信息？
- 当前程序代码
- 当前程序的评测指标
- **历史尝试**（哪些改法失败了、怎么失败的） ← 这是 AlphaEvolve 的关键洞察
- **灵感程序**（来自其他岛屿或存档的不同思路）
- 程序产生的 artifacts（运行日志、错误信息）
- 特征维度说明

**要理解的机制**：
- **双选择**：`build_prompt` 的 `parent_program` 和 `inspirations` 来自不同的采样路径，前者求稳、后者求新。
- **模板覆盖**：`templates.py` 支持从文件系统加载自定义模板，可以不改代码就换提示词。
- **`programs_as_changes_description`**：把"程序"当成"变更描述的累积"来进化。适合超大代码库——LLM 只改描述，代码由描述生成。

**必做练习**：
1. 找到 `system_message` 的默认值，把它改写成针对你自己任务的版本，对比结果。
2. 打印一次完整的 prompt（`logger.setLevel(logging.DEBUG)`），看看 LLM 到底收到了什么。

---

### 阶段 4：并行与容错（约 3–4 小时）

**`openevolve/process_parallel.py`**

**核心机制**：
- 每个迭代在**全新子进程**里跑（不是线程池），因为 LLM 调用和评测都可能阻塞、可能泄漏内存。
- `max_tasks_per_child`（Python 3.11+）定期重启 worker，防止长期运行的内存膨胀。
- 数据库以**快照**形式传给 worker：子进程读的是某个时间点的副本，写入由主进程统一做，**避免并发写冲突**。
- **超时保护**：`evaluator.timeout + 30s` 的缓冲；超时的进程会被 `_terminate_process_pool` 清理。

**要读的细节**：
- `_wait_for_processes` —— 优雅关闭
- `_CandidateRejected` 异常 —— 准入被拒时的"正常失败"路径（不是错误）
- `IslandSelector`（在 `selection.py`，只有 27 行）—— 自定义岛屿调度策略的钩子

**练习**：写一个自定义 `IslandSelector`，总是选当前平均分最高的岛屿，观察是否加速收敛（或加速崩溃）。

---

### 阶段 5：可观测性（约 2 小时）

**三个层次**：

1. **Artifacts（`evaluator.py` + `database.py`）**
   - 评测函数可以返回辅助数据（日志、图表、中间结果），供**下一代** LLM 参考。
   - 小 artifact（默认 < 32KB，见 `config.py` 的 `artifact_size_threshold`）存数据库，大的存磁盘。
   - 看 `evaluation_result.py`（68 行）：`EvaluationResult(metrics, artifacts)`。
   - 例子：`examples/circle_packing_with_artifacts/`

2. **Evolution Trace（`evolution_trace.py` 626 行 + `utils/trace_export_utils.py`）**
   - 记录每次迭代的 `(state, action, reward)` 三元组，用于 RL 训练或分析。
   - 支持 JSONL / gzip JSONL / JSON / HDF5 导出。

3. **Visualizer（`scripts/visualizer.py`）**
   - Flask 应用，展示进化树、岛屿状态、指标曲线。

**练习**：打开 `evolution_trace`，跑 10 次迭代，导出 trace 并用 pandas 分析"哪种 LLM 响应被采纳了"。

---

### 阶段 6：扩展点（按需）

到这里你应该能按需求去读对应文件了：

| 想做的事 | 读哪个文件 |
|---|---|
| 接一个新的 LLM 服务 | `llm/openai.py`（OpenAI 兼容的都走这里）、`llm/base.py`（接口） |
| 接一个 CLI 型 LLM | `llm/claude_code.py`（162 行，最简模板）、`llm/copilot_cli.py` |
| 用多个模型集成 | `llm/ensemble.py`（142 行，按权重随机选） |
| 当 Python 库用（不用 CLI） | `api.py` —— `evolve_function` / `evolve_code` / `evolve_algorithm` / `run_evolution` |
| 人工审核模式 | `llm/openai.py` 的 `manual_mode`（写问题到队列目录，等人工回答文件） |
| 自定义进化准入/淘汰 | `population.py` 的 `PopulationStrategy` |
| 自定义岛屿调度 | `selection.py` 的 `IslandSelector` |

**库 API 快速示例**：

```python
from openevolve import evolve_function

def solve(x):
    return x * 2

result = evolve_function(solve, iterations=50)
print(result.best_score, result.best_code)
```

---

## 3. 关键概念速查表

| 术语 | 含义 | 在哪 |
|---|---|---|
| **EVOLVE-BLOCK** | 标记"允许 LLM 修改"的代码区域 | `utils/code_utils.py` |
| **combined_score** | 必须有！进化的主目标分数 | 你的 `evaluator.py` |
| **Cascade evaluation** | 分级评测，早淘汰省钱 | `evaluator.py` |
| **Island** | 独立演化的子种群 | `database.py` |
| **Migration** | 岛屿间搬运程序 | `database.py` |
| **Archive** | 全局最佳档案 | `database.py` |
| **Artifact** | 评测产生的辅助数据，喂给下一代 | `evaluation_result.py` |
| **Double selection** | 父本 ≠ 灵感来源 | `prompt/sampler.py` |
| **Diff-based evolution** | LLM 输出 SEARCH/REPLACE 补丁而非整文件 | `utils/code_utils.py` |
| **Changes description** | 把"变更描述"当进化单元（大代码库用） | `config.prompt` |
| **Novelty judge** | 用 embedding 判断是否真的不同 | `embedding.py`, `novelty_judge.py` |
| **Manual mode** | 人工在环，不用 API | `llm/openai.py` |

---

## 4. 工程实践：测试与类型检查

这部分是本仓库当前的状态（已完善）。

### 跑测试

```bash
make test              # 全套：单元 + 集成（无 LLM 服务时集成自动 skip）
make test-unit         # 只跑单元（快，无需 LLM）
make test-cov          # 单元 + 覆盖率报告
make test-unittest     # 旧版 unittest 发现器（保留兼容）

# 直接用 pytest
pytest                                    # 全套
pytest tests --ignore=tests/integration   # 只跑单元
pytest -m integration                     # 只跑集成
pytest tests/test_database.py -k island   # 单个文件/按名字过滤
pytest --cov=openevolve --cov-report=term-missing   # 看哪个模块没测到
```

### 跑类型检查

```bash
make typecheck     # 等价于 python -m mypy（配置在 pyproject.toml）
```

### 一条命令跑完 PR 门槛

```bash
make check         # typecheck + 单元测试
```

### 测试的组织方式

```
tests/
├── test_*.py              # ~70 个单元测试文件，按功能域划分
│   ├── test_database.py           # MAP-Elites、岛屿、迁移
│   ├── test_island_*.py           # 岛屿隔离/迁移/跟踪/父子一致性
│   ├── test_evaluator_timeout.py  # 评测超时与容错
│   ├── test_reasoning_model_params.py  # 推理模型的参数路由
│   ├── test_cli.py                # 命令行入口
│   └── ...
├── integration/
│   ├── conftest.py        # optillm 服务 fixture + integration 标记自动注入
│   └── test_*_with_llm.py # 真实 LLM 端到端（需要 optillm）
└── test_utils.py          # 共享测试工具
```

### 当前测试与覆盖率现状

```
883 passed, 20 skipped          # 全套（3 个 skip 是没装 h5py，17 个是没起 optillm）
859 passed,  3 skipped          # 仅单元测试
覆盖率 78%（单元测试口径）
```

**还有哪些模块覆盖偏低** —— 想练手写测试就从这里挑：

| 模块 | 覆盖率 | 主要缺口 |
|---|---:|---|
| `evaluator.py` | 63% | 级联评测的 stage2/3 分支、LLM 反馈解析 |
| `llm/openai.py` | 66% | `manual_mode` 的排队与轮询逻辑 |
| `process_parallel.py` | 71% | 进程池启停、超时终止、优雅退出 |
| `controller.py` | 71% | 信号处理、检查点保存/恢复的组合路径 |
| `prompt/sampler.py` | 74% | 提示词分支组合、模板文件加载 |
| `utils/trace_export_utils.py` | 62% | HDF5 导出（需装 `h5py`） |

### 几个值得学的测试设计模式

1. **集成测试优雅降级**
   `tests/integration/conftest.py` 的 `optillm_server` fixture：有服务就用，没有就 `pytest.skip`。
   但 CI 里设了 `OPENEVOLVE_REQUIRE_LLM_SERVER=1`，这时"没有服务"变成**硬失败** —— 保证 CI 不会因为服务挂了而"假装绿"。

   ```python
   def _require_or_skip(reason):
       if os.environ.get(REQUIRE_SERVER_ENV, "").strip().lower() in {"1", "true", "yes"}:
           pytest.fail(reason, pytrace=False)   # CI：硬失败
       pytest.skip(reason)                       # 本地：跳过
   ```

2. **禁止"假测试"**
   `pyproject.toml` 里：

   ```toml
   filterwarnings = ["error::pytest.PytestReturnNotNoneWarning"]
   ```

   测试函数 `return True` 而不是 `assert` 会**直接失败**。这个仓库之前真的有一个这样的"测试"，永远通过、实际什么都没验证。

3. **对着真实逻辑写测试，不要复刻逻辑**
   `tests/test_reasoning_model_params.py` 的做法：mock 掉 `openai.OpenAI` 客户端，捕获传给 `chat.completions.create` 的参数，然后断言真实的 `OpenAILLM` 行为。
   反例（已删除）：在测试里**复制一份**参数选择逻辑再做断言 —— 那样只能证明"测试和代码写得一样"，不能证明代码是对的。

4. **用 TypedDict 给"混合字段字典"定型**
   `database.py` 的 `FeatureStats(TypedDict)`：一个字段是 float，另一个是 `List[float]`。
   写成 `Dict[str, Union[float, List[float]]]` 会让**每个**字段读取都变成联合类型，后续所有算术/索引操作都报错。TypedDict 才能精确表达"标量 + 列表"的混合记录。

5. **用 `cast` 而不是 `# type: ignore`**
   `database.py` 里 `cast(List[float], self.embedding_client.get_embedding(...))`：被调用方缺少 `@overload`，但传 `str` 必然得到一个向量。`cast` 是**有依据的断言**，`# type: ignore` 是**关掉检查**。

---

## 5. 已知陷阱（本轮审计中发现的真实 bug，现已修复）

这一节是"读源码时容易踩的坑"，也是**理解这个项目设计取舍的好材料**。

| 问题 | 症状 | 根因 | 修复 |
|---|---|---|---|
| `LLMModelConfig` 直接用会崩 | 第一次 `generate()` 抛 `TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'` | `retries` 默认 `None`（只有 `LLMConfig` 会填），`range(retries + 1)` 直接炸；而 `__init__` 里对同一个 `None` 已经做了保护，说明是遗漏 | `llm/openai.py` 在 `__init__` 里归一化为 `int` |
| `generate_all_with_context` 类型撒谎 | 调用方把它当 `str` 用，实际是 `list[str]` | 返回注解写成 `-> str`，函数体 `return responses`（列表） | 注解改为 `List[str]` |
| embedding 失败路径返回元组 | 失败时把 `([], 0.0)` 存进数据库，`_cosine_similarity` 的 `if not vec1` 空值保护**失效**（二元组为真），novelty 判断静默退化 | 从 ShinkaEvolve 移植时改了签名，错误分支没跟着改 | 返回 `[]` / `[[]]` |
| novelty LLM 为 `None` 时崩溃 | 开启 embedding 但没有 novelty LLM → `AttributeError` | `novelty_llm: Optional[...]`，配置默认 `None`，worker 里也显式置 `None` | 在 `_llm_judge_novelty` 里加 None 保护，返回"假定新颖" |
| `changes_description` 为 `None` 时崩溃 | 开 `programs_as_changes_description` 且描述为空 → `AttributeError` | `current_changes_description.rstrip()` 没防 `None` | `(current_changes_description or "").rstrip()` |
| 日志器上挂状态 | `logger._initialized_models = set()` | `Logger` 是**进程级单例**，往上挂属性会跨模块泄漏状态 | 改成模块级变量（`_announced_models` / `_ensemble_logged` / `_prompt_sampler_logged`） |
| `EvaluationResult.artifacts` 类型过窄 | `artifacts={"timeout": True}` 类型报错 | 注解写成 `Dict[str, Union[str, bytes]]`，但评测器自己就在存 bool/int，`get_artifact_size` 也有非 str/bytes 分支 | 放宽为 `Dict[str, Any]` |
| 同名局部变量复用 | `parent` / `content` 在一个函数里先后表示两种东西 | 长函数里重名绑定 | 重命名为 `sampled_parent` / `binary_content` |
| `_is_novel` 参数类型写错 | 注解说 `int`，调用方传 `str` | `self.programs` 是 `Dict[str, Program]`，key 是 `str` | 注解改为 `str` |
| CLI 最佳程序为 `None` | 抛 `AttributeError` 被宽 `except` 吞掉，只留 traceback | `run()` 可能返回 `None`，取值前没判空 | 加显式判断，返回退出码 1 和清晰信息 |
| `api.py` 用临时平均算 `best_score` | 超时的程序 `best_score` 报成 **0.5**（`{"timeout": True}` 单独出现时甚至是 1.0） | `bool` 是 `int` 子类，`isinstance(v, (int, float))` 把 `timeout: True` 当成 1.0 分。仓库里 `safe_numeric_average` 早就为此加了 bool 排除（见 `tests/test_boolean_metrics.py` 的注释），但 `api.py` 自己又写了一遍临时平均 | 改用 `safe_numeric_average` |
| `safe_numeric_sum` 统计 bool | 与 `safe_numeric_average` 对同一个 dict 给出不一致的结果 | 漏了 bool 排除 | 补上同样的 bool 判断 |

### 尚未修复（已知问题，留作练习）

以下 3 个是本轮写测试时发现、但**刻意没有改**的问题 —— 都在 `openevolve/api.py`，适合作为"读代码 + 提 PR"的练手题：

| 位置 | 问题 | 最小修复 |
|---|---|---|
| `api.py` 的 `evolve_function` | 传 lambda 进去会从公共 API 里**泄漏出裸的 `StopIteration`**（`next(...)` 没有默认值，而 lambda 的源码里没有 `def ` 行） | 给 `next(...)` 加默认值 `None`，然后抛带说明的 `ValueError` |
| `api.py` 的 `evolve_algorithm` | 用 lambda 当 benchmark 会**静默生成无法编译的 evaluator 代码**（源码里插值出 `metrics = <lambda>(instance)`），报错要等到 worker 加载时 | 复用 `_extract_lambda_source` 做序列化，或者直接拒绝并给出清晰报错 |
| `utils/trace_export_utils.py` | `output_metadata = metadata or {}` 让"是否修改调用方 dict"变得不一致：非空 dict 会被 `setdefault` 改写，空 dict 不会 | 改为 `metadata if metadata is not None else {}`（注意这是**行为变更**，要先确认调用方依赖哪种语义） |

**读源码时的通用建议**：这个项目 `Optional[...]` 用得很多（配置项几乎全是 Optional）。看到 `Optional` 就要问一句"**如果它真的是 None，这里会怎样**" —— 上表一半的 bug 都是这么发现的。

另一个高频陷阱是 **`bool` 是 `int` 的子类**：凡是 `isinstance(x, (int, float))` 出现在"算分数"的地方，都要问一句"如果 x 是布尔标志会怎样"。这个项目里 `timeout: True` 这类标志非常常见。

---

## 6. 练习建议（从易到难）

1. **【易】** 换一个自己的优化问题（比如排序算法、背包问题），写 `initial_program.py` + `evaluator.py`，跑 50 次迭代。
2. **【易】** 关掉 diff-based evolution，对比 prompt 长度和结果质量。
3. **【中】** 给 `evaluator.py` 加 artifact 输出（把中间结果写进 artifacts），观察下一代 LLM 是否利用了它。
4. **【中】** 写一个自定义 `PopulationStrategy`，实现"只接受 generation ≥ 2 的程序"，观察种群多样性和收敛速度。
5. **【中】** 写一个自定义 `IslandSelector`，对比"轮询"、"最高分优先"、"最少负载优先"三种调度策略。
6. **【中】** 接一个新的 LLM 后端（实现 `LLMInterface` 的两个方法即可，参考 `llm/claude_code.py`（162 行））。
7. **【难】** 给 `database.py` 加一个新的 feature dimension 类型（比如基于 embedding 距离的分箱）。
8. **【难】** 用 evolution trace 导出数据，训练一个"预测哪个 LLM 响应会被采纳"的小模型。
9. **【难】** 实现一个"多层次进化"：外层进化提示词，内层进化程序。

---

## 7. 常见问题

**Q: 为什么我的程序完全没进化？**
A: 按顺序检查：① `evaluator.py` 是否返回了 `combined_score`？② `EVOLVE-BLOCK` 标记范围是否覆盖了你真正想改的代码？③ `logger.setLevel(logging.DEBUG)` 看 LLM 到底返回了什么 —— 大部分情况是 diff 格式不对，程序被 `_CandidateRejected` 丢掉了。

**Q: 进化结果每次都不一样？**
A: 设 `random_seed`。它会同时固定 LLM 采样 seed 和 ensemble 的模型选择。

**Q: 怎么省钱？**
A: ① `cascade_evaluation` 一定要开，且阶段 1 要足够便宜；② `diff_based_evolution: true` 输出短得多；③ 选便宜的模型做 stage1；④ 先用 `--iterations 10` 验证流程通了再放大。

**Q: 想读论文从哪里开始？**
A: 先看 `README.md` 的 "How OpenEvolve Works" 和 `openevolve-architecture.png`，再去读 AlphaEvolve 论文。代码里的注释经常直接引用 issue 编号（如 "GitHub issue #246"），顺着能找到设计讨论。

**Q: `mypy` 报错了怎么办？**
A: 先看是不是 `Optional` 没处理（本项目最常见）。**不要**用 `# type: ignore` 关掉检查 —— 那通常意味着你漏了一个真实的 `None` 情形。参考上面的"已知陷阱"表。

---

## 8. 一页速查：文件 → 职责

| 文件 | 行数 | 一句话职责 |
|---|---:|---|
| `database.py` | 2840 | MAP-Elites + 岛屿 + 存档 + 迁移，**最大的文件** |
| `process_parallel.py` | 1113 | 子进程并行执行迭代 |
| `evaluator.py` | 752 | 级联评测、超时、artifacts |
| `prompt/sampler.py` | 751 | 提示词组装 |
| `api.py` | 681 | 库 API（不开 CLI 直接用） |
| `evolution_trace.py` | 626 | (state, action, reward) 轨迹记录 |
| `controller.py` | 579 | 总编排 |
| `config.py` | 547 | 全部配置数据结构 |
| `utils/code_utils.py` | 433 | diff 提取/应用、EVOLVE-BLOCK 解析 |
| `utils/trace_export_utils.py` | 369 | trace 导出（JSON/JSONL/gzip/HDF5） |
| `llm/openai.py` | 342 | OpenAI 兼容后端 + 人工模式 |
| `prompt/templates.py` | 236 | 提示词模板 |
| `utils/async_utils.py` | 227 | 重试/超时工具 |
| `cli.py` | 191 | `openevolve-run` 命令行入口 |
| `llm/copilot_cli.py` | 183 | GitHub Copilot CLI 后端 |
| `utils/metrics_utils.py` | 149 | 指标安全聚合 |
| `llm/claude_code.py` | 162 | Claude Code CLI 后端（**新后端的模板**） |
| `llm/ensemble.py` | 142 | 多模型按权重集成 |
| `embedding.py` | 128 | 文本 embedding（novelty 判断用） |
| `utils/format_utils.py` | 72 | 输出格式化 |
| `evaluation_result.py` | 68 | `EvaluationResult`（metrics + artifacts） |
| `population.py` | 67 | 种群管理策略钩子 |
| `novelty_judge.py` | 43 | LLM 判新颖性 |
| `selection.py` | 27 | 岛屿选择策略钩子（**最小的扩展点**） |
| `llm/base.py` | 22 | `LLMInterface` 抽象基类 |

---

**最后一句建议**：这个项目最值得学的不是"怎么调 LLM"，而是**怎么把一个随机的、会失败的、慢的外部依赖（LLM）包装成一个可靠的搜索算子** —— 重试、超时、级联淘汰、快照隔离、准入拒绝、类型契约。阶段 1 和阶段 4 就是讲这件事的。
