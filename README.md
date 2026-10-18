# NetSleuth

NetSleuth 是一个面向网络故障诊断的 LangGraph Agent。它负责规划诊断步骤、调用工具收集证据、验证根因并提交结果。

> NetSleuth 是独立维护的社区项目，不是 NIKA 官方发行版，也不代表 NIKA 项目或其作者。

## 与 NIKA 的关系

本仓库只包含 NetSleuth Agent，不包含 NIKA 平台。NetSleuth 当前不是完全独立运行的软件：它复用 NIKA 的会话、模型、日志和提交接口，并通过 NIKA 为当前实验会话生成的 MCP 配置获取网络诊断工具。

```text
NIKA：创建场景、注入故障、提供会话和 MCP 工具、评测结果
                              ↓
NetSleuth：规划、调用只读诊断工具、分析证据、验证和提交根因
```

因此，运行 NetSleuth 前需要先安装并启动一个兼容的 NIKA 项目。

## 目录结构

```text
.
├── agent.py
├── phases.py
├── harness.py
├── tool_policy.py
├── graph/
├── prompts/
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── scripts/install_into_nika.py
```

## 嵌入 NIKA

要求：

- Python 3.12 或更高版本；
- 一个使用 `src/agent` 布局的 NIKA checkout；
- NIKA 使用 `uv` 管理依赖；
- 安装脚本所针对的 NIKA registry/config 文件结构与当前版本兼容。

先预览变更：

```bash
python3.12 scripts/install_into_nika.py /path/to/nika --dry-run
```

确认后安装：

```bash
python3.12 scripts/install_into_nika.py /path/to/nika
```

脚本会：

1. 将 Agent 源码复制到 `NIKA/src/agent/community/netsleuth`；
2. 在 NIKA 的 registry、CLI、provider 和 model 配置中注册 `community.netsleuth`；
3. 保留 NIKA 中已有且内容相同的接入项，因此可以重复执行。

如果目标目录已有不同版本，脚本默认停止。确认需要更新时使用：

```bash
python3.12 scripts/install_into_nika.py /path/to/nika --force
```

建议在干净的 NIKA 工作树中执行，并在安装后检查 `git diff`。脚本只负责接入源码，不负责安装或启动 NIKA。

## 在 NIKA 中运行

按照 NIKA 的正常流程创建并启动 session，然后选择 `community.netsleuth`：

```bash
uv run nika agent run \
  -a community.netsleuth \
  -p openai \
  -m gpt-5-mini \
  <SESSION_ID>
```

具体参数以目标 NIKA 版本的帮助为准：

```bash
uv run nika agent run --help
```

NetSleuth 支持 NIKA 为 LangGraph Agent 提供的 `openai`、`anthropic`、`deepseek` 和 `custom` provider。对应凭据和 endpoint 应在 NIKA 中配置，不要提交 `.env` 或 API key。

## 验证接入

在 NIKA 根目录执行：

```bash
uv run pytest tests/agent/test_registry.py tests/agent/test_provider_env.py tests/agent/test_agent_config.py
uv run ruff check src/agent/community/netsleuth
```

最终验证应使用真实 NIKA session，让 NetSleuth 通过 MCP 工具诊断一个实际场景，并确认 session 由 NIKA 正常清理。

## Citation

如果在学术研究中使用 NetSleuth 和 NIKA，请引用 NIKA 项目要求的论文：

```bibtex
@misc{nika25long,
  title          = {A Network Arena for Benchmarking AI Agents on Network Troubleshooting},
  author         = {Zhihao Wang and Alessandro Cornacchia and Alessio Sacco and Franco Galante and Marco Canini and Dingde Jiang},
  year           = {2025},
  eprint         = {2512.16381},
  archivePrefix  = {arXiv},
  primaryClass   = {cs.NI},
  url            = {https://arxiv.org/abs/2512.16381}
}

@inproceedings{nika25ngno,
  title        = {Towards a Playground to Democratize Experimentation and Benchmarking of AI Agents for Network Troubleshooting},
  author       = {Wang, Zhihao and Cornacchia, Alessandro and Galante, Franco and Centofanti, Carlo and Sacco, Alessio and Jiang, Dingde},
  year         = {2025},
  publisher    = {Association for Computing Machinery},
  url          = {https://doi.org/10.1145/3748496.3748990},
  booktitle    = {Proceedings of the 1st Workshop on Next-Generation Network Observability},
  location     = {Coimbra, Portugal},
  series       = {NGNO '25}
}
```

## License 与第三方声明

NetSleuth 以 [MIT License](LICENSE) 发布，第三方项目说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。NIKA 官方仓库声明 NIKA 使用 MIT License；由于其仓库目前没有单独的完整许可证文件，本仓库不替 NIKA 作者补写或推断版权主体。

## 发布为不带历史的新 GitHub 仓库

不要在 NIKA 仓库内直接执行 `git init`。把本目录复制到 NIKA 之外的新目录，并排除父项目的 Git 元数据和运行产物：

```bash
mkdir -p /path/to/standalone-netsleuth
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  /path/to/nika/src/agent/community/netsleuth/ \
  /path/to/standalone-netsleuth/

cd /path/to/standalone-netsleuth
git init -b main
git add .
git commit -m "Initial import of NetSleuth"
git remote add origin git@github.com:<YOUR_ACCOUNT>/<YOUR_REPOSITORY>.git
git push -u origin main
```

这样生成的是一个只有首次导入提交的新仓库，不包含 NIKA 的 Git 历史。独立仓库会包含 NetSleuth 的 `LICENSE` 和 NIKA 的第三方声明；如果 NIKA 后续补充或修改正式许可证及版权声明，应同步更新本仓库中的相关说明。

在首次提交前检查待发布内容，确认没有密钥、实验结果或 NIKA 的 Git 元数据：

```bash
git status --short
git ls-files
git log --oneline
```

其中 `git log --oneline` 在首次提交后应只有一条 `Initial import of NetSleuth` 记录。

## 安全边界

诊断阶段会对 MCP 命令工具应用只读策略，阻止明显的修复、配置变更和 benchmark 答案读取。提交阶段则使用 NIKA 的正式 submission MCP 流程。该策略是 Agent 层的防护，不应替代 NIKA 自身的权限隔离和两阶段工具网关。
