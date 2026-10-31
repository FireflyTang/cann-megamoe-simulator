# 复现、精简回传与验收

源码真值为本地 `~/sources/ops-transformer-sim`。初始迁移基线是 `b8bb9a7292c7c7635977f3862f13e80999cdbe7e`，每次以该 run 的 `reproduce/snapshot/base_commit.txt` 为准。CANN、当前用户 tang、第三方依赖和 Git 基线是外部前提，不把只含结果的包描述为无环境依赖的完整安装包。

## 三种不同的复现

1. **重新查看、校验、生成阶段流水**：使用本地实际输出、原始 DB、已归档反汇编和实际分析脚本；不访问远端、不仿真。
2. **同一二进制重新仿真**：从已保存 ELF、tiling、fixture 配置和脚本重建 case，再为它创建新的运行目录及任务；属于新一轮采集，不覆盖旧结果。
3. **恢复源码并重新编译**：恢复 Git 基线、补丁、未跟踪源码和冻结脚本，使用记录的依赖与 CANN。不同工具版本不保证产物/时间戳位级一致。

## 离线重建

在本地加载环境并选定已有 run 和一个不存在的新输出目录：

```bash
cd ~/sources/ops-transformer-sim
source ~/Ascend/local-build-env.sh
RUN=/绝对路径/已有run
OUT=/绝对路径/新的离线复现目录
~/miniforge3/envs/tang/bin/python megamoe_profile/replay.py --run "$RUN" --output "$OUT"
```

该入口用保存的 ELF/tiling 与配置重建输入，再对保存的实际输出执行原数值/计数检查，并从原始记录重建阶段流水。原 run 不覆盖。具体证据路径和结果见新目录 README；不能用新 golden 代替实际输出。

原生 Insight 报告保留一份可直接查看的结果。省略的 msprof dump 可从 `chip*_instr.bin` 用已记录版本的 `export_msprof.py` 离线重导出；需要同版远端工具时只上传这些必要文件进行导出，不重跑 kernel。本地 beta.2 不具备所有 weekly 解码能力，因此不宣称任意机器都能独立重新导出原生报告。

## 源码恢复

以 run 内自动生成的 `reproduce/snapshot/RESTORE.md` 为准，步骤为：在含基线 commit 的 Git 仓库建立新 worktree，应用 `changes.patch`，还原 `snapshot/untracked/`，再把 `reproduce/scripts/` 的实际脚本/头覆盖至新 worktree 的 `megamoe_profile/`。恢复过程不修改当前开发目录，也不要求 git add。

配置中的解释器、CANN、依赖绝对路径按新机器调整并保留新配置；不改 shape、dtype、算法或精度阈值来凑成功。依赖版本见 `dependencies.json`，构建命令和生成源码兼容补丁保留在日志及 snapshot 中。新目录可用：

```bash
~/miniforge3/envs/tang/bin/python megamoe_profile/replay.py \
  --run "$RUN" --output "$OUT" --source /绝对路径/恢复的源码 --build
```

该模式只重新编译并生成 case，终态为 `prepared`；新内核的执行与数值结论需要使用 `run_remote.py` 创建新的仿真任务。若解释器或记录依赖不存在，报告缺失项，不借用他人环境。

## 传输清单

| 内容 | 保留/传输原则 |
|---|---|
| 源码、第三方依赖、构建缓存 | 留本地；按 run 保存基线/补丁/必要未跟踪输入，不整仓上传 |
| 实际 ELF、tiling、case、执行脚本、配置 | 本地归档；仅上传运行需要的部分；回传保留实际执行脚本来源 |
| 实际输出与专家计数 | 保留，以便事后逐元素重检，不用 SHA 代替数据 |
| 原始 helper DB、instr.bin、捕获元数据 | 保留，支持重新分析与原生导出 |
| 原生 msprof/npusim 报告 | 保留可查看的报告；相同 msprof 树只传一次 |
| msprof dump 文本 | compact 不传；可从 instr.bin 重建 |
| 日志、退出码、终态、环境、反汇编及哈希 | 保留；成功时省略重复的模型调试 dump，失败时保留诊断 |

`result_compact_manifest.json` 记录逐文件 SHA 与保留策略，`result_compact_ready.json` 记录包名、SHA、大小及退出码。原 full 包与 ready 不被旧任务的精简整理覆盖。

既有已完成任务可在 model1 使用当前 `pack_result.py --run REMOTE_RUN --mode compact` 离线打包；新任务默认产出 compact。然后在本地：

```bash
~/miniforge3/envs/tang/bin/python megamoe_profile/run_remote.py collect \
  --run "$RUN" --result-format compact
```

`auto` 优先 compact 并兼容旧 full；明确需要完整旧包时选 `full`。重复 collect 不提交或仿真。旧采集缺少反汇编时，从同一 ELF 离线补导出，记工具与 ELF/文本哈希；更换分析脚本前保存原件、替换清单和原因。不得把补导出说成当初执行时已生成。

## README 与最终验收

每次 run 的 README 至少包含：背景及入口；参数/每 rank shape；源码/环境/兼容处理；实际数值、计数、打点结果；文件索引及精简清单；离线重建、源码恢复、新任务执行方法。

完整归档不是只留最终 JSON：还须保留实际配置、脚本、ELF/tiling、源码增量、原始测量、实际输出及终态。校验结果与运行退出码分开记录；失败不能改写为通过。

先验证传输包/逐文件 SHA，再校验数值、DFX 配对、原生报告存在且可解析；最后用新目录 replay 验证材料足够重建。已有原始包、现场和本地报告不因“少传输”自动删除。
