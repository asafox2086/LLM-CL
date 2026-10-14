# 服务器迁移与旧资源清理（2026-10-14）

当前工作服务器：`liyapeng@100.87.141.22:8666`（Tailscale），仓库 `/data1/liyapeng/program/LLMCL`。

原仓库约 177 GiB，包括 Git 历史、未提交文件、论文、源码、数据、模型与旧实验输出，已经复制到新服务器并通过 rsync 内容校验；外部 MM 与数据工具环境也已迁移。旧环境运行中的 `tailscaled.sock` 和 `tailscaled.log` 单独排除静态校验。随后修复绝对链接和 Python 入口，CUDA、PyTorch/Transformers、pyarrow、matplotlib 及已知 AP/ForgetRate/FWT/BWT 测试通过。

用户授权后，删除旧仓库 `/data2/liyapeng_grp/program/LLMCL` 和已迁移的数据工具环境 `/tmp/llmcl-assets-env`。清理前检查源目录无校验后修改、无目录内挂载、无正在写入源文件的进程；清理不跟随外部符号链接。磁盘可用空间观测增加约 176.6 GiB（共享磁盘，含同期其他进程影响）。

保留资源：

- 尚未整包迁移的约 95 GiB GeoChat 原始分卷：`/var/tmp/liyapeng_grp_LLMCL_assets/GeoChat_Instruct/raw`。
- GeoChat 原始元数据已从旧仓库另存：`/data2/liyapeng_grp/.llmcl-migration/retained_geochat_metadata`，5 个文件逐一 SHA-256 核验一致。
- 旧端 `/data2/liyapeng_grp/.conda/envs/MM` 仍用于运行 Tailscale；本轮不删除，避免旧端保留数据失联。
- 清理审计保存在 [source_cleanup_audit.json](source_cleanup_audit.json)；迁移前分数、状态在 [历史归档](../ago/old_server_before_migration_20261014/ARCHIVE_NOTE.md)。

新服务器用户级 Tailscale 服务已启用，Linger=yes。临时迁移 SSH 授权和传输私钥已删除；正常用户 SSH 和 GitHub 推送密钥保留。正式训练仍关闭，等待 T5/T6 地理／遥感及 T8 金融图表数据核验，不将旧任务结果标记为新任务实验结果。
