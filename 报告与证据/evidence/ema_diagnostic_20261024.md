# EMA 对照追加记录

日期：2026-10-24。已从 WSL 运行目录归档 12 个原始文件至 [EMA 对照目录](ema_comparison_20261024/summary.json)，逐个核对源文件与副本的 SHA256。六份逐题 JSON 均 complete=true、各含 128 条记录；其 correct 字段合计与 summary.json 一致。此核对没有重新执行评分器。EMA 源码和更新计数检查、语法分类统计仍依据运行时的终端输出。

## 设置与结果

- checkpoint：`outputs/sft_sepfix_20261022_171041_876248/checkpoints/best-step000400.ckpt`
- 输出目录：`/home/chen/projects/s-flm/outputs/sft_ema_vs_raw_20261024_165836_733187`
- 脚本：`scripts/local/check_sft_ema_vs_raw.py`
- 数据：既有 128 道 RL 验证题；种子 202、303、404；每种权重共 384 次生成。
- 固定长度 512、采样 32 步、batch size 1；无训练，未使用测试集。
- EMA 正确数：9、8、9，累计 26；普通权重：4、5、3，累计 12。
- 普通权重相对 EMA 得失：各种子 gained/lost 分别为 2/7、2/5、2/8。
- 原始文本语法通过：EMA 203，普通权重 156。
- 包含预期函数名字符串：两者均 384。
- 语法通过但评分错误：EMA 177，普通权重 144。
- 耗时 235.7999885519997 秒；峰值 allocated 显存 1.6876554489135742 GiB。
- 对照与后续语法统计均报告 exit code 0。

## EMA 状态检查

原始、第 400 步和第 1000 步 checkpoint 的 `num_updates` 分别为 250000、250400、251000，`decay` 均为 0.9999。源码的有效衰减公式在该计数范围内取 0.9999；初始 EMA 系数在 400／1000 步后约为 96.08%／90.48%，不等于知识保留比例。

## 结论边界

该固定 checkpoint 上，EMA 三个种子均优于普通权重；不支持直接停用 EMA 改善表现。没有测试其他衰减系数、其他 checkpoint 或重新训练，未证明最优系数及退化原因。384 次生成复用 128 道问题，不是独立的 384 道题。

## 归档范围与剩余边界

已复制并校验 `protocol.json`、`weight_check.json`、`config.yaml`、`paired_results.json`、`summary.json`、六份逐题 JSON 和运行目录内实际执行脚本。`weight_check.json` 记录两套参数最大绝对差为 0.13535380363464355。未复制运行目录外的终端日志；语法诊断目前只有运行时的终端输出，未在归档时重新统计。

