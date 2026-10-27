# 两条10% SeqFT资源测量：故障终态与环境边界

本轮只读本地原worker、queue及已取回的故障收据；没有服务器访问、源码/队列修改、重新加载或测量。

原测量没有产出资源数。OneVision worker在2.735200秒后exit1：`runtime_utils.py`导入`PIL.Image`失败。原worker已执行`import torch`和`torch.cuda.set_device(0)`，随后在导入checkpoints/runtime_utils时失败；`checkpoints.load`、完整CP载入及所有generation都尚未到达。因此是0模型加载、0生成/评分，不能写0CUDA初始化。Qwen phase没有STARTED，两个骨干的五窗GPU峰、TTFT、速度仍均缺测；不能把环境故障当模型质量或资源负结果。

依据：OneVision原log SHA `0f3887ff00543571e3f0cc5d8056fe93692ff7aaaa1847e2bff557a3c065602d`；SUPERVISOR_FAILURE SHA `13533134b412f45723277c1c39284cc9bfa25101d7c86cd92199048b2eb79415`。原slot是`failed_closed/released=true`。保留原RUN、slot/request、日志、STARTED/EXIT/FAILURE及费用，不改为成功或清空后重用。

补齐正式基线资源指标有必要，可另立明确披露来源的新测量输出；它不是重开方法分支，也不改变1%/10%训练。固定原S19 CP/配置/完整未fold图、原五个hash-min raw QA、每题32新token、同卡串行次序及原worker/helpers/计时定义，都能沿用；不重训、不重建CP、不选择F1。但“仅换`.vision-env`即可”尚不能确认：`validate_metadata`第118–122行还核原environment/runtime_files及torch、transformers、accelerate、cloudpickle版本，解释器已用过不等于与该绑定相同。

后续最少必要检查：先在拟用环境核这三个来源集合与原绑定，并确认PIL及原完整导入链可用（不加载模型、不用checkpoint_probe）；记录实际解释器、Pillow/处理器依赖与相关包版本。若版本/导入源不一致，应明确新环境差异并据事实审阅，不能静默放松旧断言。Pillow/处理器差异可能改变预处理，原图字节相同本身不能保证prepared tensor相同；仍须保原输入/预处理合同及实际prepared摘要。

原worker峰值从input preparation之前reset至generation后捕获，含视觉/decoder；文件校验在计时前，张量审计摘要在计时和峰值后。TTFT从prepare前至首token；速率为31/(末token时刻−首token时刻)。有限logit检查、逐token同步仍计时，属于instrumented五个串行窗口，不是生产吞吐、全TEST峰或每题独立冷启动。切换环境不授权改这些定义。新测量应显式保留旧失败成本和独立输出/预算收据，不能沿旧closed slot自动重试；仍由原分配器在type研究之后的合法空卡机会安排。本轮不建立、排队或执行该新测量。

审查源：worker `aedc7e7b97d4f5be22a98d022c758d3c3c285dbc805e087b2a952c09de302400`；queue `6a833c438c335109cf000a64f7ef4f004887625a1efaf3aa54e712c79f3f4a2b`。作者曾编写这两个原文件，本次是故障/合同复核，不冒称新的独立源码作者审查。
